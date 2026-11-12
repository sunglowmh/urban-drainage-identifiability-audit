# -*- coding: utf-8 -*-
"""关口 8 (prereg v1.9 A28): 锚点分段复验 + 簇级 bootstrap + ev08/边界堆积归因孪生 + E2 trade-off
用法:
  python p2h_gate8.py --seg    # A28-1 锚点前后两段复验 (稳健性, I3)
  python p2h_gate8.py --boot   # A28-2 K=7 簇 bootstrap B=1000 (+K=6 E1 敏感性)
  python p2h_gate8.py --twin   # A28-3 归因孪生 (θ*=先验中点; ev08 vs ev01 regime; Q1/Q2 预登记)
  python p2h_gate8.py --bias   # A28-4 E2 偏置吸收量与 n 位移的跨事件关联
  python p2h_gate8.py --report # 汇总 gate8_report_v2.md
前置: p2f_raincmp --corr 已完成; runs_gate8/{ev08,ev01}_truth 完成 (--twin)
判级: A25; 措辞纪律: 分段/bootstrap=稳健性检验(I3), 孪生=机制归因证据(conditional on DGP)
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_xev_posterior import load_event_runs
from p2b_realdata_v3 import (fit_gp, phi_from_residuals, invert_real, ident_flags)
from p2f_raincmp import (theta90, alignment, load_runs_from, CORR_ROOT, OUT)
from p2_posterior_matrixA import recon_series, zoom_eval, posterior_diag, TAU_GRID
from p2_likelihoods_v3 import DiagLikV3
from p2f_raincmp import s2_star_e1

OUTG = os.path.join(p2lib.P2, "gate8")
G8 = os.path.join(p2lib.P2, "runs_gate8")
CORR_EVENTS = ["ev00", "ev01", "ev04", "ev05", "ev06", "ev07", "ev08"]
TAU_B = 0.10
SIGMA_EFF = 0.050


# ---------------- 公共: 锚点修正臂管线一次性加载 ----------------
def load_anchor():
    X = theta90()
    tidx, cols = alignment("ev00")
    Y = load_runs_from(os.path.join(CORR_ROOT, "ev00"), tidx, cols)
    gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
    xlim = (X.min(0), X.max(0))
    O, M0 = p2lib.load_obs_matrix(tidx, cols)
    M, qc = p2lib.qc_stations(O, M0, Y.mean(0).T)
    W = np.load(os.path.join(OUT, "wmap_ev00_corr.npy"))
    return {"X": X, "tidx": tidx, "cols": cols, "Y": Y, "gps": gps, "Vt": Vt,
            "Ym": Ym, "Ys": Ys, "n_pc": n_pc, "Zmean": Zmean, "xlim": xlim,
            "O": O, "M": M, "qc": qc, "W": W, "T": Y.shape[1], "n_st": Y.shape[2]}


def invert_seg(model, O_seg, M_seg, W_seg, phi_s, ctx, t0, t1, N_eff):
    """分段反演: 全窗 recon 后切片 (GP 为全窗响应面)"""
    T_seg = t1 - t0
    lik = DiagLikV3(W_seg, M_seg, T_seg, O_seg.shape[0], TAU_GRID)
    lik.set_phi(phi_s)
    n_st = ctx["n_st"]

    def ev(grid):
        Yg = recon_series(grid, ctx["gps"], ctx["Vt"], ctx["Zmean"], ctx["Ys"],
                          ctx["Ym"], ctx["n_pc"], ctx["T"], n_st,
                          xlim=ctx["xlim"]).astype(np.float64)
        e = (O_seg.T[None] - Yg[:, t0:t1, :])
        e = np.where(M_seg.T[None], e, 0.0)
        s2_hi = max((e[0] ** 2).sum() / N_eff, 1e-6)
        s2_grid = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
        return {"E0": lik.e0_ll, "E1": lik.e1_ll, "E2": lik.e2_ll}[model](e, s2_grid)

    ll, grid, bnd, resolved = zoom_eval(ev)
    diag, _, _, support = posterior_diag(ll, grid)
    return {"R_j": [d["R_j"] for d in diag], "sd": [d["sd"] for d in diag],
            "mean": [d["mean"] for d in diag], "info": [d["info_nats"] for d in diag]}


def gate8_seg(ctx):
    res = {}
    T = ctx["T"]
    for name, (t0, t1) in {"seg1": (0, T // 2), "seg2": (T // 2, T)}.items():
        O_s = ctx["O"][:, t0:t1]
        M_s = ctx["M"][:, t0:t1]
        W_s = ctx["W"][t0:t1, :]
        Yb_s = ctx["Y"].mean(0).T[:, t0:t1]
        phi_s = phi_from_residuals(O_s, M_s, Yb_s)
        n_eff = int(M_s.sum())
        for m in ("E0", "E1", "E2"):
            r = invert_seg(m, O_s, M_s, W_s, phi_s, ctx, t0, t1, n_eff)
            res[f"{name}_{m}"] = r
            print(f"{name}/{m}: R_j={[round(x,3) for x in r['R_j']]} "
                  f"mean={[round(x,4) for x in r['mean']]}", flush=True)
    json.dump(res, open(os.path.join(OUTG, "seg_results.json"), "w"), indent=1, default=float)
    print("seg done", flush=True)


# ---------------- A28-2 簇 bootstrap (固定网格预计算加速) ----------------
def _fixed_grid_ctx(ctx, model):
    """全站 corr 反演一次 → zoom 网格 + 预计算 Yg (首行=先验最低角, s2_hi 参照)"""
    O, M, W, phi = ctx["O"], ctx["M"], ctx["W"], phi_from_residuals(
        ctx["O"], ctx["M"], ctx["Y"].mean(0).T)
    lik = DiagLikV3(W, M, ctx["T"], ctx["n_st"], TAU_GRID)
    lik.set_phi(phi)
    N_eff = int(M.sum())

    def ev(grid):
        Yg = recon_series(grid, ctx["gps"], ctx["Vt"], ctx["Zmean"], ctx["Ys"],
                          ctx["Ym"], ctx["n_pc"], ctx["T"], ctx["n_st"],
                          xlim=ctx["xlim"]).astype(np.float64)
        e = (O.T[None] - Yg)
        e = np.where(M.T[None], e, 0.0)
        s2_hi = max((e[0] ** 2).sum() / N_eff, 1e-6)
        s2_grid = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
        return {"E0": lik.e0_ll, "E1": lik.e1_ll, "E2": lik.e2_ll}[model](e, s2_grid)

    ll, grid, bnd, resolved = zoom_eval(ev)
    LO = 0.009
    Yg = recon_series(grid, ctx["gps"], ctx["Vt"], ctx["Zmean"], ctx["Ys"],
                      ctx["Ym"], ctx["n_pc"], ctx["T"], ctx["n_st"],
                      xlim=ctx["xlim"]).astype(np.float64)
    Ycorner = recon_series(np.array([[LO, LO, LO]]), ctx["gps"], ctx["Vt"], ctx["Zmean"],
                           ctx["Ys"], ctx["Ym"], ctx["n_pc"], ctx["T"], ctx["n_st"],
                           xlim=ctx["xlim"])[0].astype(np.float64)
    return {"grid": grid, "ll_full": ll, "Yg": Yg, "Ycorner": Ycorner, "phi_full": phi}


def _boot_ll_fast(model, e_all, M_sub, W_sub, phi_sub, tau_grid, s2_grid):
    """bootstrap 快路径: 逐站批量归约, 无 3D 临时数组 (数学上与 DiagLikV3 完全等价).
    e_all: (n_rows, T, n_sub); M_sub/W_sub: (n_sub,T)/(T,n_sub); 返回 ll (n_rows,)"""
    MT = M_sub.T.copy()                       # (T, n_sub)
    n_rows, T, n_sub = e_all.shape
    A = e_all ** 2                            # (n_rows,T,n_sub) 一次性
    best = np.full(n_rows, -np.inf)
    idxs = [np.where(MT[:, i])[0] for i in range(n_sub)]
    ab_info = []
    for i in range(n_sub):
        idx = idxs[i]
        if len(idx) < 5:
            ab_info.append(None)
            continue
        p = max(float(phi_sub[i]), 1e-3)
        ab = ar_bands_local(p, len(idx))
        ab_info.append((idx, ab, (len(idx) - 1) * np.log(1 - p * p)))
    for s2 in s2_grid:
        Vm = np.where(MT, W_sub + s2, 1.0)    # (T,n_sub)
        ldet0 = (np.log(2 * np.pi * Vm) * MT).sum()
        g = 1.0 / np.sqrt(Vm)                 # (T,n_sub)
        if model == "E0":
            ll = -0.5 * (ldet0 + (A / Vm[None] * MT[None]).sum((1, 2)))
            best = np.maximum(best, ll)
            continue
        quad = np.zeros(n_rows)
        ldet = ldet0
        c_vQw = np.zeros((n_rows, n_sub))
        a_wQw = np.zeros(n_sub)
        for i in range(n_sub):
            info = ab_info[i]
            if info is None:
                continue
            idx, ab, ld_ar = info
            d, off = ab[1], ab[0, 1:]
            ei = e_all[:, idx, i]             # (n_rows, n_i)
            gi = g[idx, i]                    # (n_i,)
            v = ei * gi[None]                 # (n_rows, n_i)
            # quad_i = Σ_t d_t v_t² + 2 Σ_t off_t v_t v_{t+1}  (d/off 逐 t)
            quad += (v * v * d[None, :]).sum(1) \
                + 2.0 * (v[:, :-1] * v[:, 1:] * off[None, :]).sum(1)
            ldet += ld_ar
            if model == "E2":
                wi = gi
                Qwi = d * wi
                Qwi[:-1] += off * wi[1:]
                Qwi[1:] += off * wi[:-1]
                a_wQw[i] = float(wi @ Qwi)
                c_vQw[:, i] = v @ Qwi         # vᵀQw = Σ_t v_t·Qw_t, v=e·g
        if model == "E1":
            best = np.maximum(best, -0.5 * (ldet + quad))
        else:
            for tau in tau_grid:
                t2 = tau ** 2
                q = quad - (t2 * c_vQw ** 2 / (1 + t2 * a_wQw[None, :])).sum(1)
                ld = ldet + np.log(1 + t2 * a_wQw).sum()
                best = np.maximum(best, -0.5 * (ld + q))
    return best


def ar_bands_local(p, n):
    c = 1.0 / (1 - p * p)
    d = np.full(n, (1 + p * p) * c)
    d[0] = d[-1] = c
    off = np.full(n - 1, -p * c)
    ab = np.zeros((3, n))
    ab[0, 1:] = off
    ab[1, :] = d
    ab[2, :-1] = off
    return ab


def _boot_invert_fast(fg, ctx, si, model, tau_grid=TAU_GRID):
    """快路径子集反演: 预计算 e 全批 → _boot_ll_fast → posterior_diag"""
    O, W, Yb = ctx["O"], ctx["W"], ctx["Y"].mean(0).T
    M_sub = ctx["M"][si]
    W_sub = W[:, si]
    O_sub = O[si]
    phi_s = phi_from_residuals(O_sub, M_sub, Yb[si])
    Yg_all = np.concatenate([fg["Ycorner"][None], fg["Yg"]], axis=0)
    e = (O_sub.T[None] - Yg_all[:, :, si])
    e = np.where(M_sub.T[None], e, 0.0)
    s2_hi = max((e[0] ** 2).sum() / int(M_sub.sum()), 1e-6)
    s2_grid = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
    ll = _boot_ll_fast(model, e[1:], M_sub, W_sub, phi_s, tau_grid, s2_grid)
    diag, _, _, _ = posterior_diag(ll, fg["grid"])
    return {"R_j": [d["R_j"] for d in diag], "mean": [d["mean"] for d in diag],
            "sd": [d["sd"] for d in diag]}


def _boot_invert(fg, ctx, si, model, rng=None):
    """固定网格上的子集反演 (e 全批; 仅似然层变化)"""
    O, W, Yb = ctx["O"], ctx["W"], ctx["Y"].mean(0).T
    M_sub = ctx["M"][si]
    W_sub = W[:, si]
    O_sub = O[si]
    Yb_sub = Yb[si]
    T, n_sub = ctx["T"], len(si)
    phi_s = phi_from_residuals(O_sub, M_sub, Yb_sub)
    lik = DiagLikV3(W_sub, M_sub, T, n_sub, TAU_GRID)
    lik.set_phi(phi_s)
    # e: (n_grid, T, n_sub) — 首行用最低角
    Yg_all = np.concatenate([fg["Ycorner"][None], fg["Yg"]], axis=0)  # (1+n_g, T, n_st)
    e = (O_sub.T[None] - Yg_all[:, :, si])
    e = np.where(M_sub.T[None], e, 0.0)
    s2_hi = max((e[0] ** 2).sum() / int(M_sub.sum()), 1e-6)
    s2_grid = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
    ll = {"E0": lik.e0_ll, "E1": lik.e1_ll, "E2": lik.e2_ll}[model](e[1:], s2_grid)
    # 最低角行单独一个点 (用于 s2_hi 参照一致性; ll 网格仍用 fg grid)
    diag, _, _, support = posterior_diag(ll, fg["grid"])
    return {"R_j": [d["R_j"] for d in diag], "mean": [d["mean"] for d in diag],
            "sd": [d["sd"] for d in diag]}


def gate8_boot(ctx, B_E1=500, B_E0E2=150, seed=20261110):
    """B 结构 (偏离 A28-2 B=1000 的登记理由): 固定网格 501 点 × 8 s2 × AR 逐站似然
    实测 ~3.1 s/反演 → B=1000×3 模型×2 集不可行; E1 主统计量 B=500, E0/E2 B=150。
    E1 百分位 CI 在 B=500 下 MC 误差 ~2.2% (R_j 尺度 0.5), 稳健性检验可接受。"""
    cl = json.load(open(os.path.join(p2lib.P2, "station_clusters.json")))
    clusters = [np.array(v) for v in cl["clusters"].values()]
    node_order = list(cl["node_order"])
    idx_of = {n: i for i, n in enumerate(node_order)}
    cl_idx = [np.array([idx_of[n] for n in c]) for c in clusters]
    singleton = [i for i, c in enumerate(cl_idx) if len(c) == 1]
    rng = np.random.default_rng(seed)
    fgs = {m: _fixed_grid_ctx(ctx, m) for m in ("E0", "E1", "E2")}
    ref = {}
    for m in ("E0", "E1", "E2"):
        diag, _, _, _ = posterior_diag(fgs[m]["ll_full"], fgs[m]["grid"])
        ref[m] = {"R_j": [d["R_j"] for d in diag], "mean": [d["mean"] for d in diag]}
    # ---- 快路径正确性校验 (对 1 个 bootstrap 样本, 与慢路径逐点对照) ----
    rng_chk = np.random.default_rng(seed + 1)
    pick = [cl_idx[rng_chk.integers(len(cl_idx))] for _ in range(len(cl_idx))]
    si_chk = np.unique(np.concatenate(pick))
    O, W, Yb = ctx["O"], ctx["W"], ctx["Y"].mean(0).T
    M_sub = ctx["M"][si_chk]
    W_sub = W[:, si_chk]
    O_sub = O[si_chk]
    phi_s = phi_from_residuals(O_sub, M_sub, Yb[si_chk])
    for m in ("E0", "E1", "E2"):
        Yg_all = np.concatenate([fgs[m]["Ycorner"][None], fgs[m]["Yg"]], axis=0)
        e = (O_sub.T[None] - Yg_all[:, :, si_chk])
        e = np.where(M_sub.T[None], e, 0.0)
        s2_hi = max((e[0] ** 2).sum() / int(M_sub.sum()), 1e-6)
        sg = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
        lik_slow = DiagLikV3(W_sub, M_sub, ctx["T"], len(si_chk), TAU_GRID)
        lik_slow.set_phi(phi_s)
        ll_slow = {"E0": lik_slow.e0_ll, "E1": lik_slow.e1_ll, "E2": lik_slow.e2_ll}[m](e[1:], sg)
        ll_fast = _boot_ll_fast(m, e[1:], M_sub, W_sub, phi_s, TAU_GRID, sg)
        rel = float(np.max(np.abs(ll_slow - ll_fast) / (np.abs(ll_slow) + 1e-9)))
        print(f"  fast-path check {m}: max rel diff = {rel:.2e}", flush=True)
        assert rel < 1e-8, f"快路径与慢路径不一致 ({m}: {rel})"
    print("fast-path 校验通过 ✓", flush=True)
    out = {}
    for label, use_clusters in (("K7", cl_idx),
                                ("K6", [c for i, c in enumerate(cl_idx) if i not in singleton])):
        n_cl = len(use_clusters)
        plan = {"E1": B_E1, "E0": B_E0E2, "E2": B_E0E2} if label == "K7" else {"E1": B_E1}
        samples = {m: {"R_j": [], "mean": []} for m in plan}
        Bmax = max(plan.values())
        for b in range(Bmax):
            pick = [use_clusters[rng.integers(n_cl)] for _ in range(n_cl)]
            si = np.unique(np.concatenate(pick))
            for m in list(samples):
                if b >= plan[m]:
                    continue
                r = _boot_invert_fast(fgs[m], ctx, si, m)
                samples[m]["R_j"].append(r["R_j"])
                samples[m]["mean"].append(r["mean"])
            if (b + 1) % 100 == 0:
                print(f"  [{label}] {b+1}", flush=True)
        out[label] = {m: {"R_j": np.array(v["R_j"]), "mean": np.array(v["mean"])} for m, v in samples.items()}
        for m, v in out[label].items():
            q = lambda a, qs: [float(np.percentile(a, q)) for q in qs]
            print(f"[{label}/{m}] n={len(v['R_j'])} R_j n1 CI={q(v['R_j'][:,0],[2.5,97.5])} "
                  f"n2={q(v['R_j'][:,1],[2.5,97.5])} n3={q(v['R_j'][:,2],[2.5,97.5])}", flush=True)
    np.savez(os.path.join(OUTG, "boot_results.npz"),
             **{f"{lb}_{m}_Rj": v["R_j"] for lb, d in out.items() for m, v in d.items()},
             **{f"{lb}_{m}_mean": v["mean"] for lb, d in out.items() for m, v in d.items()},
             ref=json.dumps(ref))
    print("boot done", flush=True)


# ---------------- A28-3 归因孪生 ----------------
def innov_draw(Yt, M, sig_i, phi_gen, rng):
    """共享创新项 DGP (A16 同构, p2a_twin_posterior innov_draw 同法)"""
    xi = rng.normal(0, 1, Yt.shape)
    bias = rng.normal(0, TAU_B, Yt.shape[0])
    out = {}
    for kind in ("E0", "E1", "E2"):
        O = Yt.copy()
        for i in range(Yt.shape[0]):
            ok = M[i]
            if ok.sum() == 0:
                continue
            d90 = np.nanpercentile(Yt[i, ok], 90) + 1e-6
            h = 0.1 + 0.9 * np.clip(Yt[i, ok] / d90, 0, 2)
            x = xi[i, ok]
            if kind == "E0":
                e = sig_i[i] * x
            else:
                e1 = h * sig_i[i] * x
                a = np.empty(len(x))
                a[0] = e1[0]
                for t in range(1, len(x)):
                    a[t] = phi_gen * a[t - 1] + np.sqrt(max(1 - phi_gen ** 2, 0.05)) * e1[t]
                e = a if kind == "E1" else a + bias[i]
            O[i, ok] = Yt[i, ok] + e
        out[kind] = O
    return out


def gate8_twin(ctx_by_event, seed=20261110):
    """θ*=[0.0155]×3 归因孪生; ev08 vs ev01 regime; Q1/Q2 预登记判定"""
    th_star = [0.0155, 0.0155, 0.0155]
    LO, HI = 0.009, 0.022
    raw = {}
    for eid in ("ev08", "ev01"):
        ctx = ctx_by_event[eid]
        truth_dir = os.path.join(G8, f"{eid}_truth")
        yt = pd.read_csv(os.path.join(truth_dir, "series.csv"), index_col=0, parse_dates=True)
        try:
            yt.index = pd.to_datetime(yt.index, utc=True)
        except (TypeError, ValueError):
            pass
        Yt = yt.reindex(index=ctx["tidx"])[ctx["cols"]].to_numpy(dtype=np.float64).T  # (n_st, T)
        assert not np.isnan(Yt).any()
        O, M = ctx["O"], ctx["M"]
        Ybase = ctx["Y"].mean(0).T
        sig_i = np.array([np.std((O[i] - Ybase[i])[M[i]]) if M[i].sum() > 30 else SIGMA_EFF
                          for i in range(ctx["n_st"])])
        phi_ev = phi_from_residuals(O, M, Ybase)
        phi_gen = float(np.median(phi_ev))
        rng = np.random.default_rng(seed)
        draws = innov_draw(Yt, M, sig_i, phi_gen, rng)
        raw[eid] = {"phi_gen": phi_gen, "sig_i_med": float(np.median(sig_i)), "cases": {}}
        for g in ("E0", "E1", "E2"):
            O_s = draws[g]
            phi_s = phi_from_residuals(O_s, M, Ybase)
            for m in ("E0", "E1", "E2"):
                r = invert_real(m, O_s, np.arange(ctx["n_st"]), M, ctx["W"], phi_s,
                                ctx["gps"], ctx["Vt"], ctx["Zmean"], ctx["Ys"], ctx["Ym"],
                                ctx["n_pc"], ctx["T"], ctx["n_st"], ctx["xlim"])
                raw[eid]["cases"][f"{g}->{m}"] = r
                print(f"{eid} {g}->{m}: R_j={[round(x,3) for x in r['R_j']]} "
                      f"mean={[round(x,4) for x in r['mean']]}", flush=True)
        # 预登记 Q1/Q2 判定
        c = raw[eid]["cases"]
        sd2 = c["E2->E2"]["sd"]
        d12 = max(abs(c["E1->E1"]["mean"][d] - c["E2->E2"]["mean"][d]) / sd2[d] for d in range(3))
        interior = all(LO + 2 * c[m2][ "sd"][d] < c[m2]["mean"][d] < HI - 2 * c[m2]["sd"][d]
                       for m2 in ("E1->E1", "E2->E2") for d in range(3))
        if eid == "ev08":
            raw[eid]["Q1"] = {"drift_E1_E2_sigma": d12, "interior": bool(interior),
                              "pass": bool(d12 < 2.0 and interior)}
        else:
            hits = []
            for m2 in ("E1->E1", "E2->E2"):
                pm, sd = c[m2]["mean"], c[m2]["sd"]
                for d in range(3):
                    if abs(pm[d] - LO) < 2 * sd[d]:
                        hits.append(f"{m2}.n{d+1}低")
                    if abs(pm[d] - HI) < 2 * sd[d]:
                        hits.append(f"{m2}.n{d+1}高")
            raw[eid]["Q2"] = {"drift_E1_E2_sigma": d12, "boundary_hits": hits,
                              "pass": bool(len(hits) > 0 or d12 >= 2.0)}
    json.dump(raw, open(os.path.join(OUTG, "twin_attr_raw.json"), "w"), indent=1, default=float)
    print("twin done; Q1(ev08):", raw["ev08"]["Q1"], "; Q2(ev01):", raw["ev01"]["Q2"], flush=True)


def load_ctx_by_event(eids=("ev08", "ev01")):
    ctxs = {}
    for eid in eids:
        X = theta90()
        tidx, cols = alignment(eid)
        Y = load_runs_from(os.path.join(CORR_ROOT, eid), tidx, cols)
        gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
        xlim = (X.min(0), X.max(0))
        O, M0 = p2lib.load_obs_matrix(tidx, cols)
        M, _ = p2lib.qc_stations(O, M0, Y.mean(0).T)
        W = np.load(os.path.join(OUT, f"wmap_{eid}_corr.npy"))
        ctxs[eid] = {"tidx": tidx, "cols": cols, "Y": Y, "gps": gps, "Vt": Vt, "Ym": Ym,
                     "Ys": Ys, "n_pc": n_pc, "Zmean": Zmean, "xlim": xlim, "O": O,
                     "M": M, "W": W, "T": Y.shape[1], "n_st": Y.shape[2]}
    return ctxs


# ---------------- A28-4 E2 偏置吸收 trade-off ----------------
def b_absorb(O, M, W, s2, theta, ctx):
    """给定 θ 的 diag 加权偏置吸收 RMS (E2 视角; 近似 1ᵀV⁻¹e/1ᵀV⁻¹1)"""
    Yg = recon_series(np.array([theta]), ctx["gps"], ctx["Vt"], ctx["Zmean"], ctx["Ys"],
                      ctx["Ym"], ctx["n_pc"], ctx["T"], ctx["n_st"], xlim=ctx["xlim"])[0]
    e = (O - Yg.T).astype(np.float64)      # (n_st, T)
    V = (s2 + W)
    bs = []
    for i in range(O.shape[0]):
        ok = M[i]
        if ok.sum() < 5:
            continue
        w = 1.0 / V[ok, i]
        bs.append(float((w * e[i][ok]).sum() / w.sum()))
    bs = np.array(bs)
    return float(np.sqrt((bs ** 2).mean())), bs


def gate8_bias(ctx_by_event):
    raw_corr = json.load(open(os.path.join(OUT, "xev_raw_corr.json")))
    raw_v2 = json.load(open(os.path.join(OUT, "xev_raw_v2.json")))
    raw_all = {**raw_v2, **raw_corr}
    rows = []
    for eid, raw in sorted(raw_all.items()):
        ctx = ctx_by_event.get(eid)
        if ctx is None:
            continue
        s2s = raw.get("s2_star_E1")
        if s2s is None:
            continue  # 零偏差事件无 s2* (与 v2 管线一致, 覆盖 7 corr 事件)
        th = raw["post_mean"]["E1"]
        brms, _ = b_absorb(ctx["O"], ctx["M"], ctx["W"], s2s, th, ctx)
        dist = float(np.sqrt(((np.array(th) - 0.0155) ** 2).sum()))
        rows.append({"event_id": eid, "b_rms": brms, "dist_mid": dist,
                     "mean_n1": th[0], "mean_n2": th[1], "mean_n3": th[2]})
        print(f"{eid}: b_rms={brms:.4f} m  |θ̂-mid|={dist:.4f}", flush=True)
    df = pd.DataFrame(rows)
    if len(df) >= 4:
        from scipy.stats import spearmanr
        rho, p = spearmanr(df["b_rms"], df["dist_mid"])
        print(f"Spearman(b_rms, dist_mid) = {rho:.3f} (p={p:.3f}, n={len(df)})", flush=True)
        df.attrs["spearman"] = (rho, p, len(df))
    df.to_csv(os.path.join(OUTG, "bias_tradeoff.csv"), index=False)
    print("bias done", flush=True)


# ---------------- 汇总报告 ----------------
def gate8_report():
    rep = ["# 关口 8 报告: 锚点复验 + ev08/边界堆积归因 (prereg v1.9 A28)\n\n",
           "- 措辞纪律: 分段/bootstrap 为稳健性检验 (I3); 孪生为机制归因证据 (conditional on DGP)\n\n"]
    try:
        seg = json.load(open(os.path.join(OUTG, "seg_results.json")))
        rep.append("## G1. 锚点分段复验 (前/后半窗各 52 步; 稳健性检验)\n\n"
                   "| 段 | 模型 | R_j | 均值 | info |\n|---|---|---|---|---|\n")
        for k, r in seg.items():
            rep.append(f"| {k} | {r['R_j'][0]:.2f}/{r['R_j'][1]:.2f}/{r['R_j'][2]:.2f} | "
                       f"{r['mean'][0]:.4f}/{r['mean'][1]:.4f}/{r['mean'][2]:.4f} | "
                       f"{r['info'][0]:.2f}/{r['info'][1]:.2f}/{r['info'][2]:.2f} |\n")
        rep.append("\n")
    except FileNotFoundError:
        rep.append("## G1. 分段复验: 未运行\n\n")
    try:
        tw = json.load(open(os.path.join(OUTG, "twin_attr_raw.json")))
        rep.append("## G3. 归因孪生 (θ*=先验中点 0.0155; ev08 vs ev01 regime; A16 同构 DGP)\n\n"
                   "| regime | 生成->反演 | R_j | 均值 | sd |\n|---|---|---|---|---|\n")
        for eid in ("ev08", "ev01"):
            for k, r in tw[eid]["cases"].items():
                rep.append(f"| {eid} | {k} | {r['R_j'][0]:.2f}/{r['R_j'][1]:.2f}/{r['R_j'][2]:.2f} | "
                           f"{r['mean'][0]:.4f}/{r['mean'][1]:.4f}/{r['mean'][2]:.4f} | "
                           f"{r['sd'][0]:.4f}/{r['sd'][1]:.4f}/{r['sd'][2]:.4f} |\n")
        rep.append(f"\n- **Q1 (ev08 regime 跨模型一致且内部)**: {tw['ev08']['Q1']}\n")
        rep.append(f"- **Q2 (ev01 regime 堆积/漂移)**: {tw['ev01']['Q2']}\n\n")
    except FileNotFoundError:
        rep.append("## G3. 归因孪生: 未运行\n\n")
    try:
        bt = np.load(os.path.join(OUTG, "boot_results.npz"), allow_pickle=True)
        ref = json.loads(str(bt["ref"]))
        rep.append("## G2. 锚点簇级 bootstrap (A11; K7 主 + K6 边界敏感性)\n\n"
                   "| 集 | 模型 | R_j 全站 | R_j 95% CI (n1/n2/n3) | 均值 95% CI (n1) |\n|---|---|---|---|---|\n")
        for lb in ("K7", "K6"):
            for m in ("E0", "E1", "E2"):
                key = f"{lb}_{m}_Rj"
                if key not in bt:
                    continue
                R = bt[key]
                Mn = bt[f"{lb}_{m}_mean"]
                ci = lambda a: f"[{np.percentile(a,2.5):.2f},{np.percentile(a,97.5):.2f}]"
                cim = lambda a: f"[{np.percentile(a,2.5):.4f},{np.percentile(a,97.5):.4f}]"
                rj = ref[m]["R_j"] if m in ref else ref.get(m, {}).get("R_j", [float("nan")]*3)
                rep.append(f"| {lb} | {m} | {rj[0]:.2f}/{rj[1]:.2f}/{rj[2]:.2f} | "
                           f"{ci(R[:,0])} {ci(R[:,1])} {ci(R[:,2])} | {cim(Mn[:,0])} |\n")
        rep.append("\n")
    except FileNotFoundError:
        rep.append("## G2. 簇 bootstrap: 未运行\n\n")
    try:
        bdf = pd.read_csv(os.path.join(OUTG, "bias_tradeoff.csv"))
        rep.append("## G4. E2 偏置吸收 vs n 位移 (A28-4; diag 加权近似口径)\n\n"
                   "| 事件 | b_rms (m) | |θ̂-先验中点| | mean n1/n2/n3 |\n|---|---|---|---|\n")
        for _, r in bdf.iterrows():
            rep.append(f"| {r['event_id']} | {r['b_rms']:.4f} | {r['dist_mid']:.4f} | "
                       f"{r['mean_n1']:.4f}/{r['mean_n2']:.4f}/{r['mean_n3']:.4f} |\n")
        rep.append("\n")
    except FileNotFoundError:
        rep.append("## G4. E2 trade-off: 未运行\n\n")
    with open(os.path.join(p2lib.P2, "gate8_report_v2.md"), "w", encoding="utf-8") as f:
        f.writelines(rep)
    print("gate8_report_v2.md written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", action="store_true")
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTG, exist_ok=True)
    if a.seg or a.boot:
        ctx = load_anchor()
        if a.seg:
            gate8_seg(ctx)
        if a.boot:
            gate8_boot(ctx)
    if a.twin:
        gate8_twin(load_ctx_by_event())
    if a.bias:
        gate8_bias(load_ctx_by_event(CORR_EVENTS))
    if a.report:
        gate8_report()


if __name__ == "__main__":
    main()
