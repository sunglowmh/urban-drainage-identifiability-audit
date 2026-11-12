# -*- coding: utf-8 -*-
"""R-A3 (R1-C1): s²/τ profile vs 全网格求和边际化敏感性 —— 复用 gate9 存档网格与 bundle, 零新模拟
对每个 (event, model): 在最终 zoom 网格上计算逐 (s2[,tau]) 似然,
  - profile 口径 = 逐格 max (与存档 ll 同构, 用于核对)
  - 边际化口径 = logsumexp(log-uniform 权重) (E2 对 (s2,τ) 联合)
τ 敏感性: 原网格 [0.02..0.40] 与扩展网格 (+0.005 下界点) 各一份; 报 τ 边界命中占比
产出: output/p2/gate10/marginalization_sensitivity.csv + _summary.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import G9, EVENTS, MODELS, load_bundle, _anchor_s2grid
from p2_posterior_matrixA import recon_series, TAU_GRID, PRIOR_LO, PRIOR_HI
from p2_likelihoods_v3 import DiagLikV3

TAU_EXT = [0.005] + list(TAU_GRID)
G10 = os.path.join(P2DIR := "output/p2", "gate10")
VAR_PRIOR = (PRIOR_HI - PRIOR_LO) ** 2 / 12


def post_stats(grid, ll):
    w = np.exp(ll - np.nanmax(ll))
    w = np.where(np.isfinite(w), w, 0.0)
    w /= w.sum()
    out = []
    for d in range(3):
        gm = grid[:, d]
        m = (w * gm).sum()
        v = (w * (gm - m) ** 2).sum()
        out.append({"mean": m, "sd": np.sqrt(v), "R_j": 1 - v / VAR_PRIOR,
                    "info": 0.5 * np.log(VAR_PRIOR / max(v, 1e-300))})
    return out, w


def ident(d):
    return bool(d["R_j"] >= 0.5 and d["info"] >= 0.1)


def run_one(args):
    eid, model = args
    b = load_bundle(eid)
    z = np.load(os.path.join(G9, f"full_{eid}_{model}.npz"))
    grid = z["grid"]
    si = np.arange(b["n_st"])
    s2_grid = _anchor_s2grid(b, grid, si)
    Msub = b["M"][si]
    lik = DiagLikV3(b["W"][:, si], Msub, b["T"], len(si), TAU_GRID)
    lik.set_phi(b["phi"][si])
    MT = lik.MT
    n_g = len(grid)
    out = {"event": eid, "model": model, "n_grid": n_g}

    # ---- profile 复算 (与存档核对) + 边际化, chunked ----
    ll_prof, ll_marg = np.full(n_g, -np.inf), np.full(n_g, -np.inf)
    ll_marg_ext = np.full(n_g, -np.inf) if model == "E2" else None
    tau_hit = {t: 0 for t in TAU_GRID}
    tau_hit_ext = {t: 0 for t in TAU_EXT}
    s2_hit = {s: 0 for s in s2_grid}
    for c0 in range(0, n_g, 1600):
        g = grid[c0:c0 + 1600]
        Yg = recon_series(g, b["gps"], b["Vt"], b["Zmean"], b["Ys"], b["Ym"],
                          b["n_pc"], b["T"], b["n_st"], xlim=b["xlim"]).astype(np.float64)
        e = (b["O"][si].T[None] - Yg[:, :, si])
        e = np.where(MT[None], e, 0.0)
        if model in ("E0", "E1"):
            lls = np.stack([{"E0": lik.e0_ll, "E1": lik.e1_ll}[model](e, s2_grid[k:k + 1])
                            for k in range(len(s2_grid))], axis=1)  # (n, n_s2)
            ll_prof[c0:c0 + 1600] = lls.max(1)
            ll_marg[c0:c0 + 1600] = logsumexp(lls, axis=1) - np.log(len(s2_grid))
            am = lls.argmax(1)
            for k, s in enumerate(s2_grid):
                s2_hit[s] += int((am == k).sum())
        else:
            # E2: 逐 (s2, tau) 闭式值
            def e2_table(e_blk, taus):
                n = e_blk.shape[0]
                tab = np.empty((n, len(s2_grid), len(taus)))
                for k, s2 in enumerate(s2_grid):
                    V = lik.W + s2
                    Vm = np.where(MT, V, 1.0)
                    v = e_blk / np.sqrt(Vm)[None]
                    w_vec = np.where(MT, 1.0 / np.sqrt(Vm), 0.0)
                    base_quad = np.zeros(n)
                    ldet0 = (np.log(2 * np.pi * Vm) * MT).sum()
                    a_wQw = np.zeros(len(si))
                    c_vQw = np.zeros((n, len(si)))
                    for i in range(len(si)):
                        idx = lik._station_idx(i)
                        if len(idx) < 5:
                            continue
                        p = max(lik.phi[i], 1e-3)
                        ldet0 += (len(idx) - 1) * np.log(1 - p * p)
                        base_quad += lik._Qv(i, v[:, idx, i])
                        a_wQw[i] = lik._Qw(i, w_vec[idx, i])
                        Qw = lik._Qmul(i, w_vec[idx, i])
                        c_vQw[:, i] = v[:, idx, i] @ Qw
                    for t_i, tau in enumerate(taus):
                        t2 = tau ** 2
                        quad = base_quad - (t2 * c_vQw ** 2 / (1 + t2 * a_wQw[None, :])).sum(1)
                        ldet = ldet0 + np.log(1 + t2 * a_wQw).sum()
                        tab[:, k, t_i] = -0.5 * (ldet + quad)
                return tab
            tab = e2_table(e, TAU_GRID)          # (n, n_s2, n_tau)
            ll_prof[c0:c0 + 1600] = tab.max((1, 2))
            ll_marg[c0:c0 + 1600] = logsumexp(tab.reshape(len(e), -1), axis=1) - np.log(tab[0].size)
            am = np.unravel_index(tab.reshape(len(e), -1).argmax(1), tab[0].shape)
            for k in range(len(s2_grid)):
                s2_hit[s2_grid[k]] += int((am[0] == k).sum())
            for t_i, t in enumerate(TAU_GRID):
                tau_hit[t] += int((am[1] == t_i).sum())
            tab_ext = e2_table(e, TAU_EXT)
            ll_marg_ext[c0:c0 + 1600] = logsumexp(
                tab_ext.reshape(tab_ext.shape[0], -1), axis=1) - np.log(tab_ext[0].size)
            am_e = np.unravel_index(tab_ext.reshape(len(e), -1).argmax(1), tab_ext[0].shape)
            for t_i, t in enumerate(TAU_EXT):
                tau_hit_ext[t] += int((am_e[1] == t_i).sum())

    # ---- 存档 ll 核对 (profile 应一致到浮点) ----
    d_arch, _ = post_stats(grid, z["ll"])
    d_prof, _ = post_stats(grid, ll_prof)
    dev = max(abs(d_arch[j]["R_j"] - d_prof[j]["R_j"]) for j in range(3))
    out["arch_profile_Rj_maxdev"] = float(dev)

    # ---- 三口径判级对照 ----
    for tag, dset in (("profile", d_prof), ("marg", post_stats(grid, ll_marg)[0])):
        out[f"{tag}_Rj"] = [round(dset[j]["R_j"], 4) for j in range(3)]
        out[f"{tag}_ident"] = [ident(dset[j]) for j in range(3)]
        out[f"{tag}_mean"] = [round(dset[j]["mean"], 6) for j in range(3)]
        out[f"{tag}_sd"] = [float(dset[j]["sd"]) for j in range(3)]
        out[f"{tag}_pass"] = bool(all(ident(dset[j]) for j in range(3)))
    if model == "E2":
        d_ext = post_stats(grid, ll_marg_ext)[0]
        out["margext_Rj"] = [round(d_ext[j]["R_j"], 4) for j in range(3)]
        out["margext_ident"] = [ident(d_ext[j]) for j in range(3)]
        out["margext_mean"] = [round(d_ext[j]["mean"], 6) for j in range(3)]
        n_tot = sum(tau_hit_ext.values())
        out["tau_hit_ext"] = {str(t): round(v / n_tot, 4) for t, v in tau_hit_ext.items()}
    if model == "E2":
        n_tot = sum(tau_hit.values())
        out["tau_hit"] = {str(t): round(v / n_tot, 4) for t, v in tau_hit.items()}
        n_tot_s = sum(s2_hit.values())
        out["s2_hit_low"] = round(s2_hit[min(s2_grid)] / n_tot_s, 4)
        out["s2_hit_high"] = round(s2_hit[max(s2_grid)] / n_tot_s, 4)
    out["s2_grid"] = [float(x) for x in s2_grid]
    return out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(e, m) for e in EVENTS for m in MODELS]
    os.makedirs(G10, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=11) as ex:
        for res in ex.map(run_one, jobs):
            results.append(res)
            print(f"done {res['event']}/{res['model']}: "
                  f"prof_pass={res['profile_pass']} marg_pass={res['marg_pass']} "
                  f"tau_lo={res.get('tau_hit', {}).get('0.02')} "
                  f"tau_hi={res.get('tau_hit', {}).get('0.4')}", flush=True)
    pd.DataFrame(results).to_csv(os.path.join(G10, "marginalization_sensitivity.csv"), index=False)

    # ---- 汇总 ----
    df = pd.DataFrame(results)
    flips_prof_marg = []
    for e in EVENTS:
        for j in range(3):
            pi = {m: bool(df[(df.event == e) & (df.model == m)]["profile_ident"].iloc[0][j]) for m in MODELS}
            mi = {m: bool(df[(df.event == e) & (df.model == m)]["marg_ident"].iloc[0][j]) for m in MODELS}
            if pi != mi:
                flips_prof_marg.append({"event": e, "param": f"n{j+1}",
                                        "profile": {m: pi[m] for m in MODELS},
                                        "marginal": {m: mi[m] for m in MODELS}})
    summary = {
        "arch_profile_max_dev": float(df["arch_profile_Rj_maxdev"].max()),
        "pass_matrix": {f"{e}|{m}": {"profile": bool(df[(df.event == e) & (df.model == m)]["profile_pass"].iloc[0]),
                                     "marginal": bool(df[(df.event == e) & (df.model == m)]["marg_pass"].iloc[0])}
                        for e in EVENTS for m in MODELS},
        "ident_flips_profile_vs_marginal": flips_prof_marg,
        "tau_low_hit_frac_median_E2": float(df[df.model == "E2"]["tau_hit"].apply(lambda d: d["0.02"]).median()),
        "tau_high_hit_frac_median_E2": float(df[df.model == "E2"]["tau_hit"].apply(lambda d: d["0.4"]).median()),
    }
    try:
        tau_ext = df[df.model == "E2"]["tau_hit_ext"].apply(lambda d: d["0.005"])
        summary["tau_ext005_hit_frac_median_E2"] = float(tau_ext.median())
    except Exception:
        pass
    json.dump(summary, open(os.path.join(G10, "marginalization_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1, default=str))
