# -*- coding: utf-8 -*-
"""关口 9 留出预测共享库 (prereg v2.0 A29)
- 事件 bundle: GP/W/O/M/phi (ev00 = 锚点机制, ev01-10 = xev 机制), pickle 缓存
- LOEO 合并反演: 逐事件 profile 似然求和 (与 invert_real/A26-3 同构, chunked)
- 预测分布 (A29-2): θ 采样 → GP 重建 + 误差模型残差采样
    E0: eps ~ N(0, s2*+W);  E1: eps = sqrt(s2*+W)·平稳AR(1)(phi_i);  E2: + b_i ~ N(0, tau*²)
- nuisance 传递 (I18 补遗): s2*/tau* = 校准事件同 θ profile 中位数 (主);
  self-profile 仅敏感性; 预测 profiling s2 网格 geomspace(s2_hi*1e-4, s2_hi*1e2, 12)
- 指标 (A29-3): CRPS / 覆盖率50,90 / 宽度90 / PIT / 峰值 / 峰现 / 阈值超越 Brier
"""
import os
import sys
import json
import pickle
import zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_emulator_gate import load_runs
from p2_xev_posterior import load_event_runs
from p2b_realdata_v3 import fit_gp
from p2_posterior_matrixA import (recon_series, zoom_eval, posterior_diag, TAU_GRID,
                                  PRIOR_LO, PRIOR_HI)
from p2_likelihoods_v3 import DiagLikV3

G9 = os.path.join(p2lib.P2, "gate9")
XEV = os.path.join(p2lib.P2, "xev")
EVENTS = ["ev00", "ev01", "ev02", "ev03", "ev04", "ev05", "ev06", "ev07", "ev08", "ev09", "ev10"]
MODELS = ("E0", "E1", "E2")
S_SAMPLES = 500
BASE_SEED = 20261110


def seed_of(label):
    return (BASE_SEED + zlib.crc32(label.encode("utf-8"))) % (2 ** 31)


# ---------------- bundle ----------------
def load_bundle(eid):
    cache = os.path.join(G9, f"bundle_{eid}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    if eid == "ev00":
        X, Y, tidx, cols = load_runs()
        W = np.load(os.path.join(p2lib.P2, "twin", "emu_err_map_val_v2.npy"))
    else:
        X, Y, tidx, cols = load_event_runs(eid)
        W = np.load(os.path.join(XEV, f"wmap_{eid}_v2.npy"))
    gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
    xlim = (X.min(0), X.max(0))
    O, M = p2lib.load_obs_matrix(tidx, cols)
    M, qc_excl = p2lib.qc_stations(O, M, Y.mean(0).T)
    phi = phi_base(O, M, Y.mean(0).T)
    b = {"eid": eid, "gps": gps, "Vt": Vt, "Zmean": Zmean, "Ym": Ym, "Ys": Ys,
         "n_pc": n_pc, "T": int(Y.shape[1]), "n_st": int(Y.shape[2]), "xlim": xlim,
         "O": O, "M": M, "phi": phi, "W": W.astype(np.float64), "cols": list(cols),
         "qc_excl": qc_excl}
    os.makedirs(G9, exist_ok=True)
    pickle.dump(b, open(cache, "wb"))
    return b


def phi_base(O, M, Ybase):
    phi = []
    for i in range(O.shape[0]):
        ok = M[i] & np.roll(M[i], 1)
        ok[0] = False
        if ok.sum() > 30:
            a = (O[i, 1:] - Ybase[i, 1:])[ok[1:]]
            bb = (O[i, :-1] - Ybase[i, :-1])[ok[1:]]
            phi.append(float(np.clip(np.corrcoef(a, bb)[0, 1], 0.0, 0.95)) if a.std() > 1e-6 else 0.5)
        else:
            phi.append(0.5)
    return np.array(phi)


def station_index(b):
    return {name: i for i, name in enumerate(b["cols"])}


# ---------------- 反演 (A26-3 同构, chunked) ----------------
def _anchor_s2grid(b, grid0, si):
    """s2 网格锚点: 与 invert_real 同规则 — 残差取当前网格首行 (grid[0])"""
    si = np.asarray(si)
    Yg = recon_series(grid0[0:1], b["gps"], b["Vt"], b["Zmean"], b["Ys"], b["Ym"],
                      b["n_pc"], b["T"], b["n_st"], xlim=b["xlim"])[0].astype(np.float64)
    e0 = b["O"][si].T - Yg[:, si]  # (T, n_si) 全时序 (与 invert_real 的 e 一致)
    e0 = np.where(b["M"][si].T, e0, 0.0)
    N_eff = max(int(b["M"][si].sum()), 1)
    s2_hi = max((e0 ** 2).sum() / N_eff, 1e-6)
    return np.geomspace(s2_hi * 1e-4, s2_hi, 8)


def event_ll(b, model, grid, si, s2_grid=None, chunk=1600):
    """单事件 profile 似然 (A26-3 同构): s2 网格锚定 grid[0], 网格内固定"""
    si = np.asarray(si)
    Msub = b["M"][si]
    lik = DiagLikV3(b["W"][:, si], Msub, b["T"], len(si), TAU_GRID)
    lik.set_phi(b["phi"][si])
    if s2_grid is None:
        s2_grid = _anchor_s2grid(b, grid, si)
    out = np.empty(len(grid))
    for s0 in range(0, len(grid), chunk):
        g = grid[s0:s0 + chunk]
        Yg = recon_series(g, b["gps"], b["Vt"], b["Zmean"], b["Ys"], b["Ym"],
                          b["n_pc"], b["T"], b["n_st"], xlim=b["xlim"]).astype(np.float64)
        e = (b["O"][si].T[None] - Yg[:, :, si])
        e = np.where(Msub.T[None], e, 0.0)
        out[s0:s0 + chunk] = {"E0": lik.e0_ll, "E1": lik.e1_ll, "E2": lik.e2_ll}[model](e, s2_grid)
    return out


def pooled_invert(calib_eids, model, si_fn=None):
    """LOEO 合并反演: 逐事件似然求和; si_fn(b) 缺省 = 全站.
    返回 (ll, grid, bnd, resolved, s2_grids{eid: 最终层 s2 网格})"""
    bundles = {e: load_bundle(e) for e in calib_eids}

    def ev(grid):
        total = None
        for e, b in bundles.items():
            si = np.arange(b["n_st"]) if si_fn is None else si_fn(b)
            ll_e = event_ll(b, model, grid, si)  # s2 网格逐层锚定 grid[0] (A26-3 同构)
            total = ll_e if total is None else total + ll_e
        return total

    ll, grid, bnd, resolved = zoom_eval(ev)
    s2_grids = {}
    for e, b in bundles.items():
        si = np.arange(b["n_st"]) if si_fn is None else si_fn(b)
        s2_grids[e] = _anchor_s2grid(b, grid[0:1], si)
    return ll, grid, bnd, resolved, s2_grids


# ---------------- 预测 (A29-2) ----------------
def sample_theta_post(grid, ll, rng, S=S_SAMPLES):
    w = np.exp(ll - np.nanmax(ll))
    w = np.where(np.isfinite(w), w, 0.0)
    w /= w.sum()
    idx = rng.choice(len(grid), size=S, p=w)
    return grid[idx].astype(np.float64)


def sample_theta_prior(rng, S=S_SAMPLES):
    return rng.uniform(PRIOR_LO, PRIOR_HI, size=(S, 3))


def profile_pick(b, model, theta_s, si, anchor_theta):
    """逐 θ 采样的 nuisance profile argmax: 返回 (s2_star (S,), tau_star (S,) 或 None)
    预测用扩展网格 (I18 补遗): geomspace(s2_hi*1e-4, s2_hi*1e2, 12),
    锚点 = 反演最终 zoom 网格首行角点 (与反演一致)"""
    si = np.asarray(si)
    Msub = b["M"][si]
    lik = DiagLikV3(b["W"][:, si], Msub, b["T"], len(si), TAU_GRID)
    lik.set_phi(b["phi"][si])
    Yg0 = recon_series(np.atleast_2d(anchor_theta), b["gps"], b["Vt"],
                       b["Zmean"], b["Ys"], b["Ym"], b["n_pc"], b["T"], b["n_st"],
                       xlim=b["xlim"])[0].astype(np.float64)
    e0 = b["O"][si].T - Yg0[:, si]  # (T, n_si) 全时序
    e0 = np.where(Msub.T, e0, 0.0)
    N_eff = max(int(Msub.sum()), 1)
    s2_hi = max((e0 ** 2).sum() / N_eff, 1e-6)
    s2g = np.geomspace(s2_hi * 1e-4, s2_hi * 1e2, 12)
    Yg = recon_series(theta_s, b["gps"], b["Vt"], b["Zmean"], b["Ys"], b["Ym"],
                      b["n_pc"], b["T"], b["n_st"], xlim=b["xlim"]).astype(np.float64)
    e = (b["O"][si].T[None] - Yg[:, :, si])
    e = np.where(Msub.T[None], e, 0.0)
    S = len(theta_s)
    if model in ("E0", "E1"):
        lls = np.stack([{"E0": lik.e0_ll, "E1": lik.e1_ll}[model](e, s2g[k:k + 1])
                        for k in range(len(s2g))])  # (n_s2, S)
        am = np.argmax(lls, axis=0)
        return s2g[am], None
    # E2: (s2, tau) 二维 profile argmax (与 DiagLikV3.e2_ll 逐项等价)
    MT = Msub.T
    best = np.full(S, -np.inf)
    s2_star = np.full(S, s2g[0])
    tau_star = np.full(S, TAU_GRID[0])
    for k in range(len(s2g)):
        s2 = s2g[k]
        V = lik.W + s2
        Vm = np.where(MT, V, 1.0)
        v = e / np.sqrt(Vm)[None]
        w = np.where(MT, 1.0 / np.sqrt(Vm), 0.0)
        base_quad = np.zeros(S)
        ldet0 = (np.log(2 * np.pi * Vm) * MT).sum()
        a_wQw = np.zeros(len(si))
        c_vQw = np.zeros((S, len(si)))
        for i in range(len(si)):
            idx = lik._station_idx(i)
            if len(idx) < 5:
                continue
            p = max(lik.phi[i], 1e-3)
            ldet0 += (len(idx) - 1) * np.log(1 - p * p)
            base_quad += lik._Qv(i, v[:, idx, i])
            a_wQw[i] = lik._Qw(i, w[idx, i])
            Qw = lik._Qmul(i, w[idx, i])
            c_vQw[:, i] = v[:, idx, i] @ Qw
        for tau in lik.tau_grid:
            t2 = tau ** 2
            quad = base_quad - (t2 * c_vQw ** 2 / (1 + t2 * a_wQw[None, :])).sum(1)
            ldet = ldet0 + np.log(1 + t2 * a_wQw).sum()
            ll = -0.5 * (ldet + quad)
            upd = ll > best
            if upd.any():
                best[upd] = ll[upd]
                s2_star[upd] = s2
                tau_star[upd] = tau
    return s2_star, tau_star


def ar1_field(phi, T, S, rng):
    """平稳单位方差 AR(1) 场 (S, T, n_st), phi (n_st,)"""
    n_st = len(phi)
    u = np.empty((S, T, n_st))
    u[:, 0, :] = rng.standard_normal((S, n_st))
    sig = np.sqrt(np.maximum(1.0 - phi ** 2, 1e-12))[None, :]
    for t in range(1, T):
        u[:, t, :] = phi[None, :] * u[:, t - 1, :] + sig * rng.standard_normal((S, n_st))
    return u


def predictive_samples(b, model, theta_s, s2_star, tau_star, rng):
    """预测样本 (S, T, n_st) = GP 重建 + 误差模型残差 (A29-2/I18 补遗)"""
    Yg = recon_series(theta_s, b["gps"], b["Vt"], b["Zmean"], b["Ys"], b["Ym"],
                      b["n_pc"], b["T"], b["n_st"], xlim=b["xlim"]).astype(np.float64)
    V = s2_star[:, None, None] + b["W"][None]
    if model == "E0":
        return Yg + np.sqrt(V) * rng.standard_normal(Yg.shape)
    u = ar1_field(b["phi"], b["T"], len(theta_s), rng)
    X = Yg + np.sqrt(V) * u
    if model == "E2":
        X = X + rng.standard_normal((len(theta_s), 1, b["n_st"])) * tau_star[:, None, None]
    return X


# ---------------- 指标 (A29-3) ----------------
def crps_empirical(X, y):
    """X (S, C) 样本, y (C,) 观测 → 逐格 CRPS (C,)"""
    Sm = X.shape[0]
    ea = np.abs(X - y[None, :]).mean(0)
    Xs = np.sort(X, axis=0)
    wk = (2.0 * np.arange(1, Sm + 1) - Sm - 1)
    exx = 2.0 * (wk[:, None] * Xs).sum(0) / Sm ** 2
    return ea - 0.5 * exx


def pred_metrics(X, b, sh, tag):
    """A6 全项指标, 事件级汇总. X (S,T,n_st), sh 站索引"""
    S = X.shape[0]
    crps_c, pit_c, cov50_c, cov90_c, w90_c, brier_c, thr_used = [], [], [], [], [], [], []
    peak_signed, peak_abs, ptime_err = [], [], []
    n_cells_tot = 0
    for i in sh:
        vt = np.where(b["M"][i])[0]
        if len(vt) < 5:
            continue
        y = b["O"][i, vt].astype(np.float64)
        Xc = X[:, vt, i]  # (S, n_valid)
        crps_c.append(crps_empirical(Xc, y))
        pit = (np.array([(Xc[:, j] < y[j]).sum() for j in range(len(y))]) + 0.5) / (S + 1)
        pit_c.append(pit)
        q05, q25, q75, q95 = np.quantile(Xc, [0.05, 0.25, 0.75, 0.95], axis=0)
        cov50_c.append(((q25 <= y) & (y <= q75)).astype(float))
        cov90_c.append(((q05 <= y) & (y <= q95)).astype(float))
        w90_c.append(q95 - q05)
        thr = np.percentile(y, 90)
        p_exc = (Xc > thr).mean(0)
        brier_c.append((p_exc - (y > thr).astype(float)) ** 2)
        thr_used.append(thr)
        n_cells_tot += len(y)
        # 峰值/峰现 (逐站)
        obs_pk_t = int(np.argmax(y))
        samp_pk = Xc.max(axis=1)
        peak_signed.append(float(np.median(samp_pk) - y.max()))
        peak_abs.append(float(abs(np.median(samp_pk) - y.max())))
        med_series = np.median(Xc, axis=0)
        ptime_err.append(int(np.argmax(med_series) - obs_pk_t))
    if not crps_c:
        return {"tag": tag, "n_cells": 0}
    crps_all = np.concatenate(crps_c)
    pit_all = np.concatenate(pit_c)
    cov50 = np.concatenate(cov50_c)
    cov90 = np.concatenate(cov90_c)
    w90 = np.concatenate(w90_c)
    brier_all = np.concatenate(brier_c)
    return {
        "tag": tag, "n_cells": int(n_cells_tot), "n_st_used": len(crps_c),
        "crps_mean": float(crps_all.mean()), "crps_median": float(np.median(crps_all)),
        "pit_mean": float(pit_all.mean()), "pit_sd": float(pit_all.std()),
        "pit_out025_975": float(((pit_all < 0.025) | (pit_all > 0.975)).mean()),
        "cov50": float(cov50.mean()), "cov90": float(cov90.mean()),
        "width90_mean": float(w90.mean()),
        "brier_mean": float(brier_all.mean()),
        "peak_err_median": float(np.median(peak_signed)),
        "peak_err_mae": float(np.mean(peak_abs)),
        "ptime_err_median_steps": float(np.median(ptime_err)),
    }


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
