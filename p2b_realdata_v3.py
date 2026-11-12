# -*- coding: utf-8 -*-
"""真实数据 v2 重算共享模块 (prereg v1.7 A26): 似然 v3 同构管线
- fit_event_gp: 逐事件 90 LHS → PCA-GP (与 v1 同法)
- phi_from_residuals: 真实残差 lag-1 (同 v1)
- build_wmap_event: 逐事件 24 点冻结离设计验证运行 → 逐格均方误差 (T, n_st)
  (与锚点 emu_err_map_val_v2.npy 构建公式一致: recon_series xlim 归一化版)
- invert_real: DiagLikV3 反演 (s2 profile 与孪生 v3 逐字同构, A26-3)
- 判级: A25 冻结阈值 R_j>=0.5 + info>=0.1
"""
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_emulator_gate import GP
from p2_posterior_matrixA import (recon_series, zoom_eval, posterior_diag, TAU_GRID,
                                  PRIOR_LO, PRIOR_HI)
from p2_likelihoods_v3 import DiagLikV3

ELL_GRID = [0.15, 0.25, 0.4, 0.6, 0.85]
BASE = "E:/work/workBuddy/BlueHeartV2"


def fit_gp(X, Y):
    """PCA-GP: 与 p2_xev_posterior.fit_gp 完全同法; 额外返回 Zmean"""
    n_runs = len(X)
    Xs = (X - X.min(0)) / (X.max(0) - X.min(0))
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-6
    Z = ((Y - Ym[None]) / Ys[None]).reshape(n_runs, -1)
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    n_pc = int(np.searchsorted(np.cumsum(S ** 2 / (S ** 2).sum()), 0.999) + 1)
    scores = Zc @ Vt[:n_pc].T
    gps = []
    for l in range(n_pc):
        y = scores[:, l]
        best, best_rmse = 0.4, np.inf
        for ell in ELL_GRID:
            g = GP(Xs, y, ell=ell)
            Kinv_a = np.linalg.solve(g.L.T, np.linalg.solve(g.L, np.eye(len(y))))
            loo = y - g.alpha / np.diag(Kinv_a)
            rmse = float(np.sqrt(((loo - y) ** 2).mean()))
            if rmse < best_rmse:
                best, best_rmse = ell, rmse
        gps.append(GP(Xs, y, ell=best, sn2=best_rmse ** 2))
    Zmean = Z.mean(0)
    return gps, Vt[:n_pc], Ym, Ys, n_pc, Zmean


def phi_from_residuals(O, M, Ybase):
    phi = []
    for i in range(O.shape[0]):
        ok = M[i] & np.roll(M[i], 1)
        ok[0] = False
        if ok.sum() > 30:
            a = (O[i, 1:] - Ybase[i, 1:])[ok[1:]]
            b = (O[i, :-1] - Ybase[i, :-1])[ok[1:]]
            phi.append(float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.95)) if a.std() > 1e-6 else 0.5)
        else:
            phi.append(0.5)
    return np.array(phi)


def load_val_design():
    """24 点冻结离设计验证设计 (A21/A26): 返回 [(tag, rough), ...]"""
    jobs = json.load(open(os.path.join(p2lib.P2, "twin/val/jobs_val.json")))
    return [(os.path.basename(j["run_dir"]), j["rough"]) for j in jobs]


def build_wmap_event(val_runs_root, eid, gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st,
                     tidx, cols, xlim):
    """逐事件 W map: 24 冻结 θ 点在事件强迫下的逐格 GP 均方误差 (A26-1).
    公式与锚点 _v2 一致: recon_series(xlim 归一化) → sq=(recon−act)² → 对 24 运行取均值."""
    design = load_val_design()
    sq_sum = np.zeros(T * n_st)
    per_pt = []
    for tag, rough in design:
        rd = os.path.join(val_runs_root, eid, tag)
        act = pd.read_csv(os.path.join(rd, "series.csv"), index_col=0, parse_dates=True)
        try:
            act.index = pd.to_datetime(act.index, utc=True)
        except (TypeError, ValueError):
            pass
        A = act.reindex(index=tidx)[cols].to_numpy(dtype=np.float64)  # (T, n_st)
        if np.isnan(A).any():
            raise ValueError(f"{eid}/{tag}: series 与事件窗对齐失败 (NaN {int(np.isnan(A).sum())} 格)")
        Yg = recon_series(np.array([rough], dtype=float), gps, Vt, Zmean, Ys, Ym,
                          n_pc, T, n_st, xlim=xlim)[0].astype(np.float64)  # (T, n_st)
        sq = (Yg - A) ** 2
        per_pt.append(float(np.sqrt(sq.mean())))
        sq_sum += sq.ravel()
    W = (sq_sum / len(design)).reshape(T, n_st)
    return W, per_pt


def invert_real(model, O, si, M, W, phi, gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st, xlim):
    """DiagLikV3 反演 (站点子集 si): 与孪生 v3 invert() 同构 (A26-3/5).
    O (n_st, T); M (n_st, T); W (T, n_st); phi (n_st,)"""
    si = np.asarray(si)
    Msub = M[si]
    Wsub = W[:, si]
    N_eff = int(Msub.sum())
    lik = DiagLikV3(Wsub, Msub, T, len(si), TAU_GRID)
    lik.set_phi(phi[si])

    def ev(grid):
        Yg = recon_series(grid, gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st,
                          xlim=xlim).astype(np.float64)
        e = (O[si].T[None] - Yg[:, :, si])
        e = np.where(Msub.T[None], e, 0.0)
        s2_hi = max((e[0] ** 2).sum() / N_eff, 1e-6)
        s2_grid = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
        if model == "E0":
            return lik.e0_ll(e, s2_grid)
        if model == "E1":
            return lik.e1_ll(e, s2_grid)
        return lik.e2_ll(e, s2_grid)

    ll, grid, bnd, resolved = zoom_eval(ev)
    diag, _, _, support = posterior_diag(ll, grid)
    return {"R_j": [d["R_j"] for d in diag], "sd": [d["sd"] for d in diag],
            "mean": [d["mean"] for d in diag], "info": [d["info_nats"] for d in diag],
            "support": int(support), "resolved": bool(resolved)}


def ident_flags(r):
    """A25 冻结阈值判级"""
    return [bool(r["R_j"][d] >= 0.5 and r["info"][d] >= 0.1) for d in range(3)]
