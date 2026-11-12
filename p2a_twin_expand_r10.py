# -*- coding: utf-8 -*-
"""R=10 扩充 (prereg v1.1 A3 + power_budget §3 预声明规则; A18-4 MARGINAL 处置):
种子 4-9 新增 30 案例 (5 真值 × 6 新种子), 每 案例 × 3 反演; 与 R=4 主运行合并后
对角单元 n=50/格 重算 Wilson CI → 重做 A18-1。无新 SWMM 运行 (仅后验)。
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_emulator_gate import GP, load_runs
from p2_posterior_matrixA import (PRIOR_LO, PRIOR_HI, recon_series, zoom_eval,
                                  posterior_diag, TAU_GRID)
from p2a_twin_posterior import load_truth, innov_draw, TRUTHS, TAU_B, OUT
from p2_likelihoods_v3 import DiagLikV3

BASE = "E:/work/workBuddy/BlueHeartV2"
X, Y, tidx, cols_nodes = load_runs()
n_runs, T, n_st = Y.shape
Xlo, Xhi = X.min(0), X.max(0)
Xs = (X - Xlo) / (Xhi - Xlo)
Ym, Ys = Y.mean(0), Y.std(0) + 1e-6
Z = ((Y - Ym[None]) / Ys[None]).reshape(n_runs, -1)
Zc = Z - Z.mean(0)
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
n_pc = int(np.searchsorted(np.cumsum(S ** 2 / (S ** 2).sum()), 0.999) + 1)
scores = Zc @ Vt[:n_pc].T
Zmean = Z.mean(0)
ELL_GRID = [0.15, 0.25, 0.4, 0.6, 0.85]
tuned = []
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
    tuned.append((best, best_rmse))
gps = [GP(Xs, scores[:, l], ell=tuned[l][0], sn2=tuned[l][1] ** 2) for l in range(n_pc)]

O_real, M = p2lib.load_obs_matrix(tidx, cols_nodes)
M, _ = p2lib.qc_stations(O_real, M, Y.mean(0).T)
N_eff = int(M.sum())
W = np.load(os.path.join(OUT, "emu_err_map_val_v2.npy"))
lik = DiagLikV3(W, M, T, n_st, TAU_GRID)
Ybase = Y.mean(0).T
sig_i = np.array([np.std((O_real[i] - Ybase[i])[M[i]]) if M[i].sum() > 30 else 0.05
                  for i in range(n_st)])
phi_gen = 0.67
run_ids = {r: i for i, r in enumerate(list(TRUTHS) + ["m1", "m2"])}


def est_phi(O_synth):
    phis = []
    for i in range(n_st):
        ok = M[i] & np.roll(M[i], 1); ok[0] = False
        if ok.sum() > 30:
            a = (O_synth[i, 1:] - Ybase[i, 1:])[ok[1:]]
            b = (O_synth[i, :-1] - Ybase[i, :-1])[ok[1:]]
            phis.append(float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.95))
                        if a.std() > 1e-6 else 0.5)
        else:
            phis.append(0.5)
    return np.array(phis)


def wquant(ll, grid, d, qs):
    w = np.exp(ll - np.nanmax(ll))
    w = np.where(np.isfinite(w), w, 0.0); w /= w.sum()
    x = grid[:, d]; o = np.argsort(x)
    cw = np.cumsum(w[o])
    return [float(np.interp(q, cw, x[o])) for q in qs]


def invert(model, O_synth, phi_s):
    lik.set_phi(phi_s)

    def ev(grid):
        Yg = recon_series(grid, gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st,
                          xlim=(Xlo, Xhi)).astype(np.float64)
        e = (O_synth.T[None] - Yg)
        e = np.where(M.T[None], e, 0.0)
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
            "ci": [wquant(ll, grid, d, [0.025, 0.975]) for d in range(3)],
            "support": int(support), "resolved": bool(resolved)}


rows = []
k = 0
for tid, th in TRUTHS.items():
    Yt = load_truth(tid, tidx, cols_nodes)
    for seed in range(4, 10):
        rng = np.random.default_rng(run_ids[tid] * 100 + seed)
        draws = innov_draw(Yt, M, sig_i, phi_gen, rng)   # CRN: 同 ξ,b → 三生成模型
        for g in ("E0", "E1", "E2"):
            O_s = draws[g]
            phi_s = est_phi(O_s)
            r = invert(g, O_s, phi_s)
            cov = [float(r["ci"][d][0] <= th[d] <= r["ci"][d][1]) for d in range(3)]
            bias = [r["mean"][d] - th[d] for d in range(3)]
            ident = [bool(r["R_j"][d] >= 0.5 and r["info"][d] >= 0.1) for d in range(3)]
            rows.append({"case": tid, "gen": g, "invert": g, "seed": seed,
                         "mismatch": "", "truth": th,
                         "cov": cov, "bias": [round(x, 5) for x in bias],
                         "R_j": [round(x, 3) for x in r["R_j"]],
                         "post_sd": [float(f"{x:.2e}") for x in r["sd"]],
                         "ident": ident, "support": r["support"], "resolved": r["resolved"]})
        k += 1
        print(f"  {k}/30 cases done", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "twin_results_r10.csv"), index=False)
print("R=10 expansion saved: twin_results_r10.csv", flush=True)
