# -*- coding: utf-8 -*-
"""A18-3 偏置-参数混淆检查 (b̂_i 预登记口径, integrity I9/I10 后补全):
E2 对角 20 案例 (5 真值 × 4 种子), 逐站 b 后验 = θ 网格上的正态混合;
统计量: |median(b̂_i) − b_i,true| < 2×sd(b̂_i) 的 (案例×站) 比例 ≥80% → 通过
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_emulator_gate import GP, load_runs
from p2_posterior_matrixA import recon_series, zoom_eval
from p2a_twin_posterior import load_truth, TRUTHS, TAU_B, OUT
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
W = np.load(os.path.join(OUT, "emu_err_map_val_v2.npy"))
lik = DiagLikV3(W, M, T, n_st, list(np.geomspace(0.005, 0.3, 10)))
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


# b 网格 (mixture CDF)
BGRID = np.linspace(-0.5, 0.5, 4001)

rows = []
for tid in TRUTHS:
    Yt = load_truth(tid, tidx, cols_nodes)
    for seed in range(4):
        seed_base = run_ids[tid] * 100 + seed
        # 双 rng 复现: innov_draw 内部自抽 xi/bias; b_true 用同种子独立重放前两次抽样
        rng_run = np.random.default_rng(seed_base)
        from p2a_twin_posterior import innov_draw
        O_s = innov_draw(Yt, M, sig_i, phi_gen, rng_run)["E2"]
        rng_rep = np.random.default_rng(seed_base)
        xi = rng_rep.normal(0, 1, Yt.shape)
        b_true = rng_rep.normal(0, TAU_B, n_st)
        # 复现自检: E2 残差站均值 ≈ b_true (+AR 噪声均值)
        res_mean = np.array([np.mean((O_s[i] - Yt[i])[M[i]]) if M[i].sum() > 30 else np.nan
                             for i in range(n_st)])
        ok_st = np.isfinite(res_mean)
        corr = np.corrcoef(res_mean[ok_st], b_true[ok_st])[0, 1]
        if tid == "t1" and seed == 0:
            print(f"[复现自检] res_mean vs b_true: corr={corr:.3f} (应≈1)", flush=True)
        lik.set_phi(est_phi(O_s))
        store = {}

        def ev(grid, _store=store, _O=O_s):
            Yg = recon_series(grid, gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st,
                              xlim=(Xlo, Xhi)).astype(np.float64)
            e = (_O.T[None] - Yg)
            e = np.where(M.T[None], e, 0.0)
            s2_hi = max((e[0] ** 2).sum() / int(M.sum()), 1e-6)
            s2_grid = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
            ll, bm, bv = lik.e2_ll_b(e, s2_grid)
            _store["grid"] = grid.copy()
            _store["bm"] = bm.copy()
            _store["bv"] = bv.copy()
            return ll

        ll, grid, bnd, resolved = zoom_eval(ev)
        g = store["grid"]; bm = store["bm"]; bv = store["bv"]
        assert np.array_equal(g, grid)
        wts = np.exp(ll - np.nanmax(ll))
        wts = np.where(np.isfinite(wts), wts, 0.0); wts /= wts.sum()

        ok_n = tot_n = 0
        for i in range(n_st):
            idx = np.where(M.T[:, i])[0]
            if len(idx) < 5:
                continue
            mu = bm[:, i]; sd = np.sqrt(np.maximum(bv[:, i], 1e-12))
            # 混合 CDF → 中位数与 sd
            cdf = (wts[:, None] * 0.5 * (1 + np.erf if False else None)) if False else None
            from math import erf
            z = (BGRID[None, :] - mu[:, None]) / sd[:, None]
            cdf = (wts[:, None] * 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))).sum(0)
            med = float(np.interp(0.5, cdf, BGRID))
            ex = (wts * mu).sum()
            ex2 = (wts * (bv[:, i] + mu ** 2)).sum()
            sdm = float(np.sqrt(max(ex2 - ex ** 2, 1e-12)))
            hit = abs(med - b_true[i]) < 2 * sdm
            rows.append({"truth": tid, "seed": seed, "st": i, "b_true": float(b_true[i]),
                         "b_hat": med, "b_sd": sdm, "hit": bool(hit)})
            tot_n += 1; ok_n += hit
        print(f"{tid} s{seed}: 站级命中 {ok_n}/{tot_n}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "twin_bias_check.csv"), index=False)
print()
print("=== A18-3 (b̂ 预登记口径): |median−b_true|<2sd 比例 ===")
print("总体 (20 案例 × 有效站):", f"{df['hit'].mean():.3f}  (n={len(df)})")
print("t5 子集 (注册口径 27×4=108):",
      f"{df[df.truth=='t5']['hit'].mean():.3f}  (n={(df.truth=='t5').sum()})")
print("逐真值:")
print(df.groupby("truth")["hit"].mean().round(3).to_string())
