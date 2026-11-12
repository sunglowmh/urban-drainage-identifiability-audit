# -*- coding: utf-8 -*-
"""
P2 阶段二 ①: 跨事件重复 — 逐事件后验 + 可辨识性稳定性分析
每事件 (ev01-ev10): 90 组 LHS 运行 → PCA-GP → 观测/QC/φ → zoom 后验 (E0/E1/E1w/E2)
                    → E1 dropout (15/5 站)
锚点 ev00 = 矩阵 A 原事件 (复用 matrixA_stats.json / dropout_curve.csv, 不重算)
稳定性汇总:
  - 逐模型逐参数 R_j 跨事件 mean±sd、可辨识事件数
  - 可辨识性翻转矩阵 (结论是否随事件翻转)
  - 事件间 Kendall τ (E1 R_j 向量)
  - E1 dropout 阶梯模式是否事件间稳定 (15 站损失是否集中在 n2)
  - 后验均值跨事件散布 (有效糙率的事件特异性)
输出: output/p2/xev/xev_results.csv, xev_report.md, fig_xev_heatmap.png, fig_xev_dropout.png
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import sys
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_emulator_gate import GP
from p2_posterior_matrixA import (zoom_eval, posterior_diag, e0_ll, e1_ll, e1w_ll, e2_ll,
                                  PRIOR_LO, PRIOR_HI)

OUT = os.path.join(p2lib.P2, "xev")
XEV = os.path.join(p2lib.P2, "runs_xev")
ELL_GRID = [0.15, 0.25, 0.4, 0.6, 0.85]

def load_event_runs(eid):
    jobs = [j for j in json.load(open(os.path.join(p2lib.P2, "jobs_xev.json")))
            if j.get("event_id") == eid]
    runs, X = [], []
    for job in jobs:
        rd = job["run_dir"]
        d = os.path.join(rd, "done.json")
        if not os.path.exists(d) or "error" in json.load(open(d)):
            continue
        s = pd.read_csv(os.path.join(rd, "series.csv"), index_col=0, parse_dates=True)
        runs.append(s)
        X.append(job["rough"])
    idx = sorted(set().union(*[set(r.index) for r in runs]))
    idx = pd.DatetimeIndex(idx)
    mats = [r.reindex(idx).interpolate(limit=2) for r in runs]
    Y = np.stack([m.values.astype(np.float32) for m in mats])
    ok_t = np.isfinite(Y).all(axis=(0, 2))
    return np.array(X), Y[:, ok_t], idx[ok_t], mats[0].columns.tolist()

def fit_gp(X, Y):
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
    return gps, Vt[:n_pc], Ym, Ys, n_pc

def recon_series(grid, gps, Vt, Ymean, Ys, Ym, n_pc, T, n_st, batch=4000, xlim=None):
    """xlim: grid 物理单位时必须提供 (integrity I10 单位修复)"""
    if xlim is not None:
        lo, hi = (np.asarray(xlim[0]), np.asarray(xlim[1]))
        grid = (grid - lo) / (hi - lo)
    pg = np.empty((len(grid), n_pc), dtype=np.float64)
    for s in range(0, len(grid), batch):
        for l in range(n_pc):
            pg[s:s + batch, l] = gps[l].predict(grid[s:s + batch])
    z = pg @ Vt[:n_pc] + Ymean
    return (z.reshape(len(grid), T, n_st) * Ys[None]).astype(np.float32) + Ym[None].astype(np.float32)

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

def analyze_event(eid):
    X, Y, tidx, cols = load_event_runs(eid)
    n_runs, T, n_st = Y.shape
    gps, Vt, Ym, Ys, n_pc = fit_gp(X, Y)
    Zmean = Y.reshape(n_runs, -1).mean(0)

    Xlo, Xhi = X.min(0), X.max(0)

    def recon(grid):
        grid = (grid - Xlo) / (Xhi - Xlo)   # integrity I10 单位修复
        pg = np.empty((len(grid), n_pc), dtype=np.float64)
        for s in range(0, len(grid), 4000):
            for l in range(n_pc):
                pg[s:s + 4000, l] = gps[l].predict(grid[s:s + 4000])
        z = pg @ Vt + Zmean
        return (z.reshape(len(grid), T, n_st) * Ys[None]).astype(np.float32) + Ym[None].astype(np.float32)

    O, M = p2lib.load_obs_matrix(tidx, cols)
    M, qc_excl = p2lib.qc_stations(O, M, Y.mean(0).T)
    phi = phi_from_residuals(O, M, Y.mean(0).T)

    def make_eval(model, station_idx=None):
        si = np.arange(n_st) if station_idx is None else np.asarray(station_idx)
        Mi = M[si]
        def ev(grid):
            Yg = recon(grid)
            e = (O[si].T[None] - Yg[:, :, si]).astype(np.float32)
            if model == "E0":
                return e0_ll(e, Mi)
            if model == "E1":
                return e1_ll(e, Mi, Yg[:, :, si], phi[si])
            if model == "E1w":
                return e1w_ll(e, Mi, Yg[:, :, si])
            return e2_ll(e, Mi)
        return ev

    res = {}
    for model in ("E0", "E1", "E1w", "E2"):
        ll, grid, bnd, resolved = zoom_eval(make_eval(model))
        diag, w, eff, support = posterior_diag(ll, grid)
        res[model] = diag
    drop = {}
    for nsub in (15, 5):
        si = np.arange(min(nsub, n_st))
        ll, grid, bnd, resolved = zoom_eval(make_eval("E1", si))
        diag, _, _, _ = posterior_diag(ll, grid)
        drop[nsub] = [d["R_j"] for d in diag]
    return {
        "n_runs": n_runs, "T": int(T), "n_st": int(n_st),
        "n_obs": int(M.sum()), "n_valid_st": int(M.sum(1).clip(0, 1).sum()),
        "qc_excluded": len(qc_excl),
        "phi_med": float(np.median(phi)),
        "R": {m: [d["R_j"] for d in v] for m, v in res.items()},
        "post_mean": {m: [d["mean"] for d in v] for m, v in res.items()},
        "post_sd": {m: [d["sd"] for d in v] for m, v in res.items()},
        "drop15": drop[15], "drop5": drop[5],
    }

def ident(R_row):
    return all(r >= 0.5 for r in R_row)

if __name__ == "__main__":
    manifest = pd.read_csv(os.path.join(OUT, "events_manifest.csv"))
    man = {r["event_id"]: r for _, r in manifest.iterrows()}
    rows, raw = [], {}
    for eid in manifest["event_id"]:
        r = man[eid]
        print(f"=== {eid} ({r['stratum']}, {r['win_start'][:16]}, {r['depth_mm']}mm, GW={r['GW_pct']}) ===", flush=True)
        try:
            a = analyze_event(eid)
        except Exception as ex:
            print("FAIL", eid, ex)
            continue
        raw[eid] = a
        for model in ("E0", "E1", "E1w", "E2"):
            Rj = a["R"][model]
            rows.append({"event_id": eid, "stratum": r["stratum"], "depth_mm": r["depth_mm"],
                         "GW_pct": r["GW_pct"], "model": model,
                         "R_n1": Rj[0], "R_n2": Rj[1], "R_n3": Rj[2],
                         "ident_all": ident(Rj), "ident_none": not any(v >= 0.5 for v in Rj),
                         "n_obs": a["n_obs"], "n_valid_st": a["n_valid_st"], "phi_med": a["phi_med"]})
        print({m: [round(x, 3) for x in a["R"][m]] for m in ("E0", "E1", "E2")})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, "xev_results.csv"), index=False)  # 增量保存
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "xev_results.csv"), index=False)
    json.dump(raw, open(os.path.join(OUT, "xev_raw.json"), "w"), indent=1, default=float)
    print("per-event analysis done:", len(raw), "events")
