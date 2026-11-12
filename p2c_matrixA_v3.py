# -*- coding: utf-8 -*-
"""矩阵 A (锚点 ev00) 真实数据 v2 重算: 似然 v3 + W_n (已登记锚点底图) + dropout
prereg v1.7 A26. 输出: matrixA_report_v2.md / matrixA_stats_v2.json / dropout_curve_v2.csv
/ fig_posterior_marginals_v2.png / fig_dropout_v2.png
主分析模型: E0/E1/E2 (A26-2, E1w 退出主分析)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_emulator_gate import load_runs
from p2b_realdata_v3 import fit_gp, phi_from_residuals, invert_real, ident_flags

OUT = p2lib.P2

if __name__ == "__main__":
    X, Y, tidx, cols_nodes = load_runs()
    n_runs, T, n_st = Y.shape
    print(f"anchor runs={n_runs}, T={T}, stations={n_st}", flush=True)
    gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
    print(f"GP fitted ({n_pc} PCs)", flush=True)
    xlim = (X.min(0), X.max(0))

    O, M = p2lib.load_obs_matrix(tidx, cols_nodes)
    M, qc_excl = p2lib.qc_stations(O, M, Y.mean(0).T)
    print(f"obs valid cells: {M.sum()}/{M.size}; QC excluded {len(qc_excl)}: {qc_excl}", flush=True)
    phi = phi_from_residuals(O, M, Y.mean(0).T)
    print(f"phi: {phi.min():.2f}-{phi.max():.2f} (median {np.median(phi):.2f})", flush=True)

    W = np.load(os.path.join(p2lib.P2, "twin", "emu_err_map_val_v2.npy"))
    assert W.shape == (T, n_st), f"W shape {W.shape} != {(T, n_st)}"

    rep = ["# P2.3 矩阵 A v2 重算 (似然 v3: diag(s2+W_n) + 精确 AR(1) + E2 精确 SM; prereg v1.7 A26)\n\n",
           "- W_n: 锚点已登记底图 emu_err_map_val_v2.npy (24 离设计验证运行, A21; 有效格 RMS 0.062 m)\n",
           f"- GP {n_pc} PC ({n_runs} LHS); recon_series xlim 归一化修复版 (I10)\n",
           f"- 站数 {n_st}, QC 排除 {len(qc_excl)}: " + "; ".join(f"#{i}:{r}" for i, r in qc_excl) + "\n",
           f"- 有效观测格 {M.sum()}; φ(lag-1) 中位 {np.median(phi):.2f}\n",
           "- s2 profile 与孪生 v3 同构 (geomspace 8 点, s2_hi 自最低角格); τ 网格 TAU_GRID\n",
           "- 判级: A25 冻结阈值 R_j≥0.5 且 info≥0.1 nats; E1/E2 n1/n2 反保守 caveat 随行\n\n"]

    all_res = {}
    for model in ("E0", "E1", "E2"):
        r = invert_real(model, O, np.arange(n_st), M, W, phi, gps, Vt, Zmean, Ys, Ym,
                        n_pc, T, n_st, xlim)
        all_res[model] = r
        ident = ident_flags(r)
        rep.append(f"### {model}  (后验支撑 {r['support']} 格, 分辨率达标: {'是' if r['resolved'] else '否(达轮次上限)'})\n\n")
        rep.append("| 参数 | 后验均值 | 后验σ | R_j | info(nats) | 可辨识 |\n|---|---|---|---|---|---|\n")
        for d in range(3):
            rep.append(f"| n{d+1} | {r['mean'][d]:.5f} | {r['sd'][d]:.2e} | {r['R_j'][d]:.3f} | "
                       f"{r['info'][d]:.2f} | {'✅' if ident[d] else '❌'} |\n")
        rep.append("\n")
        print(model, "R_j:", [round(x, 3) for x in r["R_j"]], flush=True)

    # ---- dropout 阶梯 {E0, E1} × {30, 15, 5} (A26-5) ----
    dc = []
    for model in ("E1", "E0"):
        for nsub in (30, 15, 5):
            si = np.arange(min(nsub, n_st))
            r = invert_real(model, O, si, M, W, phi, gps, Vt, Zmean, Ys, Ym,
                            n_pc, T, n_st, xlim)
            dc.append({"model": model, "n_stations": min(nsub, n_st), "support": r["support"],
                       **{f"R_j_n{d+1}": round(r["R_j"][d], 3) for d in range(3)},
                       **{f"postsd_n{d+1}": float(f"{r['sd'][d]:.2e}") for d in range(3)}})
            rep.append(f"### Dropout {model} — {nsub} 站 (支撑 {r['support']} 格): " +
                       ", ".join(f"n{d+1} R_j={r['R_j'][d]:.2f}, σ={r['sd'][d]:.1e}" for d in range(3)) + "\n\n")
            print(f"dropout {model}/{nsub} done", flush=True)
    pd.DataFrame(dc).to_csv(os.path.join(OUT, "dropout_curve_v2.csv"), index=False)

    stats = {"W_source": "twin/emu_err_map_val_v2.npy (A21)",
             "likelihood": "v3 diag(s2+W_n) + exact AR(1) + exact SM (A21/A26)",
             "phi_med": float(np.median(phi)),
             "qc_excluded": [[int(i), r] for i, r in qc_excl],
             "n_valid_stations": int(M.sum(1).clip(0, 1).sum()),
             "R_j": {k: [round(x, 4) for x in v["R_j"]] for k, v in all_res.items()},
             "info": {k: [round(x, 3) for x in v["info"]] for k, v in all_res.items()},
             "post_sd": {k: [float(f"{x:.2e}") for x in v["sd"]] for k, v in all_res.items()},
             "post_mean": {k: [round(x, 6) for x in v["mean"]] for k, v in all_res.items()},
             "ident": {k: ident_flags(v) for k, v in all_res.items()},
             "dropout": dc}
    with open(os.path.join(OUT, "matrixA_report_v2.md"), "w", encoding="utf-8") as f:
        f.writelines(rep)
    json.dump(stats, open(os.path.join(OUT, "matrixA_stats_v2.json"), "w"), indent=1)
    print("matrix A v2 report written (figures next)", flush=True)

    # ---- 图 (失败不阻断已落盘结果) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from p2_posterior_matrixA import zoom_eval, posterior_diag, TAU_GRID, recon_series
        from p2_likelihoods_v3 import DiagLikV3
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        N_eff = int(M.sum())
        for model, c in (("E0", "#1f77b4"), ("E1", "#ff7f0e"), ("E2", "#2ca02c")):
            lik = DiagLikV3(W, M, T, n_st, TAU_GRID)
            lik.set_phi(phi)
            def ev(grid):
                Yg = recon_series(grid, gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st, xlim=xlim).astype(np.float64)
                e = (O.T[None] - Yg)
                e = np.where(M.T[None], e, 0.0)
                s2_hi = max((e[0] ** 2).sum() / N_eff, 1e-6)
                s2g = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
                return {"E0": lik.e0_ll, "E1": lik.e1_ll, "E2": lik.e2_ll}[model](e, s2g)
            ll, grid, bnd, res = zoom_eval(ev)
            for d in range(3):
                w = np.exp(ll - np.nanmax(ll)); w = np.where(np.isfinite(w), w, 0); w /= w.sum()
                gm = grid[:, d]
                uniq = np.unique(np.round(gm, 6))
                dens = np.array([w[np.round(gm, 6) == u].sum() for u in uniq])
                axes[d].plot(uniq, dens / dens.sum(), color=c, label=model)
        for d in range(3):
            axes[d].set_title(f"Manning n zone {d+1}")
            axes[d].set_xlabel("n"); axes[d].set_ylabel("posterior mass")
            axes[d].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_posterior_marginals_v2.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4))
        dcsv = pd.DataFrame(dc)
        for model, col in (("E1", "#1f77b4"), ("E0", "#bbbbbb")):
            sub = dcsv[dcsv["model"] == model]
            for d in (1, 2, 3):
                ax.plot(sub["n_stations"] + (0.3 if model == "E1" else -0.3),
                        sub[f"R_j_n{d}"], "o-" if model == "E1" else "o--",
                        color=col, alpha=1.0 if model == "E1" else 0.6,
                        label=f"{model} n{d}" if d == 1 else None)
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_xlabel("n sewer sensors used")
        ax.set_ylabel("posterior shrink R_j")
        ax.set_title("Dropout ladder v2 (likelihood v3): E1 vs E0")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_dropout_v2.png"), dpi=150)
        plt.close(fig)
        print("figures written", flush=True)
    except Exception as ex:
        print(f"FIGURE BLOCK FAILED (results already persisted): {ex}", flush=True)
