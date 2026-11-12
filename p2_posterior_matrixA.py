# -*- coding: utf-8 -*-
"""
P2.3: 迭代细化网格后验 + 矩阵 A (E0/E1/E2) + dropout 阶梯
网格法在 N≈2800 残差下后验宽度 << 粗网格间距 → 采用 zoomed grid (3 轮细化, n=25)
似然:
  E0 iid:    ll = −N/2(ln2πσ² + 1), σ²_ML profile
  E1 het+AR: h=0.1+0.9·d̂; 连续时间 AR(1) 白化 (φ_i 由基准残差 lag-1 自相关标定)
  E2 offset: + δ_i~N(0,τ²) 解析边缘化 (Sherman-Morrison), (σ²,τ) 二重 profile
判级: 可辨识 = R_j≥0.5 且 info≥0.1 nats; robust = 三模型全过 + 中位漂移<1σ
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import sys
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_emulator_gate import GP, load_runs

OUT = p2lib.P2
PRIOR_LO, PRIOR_HI = p2lib.PRIOR
TAU_GRID = [0.02, 0.05, 0.10, 0.20, 0.40]
N_SIG = 20
ZOOM_LEVELS, ZOOM_N = 3, 25

# ---------------- 基础 ----------------
def build_grid():
    """兼容入口: 粗网格 24³"""
    g = np.linspace(PRIOR_LO, PRIOR_HI, 24)
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)

def recon_series(grid, gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st, batch=4000, xlim=None):
    """xlim=(X.min(0), X.max(0)): grid 为物理单位时必须提供, 内部归一化到 GP 训练空间.
    (2026-11-09 修复: 原实现把物理网格直接喂给归一化训练的 GP —— 先验区间落在
    单位立方体一角, 预测对 θ 近乎不变, 似然准平坦 → 后验无效. integrity I10)"""
    if xlim is not None:
        lo, hi = (np.asarray(xlim[0]), np.asarray(xlim[1]))
        grid = (grid - lo) / (hi - lo)
    pg = np.empty((len(grid), n_pc), dtype=np.float64)
    for s in range(0, len(grid), batch):
        for l in range(n_pc):
            pg[s:s + batch, l] = gps[l].predict(grid[s:s + batch])
    z = pg @ Vt[:n_pc] + Zmean
    return (z.reshape(len(grid), T, n_st) * Ys[None]).astype(np.float32) + Ym[None].astype(np.float32)

def zoom_eval(ll_fn, lo=PRIOR_LO, hi=PRIOR_HI, levels=4, n=21):
    """自适应各向异性细化: 逐参数 sd 与格距比较, sd≥2×格距 即认为分辨率足够
    返回 (ll, grid, bounds, resolved_flag)"""
    bnd = np.array([[lo, hi]] * 3, dtype=float)
    resolved = False
    ll = grid = None
    for lv in range(levels):
        g1 = [np.linspace(bnd[d, 0], bnd[d, 1], n) for d in range(3)]
        grid = np.stack(np.meshgrid(*g1, indexing="ij"), -1).reshape(-1, 3)
        ll = ll_fn(grid)
        w = np.exp(ll - np.nanmax(ll))
        w = np.where(np.isfinite(w), w, 0.0)
        w /= w.sum()
        spacing = (bnd[:, 1] - bnd[:, 0]) / (n - 1)
        # 正确的加权 std (修正: 之前误用 (w*x).std())
        sds = np.array([np.sqrt((w * (grid[:, d] - (w * grid[:, d]).sum()) ** 2).sum())
                        for d in range(3)])
        if np.all(sds >= 2.0 * spacing):
            resolved = True
            break
        # 逐维收缩: 中心=加权中位, 半宽=max(3sd, 5格距)
        for d in range(3):
            c = (w * grid[:, d]).sum()
            half = max(3.0 * sds[d], 5.0 * spacing[d])
            bnd[d] = [max(c - half, lo), min(c + half, hi)]
    return ll, grid, bnd, resolved

def posterior_diag(ll, grid):
    w = np.exp(ll - np.nanmax(ll))
    w = np.where(np.isfinite(w), w, 0.0)
    w /= w.sum()
    eff = 1.0 / max((w ** 2).sum(), 1e-12)
    support = int((w > 1e-9).sum())
    var_prior = (PRIOR_HI - PRIOR_LO) ** 2 / 12
    out = []
    for d in range(3):
        gm = grid[:, d]
        m = (w * gm).sum()
        v = (w * (gm - m) ** 2).sum()
        out.append({"mean": m, "sd": np.sqrt(v), "R_j": 1 - v / var_prior,
                    "info_nats": 0.5 * np.log(var_prior / max(v, 1e-300))})
    return out, w, eff, support

# ---------------- 似然 ----------------
def e0_ll(e, M):
    ez = np.where(M.T[None], e, 0.0)
    s2 = (ez ** 2).sum((1, 2)) / M.sum()
    return -0.5 * M.sum() * (np.log(2 * np.pi * s2) + 1)

def e1_ll(e, M, Yg, phi):
    N = M.sum()
    d_p90 = np.nanpercentile(Yg, 90, axis=(0, 1))
    h = (0.1 + 0.9 * np.clip(Yg / (d_p90[None, None] + 1e-6), 0, 2)).astype(np.float32)
    u = np.where(M.T[None], e / h, 0.0).astype(np.float32)
    quad = np.zeros(e.shape[0])
    for i in range(e.shape[2]):
        if M[i].sum() < 20:
            ui = u[:, :, i]
        else:
            p = phi[i]
            ui = u[:, :, i].copy()
            ui[:, 1:] = ui[:, 1:] - p * u[:, :-1, i]
        quad += (ui ** 2).sum(1)
    s2 = quad / N
    return -0.5 * N * (np.log(2 * np.pi * s2) + 1)

def e1w_ll(e, M, Yg):
    """仅异方差 (无 AR 白化) — E1 归因诊断"""
    N = M.sum()
    d_p90 = np.nanpercentile(Yg, 90, axis=(0, 1))
    h = (0.1 + 0.9 * np.clip(Yg / (d_p90[None, None] + 1e-6), 0, 2)).astype(np.float32)
    ez = np.where(M.T[None], e / h, 0.0).astype(np.float32)
    s2 = (ez ** 2).sum((1, 2)) / N
    return -0.5 * N * (np.log(2 * np.pi * s2) + 1)

def e2_ll(e, M):
    ez = np.where(M.T[None], e, 0.0)
    S1 = (ez ** 2).sum(1)
    S2 = ez.sum(1) ** 2
    n_i = M.sum(1)
    s2_hi = max(S1.sum(1).max() / M.sum(), 1e-4)
    sig_grid = np.geomspace(max(s2_hi * 1e-4, 1e-8), s2_hi, N_SIG)
    best = np.full(e.shape[0], -np.inf)
    for tau in TAU_GRID:
        for s2 in sig_grid:
            ll = np.zeros(e.shape[0])
            for i in range(len(n_i)):
                n = n_i[i]
                if n < 5:
                    continue
                quad = S1[:, i] / s2 - tau ** 2 * S2[:, i] / (s2 * (s2 + n * tau ** 2))
                logdet = (n - 1) * np.log(s2) + np.log(s2 + n * tau ** 2)
                ll += -0.5 * (n * np.log(2 * np.pi) + logdet + quad)
            best = np.maximum(best, ll)
    return best

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    X, Y, tidx, cols_nodes = load_runs()
    n_runs, T, n_st = Y.shape
    print(f"runs={n_runs}, T={T}, stations={n_st}")
    Xs = (X - X.min(0)) / (X.max(0) - X.min(0))
    Ym, Ys = Y.mean(0), Y.std(0) + 1e-6
    Z = ((Y - Ym[None]) / Ys[None]).reshape(n_runs, -1)
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    n_pc = int(np.searchsorted(np.cumsum(S ** 2 / (S ** 2).sum()), 0.999) + 1)
    scores = Zc @ Vt[:n_pc].T
    Zmean = Z.mean(0)
    ELL_GRID = [0.15, 0.25, 0.4, 0.6, 0.85]
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
    print(f"GP ensemble fitted ({n_pc} PCs)")

    O, M = p2lib.load_obs_matrix(tidx, cols_nodes)
    M, qc_excl = p2lib.qc_stations(O, M, Y.mean(0).T)
    print(f"obs valid cells: {M.sum()}/{M.size}; QC excluded {len(qc_excl)}: {qc_excl}")
    # phi (E1): 基准残差 lag-1 自相关
    Ybase = Y.mean(0).T
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
    phi = np.array(phi)
    print(f"phi: {phi.min():.2f}-{phi.max():.2f} (median {np.median(phi):.2f})")

    def make_eval(model, station_idx=None):
        si = np.arange(n_st) if station_idx is None else np.asarray(station_idx)
        Mi = M[si]
        def ev(grid):
            Yg = recon_series(grid, gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st,
                              xlim=(X.min(0), X.max(0)))
            e = (O[si].T[None] - Yg[:, :, si]).astype(np.float32)
            if model == "E0":
                return e0_ll(e, Mi)
            if model == "E1":
                return e1_ll(e, Mi, Yg[:, :, si], phi[si])
            if model == "E1w":
                return e1w_ll(e, Mi, Yg[:, :, si])
            return e2_ll(e, Mi)
        return ev

    rep = ["# P2.3 矩阵 A: 参数可辨识性 (误差结构 × dropout)\n\n",
           "- zoomed grid: 自适应各向异性细化 (≤4 轮 × 21³, 每维 sd≥2×格距 即停)\n",
           f"- 代理 {n_pc} PC-GP (全 {n_runs} 组, LOO 定标噪声底); 事件窗 {T}×15min\n",
           f"- 站数 {n_st} → QC 后有效 {int(M.sum(1).clip(0,1).sum())} (排除 {len(qc_excl)}: "
           + "; ".join(f"#{i}:{r}" for i, r in qc_excl) + ")\n",
           f"- 有效观测格 {M.sum()}; φ(lag-1) 中位 {np.median(phi):.2f}\n\n",
           "## 判级标准 (预注册)\n\n- 可辨识: R_j≥0.5 且 info≥0.1 nats\n"
           "- Robust: E0/E1/E2 全过 + 后验均值漂移<1σ(E0)\n\n"]

    all_res, stats = {}, {}
    for model in ("E0", "E1", "E2", "E1w"):
        ll, grid, bnd, resolved = zoom_eval(make_eval(model))
        diag, w, eff, support = posterior_diag(ll, grid)
        all_res[model] = diag
        sp = (bnd[:, 1] - bnd[:, 0]) / 20
        rep.append(f"### {model}  (后验支撑 {support} 格, 有效样本 {eff:.0f}, "
                   f"格距 {sp.min():.1e}-{sp.max():.1e}, 分辨率达标: {'是' if resolved else '否(达轮次上限)'})\n\n")
        rep.append("| 参数 | 后验均值 | 后验σ | R_j | info(nats) | 可辨识 |\n|---|---|---|---|---|---|\n")
        for d, dg in enumerate(diag):
            ok = dg["R_j"] >= 0.5 and dg["info_nats"] >= 0.1
            rep.append(f"| n{d+1} | {dg['mean']:.5f} | {dg['sd']:.2e} | {dg['R_j']:.3f} | "
                       f"{dg['info_nats']:.2f} | {'✅' if ok else '❌'} |\n")
        rep.append("\n")
        print(model, "R_j:", [round(d["R_j"], 3) for d in diag], "sd:", [f"{d['sd']:.1e}" for d in diag])

    R = {k: np.array([d["R_j"] for d in v]) for k, v in all_res.items()}
    ident = {k: set(np.where((R[k] >= 0.5) &
                             (np.array([d["info_nats"] for d in all_res[k]]) >= 0.1))[0].tolist())
             for k in all_res}
    tau_kt = kendalltau(R["E0"], R["E1"]).statistic
    jac = len(ident["E0"] & ident["E1"] & ident["E2"]) / max(len(ident["E0"] | ident["E1"] | ident["E2"]), 1)
    drift = max(abs(all_res[m][d]["mean"] - all_res["E0"][d]["mean"]) / max(all_res["E0"][d]["sd"], 1e-12)
                for m in ("E1", "E2") for d in range(3))
    drift_w = max(abs(all_res["E1w"][d]["mean"] - all_res["E0"][d]["mean"]) / max(all_res["E0"][d]["sd"], 1e-12)
                  for d in range(3))
    robust = all(all_res[m][d]["R_j"] >= 0.5 and all_res[m][d]["info_nats"] >= 0.1
                 for m in all_res for d in range(3)) and drift < 1.0
    rep.append(f"**一致性**: Kendall τ(E0,E1)={tau_kt:.2f}; 可辨识集 Jaccard={jac:.2f}; "
               f"均值漂移 E1={drift:.2f}σ, E2={max(abs(all_res['E2'][d]['mean']-all_res['E0'][d]['mean'])/max(all_res['E0'][d]['sd'],1e-12) for d in range(3)):.2f}σ, "
               f"E1w(仅异方差)={drift_w:.2f}σ → E1-E0 漂移归因: "
               f"{'异方差权重主导' if abs(drift_w) > 0.5 * drift else 'AR 白化主导'}\n\n")
    rep.append(f"**Robust practical identifiability: {'✅ 是' if robust else '❌ 否'}**\n\n")
    stats = {"kendall_tau_E0E1": float(tau_kt), "jaccard": float(jac), "drift": float(drift),
             "robust": bool(robust),
             "R_j": {k: [round(d["R_j"], 3) for d in v] for k, v in all_res.items()},
             "post_sd": {k: [float(f"{d['sd']:.2e}") for d in v] for k, v in all_res.items()},
             "qc_excluded": [[int(i), r] for i, r in qc_excl],
             "n_valid_stations": int(M.sum(1).clip(0, 1).sum())}

    # ---- dropout 阶梯 (E0 = 误差结构瓶颈对照; E1 = killer figure) ----
    dc = []
    for model in ("E1", "E0"):
        for nsub in (30, 15, 5):
            si = np.arange(min(nsub, n_st))
            ll, grid, bnd, resolved = zoom_eval(make_eval(model, si))
            diag, _, _, support = posterior_diag(ll, grid)
            dc.append({"model": model, "n_stations": min(nsub, n_st), "support": support,
                       **{f"R_j_n{d+1}": round(diag[d]["R_j"], 3) for d in range(3)},
                       **{f"postsd_n{d+1}": float(f"{diag[d]['sd']:.2e}") for d in range(3)}})
            rep.append(f"### Dropout {model} — {nsub} 站 (支撑 {support} 格): " +
                       ", ".join(f"n{d+1} R_j={diag[d]['R_j']:.2f}, σ={diag[d]['sd']:.1e}" for d in range(3)) + "\n\n")
    pd.DataFrame(dc).to_csv(os.path.join(OUT, "dropout_curve.csv"), index=False)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for model, c in (("E0", "#1f77b4"), ("E1", "#ff7f0e"), ("E2", "#2ca02c")):
        ll, grid, bnd, resolved = zoom_eval(make_eval(model))
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
    fig.savefig(os.path.join(OUT, "fig_posterior_marginals.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    dcsv = pd.DataFrame(dc)
    for (model, d), c in ((( "E1", 1), "#1f77b4"), (("E1", 2), "#1f77b4"), (("E1", 3), "#1f77b4"),
                          (("E0", 1), "#bbbbbb"), (("E0", 2), "#bbbbbb"), (("E0", 3), "#bbbbbb")):
        sub = dcsv[(dcsv["model"] == model)]
        ax.plot(sub["n_stations"] + (0.3 if model == "E1" else -0.3),
                sub[f"R_j_n{d}"], "o-" if model == "E1" else "o--",
                color=c, alpha=1.0 if model == "E1" else 0.6,
                label=f"{model} n{d}" if d == 1 else None)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("n sewer sensors used")
    ax.set_ylabel("posterior shrink R_j")
    ax.set_title("Dropout ladder: E1 (error-aware) vs E0 (naive iid)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_dropout.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "matrixA_report.md"), "w", encoding="utf-8") as f:
        f.writelines(rep)
    json.dump(stats, open(os.path.join(OUT, "matrixA_stats.json"), "w"), indent=1)
    print("matrix A report written")
