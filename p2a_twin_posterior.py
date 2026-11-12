# -*- coding: utf-8 -*-
"""P0-A 仿真孪生 v3: 多真值 × 三误差模型生成/反演 → 判级校准
似然: preregistration v1.4 — diag(s2+W) 逐格模拟器误差底图 (24 验证运行定标) + 精确闭式 E0/E1/E2
  (p2_likelihoods_v3); 替换 v2 rank-K Woodbury (integrity I9/I10)
设计 (v1.1 A3): 5 真值 × 3 生成 × 4 种子全 3×3; 失配 m1/m2 E2-gen; φ 逐案例从合成残差估计
修复 (integrity I10): recon_series 网格归一化 (原物理网格直喂归一化 GP, 似然准平坦 → 后验无效)
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_emulator_gate import GP, load_runs
from p2_posterior_matrixA import (PRIOR_LO, PRIOR_HI, recon_series, zoom_eval,
                                  posterior_diag, TAU_GRID)
from p2_likelihoods_v3 import DiagLikV3

BASE = "E:/work/workBuddy/BlueHeartV2"
TRUTHS = {"t1": [0.0100, 0.0100, 0.0100], "t2": [0.0210, 0.0210, 0.0210],
          "t3": [0.0090, 0.0220, 0.0090], "t4": [0.0120, 0.0160, 0.0200],
          "t5": [0.0155, 0.0155, 0.0155]}
MISMATCH = {"m1": "t5", "m2": "t5"}
SEEDS = [0, 1, 2, 3]
TAU_B = 0.10
OUT = os.path.join(p2lib.P2, "twin")

def load_truth(run_id, tidx, cols_nodes):
    s = pd.read_csv(os.path.join(BASE, f"output/p2/twin/runs/{run_id}/series.csv"),
                    index_col=0, parse_dates=True)
    s.index = pd.to_datetime(s.index, utc=True)
    s = s.reindex(columns=cols_nodes)
    assert not s.isna().any().any(), f"{run_id}: column mismatch"
    return s.reindex(index=tidx).to_numpy(dtype=np.float64).T  # (n_st, T)

def innov_draw(Yt, M, sig_i, phi_gen, rng):
    """共享底层创新项 DGP (prereg v1.3 A16): 每 (真值运行, 种子) 抽一次 xi~N(0,1) 与 b~N(0,TAU_B),
    三种误差模型由同一 xi 变换而来——E0: e=sigma_i*xi; E1: AR(1)(h*sigma_i*xi); E2: E1+b_i (精确嵌套).
    最终残差序列跨生成模型不同 (共享的是创新项, 不是最终残差)."""
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

def main():
    os.makedirs(OUT, exist_ok=True)
    X, Y, tidx, cols_nodes = load_runs()
    n_runs, T, n_st = Y.shape
    print(f"ensemble runs={n_runs}, T={T}, stations={n_st}", flush=True)
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
    print(f"GP fitted ({n_pc} PCs)", flush=True)

    O_real, M = p2lib.load_obs_matrix(tidx, cols_nodes)
    M, qc_excl = p2lib.qc_stations(O_real, M, Y.mean(0).T)
    N_eff = int(M.sum())
    print(f"mask valid: {N_eff}; QC excluded: {qc_excl}", flush=True)
    Ybase = Y.mean(0).T
    sig_i = np.array([np.std((O_real[i] - Ybase[i])[M[i]]) if M[i].sum() > 30 else 0.05
                      for i in range(n_st)])
    phi_list = []
    for i in range(n_st):
        ok = M[i] & np.roll(M[i], 1); ok[0] = False
        if ok.sum() > 30:
            a = (O_real[i, 1:] - Ybase[i, 1:])[ok[1:]]
            b = (O_real[i, :-1] - Ybase[i, :-1])[ok[1:]]
            if a.std() > 1e-6:
                phi_list.append(float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.95)))
    phi_gen = float(np.median(phi_list)) if phi_list else 0.5
    print(f"sigma_i median={np.median(sig_i):.4f}; phi_gen={phi_gen:.2f}", flush=True)

    W_MAP = np.load(os.path.join(OUT, "emu_err_map_val_v2.npy"))
    assert W_MAP.shape == (T, n_st), f"error map shape {W_MAP.shape} != {(T, n_st)}"
    lik = DiagLikV3(W_MAP, M, T, n_st, TAU_GRID)

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
                              xlim=(X.min(0), X.max(0))).astype(np.float64)
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

    case_ids = {k: i for i, k in enumerate(
        [f"{c}_{g}" for c in list(TRUTHS) + list(MISMATCH) for g in ("E0", "E1", "E2")])}
    cases = []
    for tid, th in TRUTHS.items():
        Yt = load_truth(tid, tidx, cols_nodes)
        for g in ("E0", "E1", "E2"):
            for seed in SEEDS:
                cases.append({"case": tid, "gen": g, "seed": seed, "truth": th, "Yt": Yt})
    for mid, src in MISMATCH.items():
        Yt = load_truth(mid, tidx, cols_nodes)
        for seed in SEEDS:
            cases.append({"case": mid, "gen": "E2", "seed": seed,
                          "truth": TRUTHS[src], "Yt": Yt, "mismatch": mid})
    print(f"total cases: {len(cases)} (each inverted under E0/E1/E2)", flush=True)

    rows, raw = [], {}
    obs_cache = {}
    run_ids = {r: i for i, r in enumerate(list(TRUTHS) + list(MISMATCH))}
    for k, c in enumerate(cases):
        key = (c["case"], c["seed"])
        if key not in obs_cache:  # CRN: 每 (真值运行, 种子) 只抽一次创新项, 三生成模型共享
            rng = np.random.default_rng(run_ids[c["case"]] * 100 + c["seed"])
            obs_cache[key] = innov_draw(c["Yt"], M, sig_i, phi_gen, rng)
        O_s = obs_cache[key][c["gen"]]
        phi_s = est_phi(O_s)
        raw.setdefault(c["case"], {}).setdefault(c["gen"], {}).setdefault(str(c["seed"]), {})
        for m in ("E0", "E1", "E2"):
            r = invert(m, O_s, phi_s)
            raw[c["case"]][c["gen"]][str(c["seed"])][m] = r
            cov = [float(r["ci"][d][0] <= c["truth"][d] <= r["ci"][d][1]) for d in range(3)]
            bias = [r["mean"][d] - c["truth"][d] for d in range(3)]
            ident = [bool(r["R_j"][d] >= 0.5 and r["info"][d] >= 0.1) for d in range(3)]
            rows.append({"case": c["case"], "gen": c["gen"], "invert": m, "seed": c["seed"],
                         "mismatch": c.get("mismatch", ""), "truth": c["truth"],
                         "cov": cov, "bias": [round(x, 5) for x in bias],
                         "R_j": [round(x, 3) for x in r["R_j"]],
                         "post_sd": [float(f"{x:.2e}") for x in r["sd"]],
                         "ident": ident, "support": r["support"], "resolved": r["resolved"]})
        if (k + 1) % 8 == 0:
            print(f"  {k+1}/{len(cases)} cases done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "twin_results.csv"), index=False)
    json.dump(raw, open(os.path.join(OUT, "twin_raw.json"), "w"), indent=1, default=float)

    lines = ["# P0-A 仿真孪生结果 (v2 似然: 代理误差 Woodbury 边缘化, prereg v1.2 B2)\n\n",
             f"- 注入 (共享创新项 DGP, prereg v1.3 A16): σ_i 中位 {np.median(sig_i):.4f} m, φ_gen={phi_gen:.2f}, τ_b={TAU_B} m; 每(真值,种子)抽一次 ξ,b, E0/E1/E2 由同一 ξ 变换 (E2=E1+b 精确嵌套); φ 反演逐案例从合成残差估计\n",
             f"- 设计: 5 真值 × 3 生成 × 4 种子全 3×3; 失配 m1(dwfk=0.5)/m2(width+10%) E2-gen\n\n"]
    diag_df = df[(df["gen"] == df["invert"])]
    lines.append("## 覆盖率 (对角: 生成=反演, 校准性主判据; 名义 95%)\n\n")
    lines.append("| 反演模型 | 参数 | 覆盖数/n | 覆盖率 |\n|---|---|---|---|\n")
    cov_summary = {}
    for m in ("E0", "E1", "E2"):
        sub = diag_df[diag_df["invert"] == m]
        for d in range(3):
            n_cov = int(sub["cov"].apply(lambda x: x[d]).sum())
            cov_summary[f"{m}_n{d+1}"] = n_cov / len(sub)
            lines.append(f"| {m} | n{d+1} | {n_cov}/{len(sub)} | {n_cov/len(sub):.2f} |\n")
    lines.append("\n## 判级稳定性 (对角)\n\n")
    for m in ("E0", "E1", "E2"):
        sub = diag_df[diag_df["invert"] == m]
        for d in range(3):
            ident_rate = float(sub["ident"].apply(lambda x: x[d]).mean())
            cov_rate = cov_summary[f"{m}_n{d+1}"]
            lines.append(f"- {m} n{d+1}: 判级可辨识比例 {ident_rate:.2f}, 覆盖率 {cov_rate:.2f} → "
                         f"{'一致' if (ident_rate > 0.5) == (cov_rate >= 0.5) else '不一致⚠️'}\n")
    lines.append("\n## E2 混淆检查 (E2-gen 反演 E2: 偏置吸收且参数恢复 |mean−θ*|<2σ)\n\n")
    sub = df[(df["gen"] == "E2") & (df["invert"] == "E2") & (df["mismatch"] == "")]
    n_ok = int(sub.apply(lambda r: all(abs(r["bias"][d]) < 2 * r["post_sd"][d] for d in range(3)), axis=1).sum())
    lines.append(f"- 参数恢复: {n_ok}/{len(sub)}\n")
    sub_m = df[(df["mismatch"] != "") & (df["invert"] == "E2")]
    n_ok_m = int(sub_m.apply(lambda r: all(abs(r["bias"][d]) < 2 * r["post_sd"][d] for d in range(3)), axis=1).sum())
    lines.append(f"- 失配情景 m1/m2 (E2 反演) 参数恢复: {n_ok_m}/{len(sub_m)}\n")
    lines.append("\n## 失配情景覆盖率\n\n")
    for mm in ("m1", "m2"):
        sub = df[df["mismatch"] == mm]
        for d in range(3):
            n_cov = int(sub["cov"].apply(lambda x: x[d]).sum())
            lines.append(f"- {mm} n{d+1}: {n_cov}/{len(sub)} ({n_cov/len(sub):.2f})\n")
    with open(os.path.join(OUT, "twin_report.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("twin report written", flush=True)

if __name__ == "__main__":
    main()
