# -*- coding: utf-8 -*-
"""C2-b (D1=b) 阶段 2: 高激励 twin 判级反演 —— 推断层
输入: output/p2/twin_high/ (阶段 1 SWMM 运行, AMP=2.5)
管线: 90 LHS → PCA-GP → 24 验证 W map → 真值 + 共享创新项注入 (σ_i/φ_gen/τ_b 同注册 twin,
      仅强迫改变) → 5 真值 × 3 生成 × 2 种子 × {E0,E1,E2} 反演 → 冻结判级 (A25)
对照: 原锚点强迫 twin (output/p2/twin/twin_results.csv, 同种子子集)
产出: output/p2/twin_high/stage2_grading.csv + stage2_summary.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2b_realdata_v3 import fit_gp, phi_from_residuals, build_wmap_event, invert_real, ident_flags
from p2a_twin_posterior import innov_draw, TRUTHS, TAU_B

ROOT = "E:/work/workBuddy/BlueHeartV2/output/p2/twin_high"
SEEDS = [0, 1]
G10 = "E:/work/workBuddy/BlueHeartV2/output/p2/gate10"

# ---------- 全局: 载入 LHS + GP + W + 注入参数 ----------
print("loading LHS ensemble...", flush=True)
jobs = json.load(open(os.path.join(ROOT, "jobs_high.json")))
lhs_jobs = [j for j in jobs if j["tag"].startswith("lhs/")]
X = np.array([j["rough"] for j in lhs_jobs])
s0 = pd.read_csv(os.path.join(lhs_jobs[0]["run_dir"], "series.csv"), index_col=0, parse_dates=True)
s0.index = pd.to_datetime(s0.index, utc=True)
tidx = s0.index
sel30 = pd.read_csv(os.path.join(p2lib.P1, "pilot_30_stations.csv"))
cols_nodes = sel30["node_id"].unique().tolist()


def load_run(rd):
    s = pd.read_csv(os.path.join(rd, "series.csv"), index_col=0, parse_dates=True)
    s.index = pd.to_datetime(s.index, utc=True)
    A = s.reindex(columns=cols_nodes).reindex(index=tidx)
    assert not A.isna().any().any(), f"{rd}: alignment failure"
    return A.to_numpy(dtype=np.float64)  # (T, n_st)


Y = np.stack([load_run(j["run_dir"]) for j in lhs_jobs], 0)  # (90, T, n_st)
T, n_st = Y.shape[1], Y.shape[2]
print(f"ensemble runs={len(X)}, T={T}, stations={n_st}", flush=True)
gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
xlim = (X.min(0), X.max(0))
print(f"GP fitted ({n_pc} PCs)", flush=True)

W, per_pt = build_wmap_event(os.path.join(ROOT, "runs_val"), "", gps, Vt, Zmean, Ys, Ym,
                             n_pc, T, n_st, tidx, cols_nodes, xlim)
print(f"W map built (24 val pts, rmse {min(per_pt):.4f}-{max(per_pt):.4f})", flush=True)

O_real, M = p2lib.load_obs_matrix(tidx, cols_nodes)
M, qc_excl = p2lib.qc_stations(O_real, M, Y.mean(0).T)
N_eff = int(M.sum())
Ybase = Y.mean(0).T
sig_i = np.array([np.std((O_real[i] - Ybase[i])[M[i]]) if M[i].sum() > 30 else 0.05
                  for i in range(n_st)])
phi_list = []
for i in range(n_st):
    ok = M[i] & np.roll(M[i], 1)
    ok[0] = False
    if ok.sum() > 30:
        a = (O_real[i, 1:] - Ybase[i, 1:])[ok[1:]]
        b = (O_real[i, :-1] - Ybase[i, :-1])[ok[1:]]
        if a.std() > 1e-6:
            phi_list.append(float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.95)))
phi_gen = float(np.median(phi_list)) if phi_list else 0.5
print(f"mask valid: {N_eff}; sigma_i median={np.median(sig_i):.4f}; phi_gen={phi_gen:.2f}", flush=True)

YT = {tid: load_run(os.path.join(ROOT, "runs_truth", tid)) for tid in list(TRUTHS) + ["m1", "m2"]}
CASES = []
run_ids = {k: i for i, k in enumerate(list(TRUTHS) + ["m1", "m2"])}
for tid, th in TRUTHS.items():
    for g in ("E0", "E1", "E2"):
        for seed in SEEDS:
            CASES.append({"case": tid, "gen": g, "seed": seed, "truth": th, "Yt": YT[tid].T})
for mid in ("m1", "m2"):
    for seed in SEEDS:
        CASES.append({"case": mid, "gen": "E2", "seed": seed,
                      "truth": TRUTHS["t5"], "Yt": YT[mid].T, "mismatch": mid})
print(f"cases: {len(CASES)}", flush=True)


def run_case(k):
    c = CASES[k]
    key = (c["case"], c["seed"])
    rng = np.random.default_rng(run_ids[c["case"]] * 100 + c["seed"])
    O_s = innov_draw(c["Yt"], M, sig_i, phi_gen, rng)[c["gen"]]
    phi_s = phi_from_residuals(O_s, M, Ybase)
    row = {"case": c["case"], "gen": c["gen"], "seed": c["seed"],
           "mismatch": c.get("mismatch", "")}
    for m in ("E0", "E1", "E2"):
        r = invert_real(m, O_s, np.arange(n_st), M, W, phi_s, gps, Vt, Zmean, Ys, Ym,
                        n_pc, T, n_st, xlim)
        flags = ident_flags(r)
        row[f"Rj_{m}"] = [round(x, 3) for x in r["R_j"]]
        row[f"ident_{m}"] = flags
    row["flip"] = [bool(row["ident_E0"][d] != row["ident_E1"][d]
                        or row["ident_E1"][d] != row["ident_E2"][d]
                        or row["ident_E0"][d] != row["ident_E2"][d]) for d in range(3)]
    return row


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(run_case, range(len(CASES))))
        print("all inversions done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(G10, "c2b_stage2_grading.csv"), index=False)
    # 汇总: 对角 (gen==反演) 判级率; 逐 (case,seed,param) 跨结构翻转
    flip_cells = int(sum(sum(r) for r in df["flip"]))
    n_cells = len(df) * 3
    # 对角判级率: gen==反演模型 (gen=m 时反演 m 的识别率)
    ident_rates = {m: {"n1": float(df[df["gen"] == m][f"ident_{m}"].apply(lambda x: x[0]).mean()),
                       "n2": float(df[df["gen"] == m][f"ident_{m}"].apply(lambda x: x[1]).mean()),
                       "n3": float(df[df["gen"] == m][f"ident_{m}"].apply(lambda x: x[2]).mean())}
                   for m in ("E0", "E1", "E2")}
    # 原锚点 twin 对照 (同种子 {0,1}, 对角)
    try:
        import ast
        orig = pd.read_csv("E:/work/workBuddy/BlueHeartV2/output/p2/twin/twin_results.csv")
        orig = orig[orig.seed.isin(SEEDS) & (orig.gen == orig["invert"])].copy()
        orig["ident_l"] = orig["ident"].apply(
            lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
        orig_rates = {m: {"n1": float(orig[orig["invert"] == m]["ident_l"].apply(lambda x: x[0]).mean()),
                          "n2": float(orig[orig["invert"] == m]["ident_l"].apply(lambda x: x[1]).mean()),
                          "n3": float(orig[orig["invert"] == m]["ident_l"].apply(lambda x: x[2]).mean())}
                      for m in ("E0", "E1", "E2")}
    except Exception as e:
        orig_rates = None
        print("orig twin load failed:", e)
    summary = {
        "amp": 2.5, "seeds": SEEDS, "n_cases": len(CASES), "n_cells": n_cells,
        "flip_cells_high": flip_cells,
        "ident_rate_diag_high": ident_rates,
        "ident_rate_diag_orig_anchor": orig_rates,
        "injection": {"sigma_i_median": float(np.median(sig_i)), "phi_gen": phi_gen, "tau_b": TAU_B},
        "note": "High-excitation arm: anchor forcing x2.5, all else identical to the "
                "registered twin (injection levels, 90-LHS design, 24-point validation, "
                "frozen A25 thresholds).",
    }
    json.dump(summary, open(os.path.join(G10, "c2b_stage2_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
