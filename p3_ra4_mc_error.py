# -*- coding: utf-8 -*-
"""R-A4 (R1-C2): LOEO 合并 CRPS 配对差的 MC 误差传播 —— 推断层重抽样, 零新模拟
对 33 个 (fold, model) 单元各做 N_REP=20 次独立 S=500 预测重复
(θ 与残差噪声均重抽; post/prior 流独立 seed, 与原 worker 同构非 CRN —— 如实记录):
  d_r(fold,model) = crps_post_r − crps_prior_r (事件级有效格均值, m)
pooled_r = mean over 33 folds; MC se = std(pooled_r) (ddof=1)
合成: 总 se = sqrt(boot_se² + mc_se²), boot_se 由原 bootstrap CI 宽度反推
产出: output/p2/gate10/mc_error_propagation.csv (逐 rep 逐 fold) + _summary.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import (G9, EVENTS, MODELS, load_bundle, sample_theta_post,
                           sample_theta_prior, profile_pick, predictive_samples,
                           pred_metrics, seed_of)

N_REP = 20
G10 = "output/p2/gate10"


def run_job(args):
    fold, model = args
    calib = [e for e in EVENTS if e != fold]
    b_h = load_bundle(fold)
    b_cal = {e: load_bundle(e) for e in calib}
    z = np.load(os.path.join(G9, f"loeo_{fold}_{model}.npz"))
    grid, ll = z["grid"], z["ll"]
    anchor = grid[0]
    si = np.arange(b_h["n_st"])
    rows = []
    for r in range(N_REP):
        # ---- post arm ----
        rng = np.random.default_rng(seed_of(f"mc{r}|{fold}|{model}|post"))
        theta_post = sample_theta_post(grid, ll, rng)
        s2_med = np.zeros(len(theta_post))
        tau_med = np.zeros(len(theta_post)) if model == "E2" else None
        for e in calib:
            b_e = b_cal[e]
            s2_e, tau_e = profile_pick(b_e, model, theta_post, si, anchor)
            s2_med += s2_e
            if model == "E2":
                tau_med += tau_e
        s2_med /= len(calib)
        if model == "E2":
            tau_med /= len(calib)
        rngp = np.random.default_rng(seed_of(f"mc{r}|{fold}|{model}|postres"))
        X_post = predictive_samples(b_h, model, theta_post, s2_med, tau_med, rngp)
        m_post = pred_metrics(X_post, b_h, si, f"mc{r}|{fold}|{model}|post")
        # ---- prior arm ----
        rng2 = np.random.default_rng(seed_of(f"mc{r}|{fold}|{model}|prior"))
        theta_prior = sample_theta_prior(rng2)
        s2_med_p = np.zeros(len(theta_prior))
        tau_med_p = np.zeros(len(theta_prior)) if model == "E2" else None
        for e in calib:
            b_e = b_cal[e]
            s2_e, tau_e = profile_pick(b_e, model, theta_prior, si, anchor)
            s2_med_p += s2_e
            if model == "E2":
                tau_med_p += tau_e
        s2_med_p /= len(calib)
        if model == "E2":
            tau_med_p /= len(calib)
        rng3 = np.random.default_rng(seed_of(f"mc{r}|{fold}|{model}|priorres"))
        X_prior = predictive_samples(b_h, model, theta_prior, s2_med_p, tau_med_p, rng3)
        m_prior = pred_metrics(X_prior, b_h, si, f"mc{r}|{fold}|{model}|prior")
        rows.append({"fold": fold, "model": model, "rep": r,
                     "crps_post": m_post["crps_mean"], "crps_prior": m_prior["crps_mean"],
                     "diff": m_post["crps_mean"] - m_prior["crps_mean"]})
    return rows


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(e, m) for e in EVENTS for m in MODELS]
    os.makedirs(G10, exist_ok=True)
    all_rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for rows in ex.map(run_job, jobs):
            all_rows.extend(rows)
            print(f"done {rows[0]['fold']}/{rows[0]['model']}", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(G10, "mc_error_propagation.csv"), index=False)

    # pooled per rep (33 folds 均值)
    pooled = df.groupby("rep")["diff"].mean()
    mc_se = float(pooled.std(ddof=1))
    pooled_mean = float(pooled.mean())

    # 原始口径对照: 33 fold 均值 (rep 间均值)
    by_fold = df.groupby(["fold", "model"])["diff"].mean()

    # 原始 bootstrap CI (审计 I22): [+0.0004, +0.0008] → 反推 boot se
    boot_lo, boot_hi = 0.0004, 0.0008
    boot_se = (boot_hi - boot_lo) / (2 * 1.96)
    total_se = float(np.sqrt(boot_se ** 2 + mc_se ** 2))
    summary = {
        "n_rep": N_REP, "n_folds": 33,
        "pooled_diff_mean_across_reps": pooled_mean,
        "pooled_diff_per_rep": [float(x) for x in pooled],
        "mc_se": mc_se,
        "boot_se_from_orig_ci": float(boot_se),
        "total_se": total_se,
        "combined_ci95_normal": [pooled_mean - 1.96 * total_se, pooled_mean + 1.96 * total_se],
        "zero_in_combined_ci": bool(pooled_mean - 1.96 * total_se <= 0 <= pooled_mean + 1.96 * total_se),
        "by_model": {m: {"pooled_mean": float(df[df.model == m].groupby("rep")["diff"].mean().mean()),
                         "mc_se": float(df[df.model == m].groupby("rep")["diff"].mean().std(ddof=1))}
                     for m in MODELS},
        "note": "post/prior 残差与 θ 流独立 seed (非 CRN), 与原 loeo worker 同构; "
                "bootstrap 与 MC 视为独立方差成分合成 (近似)",
    }
    json.dump(summary, open(os.path.join(G10, "mc_error_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
