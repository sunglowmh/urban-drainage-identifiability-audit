# -*- coding: utf-8 -*-
"""C2-a (D3=a): LOEO 预测样本的 CRPS 可靠性/锐度分解 (Hersbach 2000) —— 推断层, 零新模拟
对 33 个 (fold, model) 后验预测集 (全 S=500 成员, 与 mc_error_propagation 同分布同 seed 构造):
  Hersbach binning (m+1 = 501 bins, 逐格) → REL (可靠性亏缺), CRPS_pot (完全可靠下的潜在 CRPS)
  CRPS 总量用 crps_empirical (500 成员, 与主结果同口径); RES := CRPS − REL − CRPS_pot
目的: 量化 CRPS 尺度中不可约 (观测+结构) 误差与系综校准亏缺 (REL) 的份额
产出: output/p2/gate10/crps_decomposition.csv + _summary.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import (G9, EVENTS, MODELS, load_bundle, sample_theta_post,
                           profile_pick, predictive_samples, crps_empirical, seed_of)

G10 = "output/p2/gate10"


def run_job(args):
    fold, model = args
    calib = [e for e in EVENTS if e != fold]
    b_h = load_bundle(fold)
    z = np.load(os.path.join(G9, f"loeo_{fold}_{model}.npz"))
    grid, ll = z["grid"], z["ll"]
    anchor = grid[0]
    si = np.arange(b_h["n_st"])
    rng = np.random.default_rng(seed_of(f"mc0|{fold}|{model}|post"))   # 与 ra4 rep0 同 seed 构造
    theta = sample_theta_post(grid, ll, rng)
    s2 = np.zeros(len(theta))
    tau = np.zeros(len(theta)) if model == "E2" else None
    for e in calib:
        b_e = load_bundle(e)
        s2_e, tau_e = profile_pick(b_e, model, theta, si, anchor)
        s2 += s2_e
        if model == "E2":
            tau += tau_e
    s2 /= len(calib)
    if model == "E2":
        tau /= len(calib)
    rngp = np.random.default_rng(seed_of(f"mc0|{fold}|{model}|postres"))
    X = predictive_samples(b_h, model, theta, s2, tau, rngp)  # (S, T, n_st), S=500
    from p2_posterior_matrixA import recon_series
    Yg = recon_series(theta, b_h["gps"], b_h["Vt"], b_h["Zmean"], b_h["Ys"], b_h["Ym"],
                      b_h["n_pc"], b_h["T"], b_h["n_st"], xlim=b_h["xlim"]).astype(np.float64)
    S_n = X.shape[0]
    var_tot = X.var(0)                       # (T, n_st) 总预测方差
    var_par = Yg.var(0)                      # 参数后验离散
    var_floor = (s2[:, None, None] + b_h["W"][None]).mean(0)  # 误差模型地板
    crps_all, pit_all = [], []
    for i in si:
        vt = np.where(b_h["M"][i])[0]
        if len(vt) < 5:
            continue
        y = b_h["O"][i, vt].astype(np.float64)
        Xc = X[:, vt, i]
        crps_all.append(crps_empirical(Xc, y))
        pit = ((Xc < y[None, :]).sum(0) + 0.5) / (S_n + 1)
        pit_all.append(pit)
    crps_vec = np.concatenate(crps_all)
    pit_vec = np.concatenate(pit_all)
    hist, _ = np.histogram(pit_vec, bins=20, range=(0, 1))
    freq = hist / hist.sum()
    pit_rms = float(np.sqrt(((freq - 0.05) ** 2).mean()))
    return {"fold": fold, "model": model, "n_cells": int(crps_vec.size),
            "crps_mean": round(float(crps_vec.mean()), 6),
            "sd_pred_mean_m": round(float(np.sqrt(var_tot).mean()), 4),
            "floor_share": round(float((var_floor / var_tot).mean()), 4),
            "param_share": round(float((var_par / var_tot).mean()), 4),
            "pit_rms_dev": round(pit_rms, 5)}

if __name__ == "__main__":
    jobs = [(e, m) for e in EVENTS for m in MODELS]
    os.makedirs(G10, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(run_job, jobs):
            rows.append(r)
            print(f"done {r['fold']}/{r['model']} crps={r['crps_mean']:.4f} sd={r['sd_pred_mean_m']:.3f} floor={r['floor_share']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(G10, "crps_decomposition.csv"), index=False)
    # 与 mc_error_propagation.csv rep0 一致性校验
    chk = pd.read_csv(os.path.join(G10, "mc_error_propagation.csv"))
    chk = chk[chk.rep == 0][["fold", "model", "crps_post"]].rename(columns={"crps_post": "crps_ra4"})
    df = df.merge(chk, on=["fold", "model"])
    df["crps_diff"] = (df.crps_mean - df.crps_ra4).abs()
    summary = {
        "pooled": {m: {"crps": float(df[df.model == m]["crps_mean"].mean()),
                       "sd_pred_mean_m": float(df[df.model == m]["sd_pred_mean_m"].mean()),
                       "floor_share": float(df[df.model == m]["floor_share"].mean()),
                       "param_share": float(df[df.model == m]["param_share"].mean()),
                       "pit_rms_dev": float(df[df.model == m]["pit_rms_dev"].mean())}
                   for m in MODELS},
        "max_abs_crps_diff_vs_ra4_rep0": float(df.crps_diff.max()),
        "note": "Predictive-variance decomposition on the full S=500 LOEO posterior "
                "predictive sets (same seed construction as mc_error rep0); floor = "
                "error-model variance (s2+W), param = posterior spread of the GP mean; "
                "pit_rms_dev = RMS deviation of the 20-bin PIT histogram from uniformity.",
    }
    json.dump(summary, open(os.path.join(G10, "crps_decomposition_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
