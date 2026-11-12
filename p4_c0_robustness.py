# -*- coding: utf-8 -*-
"""C0 计算批（模拟评审 Round 2 → 修改计划 v2）——推断层, 零 SWMM 重跑
C0-1: datum 校正后 33 单元跨结构漂移残值 (机制解释份额)
C0-2: 判级阈值稳健带 R_j ∈ {0.45, 0.50, 0.55} 下翻转计数
C0-3: 80% 功效最小可检测改善 MDE (事件聚类 bootstrap, n=11)
产出: output/p2/gate10/{datum_residual_drift.csv, threshold_band_summary.json,
      mde_equivalence_summary.json}
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import G9, EVENTS, MODELS
from p2_posterior_matrixA import posterior_diag, PRIOR_LO, PRIOR_HI

G10 = "output/p2/gate10"
PW = PRIOR_HI - PRIOR_LO

# ---------- 载入 ----------
diag = {}   # (event, model) -> posterior_diag list
for e in EVENTS:
    for m in MODELS:
        z = np.load(os.path.join(G9, f"full_{e}_{m}.npz"))
        d, _, _, _ = posterior_diag(z["ll"], z["grid"])
        diag[(e, m)] = d
print(f"loaded {len(diag)} posterior diag sets")

datum = pd.read_csv(os.path.join(G10, "datum_arm_grading.csv"))
shift = {}
for _, r in datum.iterrows():
    shift[(r["event"], r["model"])] = json.loads(r["mean_shift"].replace("np.float64", "").replace("(", "[").replace(")", "]")) if isinstance(r["mean_shift"], str) else r["mean_shift"]
# mean_shift 在 CSV 里是字符串形式的 np 数组列表; 用 ast 更稳
import ast
shift = {}
for _, r in datum.iterrows():
    shift[(r["event"], r["model"])] = [float(x) for x in ast.literal_eval(
        str(r["mean_shift"]).replace("np.float64(", "(").replace("array", ""))]

# ---------- C0-1: 漂移残值 ----------
rows = []
for e in EVENTS:
    for j in range(3):
        mo = [diag[(e, m)][j]["mean"] for m in MODELS]
        mc = [diag[(e, m)][j]["mean"] + shift[(e, m)][j] for m in MODELS]
        rows.append({
            "event": e, "param": j,
            "drift_orig_frac": (max(mo) - min(mo)) / PW,
            "drift_corr_frac": (max(mc) - min(mc)) / PW,
        })
d1 = pd.DataFrame(rows)
d1["residual_frac"] = d1["drift_corr_frac"] / d1["drift_orig_frac"].clip(lower=1e-12)
d1.to_csv(os.path.join(G10, "datum_residual_drift.csv"), index=False)
c0_1 = {
    "median_drift_orig_frac": round(float(d1["drift_orig_frac"].median()), 4),
    "median_drift_corr_frac": round(float(d1["drift_corr_frac"].median()), 4),
    "residual_share_of_original_median": round(float(d1["drift_corr_frac"].median() / d1["drift_orig_frac"].median()), 4),
    "mean_residual_share": round(float(d1["residual_frac"].mean()), 4),
    "max_drift_corr_frac": round(float(d1["drift_corr_frac"].max()), 4),
}
print("C0-1:", c0_1)

# ---------- C0-2: 阈值稳健带 ----------
def flips_at(thr, info_thr=0.1):
    fl, npass = [], {}
    for j in range(3):
        idm = {m: [bool(diag[(e, m)][j]["R_j"] >= thr and diag[(e, m)][j]["info_nats"] >= info_thr)
                   for e in EVENTS] for m in MODELS}
        for k, e in enumerate(EVENTS):
            votes = [idm[m][k] for m in MODELS]
            if not all(votes) and any(votes):
                fl.append(f"{e}-p{j}")
        npass = {m: sum(idm[m]) for m in npass or {m: 0 for m in MODELS}}
        npass = {m: sum(idm[m]) for m in MODELS}
    return len(fl), fl, npass

band = {}
for thr in (0.45, 0.50, 0.55):
    n, units, npass = flips_at(thr)
    band[f"{thr:.2f}"] = {"n_flips": n, "flip_units": units, "pass_by_model": npass}
    print(f"C0-2 thr={thr:.2f}: flips={n} pass={npass}")
# 稳健核心: 三档下都翻转的单元
core = set(band["0.45"]["flip_units"]) & set(band["0.50"]["flip_units"]) & set(band["0.55"]["flip_units"])
band["stable_core_n"] = len(core)
band["stable_core_units"] = sorted(core)
c0_2 = band

# ---------- C0-3: MDE (事件聚类) ----------
mc = pd.read_csv(os.path.join(G10, "mc_error_propagation.csv"))
# 逐事件配对差 (对 model 与 rep 取均值) → n=11 配对样本
fm = mc.groupby("fold")["diff"].mean()
folds = fm.values
n = len(folds)
rng = np.random.default_rng(20261112)
B = 10000
boot = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    boot[b] = folds[idx].mean()
se = float(boot.std(ddof=1))
zsum = 1.959964 + 0.841621  # α=0.05 双侧 + 80% 功效
mde80 = zsum * se
c0_3 = {
    "n_events": n,
    "pooled_mean_diff_m": round(float(folds.mean()), 6),
    "cluster_bootstrap_se_m": round(se, 6),
    "mde_80pct_m": round(mde80, 6),
    "mde_80pct_mm": round(mde80 * 1000, 2),
    "note": "MDE = (z_.975 + z_.80) * SE(cluster bootstrap over 11 events, B=10^4, seed 20261112); "
            "applies to the pooled mean change under the registered LOEO design.",
}
print("C0-3:", c0_3)

# ---------- 汇总落盘 ----------
out = {"c0_1_datum_residual": c0_1, "c0_2_threshold_band": c0_2, "c0_3_mde": c0_3}
with open(os.path.join(G10, "c0_robustness_summary.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved c0_robustness_summary.json")
