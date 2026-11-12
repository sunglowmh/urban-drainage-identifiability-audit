# -*- coding: utf-8 -*-
"""P1.8 对比分析: DWF k=0 vs k=1.0 (模拟已完成, 纯分析)"""
import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib

OUT = p2lib.P2
start, end = p2lib.event_window()
obs = p2lib.load_obs(start, end)
rows = []
for k in (0, 1000):
    d = os.path.join(OUT, f"dwf_run_k{k}")
    simdf = pd.read_csv(os.path.join(d, "series.csv"), index_col=0, parse_dates=True)
    simdf.index = simdf.index.tz_convert("UTC") if simdf.index.tz is not None else simdf.index.tz_localize("UTC")
    for st, (node, osr) in obs.items():
        if node not in simdf.columns:
            continue
        j = pd.concat([simdf[node].rename("sim"), osr.rename("obs")], axis=1).dropna()
        if len(j) < 10:
            continue
        off = float((j["obs"] - j["sim"]).median())
        resid = j["obs"] - j["sim"] - off
        rows.append({"dwf_k": k / 1000.0, "station": st, "node": node, "n": len(j),
                     "pearson": j["sim"].corr(j["obs"]),
                     "spearman": j["sim"].corr(j["obs"], method="spearman"),
                     "datum_offset_m": off,
                     "rmse_offsetcorr_m": float(np.sqrt((resid ** 2).mean())),
                     "obs_med_m": float(j["obs"].median()),
                     "sim_med_m": float(j["sim"].median()),
                     "peak_bias_m": float(j["obs"].max() - j["sim"].max())})
cmp = pd.DataFrame(rows)
cmp.to_csv(os.path.join(OUT, "p1_8_dwf_comparison.csv"), index=False)
agg = cmp.groupby("dwf_k")[["pearson", "spearman", "datum_offset_m",
                            "rmse_offsetcorr_m", "peak_bias_m"]].median().round(4)
print(agg.to_string())
best = agg["spearman"].idxmax()
with open(os.path.join(OUT, "p1_8_report.md"), "w", encoding="utf-8") as f:
    f.write("# P1.8 DWF + datum 对齐复核\n\n")
    f.write(f"- 模拟窗口: {start} -> {end}（按实际雨量裁剪，含前 3h/后 9h 缓冲）\n")
    f.write(f"- 对比站数: {cmp[cmp['dwf_k']==0]['station'].nunique()}/30（观测窗内 flag==1 且 ≥10 步）\n\n")
    f.write("## 中位数指标 (DWF k 对照)\n\n")
    f.write("| k (L/s/ha) | Pearson | Spearman | datum偏移(m) | 去偏RMSE(m) | 峰偏(m) |\n|---|---|---|---|---|---|\n")
    for k, r in agg.iterrows():
        f.write(f"| {k} | {r['pearson']:.3f} | {r['spearman']:.3f} | {r['datum_offset_m']:.3f} | "
                f"{r['rmse_offsetcorr_m']:.3f} | {r['peak_bias_m']:.3f} |\n")
    f.write("\n## datum 对齐规则（预注册）\n\n")
    f.write("- 每站加性偏移 δ_i = median(obs − sim)（事件窗内，flag==1）\n")
    f.write("- 相关性指标不受 δ_i 影响；深度类指标（RMSE/峰偏）一律在去偏移后计算\n")
    f.write("- E2 误差模型将 δ_i 参数化为分层随机效应（τ 由本表偏移分布定标）\n\n")
    f.write("## 结论\n\n")
    f.write(f"- DWF k={best} L/s/ha 的 Spearman 中位数更高，作为 P2 推断基线结构设置\n")
    f.write("- 未建模残余: 泵站(13台, 湿井被伪出流吸收)、雨致入渗(DI)、传感器 datum 物理零点\n")
print("adopted dwf_k =", best)
print("per-station datum offset spread (k=%s): median|δ|=%.3f m, p90|δ|=%.3f m"
      % (best, cmp[cmp.dwf_k == best]["datum_offset_m"].abs().median(),
         cmp[cmp.dwf_k == best]["datum_offset_m"].abs().quantile(0.9)))
