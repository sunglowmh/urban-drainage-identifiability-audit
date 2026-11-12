# -*- coding: utf-8 -*-
"""
P1.8: DWF 对照试验 (k=0 vs 1.0 L/s/ha) + datum 对齐复核
- 模拟窗口按实际雨量裁剪
- 每站指标: Pearson/Spearman (平移不变) + datum 偏移(中位数差) + 去偏移 RMSE/峰偏
- 输出: output/p2/p1_8_dwf_comparison.csv + p1_8_report.md
"""
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import p2lib

OUT = p2lib.P2

def run_one(args):
    k, start, end = args
    txt = p2lib.build_inp(rough=(0.013, 0.013, 0.013), dwf_k=k, start=start, end=end)
    d = os.path.join(OUT, "dwf_run_k%d" % int(k * 1000))
    df, ferr = p2lib.run_sim(txt, d, None or [])
    return k, df, ferr

if __name__ == "__main__":
    start, end = p2lib.event_window()
    print("sim window:", start, "->", end)
    sel30 = pd.read_csv(os.path.join(p2lib.P1, "pilot_30_stations.csv"))
    targets = sel30["node_id"].unique().tolist()
    obs = p2lib.load_obs(start, end)
    print("obs stations loaded:", len(obs))

    # 串行跑 2 个 (每个 ~7 min, 2 个并行用 spawn 麻烦, 直接并行 2 进程)
    def worker(k):
        txt = p2lib.build_inp(rough=(0.013, 0.013, 0.013), dwf_k=k, start=start, end=end)
        d = os.path.join(OUT, f"dwf_run_k{k}")
        df, ferr = p2lib.run_sim(txt, d, targets)
        df.to_csv(os.path.join(d, "series.csv"))
        return k, ferr

    with mp.Pool(2) as pool:
        res = pool.map(worker, [0.0, 1.0])
    print("runs done:", res)

    rows = []
    for k, ferr in res:
        d = os.path.join(OUT, f"dwf_run_k{k}")
        simdf = pd.read_csv(os.path.join(d, "series.csv"), index_col=0, parse_dates=True)
        simdf.index = simdf.index.tz_localize("UTC")
        for st, (node, osr) in obs.items():
            if node not in simdf.columns:
                continue
            j = pd.concat([simdf[node].rename("sim"), osr.rename("obs")], axis=1).dropna()
            if len(j) < 10:
                continue
            off = float((j["obs"] - j["sim"]).median())
            resid = j["obs"] - j["sim"] - off
            rows.append({"dwf_k": k, "station": st, "node": node, "n": len(j),
                         "pearson": j["sim"].corr(j["obs"]),
                         "spearman": j["sim"].corr(j["obs"], method="spearman"),
                         "datum_offset_m": off,
                         "rmse_offsetcorr_m": float(np.sqrt((resid ** 2).mean())),
                         "obs_med_m": float(j["obs"].median()),
                         "sim_med_m": float(j["sim"].median()),
                         "peak_bias_m": float(j["obs"].max() - j["sim"].max())})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(OUT, "p1_8_dwf_comparison.csv"), index=False)

    with open(os.path.join(OUT, "p1_8_report.md"), "w", encoding="utf-8") as f:
        f.write("# P1.8 DWF + datum 对齐复核\n\n")
        f.write(f"- 模拟窗口: {start} -> {end}\n")
        f.write(f"- 对比站数: {cmp[cmp['dwf_k']==0]['station'].nunique()}\n\n")
        agg = cmp.groupby("dwf_k")[["pearson", "spearman", "datum_offset_m",
                                    "rmse_offsetcorr_m", "peak_bias_m"]].median().round(4)
        f.write("## 中位数指标 (DWF k 对照)\n\n")
        f.write("| k (L/s/ha) | Pearson | Spearman | datum偏移(m) | 去偏RMSE(m) | 峰偏(m) |\n|---|---|---|---|---|---|\n")
        for k, r in agg.iterrows():
            f.write(f"| {k} | {r['pearson']:.3f} | {r['spearman']:.3f} | {r['datum_offset_m']:.3f} | "
                    f"{r['rmse_offsetcorr_m']:.3f} | {r['peak_bias_m']:.3f} |\n")
        f.write("\n## datum 对齐规则 (预注册)\n\n")
        f.write("- 每站加性偏移 δ_i = median(obs - sim)（事件窗内，flag==1）\n")
        f.write("- 相关性指标不受 δ_i 影响; 深度类指标 (RMSE/峰偏) 一律在去偏移后计算\n")
        f.write("- E2 误差模型将 δ_i 作为分层随机效应参数化 (τ 由本表偏移分布定标)\n\n")
        f.write("## 结论\n\n")
        best = agg["spearman"].idxmax()
        f.write(f"- 采用 DWF k={best} L/s/ha 作为 P2 推断的基线结构设置\n")
        f.write("- 未建模残余: 泵站(13台, 湿井被伪出流吸收), 雨致入渗(DI), 传感器 datum 物理零点\n")
    print(agg.to_string())
