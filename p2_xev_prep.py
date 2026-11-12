# -*- coding: utf-8 -*-
"""
P2 阶段二 ①: 跨事件重复 — 事件选择 + 作业生成
- 事件库 38 场窗口主体事件, 按雨强三分位(T) × 前期地下水(GW) 分层抽 10 场
- 锚点事件 ev00 = 矩阵 A 原事件 (2025-10-22, 复用 output/p2/runs, 不重跑)
- 每事件窗口: 按实际雨量裁剪 (rain>0.1mm, 前 3h/后 9h 缓冲, 与 event_window 同规则)
- 雨输入: 15 站面平均 15-min VOLUME 时序 (与模板 inp 完全同口径)
- 90 组 LHS 参数与矩阵 A 完全相同 (jobs_lhs.json), dwf_k=1.0 (与锚点运行一致, 完整性日志注明)
输出: output/p2/xev/events_manifest.csv + jobs_xev.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib

OUT = os.path.join(p2lib.P2, "xev")
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. 事件库分层 ----------------
ev = pd.read_csv(os.path.join(p2lib.BASE, "p0_audit", "event_library.csv"))
ev["start"] = pd.to_datetime(ev["start"], utc=True)
ev["end"] = pd.to_datetime(ev["end"], utc=True)
w = ev[(ev["in_common_window"]) & (ev["depth_mm"] >= 5)].copy().reset_index(drop=True)
w["T"] = np.where(w["peak_1h_mmh"] <= 2.86, "T1",
                  np.where(w["peak_1h_mmh"] <= 3.99, "T2", "T3"))
w["GW"] = w["GW_class"].fillna("NA")
print("strata:\n", w.groupby(["T", "GW"]).size())

# ---------------- 2. 分层选择 (确定性, 手工指定行号依据) ----------------
def pick(T, GW, n, exclude=()):
    c = w[(w["T"] == T) & (w["GW"] == GW) & (~w["start"].isin(exclude))]
    # 优先: 雨量代表性 (depth 接近层内中位) + 时间分散
    c = c.copy()
    c["dist"] = abs(c["depth_mm"] - c["depth_mm"].median())
    c = c.sort_values(["dist", "start"])
    # 时间分散: 贪心取间隔最远
    chosen = []
    for _, r in c.iterrows():
        if all(abs((r["start"] - s).total_seconds()) > 14 * 86400 for s in chosen):
            chosen.append(r["start"])
        if len(chosen) == n:
            break
    while len(chosen) < n:  # 不够则按 dist 补
        for _, r in c.iterrows():
            if r["start"] not in chosen:
                chosen.append(r["start"])
                break
    return chosen

sel = []
sel += [("T1_Low", pick("T1", "Low", 1))]
sel += [("T1_High", pick("T1", "High", 2))]
sel += [("T2_Low", pick("T2", "Low", 2))]
sel += [("T2_High", pick("T2", "High", 1))]
sel += [("T3_Low", pick("T3", "Low", 2))]
sel += [("T3_High", pick("T3", "High", 2))]
sel_starts = [s for _, ss in sel for s in ss]
assert len(sel_starts) == len(set(sel_starts)) == 10, sel_starts

# ---------------- 3. 面平均雨量序列 ----------------
rain_files = [f for f in glob.glob(os.path.join(p2lib.DS, "7_rain_gauge_telemetry_data", "*.csv"))
              if not f.endswith("_quality_flags.csv")]
series = []
for f in rain_files:
    df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    s = df.set_index("timestamp")["Rainfall Depth [mm]"].astype(float)
    series.append(s[~s.index.duplicated(keep="first")])
rain = pd.concat(series, axis=1).resample("15min").mean().mean(axis=1)

# ---------------- 4. 逐事件窗口 + 作业 ----------------
lhs = json.load(open(os.path.join(p2lib.P2, "jobs_lhs.json")))
manifest, jobs = [], []
rows = w.set_index("start")
k = 0
for stratum, starts in sel:
    for s0 in starts:
        k += 1
        r = rows.loc[s0]
        win = rain.loc[s0:r["end"]]
        wet = win[win > 0.1]
        if len(wet) == 0:
            print("SKIP (no rain):", s0)
            continue
        wstart = wet.index[0] - pd.Timedelta(hours=3)
        wend = wet.index[-1] + pd.Timedelta(hours=9)
        ts = rain.loc[wstart:wend].fillna(0.0)
        eid = f"ev{k:02d}"
        manifest.append({
            "event_id": eid, "stratum": stratum, "start": str(s0), "end": str(r["end"]),
            "win_start": str(wstart), "win_end": str(wend), "win_h": round((wend - wstart).total_seconds() / 3600, 2),
            "depth_mm": round(r["depth_mm"], 2), "peak_1h_mmh": round(r["peak_1h_mmh"], 2),
            "GW_pct": round(r["GW_pct"], 3) if pd.notna(r["GW_pct"]) else None,
            "T": r["T"], "GW": r["GW"], "rain_ts_total_mm": round(float(ts.sum()), 2),
        })
        start_s = wstart.strftime("%m/%d/%Y %H:%M")
        end_s = wend.strftime("%m/%d/%Y %H:%M")
        for j, job in enumerate(lhs):
            jobs.append({"run_dir": os.path.join(p2lib.P2, "runs_xev", eid, f"run_{j:03d}"),
                         "rough": job["rough"], "dwf_k": 1.0,
                         "start": start_s, "end": end_s,
                         "event_id": eid})
manifest = pd.DataFrame(manifest)
manifest.to_csv(os.path.join(OUT, "events_manifest.csv"), index=False)
json.dump(jobs, open(os.path.join(OUT, "..", "jobs_xev.json"), "w"))
print(manifest.to_string())
print(f"jobs: {len(jobs)} ({manifest.shape[0]} events × 90)")
