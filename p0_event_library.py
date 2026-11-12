# -*- coding: utf-8 -*-
"""
P0 收尾：Blue Heart 整场暴雨事件库构建
输入: dataset/7_rain_gauge_telemetry_data, 3_borehole_telemetry_data,
      6_highway_gully_telemetry_data, p0_audit/gauge_index_summary.csv
输出: p0_audit/event_library.csv, p0_audit/event_library_report.md
口径（预注册）:
  - 雨量输入: 14 个雨量计的集水面平均序列（15 min, 仅质量有效行）
  - 事件划分: 场次间最小无雨间隔 6 h（敏感性 3/12 h）, 场次总量 >= 2 mm
  - 雨强分层: 峰值 1 h 雨强三分位 T1/T2/T3
  - 前期地下水: 各钻孔"越大越湿"序列的站内经验百分位(前期 24 h 均值)跨站平均,
    以 0.5 为界分 High/Low; API7 = 事件前 7 天(不含事件本身)累计面雨量
  - 传感器可用性: 站点活跃期(来自审计明细)覆盖事件窗口的站数
"""
import os
import glob
import numpy as np
import pandas as pd

BASE = "E:/work/workBuddy/BlueHeartV2"
DS = os.path.join(BASE, "dataset")
OUT = os.path.join(BASE, "p0_audit")
os.makedirs(OUT, exist_ok=True)

COMMON_START = pd.Timestamp("2025-06-30", tz="UTC")
COMMON_END = pd.Timestamp("2026-04-30", tz="UTC")
MIN_GAP = pd.Timedelta(hours=6)
MIN_DEPTH = 2.0

def w(s):
    with open(os.path.join(OUT, "event_library_report.md"), "a", encoding="utf-8") as f:
        f.write(s + "\n")

def md_table(df):
    df = df.copy()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |",
             "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)

def read_csv_utc(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    return df.sort_values("timestamp").set_index("timestamp")
files = [f for f in glob.glob(os.path.join(DS, "7_rain_gauge_telemetry_data", "*.csv"))
         if not f.endswith("_quality_flags.csv")]
rain_series = []
for f in files:
    name = os.path.basename(f).replace(".csv", "")
    qf = f.replace(".csv", "_quality_flags.csv")
    df = read_csv_utc(f)
    col = [c for c in df.columns if "Rainfall" in c][0]
    # 质量标记: 仅保留数值 flag 为 1(有效) 的行, 与审计口径一致(数值列为第一列)
    if os.path.exists(qf):
        q = pd.read_csv(qf)
        q["timestamp"] = pd.to_datetime(q["timestamp"], format="mixed", utc=True)
        q = q.sort_values("timestamp").set_index("timestamp")
        numflag = q.select_dtypes("number")
        if numflag.shape[1] >= 1:
            ok = numflag.iloc[:, 0] == 1
            df = df.loc[df.index.intersection(ok.index[ok])]
    rain_series.append(df[col].rename(name))
rain = pd.concat(rain_series, axis=1)
n_gauges = rain.shape[1]
coverage = rain.notna().mean(axis=1)  # 每时刻有值的站数比例
rain_mean = rain.mean(axis=1).where(coverage >= 0.6)  # >=60% 站有值才取面均值
rain_15 = rain_mean.resample("15min").sum()  # 已是 15min 累积值, resample 防重复索引
print(f"rain: {n_gauges} gauges, {rain.index.min()} -> {rain.index.max()}, "
      f"{len(rain_15)} steps")

# ---------- 2. 事件划分 ----------
wet = rain_15 > 0
grp = (wet != wet.shift()).cumsum()
events = []
for g, seg in wet[wet].groupby(grp[wet]):
    s, e = seg.index.min(), seg.index.max()
    depth = rain_15.loc[s:e].sum()
    if depth < MIN_DEPTH:
        continue
    events.append({"start": s, "end": e, "depth_mm": depth})
ev = pd.DataFrame(events)
# 合并间隔 <= 6h 的事件
merged = []
cur = None
for _, r in ev.iterrows():
    if cur is None:
        cur = dict(r)
    elif r["start"] - cur["end"] <= MIN_GAP:
        cur["end"] = r["end"]
        cur["depth_mm"] += r["depth_mm"]
    else:
        merged.append(cur)
        cur = dict(r)
if cur:
    merged.append(cur)
ev = pd.DataFrame(merged)

# ---------- 3. 事件属性 ----------
# 雨强
attrs = []
for _, r in ev.iterrows():
    s, e = r["start"], r["end"]
    seg = rain_15.loc[s:e]
    hourly = rain_mean.loc[s - pd.Timedelta(minutes=15):e].resample("1h").sum()
    api_start = s - pd.Timedelta(days=7)
    api7 = rain_15.loc[api_start:s - pd.Timedelta(minutes=15)].sum()
    attrs.append({
        "start": s, "end": e,
        "duration_h": (e - s).total_seconds() / 3600,
        "depth_mm": r["depth_mm"],
        "peak_15min_mmh": seg.max() * 4,
        "peak_1h_mmh": hourly.max() if len(hourly) else np.nan,
        "mean_mmh": r["depth_mm"] / ((e - s).total_seconds() / 3600),
        "API7_mm": api7,
    })
ev = pd.DataFrame(attrs)

# 前期地下水: 站内百分位
gw_files = [f for f in glob.glob(os.path.join(DS, "3_borehole_telemetry_data", "*.csv"))
            if not f.endswith("_quality_flags.csv")]
gw_pct = []
for f in gw_files:
    try:
        df = read_csv_utc(f)
    except Exception as ex:
        print("skip GW:", f, ex)
        continue
    if "Water Level [mAOD]" in df.columns:
        v = df["Water Level [mAOD]"]  # 越大越湿
    elif "Depth To Water [m]" in df.columns:
        v = -df["Depth To Water [m]"]  # 埋深取负, 越大越湿
    else:
        continue
    v = v[v.index >= COMMON_START - pd.Timedelta(days=7)]
    if len(v) < 500:
        continue
    rank = v.rank(pct=True)
    gw_pct.append(rank.rename(os.path.basename(f).replace(".csv", "")))
gw = pd.concat(gw_pct, axis=1)
print(f"groundwater: {gw.shape[1]} stations usable")

gw_ant = []
for _, r in ev.iterrows():
    s = r["start"]
    win = gw.loc[s - pd.Timedelta(hours=24):s]
    gw_ant.append(win.mean(axis=1).mean() if len(win) else np.nan)
ev["GW_pct"] = gw_ant

# gully 记录数(事件窗口内实际上报条数)
gully_files = [f for f in glob.glob(os.path.join(DS, "6_highway_gully_telemetry_data", "*.csv"))
               if not f.endswith("_quality_flags.csv")]
gully_counts = []
for f in gully_files:
    df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    gully_counts.append(df.set_index("timestamp").index)
gully_all = pd.DatetimeIndex([])
for idx in gully_counts:
    gully_all = gully_all.append(idx)
gully_all = gully_all.sort_values()
ev["gully_records"] = [int(((gully_all >= r["start"]) & (gully_all <= r["end"])).sum())
                       for _, r in ev.iterrows()]

# 传感器活跃期(审计明细) -> 覆盖事件窗口的活跃站数
det = pd.read_csv(os.path.join(OUT, "gauge_index_summary.csv"), dtype=str)
det["start"] = pd.to_datetime(det["start"], format="mixed", utc=True)
det["end"] = pd.to_datetime(det["end"], format="mixed", utc=True)
act = {t: g for t, g in det.groupby("type")}
def active_count(row, typ):
    g = act.get(typ)
    if g is None:
        return 0
    m = (g["start"] <= row["start"]) & (g["end"] >= row["end"])
    return int(m.sum())
ev["n_sewer_active"] = ev.apply(lambda r: active_count(r, "Sewer"), axis=1)
ev["n_fluvial_active"] = ev.apply(lambda r: active_count(r, "Fluvial"), axis=1)
ev["n_borehole_active"] = ev.apply(lambda r: active_count(r, "Borehole"), axis=1)
ev["n_rain_active"] = ev.apply(lambda r: active_count(r, "Rain"), axis=1)

ev["in_common_window"] = (ev["start"] >= COMMON_START) & (ev["end"] <= COMMON_END)
ev["full_types"] = ((ev["n_sewer_active"] > 0) & (ev["n_fluvial_active"] > 0) &
                    (ev["n_borehole_active"] > 0) & (ev["n_rain_active"] > 0))

# ---------- 4. 分层 ----------
ev["GW_class"] = np.where(ev["GW_pct"].isna(), "NA",
                          np.where(ev["GW_pct"] >= 0.5, "High", "Low"))
ev_major = ev[ev["depth_mm"] >= 5].copy()  # 主体分析子集
if len(ev_major) >= 6:
    q1, q2 = ev_major["peak_1h_mmh"].quantile([1/3, 2/3])
    def iclass(x):
        if np.isnan(x):
            return "NA"
        return "T1" if x <= q1 else ("T2" if x <= q2 else "T3")
    ev_major["I_class"] = ev_major["peak_1h_mmh"].apply(iclass)
    ev_major["stratum"] = ev_major["I_class"] + "_" + ev_major["GW_class"]

ev.to_csv(os.path.join(OUT, "event_library.csv"), index=False)

# ---------- 5. 报告 ----------
if os.path.exists(os.path.join(OUT, "event_library_report.md")):
    os.remove(os.path.join(OUT, "event_library_report.md"))
w("# P0 收尾：整场暴雨事件库构建报告\n")
w(f"- 雨量输入: {n_gauges} 站面平均(>=60% 站有值), 序列 {rain.index.min():%Y-%m-%d} → {rain.index.max():%Y-%m-%d}")
w(f"- 事件划分口径: 面雨量 ≥ {MIN_DEPTH} mm, 场次间无雨间隔 > {MIN_GAP}; 可用地下水钻孔 {gw.shape[1]} 个")
w(f"- 事件总数(全序列): **{len(ev)}**; 其中深度 ≥5 mm 的主体分析子集 **{len(ev_major)}**")
w(f"- 五类齐备窗口(2025-06-30→2026-04-30)内事件: **{int(ev['in_common_window'].sum())}**"
  f"(全部 ≥2 mm) / 主体子集内: **{int(ev_major['in_common_window'].sum())}**")
w(f"- 五类传感器均覆盖(各类型活跃站>0): 全部事件 {int(ev['full_types'].sum())}, "
  f"窗口内 {int((ev['full_types'] & ev['in_common_window']).sum())}")
w(f"- 雨强三分位阈值(按主体子集峰值 1h 雨强): T1≤{q1:.2f}, T2≤{q2:.2f}, T3>{q2:.2f} mm/h\n")

w("## 分层样本量(主体子集 depth≥5mm, 窗口内)\n")
sub = ev_major[ev_major["in_common_window"]]
tab = sub.groupby(["I_class", "GW_class"]).size().unstack(fill_value=0).reset_index()
w(md_table(tab))
w("")
w("## 分层样本量 × 五类齐备(传感器均覆盖)\n")
tab2 = sub[sub["full_types"]].groupby(["I_class", "GW_class"]).size().unstack(fill_value=0).reset_index()
w(md_table(tab2))
w("")
w("## 事件库概览(全部 ≥2mm 事件, 按深度降序前 25)\n")
cols = ["start", "end", "duration_h", "depth_mm", "peak_1h_mmh", "API7_mm",
        "GW_pct", "gully_records", "n_sewer_active", "in_common_window"]
show = ev.sort_values("depth_mm", ascending=False).head(25)[cols].copy()
show["start"] = show["start"].dt.strftime("%Y-%m-%d %H:%M")
show["end"] = show["end"].dt.strftime("%Y-%m-%d %H:%M")
w(md_table(show))
w("")
w("## 口径说明与预注册声明\n")
w("- 事件划分/分层阈值在分析前固定; 后续矩阵 A/B/C 一律使用本事件库")
w("- GW_pct 为跨钻孔站内百分位均值, High≥0.5; 埋深(WS 族)已取负统一为越大越湿")
w("- gully_records 为 15 个雨水口事件窗口内实际上报条数; 事件触发采样, 无记录≠无积水")
w("- 敏感性口径(3h/12h 间隔)可在 event_library.csv 基础上重算; 本报告以 6h 为准")
print("done:", len(ev), "events,", len(ev_major), "major; report written")
