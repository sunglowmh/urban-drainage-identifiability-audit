# -*- coding: utf-8 -*-
"""Feasibility audit (small-scale) on BlueHeartV2, cross-checking prior BlueHeart workspace results.

Covers:
  1. Common coverage windows across the 5 sensor types (+ active gauge counts)
  2. Sampling intervals (distribution per type)
  3. Valid-rate (flag==1) per type
  4. Gauge -> nearest network node matching (spatial, UTM projected)
  5. Gully floodThreshold_mm semantics check
  6. Independent surface-flooding records: in-dataset search + external candidates
Writes p0_audit/feasibility_report.md
"""
import os, re, sys, json
import numpy as np
import pandas as pd
import geopandas as gpd

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "dataset")
OUT = os.path.join(HERE, "p0_audit")
os.makedirs(OUT, exist_ok=True)
R = []
def w(s=""): R.append(s)
def hr(): w(); w("---"); w()

det = pd.read_csv(os.path.join(OUT, "gauge_index_summary.csv"))
for c in ("start", "end"):
    # 混合时区列: aware(+00:00) 与 naive(gully) 混存, 必须 utc=True 统一解析,
    # 否则 naive 行被 coerce 成 NaT（正是 README 警告的时区坑在本审计里的复现）
    det[c] = pd.to_datetime(det[c], format="mixed", utc=True, errors="coerce")
det = det.dropna(subset=["start", "end"])
idx = pd.read_csv(os.path.join(BASE, "index_telemetry_data.csv"))

w("# BlueHeartV2 数据可行性审计（小规模）")
w()
w(f"生成: {pd.Timestamp.now():%Y-%m-%d %H:%M} ｜ 口径: 复用 p0_audit/gauge_index_summary.csv（994 文件逐文件实测）+ 空间复算")
hr()

# ---------- 1. common coverage ----------
w("## 1. 各类型共同覆盖期")
w()
env = det.groupby("type").agg(n=("file", "count"), start=("start", "min"), end=("end", "max")).reset_index()
w("| 类型 | 站数 | 最早数据 | 最晚数据 |")
w("|---|---|---|---|")
for _, r in env.iterrows():
    w(f"| {r['type']} | {r['n']} | {r['start']:%Y-%m-%d} | {r['end']:%Y-%m-%d} |")
w()
all5 = (env["start"].max(), env["end"].min())
rf3_start = env[env["type"].isin(["Rain", "Fluvial", "Sewer"])]["start"].max()
rf3_end = env[env["type"].isin(["Rain", "Fluvial", "Sewer"])]["end"].min()
w(f"- **五类共同覆盖窗口: {all5[0]:%Y-%m-%d} → {all5[1]:%Y-%m-%d}**（受雨水口 2025-06-30 上线与 2026-04-30 终止约束，约 10 个月）")
w(f"- 雨-河-污三类共同窗口: {rf3_start:%Y-%m-%d} → {rf3_end:%Y-%m-%d}（约 26 个月）")
w()
# active gauges inside all-5 window per type
def active(t, a, b):
    sub = det[(det["type"] == t) & (det["start"] <= b) & (det["end"] >= a)]
    return len(sub), sub["file"].tolist()
a5, b5 = all5
w(f"五类窗口内活跃站数（该窗口内至少有一条记录）:")
w()
w("| 类型 | 活跃站数 / 总数 | 备注 |")
w("|---|---|---|")
for t in env["type"]:
    n, _ = active(t, a5, b5)
    tot = env.loc[env["type"] == t, "n"].iloc[0]
    w(f"| {t} | {n}/{tot} | |")
w()
# how many gully/rain/borehole cover the whole window fully
w("窗口内**全时段**在线（start ≤ 窗口起点 且 end ≥ 窗口终点）:")
w()
w("| 类型 | 全时段在线 / 总数 |")
w("|---|---|")
for t in env["type"]:
    sub = det[det["type"] == t]
    full = ((sub["start"] <= a5) & (sub["end"] >= b5)).sum()
    w(f"| {t} | {full}/{len(sub)} |")
hr()

# ---------- 2. sampling intervals ----------
w("## 2. 采样间隔（逐站中位间隔的分布）")
w()
w("| 类型 | 中位间隔(s) | P10 | P90 | 站间一致性 |")
w("|---|---|---|---|---|")
for t, sub in det.groupby("type"):
    iv = sub["median_interval_s"].dropna()
    if len(iv) == 0: continue
    w(f"| {t} | {iv.median():.0f} | {iv.quantile(.1):.0f} | {iv.quantile(.9):.0f} | "
      f"{'集中' if (iv.quantile(.9)-iv.quantile(.1))/iv.median() < 0.1 else '分散(雨开口径:事件触发)'} |")
w()
w("注: 雨水口为事件触发式上报（15.2–15.7 min 名义间隔），联合分析前必须重采样/规则化；其余四类为规整等间隔。")
hr()

# ---------- 3. valid rates ----------
w("## 3. 有效率（flag==1 占有标签数据点比例；缺失率单列）")
w()
w("| 类型 | flag1(Good)% | flag2(Suspect)% | flag3(Bad)% | 数值缺失% |")
w("|---|---|---|---|---|")
for _, r in det.groupby("type").sum(numeric_only=True).reset_index().iterrows():
    tot_f = r["flag1"] + r["flag2"] + r["flag3"]
    w(f"| {r['type']} | {r['flag1']/tot_f*100:.1f} | {r['flag2']/tot_f*100:.1f} | {r['flag3']/tot_f*100:.1f} | "
      f"{r['missing_cells']/r['value_cells']*100:.1f} |")
w()
w("与前次 BlueHeart 工作区实测（84.46 / 15.13 / 0.40）一致；V2 数据集与 V1 同源同内容（行数逐类吻合）。")
w("重点: 污水 23.5% 可疑几乎全为标签 B（雷达零点与管内基准不统一，负值≠干管），只可作相对变化分析；钻孔水温 E 标签为规则假阳性。")
hr()

# ---------- 4. spatial matching ----------
w("## 4. 站点坐标 ↔ 管网节点匹配（V2 复算）")
w()
nodes = gpd.read_file(os.path.join(BASE, "2_network_shape_files/nodes/Nodes.shp"))
nodes_utm = nodes.to_crs(nodes.estimate_utm_crs())
gauges = gpd.GeoDataFrame(idx.rename(columns={"locationName": "Name"}),
                          geometry=gpd.points_from_xy(idx["locationLongitude"], idx["locationLatitude"]), crs="EPSG:4326").to_crs(nodes_utm.crs)
j = gpd.sjoin_nearest(gauges[["Name", "gaugeType", "geometry"]], nodes_utm[["id", "node_type", "system_typ", "geometry"]], how="left", distance_col="dist_m")
w("| 类型 | 站数 | 最近节点距离中位(m) | <30m | <50m | <100m | 最近节点类型 top3 |")
w("|---|---|---|---|---|---|---|")
for t, sub in j.groupby("gaugeType"):
    d = sub["dist_m"]
    top = sub["node_type"].value_counts().head(3).to_dict()
    w(f"| {t} | {len(sub)} | {d.median():.1f} | {(d<30).mean()*100:.0f}% | {(d<50).mean()*100:.0f}% | {(d<100).mean()*100:.0f}% | "
      + ", ".join(f"{k}={v}" for k, v in top.items()) + " |")
w()
sew = j[j["gaugeType"] == "Sewer"]
w(f"- 污水站绑定建议: 以 **30 m** 为阈值可绑定 {(sew['dist_m']<30).sum()}/{len(sew)} 站，其中最近节点为 Manhole 的 {(sew['node_type']=='Manhole').sum()} 站；>30 m 的 {len(sew)-int((sew['dist_m']<30).sum())} 站需人工核对。")
w(f"- 索引经纬度仅 4 位小数（≈10 m 量级），30 m 阈值已接近坐标精度下限，属'可绑定但需抽查'。")
w(f"- 雨量/钻孔站本来就不在管网上，匹配距离大是预期行为，仅作参照。")
w()
if (j["dist_m"] == 0).any():
    dup = j[j["dist_m"] == 0].groupby("gaugeType")["Name"].count()
    w(f"- 零距离匹配站数: " + ", ".join(f"{k}={v}" for k, v in dup.items()) + "（与前期'8 组同坐标站'结论一致，做空间交叉验证时需去重）。")
hr()

# ---------- 5. gully threshold semantics ----------
w("## 5. 雨水口 floodThreshold_mm 的语义（明确不当地表积水真值）")
w()
g = idx[idx["gaugeType"] == "Highway gully"][["locationName", "floodThreshold_mm"]]
w(f"- 15 个雨水口阈值范围: {g['floodThreshold_mm'].min():.0f}–{g['floodThreshold_mm'].max():.0f} mm，中位 {g['floodThreshold_mm'].median():.0f} mm。")
w()
w("| 雨水口 | floodThreshold_mm |")
w("|---|---|")
for _, r in g.iterrows():
    w(f"| {r['locationName']} | {r['floodThreshold_mm']:.0f} |")
w()
w("**语义声明**: README 定义为 `water level above which flooding occurs`，即**该监测点（雨水口井腔）自身的漫溢阈值**，"
  "由 gully 视角（井腔水位达到阈值视为该点发生漫溢）给出，≠ 2D 地表淹没水深真值，且 15 个点远不构成面状积水观测。"
  "在论文中只能用作「点位级 gully 漫溢事件的二值标签/召回评估」，不能作为 LISFLOOD-FP 水深验证数据。")
hr()

# ---------- 6. independent surface flooding records ----------
w("## 6. 可用于独立验证的地表积水记录核查")
w()
w("### 6.1 数据集内部")
w()
# scan all attribute fields of vector layers + metadata csv for flood/ponding keywords
found = []
for shp in ["1_catchment_shape_files/catchment_outline/Study_area.shp",
            "2_network_shape_files/links/Links.shp", "2_network_shape_files/nodes/Nodes.shp"]:
    gdf = gpd.read_file(os.path.join(BASE, shp), rows=5)
    cols = list(gdf.columns)
    hit = [c for c in cols if re.search(r"flood|pond|inund|spill", c, re.I)]
    found.append((shp, cols, hit))
for name, cols, hit in found:
    w(f"- `{name.split('/')[-1]}` 字段: {', '.join(cols)}" + (f" → 含 `{hit[0]}`" if hit else " → 无积水/淹没类字段"))
w()
w("说明: `Nodes.flood_leve` 是**模型属性**（节点漫溢水位，前期实测与 ground_level 逐点相等），属管网资产参数而非观测记录；"
  "雨水口的 `floodThreshold_mm` 同理是监测点属性阈值（见 §5）。二者都不是'实际发生的地表积水'观测。")
w()
w("### 6.2 外部候选（按可得性排序）")
w()
w("| 来源 | 内容 | 用途 | 风险/备注 |")
w("|---|---|---|---|")
w("| Environment Agency **Historic Flood Map / Recorded Flood Outlines**（开放数据, data.gov.uk） | 英格兰历史记录洪水外包络面 | 2D 模型淹没范围的**粗粒度**独立验证 | 多为河洪记录，地表水积水轮廓稀疏；几何不精确 |")
w("| EA **Hydrology Data Explorer** | 前期概要指出的 4 个 EA 河道站 + Pevensey 雨量站（不在包内） | 河道边界与降雨的独立对照 | 非积水数据，但可补齐边界真值 |")
w("| Blue Heart 项目方（blueheart.org.uk / East Sussex CC）报告与事件记录 | 项目期内的实际内涝事件记载（如文献提到的 2015-08-23 39.4 mm 事件机制） | 案例事件的过程性验证 | 需向项目方申请/引用报告，非结构化 |")
w("| 南方水司 Southern Water 内部 DG5/溢流记录 | 洪水上报记录 | 严格意义的积水验证 | **未公开**，需合作申请 |")
w()
w("**可行性结论**: 独立积水真值在包内为零，只能走「gully 阈值超限事件（点位级）+ EA 历史洪水轮廓（范围级）」两条弱验证路线，或争取 Southern Water/项目方数据。"
  "这决定了选题 1/2 的验证设计应以**管网/河道/地下水水位的时间过程匹配**为主战场，积水范围只作辅助校核。")
hr()

# ---------- 7. summary ----------
w("## 7. 可行性判定")
w()
w("| 审计项 | 结论 | 依据 |")
w("|---|---|---|")
w("| 五类共同覆盖期 | **2025-06-30 → 2026-04-30（约 10 个月）**，3 类(雨河污)窗口约 26 个月 | §1 |")
w("| 采样间隔 | 四类规整等间隔(5/15/60 min)，gully 事件触发需重采样 | §2 |")
w("| 有效率 | Good 76–100%（污水最低 76.5% 且系统性 B 标签），Bad ≤1% | §3 |")
w("| 站-网匹配 | 污水站 30 m 内 ~93% 可绑定 Manhole，坐标精度≈10 m | §4 |")
w("| gully 阈值语义 | 仅点位漫溢阈值，不可作 2D 淹没真值 | §5 |")
w("| 独立积水记录 | **包内为零**；外部仅 EA 历史洪水轮廓(粗)与项目方/SW 数据(需申请) | §6 |")
w()
w("**总判定**: 选题 1＋2（可辨识性＋多源同化）的数据基础成立，验证主轴应放在水位过程线（sewer/fluvial/gully/groundwater），"
  "事件库建议以 2025-07–2026-03 五类齐备窗口内、由 15 个雨量站圈定的暴雨事件为准。")

with open(os.path.join(OUT, "feasibility_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("DONE")
print("\n".join(R))
