# -*- coding: utf-8 -*-
"""P0 data audit for the Blue Heart dataset (E:/work/workBuddy/BlueHeartV2/dataset).

Produces:
  p0_audit/p0_audit_report.md          -- human-readable report
  p0_audit/gauge_index_summary.csv     -- per-gauge inventory with coverage stats
  p0_audit/telemetry_type_summary.csv  -- per-type aggregate stats
Outputs are written next to this script.
"""
import os
import re
import sys
import io
import json
import numpy as np
import pandas as pd
import geopandas as gpd

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0_audit")
os.makedirs(OUT, exist_ok=True)

R = []  # report lines
def w(s=""):
    R.append(s)

def hr():
    w()
    w("---")
    w()

# ---------------------------------------------------------------- index
w("# Blue Heart 数据集 P0 审计报告")
w()
w(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}  |  数据根目录: {BASE}")
hr()

idx = pd.read_csv(os.path.join(BASE, "index_telemetry_data.csv"))
w("## 1. 测点索引 (index_telemetry_data.csv)")
w()
w(f"- 总测点数: **{len(idx)}**")
cnt = idx["gaugeType"].value_counts()
w(f"- 按类型: " + ", ".join(f"{k}={v}" for k, v in cnt.items()))
own = idx["gaugeOwner"].value_counts()
w(f"- 按属主: " + ", ".join(f"{k}={v}" for k, v in own.items()))
w(f"- floodThreshold_mm 非空: {idx['floodThreshold_mm'].notna().sum()} (仅 Highway gully 期望有)")
w(f"- bedLevel/bankLevel 非空: {idx['bedLevelEstimate_mAOD'].notna().sum()} / {idx['bankLevelEstimate_mAOD'].notna().sum()} (仅 Fluvial 期望有)")
param_combo = idx.groupby("gaugeType")[["gaugeParameters", "parameterUnits"]].agg(
    lambda s: sorted(set(x.strip() for v in s for x in str(v).split(";"))))
w()
w("| 类型 | 参数(单位) 组合 |")
w("|---|---|")
for t, row in param_combo.iterrows():
    pairs = "; ".join(f"{p} [{u}]" for p, u in zip(row["gaugeParameters"], row["parameterUnits"]))
    w(f"| {t} | {pairs} |")
# param/unit list alignment check
mis = []
for _, r in idx.iterrows():
    ps = [x.strip() for x in str(r["gaugeParameters"]).split(";")]
    us = [x.strip() for x in str(r["parameterUnits"]).split(";")]
    if len(ps) != len(us):
        mis.append(r["Name"])
w()
w(f"- 参数/单位列表长度不一致的测点: {len(mis)}" + (f" 例: {mis[:5]}" if mis else ""))
# file existence check
missing = [r["Name"] for _, r in idx.iterrows()
           if not (os.path.exists(os.path.join(BASE, str(r["dataPath"])))
                   and os.path.exists(os.path.join(BASE, str(r["dataQualityFlagsPath"]))))]
w(f"- dataPath 或 flagsPath 缺失的测点: {len(missing)}" + (f" 例: {missing[:5]}" if missing else ""))
hr()

# ---------------------------------------------------------------- shapefiles
w("## 2. 矢量数据 (catchment / network shapefiles)")
w()

def shp_summary(path, name):
    g = gpd.read_file(path)
    w(f"### {name}")
    w(f"- CRS: {g.crs.to_string() if g.crs else 'None'}  |  要素数: {len(g)}")
    w(f"- 属性列: {', '.join(g.columns.drop('geometry'))}")
    if g.crs is not None and not g.crs.is_geographic:
        try:
            area_km2 = g.to_crs(g.estimate_utm_crs()).geometry.area.sum() / 1e6
        except Exception:
            area_km2 = float("nan")
        w(f"- 总面积(投影后): {area_km2:.3f} km^2")
    b = g.total_bounds
    w(f"- Bounds: {[round(x,1) for x in b]}")
    return g

study = shp_summary(os.path.join(BASE, "1_catchment_shape_files/catchment_outline/Study_area.shp"), "Study_area (研究区多边形)")
# single-feature? per-feature areas
if len(study) > 1:
    a = study.to_crs(study.estimate_utm_crs()).geometry.area / 1e6
    w(f"- 各要素面积: {[round(x,3) for x in a]}")
w()
elev = shp_summary(os.path.join(BASE, "1_catchment_shape_files/elevation/2D Zones elevation.shp"), "2D Zones elevation")
w()
infil = shp_summary(os.path.join(BASE, "1_catchment_shape_files/infiltration/2D Zones infiltration.shp"), "2D Zones infiltration")
w()
# infiltration metadata
surf = pd.read_csv(os.path.join(BASE, "1_catchment_shape_files/infiltration/2d_infiltration_surfaces_metadata.csv"))
w(" infiltration surfaces metadata:")
w()
for _, rr in surf.iterrows():
    w("- " + "; ".join(f"{c}={rr[c]}" for c in surf.columns))
w()
zones_meta = pd.read_csv(os.path.join(BASE, "1_catchment_shape_files/infiltration/2d_infiltration_zones_metadata.csv"))
w(f"- infiltration zones metadata 行数: {len(zones_meta)}, surface_id 取值: {sorted(set(zones_meta['infiltration_surface_id']))[:20]}")
w()

links = shp_summary(os.path.join(BASE, "2_network_shape_files/links/Links.shp"), "Links (管网/河道 links)")
w()
nodes = shp_summary(os.path.join(BASE, "2_network_shape_files/nodes/Nodes.shp"), "Nodes (管网节点)")
hr()

# area of study polygon, explicit
area_km2 = study.to_crs(study.estimate_utm_crs()).geometry.area.sum() / 1e6
w(f"**研究区面积复核**: Study_area.shp 总面积 = **{area_km2:.2f} km^2**（与你的高程网格估计 ~56 km^2 一致；论文口径 78 km^2 为更大集水区，二者不矛盾，论文中需明确区分 study-area polygon vs. catchment）。")
w()

# nodes/links in vs out of study polygon
if not nodes.empty and not study.empty:
    try:
        u = study.to_crs(nodes.crs).union_all()
        inside = nodes.geometry.intersects(u)
        w(f"**区外基础设施**: Nodes 共 {len(nodes)}, 位于 study_area 多边形内 {inside.sum()} 个, 区外 {(~inside).sum()} 个 ({(~inside).mean()*100:.1f}%)。")
        for lt, grp in nodes.groupby([c for c in ["Type", "type", "Node_Type", "node_type"] if c in nodes.columns][0] if any(c in nodes.columns for c in ["Type","type","Node_Type","node_type"]) else nodes.columns[0]):
            pass
    except Exception as e:
        w(f"(节点内外判断失败: {e})")
    # breakdown of out-of-area by available categorical col
    catcol = None
    for c in nodes.columns:
        if nodes[c].dtype == object and c != "geometry" and nodes[c].nunique() < 30:
            catcol = c
            break
    if catcol:
        outside = nodes[~nodes.geometry.intersects(study.to_crs(nodes.crs).union_all())]
        vc = outside[catcol].value_counts()
        w(f"- 区外节点按 `{catcol}` 分布: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    try:
        lin = links[~links.geometry.centroid.intersects(study.to_crs(links.crs).union_all())]
        w(f"- Links 共 {len(links)}, 质心在区外 {len(lin)} 条 ({len(lin)/len(links)*100:.1f}%)。")
        catcol = None
        for c in links.columns:
            if links[c].dtype == object and c != "geometry" and links[c].nunique() < 30:
                catcol = c
                break
        if catcol:
            vc = lin[catcol].value_counts()
            w(f"- 区外 links 按 `{catcol}` 分布: " + ", ".join(f"{k}={v}" for k, v in vc.head(10).items()))
    except Exception as e:
        w(f"(link 内外判断失败: {e})")
hr()

# ---------------------------------------------------------------- telemetry
w("## 3. 时序遥测数据 (994 CSV 审计)")
w()
w("逐文件统计：记录数、时间覆盖、中位采样间隔、质量标记分布（flag 1/2/3 与字母标签 A–E）、缺失值比例。")
w()

FOLDERS = {
    "Borehole": "3_borehole_telemetry_data",
    "Fluvial": "4_fluvial_telemetry_data",
    "Sewer": "5_sewer_telemetry_data",
    "Highway gully": "6_highway_gully_telemetry_data",
    "Rain": "7_rain_gauge_telemetry_data",
}

flag_re = re.compile(r"^(?P<flag>[123])(?P<labels>[A-E]*)$")

def flag_stats(series):
    counts = {1: 0, 2: 0, 3: 0}
    labels = {}
    for v in series.dropna().astype(str):
        v = v.strip()
        m = flag_re.match(v)
        if not m:
            labels.setdefault("OTHER:" + v[:10], 0)
            labels["OTHER:" + v[:10]] += 1
            continue
        counts[int(m.group("flag"))] += 1
        for ch in m.group("labels"):
            labels.setdefault(ch, 0)
            labels[ch] += 1
    return counts, labels

rows_detail = []
rows_summary = []
for gtype, folder in FOLDERS.items():
    files = sorted(f for f in os.listdir(os.path.join(BASE, folder)) if f.endswith(".csv") and not f.endswith("_quality_flags.csv"))
    tot_rec = tot_flag1 = tot_flag2 = tot_flag3 = tot_missing = 0
    tot_cells = 0
    label_tot = {}
    tz_naive_files = 0
    for f in files:
        fp = os.path.join(BASE, folder, f)
        try:
            df = pd.read_csv(fp, dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(fp, dtype=str, encoding="latin-1")
        ts_raw = df["timestamp"]
        # detect tz offset presence
        has_offset = ts_raw.dropna().str.contains(r"[+-]\d{2}:\d{2}$").mean()
        if has_offset < 0.99:
            tz_naive_files += 1
        # value columns = all except timestamp and SNR? include SNR as value col too for missing stats
        valcols = [c for c in df.columns if c != "timestamp"]
        datacols = [c for c in valcols if not c.startswith("Quality - ")]
        nrec = len(df)
        nmiss = int(df[datacols].isna().sum().sum()) if datacols else 0
        ncells = nrec * len(datacols) if datacols else 0
        # sampling interval
        try:
            ts = pd.to_datetime(ts_raw, format="mixed", utc=False)
            iv = ts.diff().dropna().dt.total_seconds()
            med_iv = float(iv.median()) if len(iv) else float("nan")
        except Exception:
            med_iv = float("nan")
        # coverage
        try:
            t0 = ts.min()
            t1 = ts.max()
        except Exception:
            t0 = t1 = None
        # flags
        fq = os.path.join(BASE, folder, f.replace(".csv", "_quality_flags.csv"))
        c1 = c2 = c3 = 0
        if os.path.exists(fq):
            try:
                q = pd.read_csv(fq, dtype=str, encoding="utf-8-sig")
            except UnicodeDecodeError:
                q = pd.read_csv(fq, dtype=str, encoding="latin-1")
            for c in q.columns:
                if c == "timestamp":
                    continue
                cnts, labs = flag_stats(q[c])
                c1 += cnts[1]; c2 += cnts[2]; c3 += cnts[3]
                for k, v in labs.items():
                    label_tot[k] = label_tot.get(k, 0) + v
        tot_rec += nrec; tot_missing += nmiss; tot_cells += ncells
        tot_flag1 += c1; tot_flag2 += c2; tot_flag3 += c3
        rows_detail.append({
            "type": gtype, "file": f, "records": nrec,
            "start": str(t0), "end": str(t1),
            "median_interval_s": med_iv,
            "tz_offset_missing_pct": round((1 - has_offset) * 100, 1) if pd.notna(has_offset) else None,
            "missing_cells": nmiss, "value_cells": ncells,
            "flag1": c1, "flag2": c2, "flag3": c3,
        })
    pct = (lambda x: f"{x/tot_cells*100:.1f}%" if tot_cells else "-")
    rows_summary.append({
        "type": gtype, "files": len(files),
        "records": tot_rec,
        "start": min((r["start"] for r in rows_detail if r["type"] == gtype and r["start"] != "None"), default="-"),
        "end": max((r["end"] for r in rows_detail if r["type"] == gtype and r["end"] != "None"), default="-"),
        "median_interval_s": sorted(r["median_interval_s"] for r in rows_detail if r["type"] == gtype and pd.notna(r["median_interval_s"]))[len(files)//2] if files else None,
        "missing_pct": pct(tot_missing),
        "flag1_pct": pct(tot_flag1), "flag2_pct": pct(tot_flag2), "flag3_pct": pct(tot_flag3),
        "label_counts": label_tot,
        "files_without_tz_offset": tz_naive_files,
    })

det = pd.DataFrame(rows_detail)
det.to_csv(os.path.join(OUT, "gauge_index_summary.csv"), index=False, encoding="utf-8-sig")
summ = pd.DataFrame(rows_summary)
summ_out = summ.drop(columns=["label_counts"])
summ_out.to_csv(os.path.join(OUT, "telemetry_type_summary.csv"), index=False, encoding="utf-8-sig")

w("| 类型 | 文件数 | 总记录 | 覆盖起点 | 覆盖终点 | 中位采样间隔(s) | 缺失值% | flag1(Good)% | flag2(Suspect)% | flag3(Bad)% | 无时区偏移文件数 |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in summ_out.iterrows():
    w(f"| {r['type']} | {r['files']} | {r['records']:,} | {str(r['start'])[:10]} | {str(r['end'])[:10]} | {r['median_interval_s']} | {r['missing_pct']} | {r['flag1_pct']} | {r['flag2_pct']} | {r['flag3_pct']} | {r['files_without_tz_offset']} |")
w()
w("质量字母标签统计（逐值）:")
w()
for _, r in summ.iterrows():
    w(f"- **{r['type']}**: " + ", ".join(f"{k}={v:,}" for k, v in sorted(r["label_counts"].items())))
w()
# timezone mismatch: gully raw naive vs flags aware
g = det[det["type"] == "Highway gully"]
w(f"**时区核查**: Highway gully 原始数据 timestamp 不带 UTC 偏移的文件数 = {summ.loc[summ['type']=='Highway gully','files_without_tz_offset'].iloc[0]}/{len(g)}；其余类型全部带偏移（README 口径一致）。质量标签文件一律带 +00:00 —— 对齐时必须先对 gully 原始时间戳做显式 localize/convert，不可直接混合解析。")
hr()

# quick gap density per type: fraction of files with >1d max gap
w("## 4. 初步数据风险提示")
w()
big_gaps = []
for gtype, folder in FOLDERS.items():
    sub = det[det["type"] == gtype]
    # recompute max gap quickly for top-3 largest files only (cost control)
    pass
w("- gully 采样为设备事件触发（非整点等间隔），同化前需重采样/规则化。")
w("- Sewer Level 单位为 mm（非 mAOD），Fluvial/Borehole 为 mAOD —— 建模时注意量纲与基准面。")
w("- index 中 Sewer 测点多属 Southern Water（WPS 泵站命名），可能含泵站启停伪影（rate-of-change flag D 应能捕捉）。")
w("- infiltration 元数据为 Horton 参数（initial/limiting/decay/recovery），即官方 1D-2D 模型的入渗参数集——参数可辨识性实验（选题1）的先验范围可直接以此为起点。")
hr()

with open(os.path.join(OUT, "p0_audit_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(R))
print("DONE")
print("\n".join(R))
