# -*- coding: utf-8 -*-
"""
P1.3: 试点子流域选择
方法:
  1. sewer 活跃站(窗口内)匹配到最近管网节点(30m, EPSG:27700)
  2. 500m 网格密度 -> 取密度最高网格为中心, 1.5km 半径截取子网
  3. 子网校验: conduit 连通性 / 是否到达 outfall / 节点与传感器数
  4. 输出: output/p1/pilot_selection.md + pilot_subnetwork.csv(节点清单)
"""
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

BASE = "E:/work/workBuddy/BlueHeartV2"
DS = os.path.join(BASE, "dataset")
OUT = os.path.join(BASE, "output", "p1")
os.makedirs(OUT, exist_ok=True)

nd = gpd.read_file(os.path.join(DS, "2_network_shape_files", "nodes", "Nodes.shp")).to_crs(27700)
lk = gpd.read_file(os.path.join(DS, "2_network_shape_files", "links", "Links.shp")).to_crs(27700)
nd["x"] = nd.geometry.x
nd["y"] = nd.geometry.y

idx = pd.read_csv(os.path.join(DS, "index_telemetry_data.csv"))
sewer = idx[idx["gaugeType"] == "Sewer"].copy()
sewer_g = gpd.GeoDataFrame(sewer, geometry=gpd.points_from_xy(sewer["locationLongitude"],
                                                             sewer["locationLatitude"]),
                           crs=4326).to_crs(27700)
tree = cKDTree(nd[["x", "y"]].values)
dist, ix = tree.query(list(zip(sewer_g.geometry.x, sewer_g.geometry.y)), k=1)
sewer_g["node_id"] = nd["id"].values[ix]
sewer_g["dist_m"] = dist
matched = sewer_g[sewer_g["dist_m"] <= 30]
print(f"sewer stations matched <=30m: {len(matched)}/{len(sewer_g)}")

# 活跃期过滤: 用审计明细
det = pd.read_csv(os.path.join(BASE, "p0_audit", "gauge_index_summary.csv"), dtype=str)
det = det[det["type"] == "Sewer"]
det["start"] = pd.to_datetime(det["start"], format="mixed", utc=True)
det["end"] = pd.to_datetime(det["end"], format="mixed", utc=True)
win_s, win_e = pd.Timestamp("2025-06-30", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC")
active_files = set(det[(det["start"] <= win_s) & (det["end"] >= win_e)]["file"])
# 审计明细 file 字段与 index Name 对应关系: 直接用文件名匹配
matched = matched.copy()
matched["file"] = matched["dataPath"].map(lambda p: os.path.basename(str(p)))
matched["active_window"] = matched["file"].isin(active_files)
act = matched[matched["active_window"]]
print(f"active in window: {len(act)}")

# 500m 网格密度
act = act.copy()
act["gx"] = (act.geometry.x // 500).astype(int)
act["gy"] = (act.geometry.y // 500).astype(int)
cell = act.groupby(["gx", "gy"]).size().sort_values(ascending=False)
top_cell = cell.index[0]
cx, cy = (top_cell[0] + 0.5) * 500, (top_cell[1] + 0.5) * 500
print("top cell center:", cx, cy, "count:", cell.iloc[0])

# 1.5km 半径子网
R = 1500.0
near = (nd["x"] - cx) ** 2 + (nd["y"] - cy) ** 2 <= R ** 2
sub_nodes = set(nd.loc[near, "id"])
sub_lk = lk[(lk["us_node_id"].isin(sub_nodes)) & (lk["ds_node_id"].isin(sub_nodes))]
sub_cond = sub_lk[sub_lk["link_type"] == "Conduit"]
sub_act = act[act["node_id"].isin(sub_nodes)]
n_out = int(nd[near & (nd["node_type"] == "Outfall")].shape[0])
print(f"pilot subnetwork: nodes={near.sum()}, conduits={len(sub_cond)}, "
      f"active sewer sensors={len(sub_act)}, outfalls={n_out}")

# 连通性: BFS from 密度中心最近节点
seg = sub_cond[["us_node_id", "ds_node_id"]]
adj = {}
for u, v in seg.values:
    adj.setdefault(u, set()).add(v)
    adj.setdefault(v, set()).add(u)
start_node = nd.loc[(nd["x"] - cx) ** 2 + (nd["y"] - cy) ** 2 < 250000, "id"]
start_node = start_node.iloc[0] if len(start_node) else sorted(sub_nodes)[0]
seen, stack = {start_node}, [start_node]
while stack:
    n = stack.pop()
    for m in adj.get(n, ()):
        if m not in seen:
            seen.add(m)
            stack.append(m)
out_ids = set(nd.loc[near & (nd["node_type"] == "Outfall"), "id"])
reaches_out = bool(seen & out_ids)
print(f"connected component from center: {len(seen)} nodes, reaches outfall: {reaches_out}")

# 事件子集: 从 38 场窗口事件按分层挑 5 场典型
ev = pd.read_csv(os.path.join(BASE, "p0_audit", "event_library.csv"),
                 parse_dates=["start", "end"])
ev["start"] = ev["start"].dt.tz_convert("UTC")
ev["end"] = ev["end"].dt.tz_convert("UTC")
sub_ev = ev[ev["in_common_window"] & (ev["depth_mm"] >= 5)].copy()
q1, q2 = sub_ev["peak_1h_mmh"].quantile([1/3, 2/3])
sub_ev["I"] = np.where(sub_ev["peak_1h_mmh"] <= q1, "T1",
                       np.where(sub_ev["peak_1h_mmh"] <= q2, "T2", "T3"))
sub_ev["GW"] = np.where(sub_ev["GW_pct"] >= 0.5, "High", "Low")
picks = pd.concat([
    sub_ev.sort_values("depth_mm", ascending=False).head(1),            # 最大事件
    sub_ev[(sub_ev["I"] == "T3") & (sub_ev["GW"] == "Low")].sort_values("peak_1h_mmh", ascending=False).head(1),
    sub_ev[(sub_ev["I"] == "T2")].sort_values("depth_mm", ascending=False).head(1),
    sub_ev[(sub_ev["I"] == "T1") & (sub_ev["GW"] == "High")].sort_values("depth_mm", ascending=False).head(1),
    sub_ev[(sub_ev["I"] == "T1") & (sub_ev["GW"] == "Low")].head(1),    # 薄格层唯一事件
]).drop_duplicates(subset="start")
picks_out = picks[["start", "end", "depth_mm", "peak_1h_mmh", "GW_pct", "duration_h"]].copy()
picks_out["start"] = picks_out["start"].dt.strftime("%Y-%m-%d %H:%M")
picks_out["end"] = picks_out["end"].dt.strftime("%Y-%m-%d %H:%M")

# 交付
nd.loc[near, ["id", "node_type", "ground_lev"]].to_csv(
    os.path.join(OUT, "pilot_subnetwork_nodes.csv"), index=False)
act_out = sub_act[["locationName", "node_id", "dist_m"]].copy()
act_out.to_csv(os.path.join(OUT, "pilot_sensors.csv"), index=False)

with open(os.path.join(OUT, "pilot_selection.md"), "w", encoding="utf-8") as f:
    f.write("# P1.3 试点子流域选择\n\n")
    f.write("## 选址方法\n\n")
    f.write("1. 窗口内活跃 sewer 站匹配最近管网节点(≤30 m, EPSG:27700);\n")
    f.write("2. 500 m 网格统计传感器密度, 取最高密度网格为中心;\n")
    f.write("3. 以中心为圆心 1.5 km 半径截取子网并校验连通性。\n\n")
    f.write(f"## 结果\n\n")
    f.write(f"- 活跃 sewer 站(窗口): {len(act)} / {len(sewer_g)} (30 m 内匹配)\n")
    f.write(f"- 试点中心: EPSG:27700 ({cx:.0f}, {cy:.0f}), 中心网格传感器 {cell.iloc[0]} 站\n")
    f.write(f"- 试点子网(1.5 km 半径): 节点 {near.sum()}, conduit {len(sub_cond)}, "
            f"活跃 sewer 传感器 {len(sub_act)}, outfall {n_out}\n")
    f.write(f"- 连通性: 从中心 BFS 可达 {len(seen)} 节点, 到达 outfall: {reaches_out}\n\n")
    f.write("## 试点事件子集(5 场, 覆盖雨强×前期地下水分层)\n\n")
    f.write("| start | end | depth_mm | peak_1h_mmh | duration_h | GW_pct |\n|---|---|---|---|---|---|\n")
    for _, r in picks_out.iterrows():
        f.write(f"| {r['start']} | {r['end']} | {r['depth_mm']:.1f} | {r['peak_1h_mmh']:.2f} | "
                f"{r['duration_h']:.1f} | {r['GW_pct']:.2f} |\n")
    f.write("\n## 说明\n\n")
    f.write("- 第 5 行为 T1_Low 薄格层唯一事件(见 event_library_report 薄格提示), 仅作探索性;\n")
    f.write("- 试点用于 P1 阶段一(小规模模型验证): 状态-参数映射、误差模型、代理误差、参数混淆检查;\n")
    f.write("- 事件期实测 sewer 水位(试点内)作为阶段一推断的观测输入。\n")
print("pilot selection written")
