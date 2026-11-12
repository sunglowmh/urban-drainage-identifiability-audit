# -*- coding: utf-8 -*-
"""
P1.5 + P1.6: 试点模型构建(产流面+网络+雨量) 与 30 站筛选、事件模拟对比
步骤:
  1. 30 站筛选: 试点 102 活跃 sewer 站按 窗口内有效率×记录量 排序取 30
  2. 产流面: infiltration 2D zones 就近分配到试点 junction, 聚合为 subcatchment
     (%Imperv = Σ area×coe / Σ area; Horton 参数面积加权; 坡度由 elevation zones 组内 p90-p10)
  3. 试点 inp: pilot conduit 子网 + 伪出流(终端) + subcatchments + 面雨量序列
  4. 事件模拟 (2025-10-22, 36h) + 30 站模拟深度 vs 观测水位相关
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

CX, CY, R = 561750.0, 99250.0, 1500.0
EVENT = {"start": "2025-10-22 06:00", "end": "2025-10-23 20:00"}  # 含 12h 缓冲

# ================= 1. 试点节点/网络 =================
nd = gpd.read_file(os.path.join(DS, "2_network_shape_files", "nodes", "Nodes.shp")).to_crs(27700)
lk = gpd.read_file(os.path.join(DS, "2_network_shape_files", "links", "Links.shp")).to_crs(27700)
pilot_ids = set(pd.read_csv(os.path.join(OUT, "pilot_subnetwork_nodes.csv"))["id"])
nd = nd[nd["id"].isin(pilot_ids)].copy()
nd["x"], nd["y"] = nd.geometry.x, nd.geometry.y

def clean_id(s):
    s = str(s).strip()
    for ch in (" ", ";", ",", "\t"):
        s = s.replace(ch, "_")
    return s[:38]

for df, cols in ((nd, ["id"]), (lk, ["id", "us_node_id", "ds_node_id"])):
    for c in cols:
        df[c] = df[c].map(clean_id)

cond = lk[(lk["link_type"] == "Conduit") & (lk["shape"] == "CIRC") &
          lk["us_node_id"].isin(pilot_ids) & lk["ds_node_id"].isin(pilot_ids)].copy()
cond["len_m"] = cond.geometry.length
cond["diameter_m"] = cond["conduit_wi"] / 1000.0
ok = cond["diameter_m"].between(0.05, 5.0) & cond["len_m"].between(1.0, 2000.0) \
     & cond["us_invert"].notna() & cond["ds_invert"].notna() \
     & (~cond["us_node_id"].isin(nd.loc[nd["node_type"] == "Outfall", "id"]))
cond = cond[ok].copy()

us_inv = cond.groupby("us_node_id")["us_invert"].min()
ds_inv = cond.groupby("ds_node_id")["ds_invert"].min()
node_inv = pd.concat([us_inv, ds_inv], axis=1).min(axis=1)
nd["invert"] = nd["id"].map(node_inv)
nd["invert"] = nd["invert"].fillna(nd["ground_lev"] - nd["DEPNOD"])
nd["inv_map"] = nd["id"].map(node_inv)  # 仅统计用
inv_map = nd.set_index("id")["invert"]
cond["off_us"] = (cond["us_invert"] - cond["us_node_id"].map(inv_map)).clip(lower=0)
cond["off_ds"] = (cond["ds_invert"] - cond["ds_node_id"].map(inv_map)).clip(lower=0)

ds_nodes = set(cond["ds_node_id"])
outfalls = nd[nd["node_type"] == "Outfall"].copy()
# 过流型 Outfall(作为 conduit 上游节点的 *_CI 涵洞入口等) -> 转 Junction
us_used = set(cond["us_node_id"])
pass_through = outfalls[outfalls["id"].isin(us_used)].copy()
outfalls = outfalls[~outfalls["id"].isin(us_used)].copy()
print(f"pass-through outfalls converted to junctions: {len(pass_through)}")
junctions = nd[nd["node_type"].isin(["Manhole", "Break", "Storage"])].copy()
junctions = pd.concat([junctions, pass_through], ignore_index=True)
junctions = junctions[~junctions["id"].isin(outfalls["id"])]
# 伪出流: 死端节点(有入流无出流) = 在 ds_nodes 但不在 us_nodes
us_nodes = set(cond["us_node_id"])
junctions["is_terminal"] = junctions["id"].isin(ds_nodes) & ~junctions["id"].isin(us_nodes)
pseudo_out = junctions[junctions["is_terminal"]].copy()
real_junc = junctions[~junctions["is_terminal"]].copy()
print(f"pilot: cond={len(cond)}, junc={len(real_junc)}, outfall={len(outfalls)}, pseudo_out={len(pseudo_out)}")

# Outfall(含伪出流)多入流 -> 虚拟汇流节点
dummy_nodes = []
outfall_all = pd.concat([outfalls, pseudo_out])
inlets_per_out = cond[cond["ds_node_id"].isin(outfall_all["id"])].groupby("ds_node_id").size()
for oid, n_in in inlets_per_out[inlets_per_out > 1].items():
    dummy = f"OUTJ_{oid}"[:38]
    row = outfall_all.loc[outfall_all["id"] == oid, "invert"]
    oinv = float(row.iloc[0]) if len(row) and pd.notna(row.iloc[0]) else 0.0
    dummy_nodes.append((dummy, oinv))
    cond.loc[cond["ds_node_id"] == oid, "ds_node_id"] = dummy
    dmax = float(cond.loc[cond["ds_node_id"] == dummy, "diameter_m"].max())
    cond = pd.concat([cond, pd.DataFrame([{
        "id": f"DUMMY_{oid}"[:38], "us_node_id": dummy, "ds_node_id": oid,
        "len_m": 1.0, "diameter_m": dmax, "off_us": 0.0, "off_ds": 0.0,
        "us_invert": oinv, "ds_invert": oinv, "geometry": None,
    }])], ignore_index=True)
print(f"pilot outfalls with multi-inlets: {len(dummy_nodes)}, dummies added")

# ================= 2. 30 站筛选 =================
det = pd.read_csv(os.path.join(BASE, "p0_audit", "gauge_index_summary.csv"), dtype=str)
det = det[det["type"] == "Sewer"].copy()
for c in ("records", "value_cells", "flag1", "flag2", "flag3"):
    det[c] = pd.to_numeric(det[c], errors="coerce")
det["valid_share"] = det["flag1"] / det["value_cells"]
det["dur_days"] = (pd.to_datetime(det["end"], format="mixed", utc=True)
                   - pd.to_datetime(det["start"], format="mixed", utc=True)).dt.total_seconds() / 86400
pilot_sensors = pd.read_csv(os.path.join(OUT, "pilot_sensors.csv"))
pilot_sensors["file"] = pilot_sensors["locationName"].astype(str) + ".csv"
sel = pilot_sensors.merge(det[["file", "records", "valid_share", "dur_days",
                               "start", "end", "median_interval_s"]], on="file", how="left")
# 窗口内活跃(已在 pilot_sensors 定义); 排序: 有效率主序, 记录量次序
sel = sel.sort_values(["valid_share", "records"], ascending=False)
sel30 = sel.head(30).copy()
sel30["node_id"] = sel30["node_id"].map(clean_id)
sel30[["locationName", "node_id", "dist_m", "valid_share", "records", "dur_days"]].to_csv(
    os.path.join(OUT, "pilot_30_stations.csv"), index=False)
print(f"30 stations selected, valid_share range: {sel30['valid_share'].min():.3f}–{sel30['valid_share'].max():.3f}")

# ================= 3. 产流面聚合 =================
inf_path = os.path.join(DS, "1_catchment_shape_files", "infiltration", "2D Zones infiltration.shp")
inf = gpd.read_file(inf_path).to_crs(27700)
inf["cx"] = inf.geometry.centroid.x
inf["cy"] = inf.geometry.centroid.y
in_pilot = (inf["cx"] - CX) ** 2 + (inf["cy"] - CY) ** 2 <= R ** 2
inf = inf[in_pilot].copy()
print(f"infiltration zones in pilot: {len(inf)}, area km2: {inf['area'].sum():.3f}")

jtree = cKDTree(real_junc[["x", "y"]].values)
d, ix = jtree.query(inf[["cx", "cy"]].values, k=1)
inf["node"] = real_junc["id"].values[ix]
inf["area_m2"] = inf["area"] * 1e4  # area 字段为公顷(ha)
fixed = inf[inf["runoff_vol"] == "Fixed"]
hort = inf[inf["runoff_vol"] != "Fixed"]
agg = pd.DataFrame({
    "total_area": inf.groupby("node")["area_m2"].sum(),
    "fixed_area": fixed.groupby("node")["area_m2"].sum(),
    "fixed_coe_w": fixed.groupby("node").apply(
        lambda g: np.average(g["runoff_coe"].fillna(0.5), weights=g["area_m2"])
        if len(g) else 0.0, include_groups=False),
    "horton_area": hort.groupby("node")["area_m2"].sum(),
}).fillna(0.0)
hort_meta = inf[inf["runoff_vol"] != "Fixed"]
if len(hort_meta):
    agg["hort_init"] = hort_meta.groupby("node")["initial_in"].mean()
    agg["hort_limit"] = hort_meta.groupby("node")["limiting_i"].mean()
    agg["hort_decay"] = hort_meta.groupby("node")["decay_fact"].mean()
agg = agg[agg["total_area"] > 100]  # 剔除 <100 m2 的微汇水
agg["imperv_pct"] = (100 * agg["fixed_area"] * agg["fixed_coe_w"] / agg["total_area"]).clip(0, 100)
agg["width_m"] = np.sqrt(agg["total_area"])
print(f"subcatchments: {len(agg)}, total area km2: {agg['total_area'].sum()/1e6:.2f}")

# 坡度: elevation zones 就近节点分组
el_path = os.path.join(DS, "1_catchment_shape_files", "elevation", "2D Zones elevation.shp")
from pyproj import Transformer
tr = Transformer.from_crs(27700, 4326, always_xy=True)
x0, y0 = tr.transform(CX - R, CY - R)
x1, y1 = tr.transform(CX + R, CY + R)
bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
el = gpd.read_file(el_path, bbox=bbox).to_crs(27700)
el["cx"], el["cy"] = el.geometry.centroid.x, el.geometry.centroid.y
el = el[(el["cx"] - CX) ** 2 + (el["cy"] - CY) ** 2 <= R ** 2]
d2, ix2 = jtree.query(el[["cx", "cy"]].values, k=1)
el["node"] = real_junc["id"].values[ix2]
slope = el.groupby("node")["elevation2"].quantile([0.9, 0.1]).unstack()
agg["slope_pct"] = ((slope[0.9] - slope[0.1]) / (2 * np.sqrt(agg["total_area"])) * 100) \
    .reindex(agg.index).clip(0.5, 10).fillna(2.0)
print(f"slope computed from {len(el)} elevation elements")

# ================= 4. 雨量序列 =================
rain_files = [f for f in glob.glob(os.path.join(DS, "7_rain_gauge_telemetry_data", "*.csv"))
              if not f.endswith("_quality_flags.csv")]
series = []
for f in rain_files:
    df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    s = df.set_index("timestamp")["Rainfall Depth [mm]"].astype(float)
    series.append(s[~s.index.duplicated(keep="first")])
rain = pd.concat(series, axis=1).resample("15min").mean()
rain_mean = rain.mean(axis=1)
ev_win = rain_mean.loc[EVENT["start"]:EVENT["end"]].fillna(0)
print(f"rain series: {len(ev_win)} steps, total {ev_win.sum():.1f} mm")

# ================= 5. 写试点 inp =================
L = []
A = L.append
A("[TITLE]\n;;Blue Heart PILOT model v1 (subcatchments + pseudo outfalls, no pumps)")
A("[OPTIONS]")
for k, v in [("FLOW_UNITS", "CMS"), ("INFILTRATION", "HORTON"), ("FLOW_ROUTING", "DYNWAVE"),
             ("LINK_OFFSETS", "DEPTH"), ("ALLOW_PONDING", "NO"),
             ("START_DATE", "10/22/2025"), ("START_TIME", "06:00:00"),
             ("END_DATE", "10/23/2025"), ("END_TIME", "20:00:00"),
             ("REPORT_STEP", "00:15:00"), ("WET_STEP", "00:05:00"), ("DRY_STEP", "01:00:00"),
             ("ROUTING_STEP", "00:00:30"), ("MIN_SURFAREA", "1.167"),
             ("INERTIAL_DAMPING", "PARTIAL"), ("NORMAL_FLOW_LIMITED", "BOTH"),
             ("VARIABLE_STEP", "0.75"), ("MAX_TRIALS", "8"), ("HEAD_TOLERANCE", "0.0015")]:
    A(f"{k:<20}{v}")
A("\n[EVAPORATION]\nCONSTANT    0.0\nDRY_ONLY    NO")
A("\n[JUNCTIONS]")
A(";;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded")
for dummy, oinv in dummy_nodes:
    A(f"{dummy:<38} {oinv:<10.3f} {5.0:<10.3f} 0          0          0")
n_bad_j = 0
for _, r in real_junc.iterrows():
    inv = r["invert"] if np.isfinite(r["invert"]) else 0.0
    g = r["ground_lev"] if pd.notna(r["ground_lev"]) else inv + 1.5
    if not (np.isfinite(inv) and np.isfinite(g)):
        inv, g = 0.0, 1.5
        n_bad_j += 1
    md = float(np.clip(g - inv, 0.1, 50.0))
    A(f"{r['id']:<38} {inv:<10.3f} {md:<10.3f} 0          0          0")
print(f"junctions coerced from non-finite invert/ground: {n_bad_j}")
A("\n[OUTFALLS]")
A(";;Name           Elevation  Type       Stage Data       Gated    Route To")
for _, r in pd.concat([outfalls, pseudo_out]).iterrows():
    inv = r["invert"] if pd.notna(r["invert"]) else 0.0
    if not np.isfinite(inv):
        n_bad_j += 1
        continue
    A(f"{r['id']:<38} {inv:<10.3f} FREE                  NO")
A("\n[CONDUITS]")
A(";;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow")
for _, r in cond.iterrows():
    A(f"{r['id']:<38} {r['us_node_id']:<38} {r['ds_node_id']:<38} "
      f"{r['len_m']:<10.2f} {0.013:<10.4f} {r['off_us']:<10.3f} {r['off_ds']:<10.3f} 0          0")
A("\n[XSECTIONS]")
A(";;Link           Shape        Geom1      Geom2      Geom3      Geom4      Barrels")
for _, r in cond.iterrows():
    A(f"{r['id']:<38} CIRCULAR     {r['diameter_m']:<10.3f} 0          0          0          1")
A("\n[RAINGAGES]")
A(";;Name          Format    Interval  SCF  Source")
A("RG1             VOLUME    0:15      1.0  TIMESERIES RG1")
A("\n[SUBCATCHMENTS]")
A(";;Name          Rain Gauge        Outlet          Area     %Imperv  Width    Slope  CurbLen")
for i, (nid, r) in enumerate(agg.iterrows()):
    A(f"S{i:<15} RG1              {nid:<15} {r['total_area']/1e4:<8.3f} {r['imperv_pct']:<8.2f} "
      f"{r['width_m']:<8.1f} {r['slope_pct']:<6.2f} 0")
A("\n[SUBAREAS]")
A(";;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo")
for i in range(len(agg)):
    A(f"S{i:<15} 0.012      0.10       1.5        5.0        25         OUTLET")
A("\n[INFILTRATION]")
A(";;Subcatchment   MaxRate    MinRate    Decay      DryTime    MaxInfil")
for i, (nid, r) in enumerate(agg.iterrows()):
    hi = r.get("hort_init", 50.0) if pd.notna(r.get("hort_init", np.nan)) else 50.0
    hl = r.get("hort_limit", 1.0) if pd.notna(r.get("hort_limit", np.nan)) else 1.0
    hd = r.get("hort_decay", 2.0) if pd.notna(r.get("hort_decay", np.nan)) else 2.0
    A(f"S{i:<15} {max(hi, hl):<10.2f} {hl:<10.2f} {hd:<10.3f} 7          0")
A("\n[TIMESERIES]")
A(";;Name           Date       Time       Value")
for ts, v in ev_win.items():
    A(f"RG1             {ts.strftime('%m/%d/%Y')}  {ts.strftime('%H:%M')}    {v:.3f}")
A("\n[REPORT]\nINPUT NO\nCONTROLS NO\nNODES ALL\nLINKS ALL")
inp = os.path.join(OUT, "blueheart_pilot.inp")
with open(inp, "w") as f:
    f.write("\n".join(L))
print("inp written:", inp, "| subcatchments:", len(agg))

# 保存聚合表供复用
agg.reset_index().to_csv(os.path.join(OUT, "pilot_subcatchments.csv"), index=False)

# ================= 6. 事件模拟 + 30 站对比 =================
from pyswmm import Simulation, Nodes
sim_nodes = {}
with Simulation(inp) as sim:
    node_objs = {n.nodeid: n for n in Nodes(sim)}
    targets = {n: node_objs[n] for n in sel30["node_id"].unique() if n in node_objs}
    print(f"simulating, target nodes present: {len(targets)}/{sel30['node_id'].nunique()}")
    for _ in sim:
        if sim.current_time.minute % 15 == 0 and sim.current_time.second == 0:
            t = sim.current_time
            for nid, n in targets.items():
                sim_nodes.setdefault(nid, {})[t] = n.depth
    ferr = sim.flow_routing_error
print("flow routing error %:", ferr)
simdf = pd.DataFrame(sim_nodes)
simdf.index = pd.to_datetime(simdf.index).tz_localize("UTC")
simdf.to_csv(os.path.join(OUT, "pilot_sim_depth_30stations.csv"))

# 观测对比
rows = []
for _, s in sel30.iterrows():
    f = os.path.join(DS, "5_sewer_telemetry_data", s["file"])
    if not os.path.exists(f):
        continue
    obs = pd.read_csv(f)
    obs["timestamp"] = pd.to_datetime(obs["timestamp"], format="mixed", utc=True)
    obs = obs.set_index("timestamp")
    lvl = obs.select_dtypes("number").iloc[:, 0] / 1000.0  # mm -> m
    qf = os.path.join(DS, "5_sewer_telemetry_data", s["file"].replace(".csv", "_quality_flags.csv"))
    if os.path.exists(qf):
        q = pd.read_csv(qf)
        q["timestamp"] = pd.to_datetime(q["timestamp"], format="mixed", utc=True)
        q = q.set_index("timestamp")
        num = q.select_dtypes("number")
        if num.shape[1]:
            lvl = lvl[num.iloc[:, 0].reindex(lvl.index) == 1]
    lvl = lvl.loc[EVENT["start"]:EVENT["end"]]
    if len(lvl) < 10 or s["node_id"] not in simdf.columns:
        continue
    sim_s = simdf[s["node_id"]].resample("15min").mean()
    obs_s = lvl.resample("15min").mean()
    j = pd.concat([sim_s.rename("sim"), obs_s.rename("obs")], axis=1).dropna()
    if len(j) < 10:
        continue
    r_p = j["sim"].corr(j["obs"])
    r_s = j["sim"].corr(j["obs"], method="spearman")
    rows.append({"station": s["locationName"], "node": s["node_id"], "n": len(j),
                 "pearson_r": r_p, "spearman_r": r_s,
                 "obs_range_m": float(j["obs"].max() - j["obs"].min()),
                 "sim_range_m": float(j["sim"].max() - j["sim"].min())})
cmp = pd.DataFrame(rows)
cmp.to_csv(os.path.join(OUT, "pilot_sim_vs_obs_30stations.csv"), index=False)
print(cmp[["pearson_r", "spearman_r"]].describe().loc[["mean", "50%"]].to_string())
print(f"stations with n>=10 compared: {len(cmp)}/30")
