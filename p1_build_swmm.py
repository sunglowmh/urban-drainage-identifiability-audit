# -*- coding: utf-8 -*-
"""
P1.1: Blue Heart 管网 shapefile -> SWMM .inp 基线骨架 (v0)
映射规则:
  - Junction = Manhole/Break 节点: invert = ground_lev - DEPNOD; MaxDepth = ground - invert
  - Outfall  = node_type=='Outfall': FREE 出流 (潮界版本后续换 TIMESERIES 边界)
  - Conduit  = link_type=='Conduit' 且两端节点存在; CIRC: diameter=conduit_wi/1000;
    非圆形截面 v0 剔除并报告; 长度 = EPSG:27700 几何长度(裁剪 1m~2000m);
    节点 invert = 相邻 conduit us/ds_invert 的最小值(缺失时 ground-DEPNOD);
    Conduit 偏移 offset = conduit_invert - node_invert (>=0)
  - 其余 link 类型 (Weir/Pump/Orifice/...) v0 剔除并计数
输出: output/p1/blueheart_baseline.inp + build_report.md
"""
import os
import numpy as np
import geopandas as gpd
import pandas as pd

BASE = "E:/work/workBuddy/BlueHeartV2"
DS = os.path.join(BASE, "dataset", "2_network_shape_files")
OUT = os.path.join(BASE, "output", "p1")
os.makedirs(OUT, exist_ok=True)
REP = os.path.join(OUT, "build_report.md")

lk = gpd.read_file(os.path.join(DS, "links", "Links.shp"))
nd = gpd.read_file(os.path.join(DS, "nodes", "Nodes.shp"))
lk_m = lk.to_crs(27700)
lk_m["len_m"] = lk_m.geometry.length

# ---------- ID 净化 (SWMM 不允许空格等) ----------
def clean_id(s):
    s = str(s).strip()
    for ch in (" ", ";", ",", "\t"):
        s = s.replace(ch, "_")
    return s[:38]

nd["id"] = nd["id"].map(clean_id)
lk_m["id"] = lk_m["id"].map(clean_id)
lk_m["us_node_id"] = lk_m["us_node_id"].map(clean_id)
lk_m["ds_node_id"] = lk_m["ds_node_id"].map(clean_id)
dup_nodes = nd["id"].duplicated().sum()
nd = nd.drop_duplicates(subset="id")

# ---------- 节点表 ----------
nd = nd.copy()
nd["invert_est"] = nd["ground_lev"] - nd["DEPNOD"]
junctions = nd[nd["node_type"].isin(["Manhole", "Break"])].copy()
outfalls = nd[nd["node_type"] == "Outfall"].copy()
storages = nd[nd["node_type"] == "Storage"].copy()

cond = lk_m[(lk_m["link_type"] == "Conduit") & (lk_m["shape"] == "CIRC")].copy()
cond["diameter_m"] = cond["conduit_wi"] / 1000.0
has_ends = cond["us_node_id"].isin(nd["id"]) & cond["ds_node_id"].isin(nd["id"])
cond = cond[has_ends].copy()

# 节点 invert 优先用相邻 conduit 反底最小值
us_inv = cond.groupby("us_node_id")["us_invert"].min()
ds_inv = cond.groupby("ds_node_id")["ds_invert"].min()
node_inv = pd.concat([us_inv, ds_inv], axis=1).min(axis=1)

def node_invert(nid, fallback):
    v = node_inv.get(nid, np.nan)
    if np.isnan(v) or v is None:
        return fallback
    return v

# 剔除无效几何/尺寸
ok_size = cond["diameter_m"].between(0.05, 5.0)
ok_len = cond["len_m"].between(1.0, 2000.0)
ok_inv = cond["us_invert"].notna() & cond["ds_invert"].notna()
dropped = {
    "conduit_bad_size": int((~ok_size).sum()),
    "conduit_bad_len": int((~ok_len).sum()),
    "conduit_bad_invert": int((~ok_inv).sum()),
}
cond = cond[ok_size & ok_len & ok_inv].copy()
# Outfall 不能作为 conduit 上游节点 -> 剔除该类 conduit
bad_up = cond["us_node_id"].isin(outfalls["id"])
dropped["conduit_outfall_as_us"] = int(bad_up.sum())
cond = cond[~bad_up].copy()

# 节点 invert 刷新并计算偏移
all_ids = pd.concat([cond["us_node_id"], cond["ds_node_id"]])
used_nodes = nd[nd["id"].isin(all_ids)].copy()
used_nodes["invert"] = [
    node_invert(r["id"], r["invert_est"]) for _, r in used_nodes.iterrows()
]
used_nodes["invert"] = used_nodes["invert"].fillna(used_nodes["invert_est"])
inv_map = used_nodes.set_index("id")["invert"]

cond["off_us"] = (cond["us_invert"] - cond["us_node_id"].map(inv_map)).clip(lower=0)
cond["off_ds"] = (cond["ds_invert"] - cond["ds_node_id"].map(inv_map)).clip(lower=0)

# Outfall 多入流处理: 插入虚拟汇流节点 OUTJ_<oid>
dummy_nodes = []
inlets_per_out = cond[cond["ds_node_id"].isin(outfalls["id"])].groupby("ds_node_id").size()
multi_out = inlets_per_out[inlets_per_out > 1]
for oid, n_in in multi_out.items():
    dummy = f"OUTJ_{oid}"[:38]
    oinv = outfalls.loc[outfalls["id"] == oid, "invert_est"]
    oinv = float(oinv.iloc[0]) if len(oinv) and pd.notna(oinv.iloc[0]) else 0.0
    dummy_nodes.append((dummy, oinv))
    cond.loc[cond["ds_node_id"] == oid, "ds_node_id"] = dummy
    # 虚拟汇流节点 -> outfall 单一连接
    dmax = float(cond.loc[cond["ds_node_id"] == dummy, "diameter_m"].max())
    cond = pd.concat([cond, pd.DataFrame([{
        "id": f"DUMMY_{oid}"[:38], "us_node_id": dummy, "ds_node_id": oid,
        "len_m": 1.0, "diameter_m": dmax, "off_us": 0.0, "off_ds": 0.0,
        "us_invert": oinv, "ds_invert": oinv,
    }])], ignore_index=True)
print(f"outfalls with multiple inlets: {len(multi_out)}, dummy junctions added")

# ---------- 写 inp ----------
L = []
A = L.append
A("[TITLE]")
A(";;Blue Heart baseline skeleton v0 (storm/foul/other conduits, CIRC only)")
A("[OPTIONS]")
A("FLOW_UNITS          CMS")
A("INFILTRATION        HORTON")
A("FLOW_ROUTING        DYNWAVE")
A("LINK_OFFSETS        DEPTH")
A("MIN_SLOPE           0")
A("ALLOW_PONDING       NO")
A("SKIP_STEADY_STATE   NO")
A("START_DATE          01/01/2026")
A("START_TIME          00:00:00")
A("REPORT_START_DATE   01/01/2026")
A("REPORT_START_TIME   00:00:00")
A("END_DATE            01/02/2026")
A("END_TIME            00:00:00")
A("SWEEP_START         01/01")
A("SWEEP_END           12/31")
A("DRY_DAYS            7")
A("REPORT_STEP         00:15:00")
A("WET_STEP            00:05:00")
A("DRY_STEP            01:00:00")
A("ROUTING_STEP        00:01:00")
A("RULE_STEP           00:00:00")
A("INERTIAL_DAMPING    PARTIAL")
A("NORMAL_FLOW_LIMITED BOTH")
A("FORCE_MAIN_EQUATION H-W")
A("VARIABLE_STEP       0.75")
A("LENGTHENING_STEP    0")
A("MIN_SURFAREA        1.167")
A("MAX_TRIALS          8")
A("HEAD_TOLERANCE      0.0015")
A("SYS_FLOW_TOL        5")
A("LAT_FLOW_TOL        5")
A("")
A("[EVAPORATION]")
A("CONSTANT    0.0")
A("DRY_ONLY    NO")
A("")
A("[JUNCTIONS]")
A(";;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded")
for dummy, oinv in dummy_nodes:
    A(f"{dummy:<38} {oinv:<10.3f} {5.0:<10.3f} 0          0          0")
for _, r in used_nodes[used_nodes["node_type"].isin(["Manhole", "Break", "Storage"])].iterrows():
    g = r["ground_lev"] if pd.notna(r["ground_lev"]) else r["invert"] + 1.5
    md = max(g - r["invert"], 0.1)
    A(f"{r['id'][:38]:<38} {r['invert']:<10.3f} {md:<10.3f} 0          0          0")
A("")
A("[OUTFALLS]")
A(";;Name           Elevation  Type       Stage Data       Gated    Route To")
for _, r in outfalls.iterrows():
    inv = r["invert_est"] if pd.notna(r["invert_est"]) else 0.0
    A(f"{r['id'][:38]:<38} {inv:<10.3f} FREE                  NO")
A("")
A("[CONDUITS]")
A(";;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow")
for _, r in cond.iterrows():
    A(f"{r['id'][:38]:<38} {r['us_node_id'][:38]:<38} {r['ds_node_id'][:38]:<38} "
      f"{r['len_m']:<10.2f} {0.013:<10.4f} {r['off_us']:<10.3f} {r['off_ds']:<10.3f} 0          0")
A("")
A("[XSECTIONS]")
A(";;Link           Shape        Geom1      Geom2      Geom3      Geom4      Barrels")
for _, r in cond.iterrows():
    A(f"{r['id'][:38]:<38} CIRCULAR     {r['diameter_m']:<10.3f} 0          0          0          1")
A("")
A("[REPORT]")
A("INPUT      NO")
A("CONTROLS   NO")
A("SUBCATCHMENTS   ALL")
A("NODES      ALL")
A("LINKS      ALL")
inp_path = os.path.join(OUT, "blueheart_baseline.inp")
with open(inp_path, "w") as f:
    f.write("\n".join(L))

# ---------- 报告 ----------
with open(REP, "w", encoding="utf-8") as f:
    f.write("# P1.1 SWMM 基线骨架构建报告 (v0)\n\n")
    f.write(f"- 输入: {len(lk)} links / {len(nd)} nodes\n")
    f.write(f"- Conduit(CIRC) 保留: **{len(cond)}** (全部 link_type=Conduit {int((lk['link_type']=='Conduit').sum())}, CIRC {int((lk['shape']=='CIRC').sum())})\n")
    f.write(f"- Junction: {int((used_nodes['node_type'].isin(['Manhole','Break'])).sum())}, "
            f"Outfall: {len(outfalls)}, Storage(未入 v0): {len(storages)}\n")
    f.write(f"- 剔除 conduit: {dropped}\n")
    f.write(f"- 剔除非 Conduit link: {int((lk['link_type']!='Conduit').sum())} "
            f"(Weir {int((lk['link_type']=='Weir').sum())}, Pump {int((lk['link_type']=='Pump').sum())}, "
            f"Orifice {int((lk['link_type']=='Orifice').sum())}, River reach {int((lk['link_type']=='River reach').sum())} 等; v0 不含水力构筑物)\n")
    f.write(f"- 长度来源: EPSG:27700 几何长度(属性 length 仅 {lk['length'].notna().mean()*100:.0f}% 覆盖)\n")
    f.write(f"- 粗糙度: 统一初值 0.013 (后续为可辨识性参数, 分区化在 P1 后续)\n")
    f.write(f"- 输出: {inp_path}\n")
print("written:", inp_path, "| conduits:", len(cond), "| junctions:",
      int((used_nodes["node_type"].isin(["Manhole", "Break"])).sum()), "| outfalls:", len(outfalls))
