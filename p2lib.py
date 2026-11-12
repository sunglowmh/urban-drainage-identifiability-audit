# -*- coding: utf-8 -*-
"""
P2 共享基础设施: 试点 inp 参数化补丁 + 并行模拟 + 观测加载
- 参数: 3 个分区 Manning n (导管中点 x 坐标三分位分区), DWF 标量 k
- 观测: 30 站 15-min 水位 (flag==1), 与模拟网格对齐
"""
import os
import re
import glob
import numpy as np
import pandas as pd

BASE = "E:/work/workBuddy/BlueHeartV2"
DS = os.path.join(BASE, "dataset")
P1 = os.path.join(BASE, "output", "p1")
P2 = os.path.join(BASE, "output", "p2")
os.makedirs(P2, exist_ok=True)

TPL_INP = os.path.join(P1, "blueheart_pilot.inp")
N_LEVELS = 24          # 每参数网格数
PRIOR = (0.009, 0.022)  # Manning n 先验区间(均匀)

_CACHE = {}

# ---------------- 模板解析 ----------------
def parse_inp(path=TPL_INP):
    """把 inp 解析为 {section_name: [lines]} (保留注释行)"""
    secs, cur = {}, None
    for ln in open(path, encoding="utf-8", errors="ignore").read().splitlines():
        m = re.match(r"^\[([A-Z_]+)\]", ln.strip())
        if m:
            cur = m.group(1)
            secs[cur] = []
        elif cur is not None:
            secs[cur].append(ln)
    return secs

# ---------------- 分区定义 ----------------
def zone_thresholds():
    """导管中点 x (EPSG:27700) 三分位阈值"""
    if "zone" in _CACHE:
        return _CACHE["zone"]
    import geopandas as gpd
    nd = gpd.read_file(os.path.join(DS, "2_network_shape_files", "nodes", "Nodes.shp")).to_crs(27700)
    pilot_ids = set(pd.read_csv(os.path.join(P1, "pilot_subnetwork_nodes.csv"))["id"].astype(str).str.strip())
    nd = nd[nd["id"].isin(pilot_ids)]
    xy = {str(i).strip(): (g.x, g.y) for i, g in zip(nd["id"], nd.geometry)}
    secs = parse_inp()
    xs = []
    for ln in secs["CONDUITS"]:
        if ln.strip().startswith(";") or not ln.strip():
            continue
        p = ln.split()
        u, d = xy.get(p[1]), xy.get(p[2])
        if u and d:
            xs.append(0.5 * (u[0] + d[0]))
    q1, q2 = np.percentile(xs, [100 / 3, 200 / 3])
    _CACHE["zone"] = (q1, q2, xy)
    return _CACHE["zone"]

def conduit_zone(us_id, ds_id):
    q1, q2, xy = zone_thresholds()
    u, d = xy.get(us_id), xy.get(ds_id)
    if not u or not d:
        return 0
    x = 0.5 * (u[0] + d[0])
    return 0 if x <= q1 else (1 if x <= q2 else 2)

# ---------------- DWF ----------------
def dwf_flows(k_lspers_ha):
    """节点 -> DWF 流量(CMS); k 单位 L/s/ha, 按子汇水区面积分配"""
    if k_lspers_ha <= 0:
        return {}
    sc = pd.read_csv(os.path.join(P1, "pilot_subcatchments.csv"))
    sc = sc[sc["total_area"] > 100]
    ha = sc["total_area"] / 1e4
    cms = k_lspers_ha * ha / 1000.0
    return dict(zip(sc["node"].astype(str).str.strip(), cms))

# ---------------- inp 生成 ----------------
def build_inp(rough=(0.013, 0.013, 0.013), dwf_k=1.0, start="10/22/2025 06:00", end="10/23/2025 20:00"):
    """从模板生成参数化 inp 文本; rough=三分区 Manning n"""
    secs = parse_inp()
    s0 = start.split(); e0 = end.split()
    opts = {}
    for ln in secs["OPTIONS"]:
        p = ln.split()
        if len(p) >= 2:
            opts[p[0]] = p[1]
    opts["START_DATE"], opts["START_TIME"] = s0[0], s0[1] if len(s0) > 1 else "00:00:00"
    opts["END_DATE"], opts["END_TIME"] = e0[0], e0[1] if len(e0) > 1 else "00:00:00"
    out = ["[OPTIONS]"]
    for k, v in opts.items():
        out.append(f"{k:<20}{v}")
    order = [s for s in secs if s != "OPTIONS"]
    for sec in order:
        lines = secs[sec]
        if sec == "CONDUITS":
            newl = []
            for ln in lines:
                if ln.strip().startswith(";") or not ln.strip():
                    newl.append(ln)
                    continue
                p = ln.split()
                z = conduit_zone(p[1], p[2])
                p[4] = f"{rough[z]:.4f}"
                newl.append(" ".join([f"{v:<38}" if i < 3 else f"{v:<10}" for i, v in enumerate(p)]))
            lines = newl
        if sec == "JUNCTIONS" and dwf_k > 0:
            pass  # DWF 独立节, 在 JUNCTIONS 后插入
        out.append(f"[{sec}]")
        out.extend(lines)
        if sec == "JUNCTIONS":
            out.append("")
            out.append("[DWF]")
            out.append(";;Node           Constituent      Avg Value")
            for node, q in sorted(dwf_flows(dwf_k).items()):
                out.append(f"{node:<38} FLOW             {q:.6f}")
    return "\n".join(out) + "\n"

# ---------------- 运行 ----------------
def run_sim(inp_text, run_dir, targets):
    """写 inp 并模拟; 返回 (time_index, DataFrame[targets], routing_err)"""
    os.makedirs(run_dir, exist_ok=True)
    ip = os.path.join(run_dir, "run.inp")
    with open(ip, "w", encoding="utf-8") as f:
        f.write(inp_text)
    from pyswmm import Simulation, Nodes
    series = {t: {} for t in targets}
    with Simulation(ip) as sim:
        objs = {n.nodeid: n for n in Nodes(sim)}
        tgt = {t: objs[t] for t in targets if t in objs}
        times, buf = [], []
        next_cap = None
        for _ in sim:
            t = sim.current_time
            if next_cap is None:
                next_cap = t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0)
            if t >= next_cap:
                for nid, n in tgt.items():
                    series[nid][next_cap] = n.depth
                next_cap = next_cap + pd.Timedelta(minutes=15)
        ferr = sim.flow_routing_error
    df = pd.DataFrame(series)
    return df, ferr

# ---------------- 观测 ----------------
def load_obs(start, end):
    """30 站观测: 15-min 平均, flag==1, m 单位; 返回 {station: (node, series)}"""
    sel30 = pd.read_csv(os.path.join(P1, "pilot_30_stations.csv"))
    if "file" not in sel30.columns:
        sel30["file"] = sel30["locationName"].astype(str) + ".csv"
    obs = {}
    for _, s in sel30.iterrows():
        f = os.path.join(DS, "5_sewer_telemetry_data", s["file"])
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        df = df.set_index("timestamp")
        lvl = df.select_dtypes("number").iloc[:, 0] / 1000.0
        qf = f.replace(".csv", "_quality_flags.csv")
        if os.path.exists(qf):
            q = pd.read_csv(qf)
            q["timestamp"] = pd.to_datetime(q["timestamp"], format="mixed", utc=True)
            q = q.set_index("timestamp")
            num = q.select_dtypes("number")
            if num.shape[1]:
                lvl = lvl[num.iloc[:, 0].reindex(lvl.index) == 1]
        lvl = lvl.loc[start:end]
        if len(lvl) >= 10:
            obs[s["locationName"]] = (s["node_id"], lvl.resample("15min").mean())
    return obs

# ---------------- 观测矩阵 ----------------
def load_obs_matrix(tidx, stations_nodes):
    """观测矩阵 O(n_st, T)+mask M; station 顺序与模拟列一致, 对齐 tidx"""
    tidx = pd.DatetimeIndex(tidx)
    if tidx.tz is None:
        tidx = tidx.tz_localize("UTC")
    start = tidx[0].strftime("%m/%d/%Y %H:%M")
    end = tidx[-1].strftime("%m/%d/%Y %H:%M")
    obs = load_obs(start, end)
    node2s = {}
    for st, (node, s) in obs.items():
        node2s.setdefault(node, s)
    O = np.full((len(stations_nodes), len(tidx)), np.nan)
    for i, node in enumerate(stations_nodes):
        if node in node2s:
            O[i] = node2s[node].reindex(tidx).values
    return O, np.isfinite(O)

# ---------------- QC ----------------
def qc_stations(O, M, Ybase, max_abs_offset=1.0, max_rmse=0.5):
    """基准运行 vs 观测的逐站 QC; 返回 (M_filtered, 排除清单)
    规则(预注册): |δ_i|=|median(obs-sim)|>1 m (datum 物理不可能) 或 去偏 RMSE>0.5 m → 排除"""
    excluded = []
    M = M.copy()
    for i in range(O.shape[0]):
        ok = M[i]
        if ok.sum() < 20:
            excluded.append((i, "insufficient_obs"))
            M[i] = False
            continue
        e = O[i, ok] - Ybase[i, ok] if Ybase is not None else O[i, ok]
        off = float(np.median(e))
        rmse = float(np.sqrt(((e - off) ** 2).mean()))
        if abs(off) > max_abs_offset or rmse > max_rmse:
            excluded.append((i, f"offset={off:.2f}m,rmse={rmse:.2f}m"))
            M[i] = False
    return M, excluded

# ---------------- 雨量窗口 ----------------
AREAL_RAIN_VERSION = "v2.0-correction-2026-11-09"  # 修正口径版本标记 (integrity I8); 旧管线输出不受影响: 本函数从未被既有运行调用

def areal_rain_15(files=None):
    """正确口径的面平均雨量 (2026-11-09 审计后新增, 版本见 AREAL_RAIN_VERSION):
    逐站 15-min bin 内增量求和 (resample.sum, 处理 Parkland 8.8% off-grid 时间戳
    与 782 个重复 bin) -> 跨站均值 (>=60% 覆盖才取值)。
    旧管线 (union 索引 + 行均值 + bin 级 mean/sum) 有 bin 聚合伪影:
    注入 SWMM 的强迫项偏高最多 14.3% (ev05), 矩阵 A 锚点偏高 7.7%。
    返回: (15-min 面平均序列, 各站 bin 求和后的 DataFrame)
    """
    if files is None:
        files = [f for f in glob.glob(os.path.join(DS, "7_rain_gauge_telemetry_data", "*.csv"))
                 if not f.endswith("_quality_flags.csv")]
    g = {}
    for f in files:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        s = df.set_index("timestamp")["Rainfall Depth [mm]"].astype(float)
        s = s[~s.index.duplicated(keep="first")]
        g[os.path.basename(f)[:-4]] = s.resample("15min").sum()
    rain = pd.concat(g, axis=1)
    cov = rain.notna().mean(axis=1)
    return rain.mean(axis=1).where(cov >= 0.6), rain

def event_window(pad_before_h=3, pad_after_h=9):
    """按实际雨量裁剪模拟窗口"""
    if "win" in _CACHE:
        return _CACHE["win"]
    rain_files = [f for f in glob.glob(os.path.join(DS, "7_rain_gauge_telemetry_data", "*.csv"))
                  if not f.endswith("_quality_flags.csv")]
    series = []
    for f in rain_files:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        s = df.set_index("timestamp")["Rainfall Depth [mm]"].astype(float)
        series.append(s[~s.index.duplicated(keep="first")])
    rain = pd.concat(series, axis=1).resample("15min").mean().mean(axis=1)
    w = rain.loc["2025-10-22 06:00":"2025-10-23 20:00"]
    wet = w[w > 0.1]
    start = (wet.index[0] - pd.Timedelta(hours=pad_before_h)).strftime("%m/%d/%Y %H:%M")
    end = (wet.index[-1] + pd.Timedelta(hours=pad_after_h)).strftime("%m/%d/%Y %H:%M")
    _CACHE["win"] = (start, end)
    return _CACHE["win"]
