# -*- coding: utf-8 -*-
"""P2 跨事件 worker: 单次模拟, 逐事件注入降雨时序
用法: python p2_xev_worker.py job.json
job = {run_dir, rough, dwf_k, start, end, event_id}
与 p2_worker.py 唯一区别: [TIMESERIES] 段替换为该事件的面平均雨量
(15 站 15-min VOLUME, 与模板同口径; 窗口 = 事件雨窗)
"""
import sys
import json
import os
import re
import numpy as np
import pandas as pd

BASE = "E:/work/workBuddy/BlueHeartV2"

def event_rain_series(wstart, wend):
    import glob
    DS = os.path.join(BASE, "dataset")
    rain_files = [f for f in glob.glob(os.path.join(DS, "7_rain_gauge_telemetry_data", "*.csv"))
                  if not f.endswith("_quality_flags.csv")]
    series = []
    for f in rain_files:
        df = pd.read_csv(f)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        s = df.set_index("timestamp")["Rainfall Depth [mm]"].astype(float)
        series.append(s[~s.index.duplicated(keep="first")])
    rain = pd.concat(series, axis=1).resample("15min").mean().mean(axis=1)
    return rain.loc[wstart:wend].fillna(0.0)

def inject_rain(inp_text, ts):
    lines = inp_text.splitlines()
    out, cur, written = [], None, False
    for ln in lines:
        m = re.match(r"^\[([A-Z_]+)\]", ln.strip())
        if m:
            cur = m.group(1)
            out.append(ln)
            if cur == "TIMESERIES" and not written:
                out.append(";;Name           Date       Time       Value")
                for t, v in ts.items():
                    out.append(f"RG1             {t.strftime('%m/%d/%Y')}  {t.strftime('%H:%M')}    {v:.3f}")
                written = True
            continue
        if cur == "TIMESERIES":
            continue  # 丢弃模板时序
        out.append(ln)
    return "\n".join(out) + "\n"

def main(job_path):
    job = json.load(open(job_path))
    sys.path.insert(0, BASE)
    import p2lib
    rd = job["run_dir"]
    os.makedirs(rd, exist_ok=True)
    done = os.path.join(rd, "done.json")
    if os.path.exists(done):
        return
    sel30 = pd.read_csv(os.path.join(p2lib.P1, "pilot_30_stations.csv"))
    targets = sel30["node_id"].unique().tolist()
    txt = p2lib.build_inp(rough=tuple(job["rough"]), dwf_k=job["dwf_k"],
                          start=job["start"], end=job["end"])
    ts = event_rain_series(pd.Timestamp(job["start"]).tz_localize("UTC"),
                           pd.Timestamp(job["end"]).tz_localize("UTC"))
    txt = inject_rain(txt, ts)
    try:
        df, ferr = p2lib.run_sim(txt, rd, targets)
        df.index = df.index.tz_localize("UTC")
        df.to_csv(os.path.join(rd, "series.csv"))
        json.dump({"routing_err": float(ferr), "n_steps": int(len(df)),
                   "n_targets": int(df.shape[1]),
                   "rain_mm": round(float(ts.sum()), 2)}, open(done, "w"))
    except Exception as e:
        json.dump({"error": str(e)[:500]}, open(done, "w"))
        raise

if __name__ == "__main__":
    main(sys.argv[1])
