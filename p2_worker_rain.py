# -*- coding: utf-8 -*-
"""P2 降雨对照 worker (prereg v1.8 A27): 修正口径/扰动强迫注入
用法: python p2_worker_rain.py job.json
job = {run_dir, rough:[n1,n2,n3], dwf_k, start, end,
       rain_mode: "corr"|"pert", eps: float (pert 时必需), event_id}
- corr: 注入 p2lib.areal_rain_15() 修正口径面雨量 (v2.0-correction-2026-11-09)
- pert: 修正口径序列 × (1+eps), 事件内常数乘子 (A27-4)
窗口裁剪与 fillna(0) 同旧管线; 注入机制与 p2_xev_worker.inject_rain 相同。
"""
import sys
import json
import os
import pandas as pd

BASE = "E:/work/workBuddy/BlueHeartV2"

def corrected_rain_series(wstart, wend):
    sys.path.insert(0, BASE)
    import p2lib
    rain, _ = p2lib.areal_rain_15()
    return rain.loc[wstart:wend].fillna(0.0)

def main(job_path):
    job = json.load(open(job_path))
    sys.path.insert(0, BASE)
    import p2lib
    from p2_xev_worker import inject_rain
    rd = job["run_dir"]
    os.makedirs(rd, exist_ok=True)
    done = os.path.join(rd, "done.json")
    if os.path.exists(done):
        return
    targets = pd.read_csv(os.path.join(p2lib.P1, "pilot_30_stations.csv"))["node_id"].unique().tolist()
    txt = p2lib.build_inp(rough=tuple(job["rough"]), dwf_k=job["dwf_k"],
                          start=job["start"], end=job["end"])
    wst = pd.Timestamp(job["start"]).tz_localize("UTC")
    wen = pd.Timestamp(job["end"]).tz_localize("UTC")
    ts = corrected_rain_series(wst, wen)
    eps = float(job.get("eps", 0.0))
    if job.get("rain_mode", "corr") == "pert":
        ts = ts * (1.0 + eps)
    txt = inject_rain(txt, ts)
    try:
        df, ferr = p2lib.run_sim(txt, rd, targets)
        df.index = df.index.tz_localize("UTC")
        df.to_csv(os.path.join(rd, "series.csv"))
        json.dump({"routing_err": float(ferr), "n_steps": int(len(df)),
                   "n_targets": int(df.shape[1]),
                   "rain_mode": job.get("rain_mode", "corr"),
                   "eps": eps,
                   "rain_mm": round(float(ts.sum()), 2),
                   "areal_rain_version": p2lib.AREAL_RAIN_VERSION},
                  open(done, "w"))
    except Exception as e:
        json.dump({"error": str(e)[:500]}, open(done, "w"))
        raise

if __name__ == "__main__":
    main(sys.argv[1])
