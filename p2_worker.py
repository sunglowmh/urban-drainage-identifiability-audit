# -*- coding: utf-8 -*-
"""P2 worker: 单次模拟 (由 p2_driver.py 以 subprocess 并行调用)
用法: python p2_worker.py job.json
job = {run_dir, rough:[n1,n2,n3], dwf_k, start, end}
输出: run_dir/series.csv + run_dir/done.json
"""
import sys
import json
import os
import numpy as np
import pandas as pd

BASE = "E:/work/workBuddy/BlueHeartV2"

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
    try:
        df, ferr = p2lib.run_sim(txt, rd, targets)
        df.index = df.index.tz_localize("UTC")
        df.to_csv(os.path.join(rd, "series.csv"))
        json.dump({"routing_err": float(ferr), "n_steps": int(len(df)),
                   "n_targets": int(df.shape[1])}, open(done, "w"))
    except Exception as e:
        json.dump({"error": str(e)[:500]}, open(done, "w"))
        raise

if __name__ == "__main__":
    main(sys.argv[1])
