# -*- coding: utf-8 -*-
"""P0-A 孪生 worker: 真值/失配 SWMM 运行
job = {run_dir, rough:[n1,n2,n3], dwf_k, start, end, width_scale(optional)}
与 p2_worker.py 同口径（锚点窗口、模板原强迫、30 站），外加 [SUBCATCHMENTS] Width 补丁。
输出: run_dir/series.csv + done.json
"""
import sys, json, os, re
import numpy as np
import pandas as pd

BASE = "E:/work/workBuddy/BlueHeartV2"

def patch_width(txt, scale):
    """[SUBCATCHMENTS] 行格式: Name Raingage Outlet Area %Imperv Width %Slope CurbLen"""
    out, sec = [], None
    for ln in txt.splitlines():
        m = re.match(r"^\[([A-Z_]+)\]", ln.strip())
        if m:
            sec = m.group(1)
            out.append(ln); continue
        if sec == "SUBCATCHMENTS" and ln.strip() and not ln.strip().startswith(";"):
            p = ln.split()
            if len(p) >= 6:
                p[5] = f"{float(p[5]) * scale:.3f}"
                out.append(" " + " ".join(p))
                continue
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
    ws = job.get("width_scale")
    if ws:
        txt = patch_width(txt, float(ws))
    try:
        df, ferr = p2lib.run_sim(txt, rd, targets)
        df.index = df.index.tz_localize("UTC")
        df.to_csv(os.path.join(rd, "series.csv"))
        json.dump({"routing_err": float(ferr), "n_steps": int(len(df)),
                   "n_targets": int(df.shape[1]), "width_scale": ws,
                   "rough": job["rough"], "dwf_k": job["dwf_k"]},
                  open(done, "w"))
    except Exception as e:
        json.dump({"error": str(e)[:500]}, open(done, "w"))
        raise

if __name__ == "__main__":
    main(sys.argv[1])
