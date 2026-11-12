# -*- coding: utf-8 -*-
"""C2-b (D1=b) 阶段 1: 高激励 twin 臂 SWMM 运行批 —— CPU 层, 与 train_hcmtnet (GPU) 零冲突
设计: 锚点强迫 (ev00, 38.55 mm) × 2.5 均匀放大 (峰值 2.85→7.13 mm/15min, 计 96.4 mm),
窗口不变; 其余注入/设计完全复用注册 twin (σ_i/φ_gen/τ_b 与 90 LHS/24 验证/5 真值/2 失配)
产出: output/p2/twin_high/{runs_lhs,runs_truth,runs_val}/<tag>/series.csv + done.json
"""
import json
import os
import re
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

BASE = "E:/work/workBuddy/BlueHeartV2"
AMP = 2.5
ROOT = os.path.join(BASE, "output/p2/twin_high")


def patch_rain(txt, amp):
    """[TIMESERIES] RG1 数值列 × amp"""
    out, sec = [], None
    for ln in txt.splitlines():
        m = re.match(r"^\[([A-Z_]+)\]", ln.strip())
        if m:
            sec = m.group(1)
            out.append(ln)
            continue
        if sec == "TIMESERIES" and ln.strip() and not ln.strip().startswith(";"):
            p = ln.split()
            if len(p) >= 4:
                try:
                    v = float(p[-1])
                    p[-1] = f"{v * amp:.4f}"
                    out.append(" ".join(p))
                    continue
                except ValueError:
                    pass
        out.append(ln)
    return "\n".join(out) + "\n"


def run_one(job):
    sys.path.insert(0, BASE)
    import p2lib
    rd = job["run_dir"]
    os.makedirs(rd, exist_ok=True)
    done = os.path.join(rd, "done.json")
    if os.path.exists(done):
        return job["tag"], "cached"
    txt = p2lib.build_inp(rough=tuple(job["rough"]), dwf_k=job.get("dwf_k", 1.0),
                          start=job.get("start", "10/22/2025 06:00"),
                          end=job.get("end", "10/23/2025 20:00"))
    txt = patch_rain(txt, AMP)
    ws = job.get("width_scale")
    if ws:
        # Width 补丁 (m2 失配用, 与 p2a_twin_worker 同式)
        out2, sec = [], None
        for ln in txt.splitlines():
            m = re.match(r"^\[([A-Z_]+)\]", ln.strip())
            if m:
                sec = m.group(1)
                out2.append(ln)
                continue
            if sec == "SUBCATCHMENTS" and ln.strip() and not ln.strip().startswith(";"):
                p = ln.split()
                if len(p) >= 6:
                    p[5] = f"{float(p[5]) * ws:.3f}"
                    out2.append(" " + " ".join(p))
                    continue
            out2.append(ln)
        txt = "\n".join(out2) + "\n"
    try:
        sel30 = pd.read_csv(os.path.join(p2lib.P1, "pilot_30_stations.csv"))
        targets = sel30["node_id"].unique().tolist()
        df, ferr = p2lib.run_sim(txt, rd, targets)
        df.index = df.index.tz_localize("UTC")
        df.to_csv(os.path.join(rd, "series.csv"))
        json.dump({"routing_err": float(ferr), "n_steps": int(len(df)),
                   "rough": job["rough"], "dwf_k": job.get("dwf_k", 1.0),
                   "amp": AMP, "width_scale": ws},
                  open(done, "w"))
        return job["tag"], "ok"
    except Exception as e:
        json.dump({"error": str(e)[:400]}, open(done, "w"))
        return job["tag"], f"ERROR {e}"


def build_jobs():
    jobs = []
    # a) 90 LHS (冻结设计, 只换强迫)
    lhs = json.load(open(os.path.join(p2lib_path(), "jobs_lhs.json"))) if False else \
        json.load(open(os.path.join(BASE, "output/p2/jobs_lhs.json")))
    for j in lhs:
        tag = os.path.basename(j["run_dir"])
        jobs.append({"tag": f"lhs/{tag}", "run_dir": os.path.join(ROOT, "runs_lhs", tag),
                     "rough": j["rough"], "dwf_k": j.get("dwf_k", 1.0)})
    # b) 5 真值 + 2 失配 (与 twin v3 同参数)
    TRUTHS = {"t1": [0.0100, 0.0100, 0.0100], "t2": [0.0210, 0.0210, 0.0210],
              "t3": [0.0090, 0.0220, 0.0090], "t4": [0.0120, 0.0160, 0.0200],
              "t5": [0.0155, 0.0155, 0.0155]}
    for tid, th in TRUTHS.items():
        jobs.append({"tag": f"truth/{tid}", "run_dir": os.path.join(ROOT, "runs_truth", tid),
                     "rough": th, "dwf_k": 1.0})
    jobs.append({"tag": "truth/m1", "run_dir": os.path.join(ROOT, "runs_truth", "m1"),
                 "rough": TRUTHS["t5"], "dwf_k": 0.5})
    jobs.append({"tag": "truth/m2", "run_dir": os.path.join(ROOT, "runs_truth", "m2"),
                 "rough": TRUTHS["t5"], "dwf_k": 1.0, "width_scale": 1.1})
    # c) 24 验证 (冻结离设计)
    val = json.load(open(os.path.join(BASE, "output/p2/twin/val/jobs_val.json")))
    for j in val:
        tag = os.path.basename(j["run_dir"])
        jobs.append({"tag": f"val/{tag}", "run_dir": os.path.join(ROOT, "runs_val", tag),
                     "rough": j["rough"], "dwf_k": j.get("dwf_k", 1.0)})
    return jobs


def p2lib_path():
    return BASE


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    jobs = build_jobs()
    json.dump(jobs, open(os.path.join(ROOT, "jobs_high.json"), "w"), indent=1)
    print(f"total jobs: {len(jobs)}", flush=True)
    n_ok = 0
    with ProcessPoolExecutor(max_workers=20) as ex:
        for tag, st in ex.map(run_one, jobs):
            n_ok += st in ("ok", "cached")
            print(f"{tag}: {st}", flush=True)
    print(f"COMPLETE {n_ok}/{len(jobs)}", flush=True)
