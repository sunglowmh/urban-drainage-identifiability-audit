# -*- coding: utf-8 -*-
"""P2 driver: subprocess 并行批量模拟
用法: python p2_driver.py <jobs.json> [n_workers]
jobs.json = [{run_dir, rough, dwf_k, start, end}, ...]
"""
import sys
import json
import os
import time
import subprocess
import glob

BASE = "E:/work/workBuddy/BlueHeartV2"
PY = "C:/Users/DELL/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

def run_jobs(jobs, n_workers=28, worker="p2_worker.py"):
    job_paths = []
    for i, job in enumerate(jobs):
        jp = os.path.join(job["run_dir"], "job.json")
        os.makedirs(job["run_dir"], exist_ok=True)
        json.dump(job, open(jp, "w"))
        job_paths.append(jp)
    pending = list(job_paths)
    running = {}
    done, failed = [], []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < n_workers:
            jp = pending.pop(0)
            p = subprocess.Popen([PY, os.path.join(BASE, worker), jp],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running[jp] = p
        for jp, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[jp]
                d = json.load(open(os.path.join(os.path.dirname(jp), "done.json"))) \
                    if os.path.exists(os.path.join(os.path.dirname(jp), "done.json")) else {}
                if rc == 0 and "error" not in d:
                    done.append(jp)
                else:
                    failed.append((jp, d.get("error", f"rc={rc}")))
        if len(done) + len(failed) and (len(done) + len(failed)) % 10 == 0:
            print(f"[{time.time()-t0:.0f}s] done={len(done)} failed={len(failed)} "
                  f"running={len(running)} pending={len(pending)}", flush=True)
        time.sleep(3)
    print(f"ALL DONE in {time.time()-t0:.0f}s: ok={len(done)} failed={len(failed)}")
    for jp, e in failed[:10]:
        print("FAILED:", os.path.dirname(jp), e)
    return done, failed

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1]))
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    worker = sys.argv[3] if len(sys.argv) > 3 else "p2_worker.py"
    import functools
    run_jobs(jobs, nw, worker)
