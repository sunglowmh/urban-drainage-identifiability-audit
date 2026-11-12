# -*- coding: utf-8 -*-
"""跨事件 (ev01–ev10) 真实数据 v2 重算: 似然 v3 + 逐事件 W_n (24 冻结离设计点, 事件强迫)
prereg v1.7 A26. 输出: xev/xev_results_v2.csv (增量) / xev_raw_v2.json / 每事件 wmap 存档
主分析模型: E0/E1/E2 + E1 dropout 15/5 (与 v1 范围可比)
前置: output/p2/xev/val_runs/{eid}/ 24 验证运行全部完成 (driver jobs_val_xev.json)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_xev_posterior import load_event_runs
from p2b_realdata_v3 import (fit_gp, phi_from_residuals, build_wmap_event,
                             invert_real, ident_flags)

OUT = os.path.join(p2lib.P2, "xev")
VAL_ROOT = os.path.join(p2lib.P2, "xev", "val_runs")


def check_val_ready(eid):
    design = json.load(open(os.path.join(p2lib.P2, "twin/val/jobs_val.json")))
    missing = []
    for j in design:
        rd = os.path.join(VAL_ROOT, eid, os.path.basename(j["run_dir"]))
        d = os.path.join(rd, "done.json")
        if not os.path.exists(d) or "error" in json.load(open(d)):
            missing.append(os.path.basename(j["run_dir"]))
    return missing


if __name__ == "__main__":
    manifest = pd.read_csv(os.path.join(OUT, "events_manifest.csv"))
    man = {r["event_id"]: r for _, r in manifest.iterrows()}
    rows, raw = [], {}
    for eid in manifest["event_id"]:
        r = man[eid]
        print(f"=== {eid} ({r['stratum']}, {r['win_start'][:16]}, {r['depth_mm']}mm, GW={r['GW_pct']}) ===", flush=True)
        missing = check_val_ready(eid)
        if missing:
            print(f"SKIP {eid}: val runs missing {len(missing)}: {missing[:4]}", flush=True)
            continue
        try:
            X, Y, tidx, cols = load_event_runs(eid)
            n_runs, T, n_st = Y.shape
            gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
            xlim = (X.min(0), X.max(0))

            O, M = p2lib.load_obs_matrix(tidx, cols)
            M, qc_excl = p2lib.qc_stations(O, M, Y.mean(0).T)
            phi = phi_from_residuals(O, M, Y.mean(0).T)

            W, per_pt = build_wmap_event(VAL_ROOT, eid, gps, Vt, Zmean, Ys, Ym,
                                         n_pc, T, n_st, tidx, cols, xlim)
            wm = W[M.T]
            print(f"  W@有效格 RMS={np.sqrt(wm.mean()):.4f} p99sd={np.sqrt(np.percentile(wm, 99)):.4f} "
                  f"(per-pt GP err {min(per_pt):.3f}-{max(per_pt):.3f})", flush=True)
            np.save(os.path.join(OUT, f"wmap_{eid}_v2.npy"), W)

            res = {}
            for model in ("E0", "E1", "E2"):
                res[model] = invert_real(model, O, np.arange(n_st), M, W, phi, gps, Vt,
                                         Zmean, Ys, Ym, n_pc, T, n_st, xlim)
            drop = {}
            for nsub in (15, 5):
                si = np.arange(min(nsub, n_st))
                drop[nsub] = invert_real("E1", O, si, M, W, phi, gps, Vt,
                                         Zmean, Ys, Ym, n_pc, T, n_st, xlim)
            raw[eid] = {
                "n_runs": n_runs, "T": int(T), "n_st": int(n_st),
                "n_obs": int(M.sum()), "n_valid_st": int(M.sum(1).clip(0, 1).sum()),
                "qc_excluded": len(qc_excl), "phi_med": float(np.median(phi)),
                "w_rms_valid": float(np.sqrt(wm.mean())),
                "w_p99sd": float(np.sqrt(np.percentile(wm, 99))),
                "R": {m: res[m]["R_j"] for m in res},
                "post_mean": {m: res[m]["mean"] for m in res},
                "post_sd": {m: res[m]["sd"] for m in res},
                "info": {m: res[m]["info"] for m in res},
                "ident": {m: ident_flags(res[m]) for m in res},
                "drop15_R": drop[15]["R_j"], "drop5_R": drop[5]["R_j"],
            }
            for model in ("E0", "E1", "E2"):
                Rj = res[model]["R_j"]
                rows.append({"event_id": eid, "stratum": r["stratum"], "depth_mm": r["depth_mm"],
                             "GW_pct": r["GW_pct"], "model": model,
                             "R_n1": Rj[0], "R_n2": Rj[1], "R_n3": Rj[2],
                             "info_n1": res[model]["info"][0], "info_n2": res[model]["info"][1],
                             "info_n3": res[model]["info"][2],
                             "ident_all": all(ident_flags(res[model])),
                             "ident_none": not any(v >= 0.5 for v in Rj),
                             "n_obs": int(M.sum()), "n_valid_st": int(M.sum(1).clip(0, 1).sum()),
                             "phi_med": float(np.median(phi)),
                             "w_rms_valid": float(np.sqrt(wm.mean()))})
            print({m: [round(x, 3) for x in res[m]["R_j"]] for m in ("E0", "E1", "E2")}, flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print("FAIL", eid, ex, flush=True)
            continue
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "xev_results_v2.csv"), index=False)
    json.dump(raw, open(os.path.join(OUT, "xev_raw_v2.json"), "w"), indent=1, default=float)
    print("per-event v2 analysis done:", len(raw), "events", flush=True)
