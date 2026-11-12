# -*- coding: utf-8 -*-
"""R-B12 (R2-#6): station 20 (TQ61003301) 边缘剔除敏感性 —— 推断层重算, 零新模拟
背景: station 20 在 §2.3 被固定排除 (offset +0.60 m, RMSE 0.67 m), 属规则驱动而非结果驱动
敏感性: 回补 station 20 (解除 QC 掩码) 后重算三结构判级 → 33 单元翻转计数
两臂: (a) raw 回补 (原观测, 含 0.6 m datum 偏移); (b) datum 校正后回补 (扣除 d_i, 同 R-A8 口径)
对照: 存档判级 (排除 station 20) 的 6 翻转
产出: output/p2/gate10/station20_reinclusion.csv + station20_reinclusion_summary.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import G9, EVENTS, MODELS, load_bundle
from p2_emulator_gate import load_runs
from p2_xev_posterior import load_event_runs
from p2_posterior_matrixA import (recon_series, posterior_diag, PRIOR_LO, PRIOR_HI)
from p2i_gate9_lib import event_ll

G10 = "output/p2/gate10"
ST20 = "TQ61003301"


def ident(d):
    return bool(d["R_j"] >= 0.5 and d["info_nats"] >= 0.1)


def st20_offset(eid, b):
    """station 20 的 datum 偏移 (与 R-A8 同口径): O − 90 LHS 均值 的中位数"""
    if eid == "ev00":
        _, Y, _, _ = load_runs()
    else:
        _, Y, _, _ = load_event_runs(eid)
    Ybase = Y.mean(0).T
    i = b["cols"].index(ST20)
    ok = np.isfinite(b["O"][i])
    return float(np.median(b["O"][i, ok] - Ybase[i, ok])) if ok.sum() >= 20 else np.nan


def run_one(args):
    eid, model, arm = args
    b = load_bundle(eid)
    z = np.load(os.path.join(G9, f"full_{eid}_{model}.npz"))
    grid = z["grid"]
    si = np.arange(b["n_st"])

    i20 = b["cols"].index(ST20)
    if not np.isfinite(b["O"][i20]).any():
        return {"event": eid, "model": model, "arm": arm, "skip": True}

    # 存档判级 (排除 station 20)
    d0, _, _, _ = posterior_diag(z["ll"], grid)
    id0 = [ident(d) for d in d0]

    # 回补臂: 解除 station 20 掩码
    b2 = dict(b)
    M2 = b["M"].copy()
    M2[i20] = np.isfinite(b["O"][i20])
    O2 = b["O"].copy()
    if arm == "datum_corr":
        off = st20_offset(eid, b)
        if np.isfinite(off):
            O2[i20] = O2[i20] - off
    b2["M"] = M2
    b2["O"] = O2
    ll = event_ll(b2, model, grid, si)
    d1, _, _, _ = posterior_diag(ll, grid)
    id1 = [ident(d) for d in d1]

    return {"event": eid, "model": model, "arm": arm,
            "ident_arch": id0, "ident_reincl": id1,
            "mean_shift": [round(d1[j]["mean"] - d0[j]["mean"], 6) for j in range(3)],
            "n_valid_st20": int(M2[i20].sum())}


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(G10, exist_ok=True)
    jobs = [(e, m, a) for e in EVENTS for m in MODELS for a in ("raw", "datum_corr")]
    results = []
    with ProcessPoolExecutor(max_workers=11) as ex:
        for res in ex.map(run_one, jobs):
            results.append(res)
            if not res.get("skip"):
                print(f"done {res['event']}/{res['model']}/{res['arm']}", flush=True)
    df = pd.DataFrame([r for r in results if not r.get("skip")])
    df.to_csv(os.path.join(G10, "station20_reinclusion.csv"), index=False)

    summary = {}
    for arm in ("raw", "datum_corr"):
        sub = df[df.arm == arm]
        flips = []
        for e in EVENTS:
            s = sub[sub.event == e]
            if s.empty:
                continue
            for j in range(3):
                ia = {m: bool(s[s.model == m]["ident_arch"].iloc[0][j]) for m in MODELS}
                ir = {m: bool(s[s.model == m]["ident_reincl"].iloc[0][j]) for m in MODELS}
                fa = [m for m in MODELS[1:] if ia[m] != ia["E0"]]
                fr = [m for m in MODELS[1:] if ir[m] != ir["E0"]]
                if fa or fr:
                    flips.append({"event": e, "param": f"n{j+1}",
                                  "flips_arch": "|".join(fa), "flips_reincl": "|".join(fr)})
        summary[arm] = {
            "n_flip_units_arch": len([f for f in flips if f["flips_arch"]]),
            "n_flip_units_reincl": len([f for f in flips if f["flips_reincl"]]),
            "flip_units": flips,
            "pass_total_arch": {m: int(sub[sub.model == m]["ident_arch"].apply(sum).sum()) for m in MODELS},
            "pass_total_reincl": {m: int(sub[sub.model == m]["ident_reincl"].apply(sum).sum()) for m in MODELS},
        }
    offs = {e: st20_offset(e, load_bundle(e)) for e in EVENTS}
    summary["st20_datum_offset_by_event"] = {e: (round(v, 4) if np.isfinite(v) else None) for e, v in offs.items()}
    json.dump(summary, open(os.path.join(G10, "station20_reinclusion_summary.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "flip_units"} if isinstance(v, dict) else v
                      for k, v in summary.items()}, indent=1))
