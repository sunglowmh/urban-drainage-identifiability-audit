# -*- coding: utf-8 -*-
"""R-A8 (R2-#5): datum 残留 × E2 优势辅助臂 —— 推断层重算, 零新模拟
逐站 datum 偏移 d_i = median(obs − 基准运行均值) (与 QC 同口径, p2lib.qc_stations 的 off)
校正臂: O' = O − d_i 后在存档最终网格上重算三结构判级 (s² 锚点随残差重定),
对照原判级: 检验 E1/E2 判级优势与 6/33 翻转是否因 E2 吸收未处理 datum 残差
说明: φ 保持原值 (隔离 datum 单一因素); QC 掩码不变
产出: output/p2/gate10/datum_offsets.csv + datum_arm_grading.csv + datum_arm_summary.json
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
from p2i_gate9_lib import _anchor_s2grid, event_ll

G10 = "output/p2/gate10"
_VAR = (PRIOR_HI - PRIOR_LO) ** 2 / 12


def ident(d):
    return bool(d["R_j"] >= 0.5 and d["info_nats"] >= 0.1)


def datum_offsets(eid, b):
    """逐站 datum 偏移 (QC 同口径): Ybase = 90 LHS 运行逐格均值"""
    if eid == "ev00":
        _, Y, _, _ = load_runs()
    else:
        _, Y, _, _ = load_event_runs(eid)
    Ybase = Y.mean(0).T  # (T, n_st)
    offs = np.full(b["n_st"], np.nan)
    for i in range(b["n_st"]):
        ok = b["M"][i]
        if ok.sum() >= 20:
            e = b["O"][i, ok] - Ybase[i, ok]
            offs[i] = float(np.median(e))
    return offs


def run_one(args):
    eid, model = args
    b = load_bundle(eid)
    offs = datum_offsets(eid, b)
    z = np.load(os.path.join(G9, f"full_{eid}_{model}.npz"))
    grid = z["grid"]
    si = np.arange(b["n_st"])

    # 原判级 (存档 ll)
    d0, _, _, _ = posterior_diag(z["ll"], grid)
    id0 = [ident(d) for d in d0]

    # 校正臂: O' = O − d_i
    b2 = dict(b)
    O2 = b["O"].copy()
    for i in range(b["n_st"]):
        if np.isfinite(offs[i]):
            O2[i] = O2[i] - offs[i]
    b2["O"] = O2
    ll_corr = event_ll(b2, model, grid, si)  # s2 锚点随校正残差重定 (同规则)
    d1, _, _, _ = posterior_diag(ll_corr, grid)
    id1 = [ident(d) for d in d1]

    return {"event": eid, "model": model,
            "Rj_orig": [round(d["R_j"], 4) for d in d0],
            "Rj_corr": [round(d["R_j"], 4) for d in d1],
            "ident_orig": id0, "ident_corr": id1,
            "mean_shift": [round(d1[j]["mean"] - d0[j]["mean"], 6) for j in range(3)],
            "offset_median": float(np.nanmedian(offs)),
            "offset_absmax": float(np.nanmax(np.abs(offs))),
            "n_offset_gt01": int((np.abs(offs) > 0.1).sum())}


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(G10, exist_ok=True)
    jobs = [(e, m) for e in EVENTS for m in MODELS]
    results = []
    with ProcessPoolExecutor(max_workers=11) as ex:
        for res in ex.map(run_one, jobs):
            results.append(res)
            print(f"done {res['event']}/{res['model']}: "
                  f"orig={sum(res['ident_orig'])} corr={sum(res['ident_corr'])} "
                  f"|off|max={res['offset_absmax']:.2f}", flush=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(G10, "datum_arm_grading.csv"), index=False)

    # 逐站偏移分布
    off_rows = []
    for e in EVENTS:
        b = load_bundle(e)
        offs = datum_offsets(e, b)
        for i, v in enumerate(offs):
            if np.isfinite(v):
                off_rows.append({"event": e, "station_idx": i, "station": b["cols"][i],
                                 "datum_offset_m": round(float(v), 4)})
    pd.DataFrame(off_rows).to_csv(os.path.join(G10, "datum_offsets.csv"), index=False)

    # 汇总: 翻转单元在校正臂下的变化
    flips = []
    for e in EVENTS:
        for j in range(3):
            io = {m: bool(df[(df.event == e) & (df.model == m)]["ident_orig"].iloc[0][j]) for m in MODELS}
            ic = {m: bool(df[(df.event == e) & (df.model == m)]["ident_corr"].iloc[0][j]) for m in MODELS}
            fo = [m for m in MODELS[1:] if io[m] != io["E0"]]
            fc = [m for m in MODELS[1:] if ic[m] != ic["E0"]]
            if fo or fc:
                flips.append({"event": e, "param": f"n{j+1}",
                              "flips_orig": "|".join(fo), "flips_corr": "|".join(fc),
                              "ident_orig": io, "ident_corr": ic})
    summary = {
        "n_flip_units_orig": len([f for f in flips if f["flips_orig"]]),
        "n_flip_units_corr": len([f for f in flips if f["flips_corr"]]),
        "flip_units": flips,
        "offset_summary": {e: {"median": float(np.nanmedian([r["datum_offset_m"] for r in off_rows if r["event"] == e])),
                               "absmax": float(np.nanmax(np.abs([r["datum_offset_m"] for r in off_rows if r["event"] == e])))}
                           for e in EVENTS},
        "pass_total_by_model_orig": {m: int(df[df.model == m]["ident_orig"].apply(sum).sum()) for m in MODELS},
        "pass_total_by_model_corr": {m: int(df[df.model == m]["ident_corr"].apply(sum).sum()) for m in MODELS},
    }
    json.dump(summary, open(os.path.join(G10, "datum_arm_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1, default=str))
