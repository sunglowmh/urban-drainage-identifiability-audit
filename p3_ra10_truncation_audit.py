# -*- coding: utf-8 -*-
"""R-A10 (R1-M3): 网格截断审计 —— 6 个翻转代表单元, 41³ 全局细网格 vs 存档 zoomed 网格
单元选取: 6 个判级翻转单元 (ev01/ev03: E1, ev01/ev02/ev05/ev06: E2 —— 覆盖全部翻转事件与两模型)
审计内容 (零新模拟, 单事件似然重评):
  1. 收缩盒内 vs 盒外 max ℓ 差 (全局网格上直接测量, 无需插桩 zoom_eval)
  2. 盒外后验质量 (全局网格归一化权重)
  3. 矩稳定性: 全局网格 vs 存档网格的 mean/sd/R_j/PASS 对照
产出: output/p2/gate10/grid_truncation_audit.csv + _summary.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import G9, load_bundle, event_ll
from p2_posterior_matrixA import posterior_diag, PRIOR_LO, PRIOR_HI

UNITS = [("ev01", "E1"), ("ev01", "E2"), ("ev02", "E2"),
         ("ev03", "E1"), ("ev05", "E2"), ("ev06", "E2")]
N_GLOBAL = 41
G10 = "output/p2/gate10"


def ident(d):
    return bool(d["R_j"] >= 0.5 and d["info_nats"] >= 0.1)


def run_one(args):
    eid, model = args
    b = load_bundle(eid)
    z = np.load(os.path.join(G9, f"full_{eid}_{model}.npz"))
    grid_z, ll_z, bnd = z["grid"], z["ll"], z["bnd"]
    si = np.arange(b["n_st"])

    # ---- 41³ 全局细网格 (不收缩) ----
    g1 = [np.linspace(PRIOR_LO, PRIOR_HI, N_GLOBAL) for _ in range(3)]
    grid_g = np.stack(np.meshgrid(*g1, indexing="ij"), -1).reshape(-1, 3)
    ll_g = event_ll(b, model, grid_g, si)  # s2 锚点 = 全局网格角点 (同规则)
    d_g, w_g, _, _ = posterior_diag(ll_g, grid_g)
    d_z, w_z, _, _ = posterior_diag(ll_z, grid_z)

    inside = np.all((grid_g >= bnd[:, 0] - 1e-12) & (grid_g <= bnd[:, 1] + 1e-12), axis=1)
    ll_in = float(ll_g[inside].max())
    ll_out = float(ll_g[~inside].max())
    mass_out = float(w_g[~inside].sum())

    row = {"event": eid, "model": model, "n_global": N_GLOBAL,
           "ll_out_minus_in": round(ll_out - ll_in, 4),
           "mass_outside_zoombox": round(mass_out, 5),
           "resolved_arch": bool(z["resolved"])}
    for j, p in enumerate(("n1", "n2", "n3")):
        row[f"{p}_mean_zoom"] = round(d_z[j]["mean"], 6)
        row[f"{p}_mean_glob"] = round(d_g[j]["mean"], 6)
        row[f"{p}_mean_shift_frac"] = round(abs(d_g[j]["mean"] - d_z[j]["mean"]) / (PRIOR_HI - PRIOR_LO), 5)
        row[f"{p}_sd_ratio_glob_zoom"] = round(d_g[j]["sd"] / max(d_z[j]["sd"], 1e-12), 3)
        row[f"{p}_Rj_zoom"] = round(d_z[j]["R_j"], 4)
        row[f"{p}_Rj_glob"] = round(d_g[j]["R_j"], 4)
        row[f"{p}_ident_zoom"] = ident(d_z[j])
        row[f"{p}_ident_glob"] = ident(d_g[j])
    return row


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(G10, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for row in ex.map(run_one, UNITS):
            rows.append(row)
            ident_changes = [p for p in ("n1", "n2", "n3")
                             if row[f"{p}_ident_zoom"] != row[f"{p}_ident_glob"]]
            print(f"done {row['event']}/{row['model']}: ll_out-in={row['ll_out_minus_in']}, "
                  f"mass_out={row['mass_outside_zoombox']}, ident_changes={ident_changes}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(G10, "grid_truncation_audit.csv"), index=False)
    ident_flips = []
    for _, r in df.iterrows():
        for p in ("n1", "n2", "n3"):
            if r[f"{p}_ident_zoom"] != r[f"{p}_ident_glob"]:
                ident_flips.append({"event": r["event"], "model": r["model"], "param": p,
                                    "zoom": bool(r[f"{p}_ident_zoom"]), "global": bool(r[f"{p}_ident_glob"])})
    summary = {
        "units": [f"{e}|{m}" for e, m in UNITS],
        "max_mass_outside_zoombox": float(df["mass_outside_zoombox"].max()),
        "max_ll_out_minus_in": float(df["ll_out_minus_in"].max()),
        "max_mean_shift_prior_frac": float(df[[f"{p}_mean_shift_frac" for p in ("n1", "n2", "n3")]].max().max()),
        "ident_flips_zoom_vs_global": ident_flips,
    }
    json.dump(summary, open(os.path.join(G10, "grid_truncation_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
