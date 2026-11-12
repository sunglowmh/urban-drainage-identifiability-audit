# -*- coding: utf-8 -*-
"""R-A6 (DA-C3/R1-M1): 边界质量审计 + 高信息子集漂移 —— 复用 gate9 存档网格, 零新模拟
产出:
  output/p2/gate10/boundary_mass_audit.csv   逐 (来源, event, model, param) 边界质量占比
  output/p2/gate10/boundary_mass_summary.json 汇总: 堆积单元清单/flip 重计/高信息子集漂移
边界堆积判据: 单参数边界质量(最近格点列) >= 0.5 → piled_low / piled_high
"""
import json
import os
import numpy as np
import pandas as pd

P2 = "output/p2"
G9 = os.path.join(P2, "gate9")
G10 = os.path.join(P2, "gate10")
EVENTS = [f"ev{i:02d}" for i in range(11)]
MODELS = ["E0", "E1", "E2"]
PARAMS = ["n1", "n2", "n3"]
PRIOR_LO, PRIOR_HI = 0.009, 0.022
I_E = {"ev00": 10.66, "ev08": 7.68, "ev10": 3.18, "ev07": 1.44, "ev09": 0.49}
HIGH_INFO = [e for e in EVENTS if I_E.get(e, 0.45) >= 5.0]  # ev00, ev08


def load_post(npz_path):
    z = np.load(npz_path)
    grid, ll = z["grid"], z["ll"]
    w = np.exp(ll - np.nanmax(ll))
    w = np.where(np.isfinite(w), w, 0.0)
    w /= w.sum()
    return grid, ll, w


def boundary_mass(grid, w, j):
    """参数 j 的低/高边界质量: 最近边界格点列的质量和"""
    vals = np.unique(grid[:, j])
    lo_col = np.isclose(grid[:, j], vals[0])
    hi_col = np.isclose(grid[:, j], vals[-1])
    return float(w[lo_col].sum()), float(w[hi_col].sum())


def diag(grid, ll):
    w = np.exp(ll - np.nanmax(ll))
    w = np.where(np.isfinite(w), w, 0.0)
    w /= w.sum()
    var_prior = (PRIOR_HI - PRIOR_LO) ** 2 / 12
    out = []
    for d in range(3):
        gm = grid[:, d]
        m = (w * gm).sum()
        v = (w * (gm - m) ** 2).sum()
        out.append({"mean": m, "sd": np.sqrt(v), "R_j": 1 - v / var_prior,
                    "info": 0.5 * np.log(var_prior / max(v, 1e-300))})
    return out


def ident_of(d):
    return [bool(d[j]["R_j"] >= 0.5 and d[j]["info"] >= 0.1) for j in range(3)]


rows, full = [], {}
# ---------- 1. 单事件 full 后验 (33 判级单元的载体) ----------
for e in EVENTS:
    for m in MODELS:
        grid, ll, w = load_post(os.path.join(G9, f"full_{e}_{m}.npz"))
        d = diag(grid, ll)
        full[(e, m)] = {"diag": d, "ident": ident_of(d)}
        for j in range(3):
            plo, phi_ = boundary_mass(grid, w, j)
            rows.append({"source": "full", "event": e, "model": m, "param": PARAMS[j],
                         "mass_low": round(plo, 4), "mass_high": round(phi_, 4),
                         "piled": bool(plo >= 0.5 or phi_ >= 0.5),
                         "R_j": round(d[j]["R_j"], 3), "pass": full[(e, m)]["ident"][j]})

# ---------- 2. LOEO 合并后验 (68.3% 漂移的载体) ----------
loeo_rows = []
for e in EVENTS:
    for m in MODELS:
        grid, ll, w = load_post(os.path.join(G9, f"loeo_{e}_{m}.npz"))
        d = diag(grid, ll)
        for j in range(3):
            plo, phi_ = boundary_mass(grid, w, j)
            rows.append({"source": "loeo", "event": e, "model": m, "param": PARAMS[j],
                         "mass_low": round(plo, 4), "mass_high": round(phi_, 4),
                         "piled": bool(plo >= 0.5 or phi_ >= 0.5),
                         "R_j": round(d[j]["R_j"], 3), "pass": None})

# ---------- 3. 逐单元漂移 × 边界堆积交叉 ----------
# 单元 = (event, param); 漂移 = full 后验均值跨模型极差 (与 p2o_audit 口径一致)
units = []
for e in EVENTS:
    for j in range(3):
        means = np.array([full[(e, m)]["diag"][j]["mean"] for m in MODELS])
        sds = np.array([full[(e, m)]["diag"][j]["sd"] for m in MODELS])
        rng_ = float(means.max() - means.min())
        piled_by = [MODELS[k] for k in range(3)
                    if rows[[r for r in range(len(rows))
                             if rows[r]["source"] == "full" and rows[r]["event"] == e
                             and rows[r]["model"] == MODELS[k] and rows[r]["param"] == PARAMS[j]][0]]["piled"]]
        ident_all = {m: full[(e, m)]["ident"][j] for m in MODELS}
        flips = [m for m in MODELS[1:] if ident_all[m] != ident_all["E0"]]
        units.append({"event": e, "param": PARAMS[j], "abs_range": round(rng_, 6),
                      "prior_frac": round(rng_ / (PRIOR_HI - PRIOR_LO), 3),
                      "any_piled": bool(piled_by), "piled_models": "|".join(piled_by),
                      "flip_models": "|".join(flips), "high_info": e in HIGH_INFO})

udf = pd.DataFrame(units)
summary = {
    "n_units": len(udf),
    "n_units_any_piled": int(udf["any_piled"].sum()),
    "piled_unit_events": sorted(udf.loc[udf["any_piled"], "event"].unique().tolist()),
    "flip_units_total": int((udf["flip_models"] != "").sum()),
    "flip_units_piled": int(((udf["flip_models"] != "") & udf["any_piled"]).sum()),
    "flip_unit_list": udf.loc[udf["flip_models"] != "", ["event", "param", "flip_models", "piled_models"]].to_dict("records"),
    "drift_median_all": float(udf["prior_frac"].median()),
    "drift_median_high_info": float(udf.loc[udf["high_info"], "prior_frac"].median()),
    "drift_median_low_info": float(udf.loc[~udf["high_info"], "prior_frac"].median()),
    "drift_median_no_piled": float(udf.loc[~udf["any_piled"], "prior_frac"].median()),
    "drift_median_piled": float(udf.loc[udf["any_piled"], "prior_frac"].median()),
    "high_info_events": HIGH_INFO,
}

# 剔除堆积单元后的 flip 重计 (仅比较三模型后验均非堆积的单元)
keep = udf[~udf["any_piled"]]
flip_keep = keep[keep["flip_models"] != ""]
summary["flip_units_after_excluding_piled"] = int(len(flip_keep))
summary["flip_units_after_excl_detail"] = flip_keep[["event", "param", "flip_models"]].to_dict("records")

# LOEO 漂移 (68.3% 主值口径复算 + 子集)
loeo_drift = []
for e in EVENTS:
    ms, ss = [], []
    for m in MODELS:
        grid, ll, _ = load_post(os.path.join(G9, f"loeo_{e}_{m}.npz"))
        d = diag(grid, ll)
        ms.append([d[j]["mean"] for j in range(3)])
        ss.append([d[j]["sd"] for j in range(3)])
    ms, ss = np.array(ms), np.array(ss)
    for j in range(3):
        loeo_drift.append({"event": e, "param": PARAMS[j],
                           "prior_frac": float((ms[:, j].max() - ms[:, j].min()) / (PRIOR_HI - PRIOR_LO)),
                           "high_info": e in HIGH_INFO})
ldf = pd.DataFrame(loeo_drift)
summary["loeo_prior_frac_median_all"] = float(ldf["prior_frac"].median())
summary["loeo_prior_frac_median_high_info"] = float(ldf.loc[ldf["high_info"], "prior_frac"].median())
summary["loeo_prior_frac_median_low_info"] = float(ldf.loc[~ldf["high_info"], "prior_frac"].median())

os.makedirs(G10, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(G10, "boundary_mass_audit.csv"), index=False)
udf.to_csv(os.path.join(G10, "boundary_mass_units.csv"), index=False)
json.dump(summary, open(os.path.join(G10, "boundary_mass_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
