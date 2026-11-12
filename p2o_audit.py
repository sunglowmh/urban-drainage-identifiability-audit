# -*- coding: utf-8 -*-
"""I22 口径审计与增补分析（全部为已登记数据的重新汇总，零新运行）"""
import json, os
import numpy as np

P2 = "output/p2"
EVENTS = [f"ev{i:02d}" for i in range(11)]
MODELS = ["E0", "E1", "E2"]
PARAMS = ["n1", "n2", "n3"]
PRIOR_LO, PRIOR_HI = 0.009, 0.022
PRIOR_W = PRIOR_HI - PRIOR_LO
rng = np.random.default_rng(20261111)

v2 = json.load(open(os.path.join(P2, "xev", "xev_raw_v2.json")))
anc = json.load(open(os.path.join(P2, "matrixA_stats_v2.json")))
v2["ev00"] = dict(R=anc["R_j"], info=anc["info"], ident=anc["ident"],
                  post_mean=anc["post_mean"], post_sd=anc["post_sd"],
                  n_valid_st=anc["n_valid_stations"])
boot = np.load(os.path.join(P2, "gate8", "boot_results.npz"), allow_pickle=True)

out = {}

# ---------- 1. 站点数审计 ----------
out["stations"] = {e: int(v2[e]["n_valid_st"]) for e in EVENTS}

# ---------- 2. 配对 CRPS 差（LOEO, 事件级有效格均值） ----------
pairs = []   # (e, m, crps_post, crps_prior)
for e in EVENTS:
    for m in MODELS:
        o = json.load(open(os.path.join(P2, "gate9", f"loeo_{e}_{m}.json")))
        pairs.append((e, m, float(o["post"]["crps_mean"]), float(o["prior"]["crps_mean"])))

def paired_stats(sub):
    """sub: list of (post, prior); 差 = post - prior (负=改善)"""
    d = np.array([p - q for p, q in sub])
    n = len(d)
    bs = np.array([np.mean(d[rng.integers(0, n, n)]) for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    try:
        from scipy import stats
        w_p = float(stats.wilcoxon(d).pvalue)
    except Exception:
        w_p = None
    return dict(n=n, mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                ci95=[float(lo), float(hi)], wilcoxon_p=w_p,
                # 事件层 bootstrap 的均值差 CI 即"可排除改善上界"依据:
                best_case_improve_bound=float(-hi))  # -hi > 0 时为可排除的改善上界(正数)

out["paired_crps_all33"] = paired_stats([(p, q) for _, _, p, q in pairs])
out["paired_crps_by_model"] = {m: paired_stats([(p, q) for e, mm, p, q in pairs if mm == m])
                               for m in MODELS}

# 逐事件逐模型差值明细
out["paired_detail"] = {f"{e}|{m}": round(p - q, 5) for e, m, p, q in pairs}

# ---------- 3. 跨模型预测极差口径复算 ----------
rel_range = {}
for e in EVENTS:
    posts = [next(p for ee, mm, p, q in pairs if ee == e and mm == m) for m in MODELS]
    pri = next(q for ee, mm, p, q in pairs if ee == e and mm == "E0")
    rel_range[e] = (max(posts) - min(posts)) / pri
out["rel_range_by_event"] = {e: round(v, 4) for e, v in rel_range.items()}
out["rel_range_median"] = float(np.median(list(rel_range.values())))

# ---------- 4. 漂移效应量三列重述 (v2 基线逐事件) ----------
drift_rows = []
for e in EVENTS:
    for j, pj in enumerate(PARAMS):
        means = [v2[e]["post_mean"][m][j] for m in MODELS]
        sds = [v2[e]["post_sd"][m][j] for m in MODELS]
        rng_abs = max(means) - min(means)
        drift_rows.append(dict(event=e, param=pj,
                               abs_range=round(rng_abs, 6),
                               prior_frac=round(rng_abs / PRIOR_W, 3),
                               sigma_mult=round(rng_abs / max(sds), 1)))
out["drift_effect_size"] = drift_rows
for key, sel in (("all", drift_rows), ("n1", [r for r in drift_rows if r["param"] == "n1"]),
                 ("n2", [r for r in drift_rows if r["param"] == "n2"]),
                 ("n3", [r for r in drift_rows if r["param"] == "n3"])):
    out[f"drift_summary_{key}"] = dict(
        abs_med=float(np.median([r["abs_range"] for r in sel])),
        prior_frac_med=float(np.median([r["prior_frac"] for r in sel])),
        sigma_med=float(np.median([r["sigma_mult"] for r in sel])))

# LOEO 合并反演的漂移 (物理量)
loeo_drift = []
for e in EVENTS:
    ms, ss = [], []
    for m in MODELS:
        o = json.load(open(os.path.join(P2, "gate9", f"loeo_{e}_{m}.json")))
        ms.append(o["mean"]); ss.append(o["sd"])
    ms = np.array(ms); ss = np.array(ss)
    for j in range(3):
        loeo_drift.append(dict(event=e, param=PARAMS[j],
                               abs_range=float(max(ms[:, j]) - min(ms[:, j])),
                               prior_frac=float((max(ms[:, j]) - min(ms[:, j])) / PRIOR_W),
                               sigma_mult=float((max(ms[:, j]) - min(ms[:, j])) / max(ss[:, j]))))
out["loeo_drift_summary"] = dict(
    abs_med=float(np.median([r["abs_range"] for r in loeo_drift])),
    prior_frac_med=float(np.median([r["prior_frac"] for r in loeo_drift])),
    sigma_med=float(np.median([r["sigma_mult"] for r in loeo_drift])),
    abs_max=float(max(r["abs_range"] for r in loeo_drift)))

# ---------- 5. W 底噪口径核对 ----------
wv = [v2[e].get("w_rms_valid") for e in EVENTS if "w_rms_valid" in v2[e]]
out["w_rms_valid"] = {e: round(float(v2[e]["w_rms_valid"]), 4) for e in EVENTS
                      if "w_rms_valid" in v2[e]}

# ---------- 6. C1–C4 独立逐格复算校验 ----------
cls_file = list(__import__("csv").DictReader(
    open(os.path.join(P2, "gate10", "classification.csv"), encoding="utf-8")))
loeo_pred = {(e, m): float(json.load(open(os.path.join(P2, "gate9", f"loeo_{e}_{m}.json")))["crps_improve"])
             for e in EVENTS for m in MODELS}
mismatch = 0
for row in cls_file:
    e, j = row["event"], int(["n1", "n2", "n3"].index(row["param"]))
    passes = [v2[e]["ident"][m][j] for m in MODELS]
    means = [v2[e]["post_mean"][m][j] for m in MODELS]
    sds = [v2[e]["post_sd"][m][j] for m in MODELS]
    drift = (max(means) - min(means)) / max(sds)
    pred = float(np.median([loeo_pred[(e, m)] for m in MODELS]))
    allp, anyp = all(passes), any(passes)
    if allp and drift < 1.0 and pred >= 0.05:
        cat = "C1"
    elif anyp:
        cat = "C2"
    elif pred >= 0.05:
        cat = "C3"
    else:
        cat = "C4"
    if cat != row["category"]:
        mismatch += 1
        print("MISMATCH", e, row["param"], cat, row["category"])
out["c1c4_recheck"] = dict(n=len(cls_file), mismatch=mismatch)

json.dump(out, open(os.path.join(P2, "gate10", "audit_i22.json"), "w"), indent=1, default=str)

# ---------- 打印摘要 ----------
print("== stations ==", out["stations"])
print("== paired all33 ==", json.dumps(out["paired_crps_all33"], default=str))
for m in MODELS:
    print(f"== paired {m} ==", json.dumps(out["paired_crps_by_model"][m], default=str))
print("== rel_range ==", out["rel_range_by_event"], "median:", out["rel_range_median"])
print("== drift summary ==", {k: v for k, v in out.items() if k.startswith("drift_summary") or k == "loeo_drift_summary"})
print("== w_rms ==", out["w_rms_valid"])
print("== C1C4 recheck ==", out["c1c4_recheck"])
