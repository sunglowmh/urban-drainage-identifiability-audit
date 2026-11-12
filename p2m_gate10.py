# -*- coding: utf-8 -*-
"""关口 10: 翻转矩阵汇总 + C1–C4 终分类 (prereg v2.1/A30, 零新运行)
全部数值直接引用 I13–I19 已登记产物."""
import json, os, csv
import numpy as np

P2 = "output/p2"
OUT = os.path.join(P2, "gate10")
EVENTS = [f"ev{i:02d}" for i in range(11)]
MODELS = ["E0", "E1", "E2"]
PARAMS = ["n1", "n2", "n3"]
PRIOR_LO, PRIOR_HI = 0.009, 0.022
THR_R, THR_KL, THR_PRED = 0.5, 0.1, 0.05

os.makedirs(OUT, exist_ok=True)

# ---------- 载入已登记证据 ----------
v2 = json.load(open(os.path.join(P2, "xev", "xev_raw_v2.json")))
_anchor = json.load(open(os.path.join(P2, "matrixA_stats_v2.json")))
v2["ev00"] = dict(R=_anchor["R_j"], info=_anchor["info"], ident=_anchor["ident"],
                  post_mean=_anchor["post_mean"], post_sd=_anchor["post_sd"])
corr = json.load(open(os.path.join(P2, "xev", "xev_raw_corr.json")))
pert = json.load(open(os.path.join(P2, "xev", "pert_raw.json")))
seg = json.load(open(os.path.join(P2, "gate8", "seg_results.json")))
boot = np.load(os.path.join(P2, "gate8", "boot_results.npz"), allow_pickle=True)

def seg_ident(key):
    d = seg[key]
    out = []
    for j in range(3):
        ok = (d["R_j"][j] >= THR_R and d["info"][j] >= THR_KL
              and PRIOR_LO <= d["mean"][j] <= PRIOR_HI)
        out.append(bool(ok))
    return out

SEG = {f"seg{s}_{m}": seg_ident(f"seg{s}_{m}") for s in (1, 2) for m in MODELS}

# A5: bootstrap P(R_j>=0.5) 仅锚点; E0/E2 用 K7, E1 另报 K6
boot_p = {}
for m in MODELS:
    rj = boot[f"K7_{m}_Rj"]          # (B,3)
    boot_p[m] = (rj >= THR_R).mean(0)
boot_p_K6 = {}
for m in MODELS:
    k = f"K6_{m}_Rj"
    if k in boot:
        boot_p_K6[m] = (boot[k] >= THR_R).mean(0)

# A6/A7: LOEO 判级与预测改善
loeo_ident = {}
loeo_pred = {}
for e in EVENTS:
    for m in MODELS:
        o = json.load(open(os.path.join(P2, "gate9", f"loeo_{e}_{m}.json")))
        loeo_ident[(e, m)] = bool(o["resolved"])
        loeo_pred[(e, m)] = float(o["crps_improve"])

# 敏感性: 单事件反演预测改善 (full) 与转移矩阵 (transfer)
full_pred = {}
for e in EVENTS:
    for m in MODELS:
        try:
            o = json.load(open(os.path.join(P2, "gate9", f"full_{e}_{m}.json")))
            full_pred[(e, m)] = float(o["crps_improve"])
        except Exception:
            full_pred[(e, m)] = None

trans_vals = []
try:
    for e in EVENTS:
        t = json.load(open(os.path.join(P2, "gate9", f"transfer_{e}.json")))
        for m in MODELS:
            for tgt, d in t[m].items():
                key = "crps_improve" if "crps_improve" in d else "crps_improve_vs_prior"
                if tgt != e and key in d:
                    trans_vals.append(float(d[key]))
except Exception:
    pass
trans_vals = [v for v in trans_vals if v is not None]
trans_off = None  # 填充于报告处(结构探测不确定时仅报 full 敏感性)

# ---------- 翻转矩阵 (逐 (e,m,j), 99 行) ----------
rows = []
for e in EVENTS:
    for m in MODELS:
        for j, pj in enumerate(PARAMS):
            a1 = bool(v2[e]["ident"][m][j])
            a2 = bool(corr[e]["ident"][m][j]) if e in corr else None
            # A3 扰动臂(仅 ev00): 严格 = 覆盖实现全部过; 另存逐实现明细供报告
            a3 = None
            if e == "ev00":
                per_impl = [bool(pert[t]["ident"][m][j]) for t in pert]
                a3 = all(per_impl)
            a4a = SEG[f"seg1_{m}"][j] if e == "ev00" else None
            a4b = SEG[f"seg2_{m}"][j] if e == "ev00" else None
            a5 = bool(boot_p[m][j] >= 0.95) if e == "ev00" else None
            a6 = loeo_ident[(e, m)]
            a7 = loeo_pred[(e, m)] >= THR_PRED
            rows.append(dict(event=e, model=m, param=pj,
                             A1_v2=a1, A2_corr=a2,
                             A3_pert=a3, A4_seg1=a4a, A4_seg2=a4b,
                             A5_bootP95=a5, A6_loeo=a6, A7_pred_ge5=a7,
                             R_j=v2[e]["R"][m][j], KL=v2[e]["info"][m][j],
                             bootP=None if a5 is None else float(boot_p[m][j])))

def wcsv(path, data, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)

wcsv(os.path.join(OUT, "flip_matrix.csv"), rows,
     ["event", "model", "param", "A1_v2", "A2_corr", "A3_pert", "A4_seg1",
      "A4_seg2", "A5_bootP95", "A6_loeo", "A7_pred_ge5", "R_j", "KL", "bootP"])

# ---------- C1–C4 判定 (逐 (e,j)) ----------
cls = []
for e in EVENTS:
    for j, pj in enumerate(PARAMS):
        passes = {m: bool(v2[e]["ident"][m][j]) for m in MODELS}
        means = [v2[e]["post_mean"][m][j] for m in MODELS]
        sds = [v2[e]["post_sd"][m][j] for m in MODELS]
        drift = (max(means) - min(means)) / max(sds)
        pred_per_m = {m: loeo_pred[(e, m)] for m in MODELS}
        pred_med = float(np.median(list(pred_per_m.values())))
        allp, anyp = all(passes.values()), any(passes.values())
        pred_ok = pred_med >= THR_PRED
        if allp and drift < 1.0 and pred_ok:
            cat, sub = "C1", ""
        elif anyp:
            cat = "C2"
            if allp and drift >= 1.0:
                sub = "C2-均值漂移(≥1σ)"
            elif allp and not pred_ok:
                sub = "C2-判级脆弱(预测无增益)"
            else:
                sub = "C2-判级翻转"
        elif pred_ok:
            cat, sub = "C3", ""
        else:
            cat, sub = "C4", ""
        flip_model_axis = len(set(passes.values())) > 1
        cls.append(dict(event=e, param=pj, E0=passes["E0"], E1=passes["E1"],
                        E2=passes["E2"], drift_sigma=round(drift, 1),
                        pred_med=round(pred_med, 4),
                        pred_E0=round(pred_per_m["E0"], 4),
                        pred_E1=round(pred_per_m["E1"], 4),
                        pred_E2=round(pred_per_m["E2"], 4),
                        category=cat, sublabel=sub,
                        flip_model_axis=flip_model_axis))

wcsv(os.path.join(OUT, "classification.csv"), cls,
     ["event", "param", "E0", "E1", "E2", "drift_sigma", "pred_med",
      "pred_E0", "pred_E1", "pred_E2", "category", "sublabel", "flip_model_axis"])

# ---------- 汇总统计 ----------
from collections import Counter
cat_cnt = Counter(c["category"] for c in cls)
sub_cnt = Counter(c["sublabel"] for c in cls if c["sublabel"])
per_param = {}
for pj in PARAMS:
    cc = Counter(c["category"] for c in cls if c["param"] == pj)
    per_param[pj] = dict(cc)
flip_units = sum(1 for c in cls if c["flip_model_axis"])

# 轴间一致率 (已覆盖对中判定一致的比例)
axis_pairs = [("A1_v2", "A6_loeo"), ("A1_v2", "A2_corr"), ("A6_loeo", "A2_corr")]
axis_agree = {}
for a, b in axis_pairs:
    n = t = 0
    for r in rows:
        va, vb = r[a], r[b]
        if va is not None and vb is not None:
            t += 1
            n += (va == vb)
    axis_agree[f"{a}|{b}"] = (n, t)

# T5: R_j Kendall τ (逐事件参数排名跨模型) + 可辨识集 Jaccard
from itertools import combinations
def kendall_tau(a, b):
    n = len(a)
    c = d = 0
    for i, jj in combinations(range(n), 2):
        s = (a[i] - a[jj]) * (b[i] - b[jj])
        c += s > 0
        d += s < 0
    return (c - d) / (n * (n - 1) / 2)

taus = {f"{m1}{m2}": [] for m1, m2 in combinations(MODELS, 2)}
for e in EVENTS:
    for m1, m2 in combinations(MODELS, 2):
        taus[f"{m1}{m2}"].append(kendall_tau(v2[e]["R"][m1], v2[e]["R"][m2]))
tau_med = {k: round(float(np.median(v)), 3) for k, v in taus.items()}

ident_sets = {m: set((e, j) for e in EVENTS for j in range(3)
                     if v2[e]["ident"][m][j]) for m in MODELS}
jac = {}
for m1, m2 in combinations(MODELS, 2):
    u = ident_sets[m1] | ident_sets[m2]
    jac[f"{m1}|{m2}"] = round(len(ident_sets[m1] & ident_sets[m2]) / len(u), 3) if u else 1.0

# 轴翻转计数 (逐 (e,m,j) 在已覆盖判级轴间不一致)
axis_cols = ["A1_v2", "A2_corr", "A3_pert", "A4_seg1", "A4_seg2", "A5_bootP95", "A6_loeo"]
axis_flip = 0
axis_covered = 0
for r in rows:
    vals = [r[c] for c in axis_cols if r[c] is not None]
    axis_covered += len(vals)
    axis_flip += (len(set(vals)) > 1)

summary = dict(cat_cnt=dict(cat_cnt), sub_cnt=dict(sub_cnt),
               per_param=per_param, flip_units_model_axis=flip_units,
               axis_agree=axis_agree, axis_flip=axis_flip,
               axis_covered=axis_covered, tau_med=tau_med, jaccard=jac,
               full_pred_by_model={m: {
                   "median": round(float(np.median([full_pred[(e, m)] for e in EVENTS])), 4),
                   "ge5": int(sum(full_pred[(e, m)] >= THR_PRED for e in EVENTS))}
                   for m in MODELS},
               transfer_walk_n=len(trans_vals),
               transfer_median=round(float(np.median(trans_vals)), 4) if trans_vals else None,
               transfer_ge5_frac=round(float(np.mean([v >= THR_PRED for v in trans_vals])), 3) if trans_vals else None)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1, ensure_ascii=False))
print("CLASSIFICATION:")
for c in cls:
    print(c["event"], c["param"], c["category"], c["sublabel"],
          "drift=%.1f" % c["drift_sigma"], "pred=%.4f" % c["pred_med"])
