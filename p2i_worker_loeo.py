# -*- coding: utf-8 -*-
"""关口 9 LOEO worker: p2i_worker_loeo.py <fold_eid> <model>
合并校准 (其余 10 事件) → 合并后验落盘 → 留出事件预测 (后验/先验/self-profile 敏感性)"""
import os
import sys
import numpy as np

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import (G9, EVENTS, S_SAMPLES, load_bundle, pooled_invert,
                           sample_theta_post, sample_theta_prior, profile_pick,
                           predictive_samples, pred_metrics, save_json, seed_of)

fold, model = sys.argv[1], sys.argv[2]
calib = [e for e in EVENTS if e != fold]
b_h = load_bundle(fold)

ll, grid, bnd, resolved, s2_grids = pooled_invert(calib, model)
np.savez_compressed(os.path.join(G9, f"loeo_{fold}_{model}.npz"),
                    grid=grid, ll=ll, bnd=bnd, resolved=resolved,
                    s2_grids=np.stack([s2_grids[e] for e in calib]),
                    calib=np.array(calib))
diag, w, eff, support = None, None, None, None
from p2_posterior_matrixA import posterior_diag
diag, w, eff, support = posterior_diag(ll, grid)

anchor = grid[0]
out = {"fold": fold, "model": model, "resolved": bool(resolved), "support": int(support),
       "eff": float(eff), "R_j": [d["R_j"] for d in diag], "sd": [d["sd"] for d in diag],
       "mean": [d["mean"] for d in diag], "info": [d["info_nats"] for d in diag]}

# ---- 后验预测 (nuisance = 校准事件中位传递, I18 补遗) ----
rng = np.random.default_rng(seed_of(f"loeo|{fold}|{model}|post"))
theta_post = sample_theta_post(grid, ll, rng)
s2_med = np.zeros(len(theta_post))
tau_med = np.zeros(len(theta_post)) if model == "E2" else None
for e in calib:
    b_e = load_bundle(e)
    s2_e, tau_e = profile_pick(b_e, model, theta_post, np.arange(b_e["n_st"]), anchor)
    s2_med += s2_e
    if model == "E2":
        tau_med += tau_e
s2_med /= len(calib)
if model == "E2":
    tau_med /= len(calib)
rngp = np.random.default_rng(seed_of(f"loeo|{fold}|{model}|postres"))
X_post = predictive_samples(b_h, model, theta_post, s2_med, tau_med, rngp)
out["post"] = pred_metrics(X_post, b_h, np.arange(b_h["n_st"]), f"loeo|{fold}|{model}|post")

# ---- self-profile 敏感性 (留出事件自身 profile — 仅敏感性, I18 补遗) ----
s2_self, tau_self = profile_pick(b_h, model, theta_post, np.arange(b_h["n_st"]), anchor)
rngs = np.random.default_rng(seed_of(f"loeo|{fold}|{model}|selfres"))
X_self = predictive_samples(b_h, model, theta_post, s2_self, tau_self, rngs)
out["post_selfprofile"] = pred_metrics(X_self, b_h, np.arange(b_h["n_st"]),
                                       f"loeo|{fold}|{model}|self")

# ---- 先验预测 (同 nuisance 传递规则, 先验 θ 处 profile) ----
rng2 = np.random.default_rng(seed_of(f"loeo|{fold}|{model}|prior"))
theta_prior = sample_theta_prior(rng2)
s2_med_p = np.zeros(len(theta_prior))
tau_med_p = np.zeros(len(theta_prior)) if model == "E2" else None
for e in calib:
    b_e = load_bundle(e)
    s2_e, tau_e = profile_pick(b_e, model, theta_prior, np.arange(b_e["n_st"]), anchor)
    s2_med_p += s2_e
    if model == "E2":
        tau_med_p += tau_e
s2_med_p /= len(calib)
if model == "E2":
    tau_med_p /= len(calib)
rng3 = np.random.default_rng(seed_of(f"loeo|{fold}|{model}|priorres"))
X_prior = predictive_samples(b_h, model, theta_prior, s2_med_p, tau_med_p, rng3)
out["prior"] = pred_metrics(X_prior, b_h, np.arange(b_h["n_st"]), f"loeo|{fold}|{model}|prior")

if "post" in out and "prior" in out and out["post"].get("n_cells", 0) > 0:
    cp, pr = out["post"]["crps_mean"], out["prior"]["crps_mean"]
    out["crps_improve"] = (pr - cp) / pr if pr > 0 else None
save_json(os.path.join(G9, f"loeo_{fold}_{model}.json"), out)
print(f"LOEO {fold}/{model} done: improve={out.get('crps_improve')}", flush=True)
