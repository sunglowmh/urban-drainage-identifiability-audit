# -*- coding: utf-8 -*-
"""关口 9 LOCO worker: p2i_worker_loco.py <eid> <cluster_label> <model>
留一簇站点留出: 该簇站点不入似然, 其余站反演 → 留出簇站点预测 (A17 三层)"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import (G9, S_SAMPLES, load_bundle, pooled_invert, station_index,
                           sample_theta_post, sample_theta_prior, profile_pick,
                           predictive_samples, pred_metrics, save_json, seed_of)
from p2_posterior_matrixA import posterior_diag

eid, clabel, model = sys.argv[1], sys.argv[2], sys.argv[3]
b = load_bundle(eid)
idx_of = station_index(b)
cl = json.load(open("E:/work/workBuddy/BlueHeartV2/output/p2/station_clusters.json"))
sh = np.array([idx_of[n] for n in cl["clusters"][clabel]])
all_idx = np.arange(b["n_st"])
si = np.setdiff1d(all_idx, sh)

ll, grid, bnd, resolved, s2_grids = pooled_invert([eid], model, si_fn=lambda bb: si)
diag, w, eff, support = posterior_diag(ll, grid)
anchor = grid[0]
out = {"eid": eid, "cluster": clabel, "model": model,
       "n_holdout_st": len(sh), "resolved": bool(resolved), "support": int(support),
       "R_j": [d["R_j"] for d in diag], "mean": [d["mean"] for d in diag],
       "sd": [d["sd"] for d in diag], "info": [d["info_nats"] for d in diag]}

# s2* / tau* 仅用校准站 (si) profile — 防泄漏 (I18 补遗)
rng = np.random.default_rng(seed_of(f"loco|{eid}|{clabel}|{model}|post"))
theta_post = sample_theta_post(grid, ll, rng)
s2_star, tau_star = profile_pick(b, model, theta_post, si, anchor)
rngp = np.random.default_rng(seed_of(f"loco|{eid}|{clabel}|{model}|postres"))
X_post = predictive_samples(b, model, theta_post, s2_star, tau_star, rngp)
out["post"] = pred_metrics(X_post, b, sh, f"loco|{eid}|{clabel}|{model}|post")

rng2 = np.random.default_rng(seed_of(f"loco|{eid}|{clabel}|{model}|prior"))
theta_prior = sample_theta_prior(rng2)
s2_p, tau_p = profile_pick(b, model, theta_prior, si, anchor)
rng3 = np.random.default_rng(seed_of(f"loco|{eid}|{clabel}|{model}|priorres"))
X_prior = predictive_samples(b, model, theta_prior, s2_p, tau_p, rng3)
out["prior"] = pred_metrics(X_prior, b, sh, f"loco|{eid}|{clabel}|{model}|prior")

if out["post"].get("n_cells", 0) > 0 and out["prior"].get("crps_mean", 0) > 0:
    out["crps_improve"] = (out["prior"]["crps_mean"] - out["post"]["crps_mean"]) / \
                          out["prior"]["crps_mean"]
save_json(os.path.join(G9, f"loco_{eid}_{clabel}_{model}.json"), out)
print(f"LOCO {eid}/{clabel}/{model} done", flush=True)
