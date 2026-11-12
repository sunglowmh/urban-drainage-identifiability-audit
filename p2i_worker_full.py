# -*- coding: utf-8 -*-
"""关口 9 全站反演 worker: p2i_worker_full.py <eid> <model>
全站反演 (权重落盘: 转移矩阵/LOCO 基准共用) + in-sample 预测 (后验/先验)"""
import os
import sys
import numpy as np

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import (G9, S_SAMPLES, load_bundle, pooled_invert,
                           sample_theta_post, sample_theta_prior, profile_pick,
                           predictive_samples, pred_metrics, save_json, seed_of)
from p2_posterior_matrixA import posterior_diag

eid, model = sys.argv[1], sys.argv[2]
b = load_bundle(eid)
si = np.arange(b["n_st"])

ll, grid, bnd, resolved, s2_grids = pooled_invert([eid], model)
np.savez_compressed(os.path.join(G9, f"full_{eid}_{model}.npz"),
                    grid=grid, ll=ll, bnd=bnd, resolved=resolved,
                    s2_grid=s2_grids[eid])
diag, w, eff, support = posterior_diag(ll, grid)
anchor = grid[0]
out = {"eid": eid, "model": model, "resolved": bool(resolved), "support": int(support),
       "eff": float(eff), "R_j": [d["R_j"] for d in diag], "sd": [d["sd"] for d in diag],
       "mean": [d["mean"] for d in diag], "info": [d["info_nats"] for d in diag]}

rng = np.random.default_rng(seed_of(f"full|{eid}|{model}|post"))
theta_post = sample_theta_post(grid, ll, rng)
s2_star, tau_star = profile_pick(b, model, theta_post, si, anchor)
rngp = np.random.default_rng(seed_of(f"full|{eid}|{model}|postres"))
X_post = predictive_samples(b, model, theta_post, s2_star, tau_star, rngp)
out["post"] = pred_metrics(X_post, b, si, f"full|{eid}|{model}|post")

rng2 = np.random.default_rng(seed_of(f"full|{eid}|{model}|prior"))
theta_prior = sample_theta_prior(rng2)
s2_p, tau_p = profile_pick(b, model, theta_prior, si, anchor)
rng3 = np.random.default_rng(seed_of(f"full|{eid}|{model}|priorres"))
X_prior = predictive_samples(b, model, theta_prior, s2_p, tau_p, rng3)
out["prior"] = pred_metrics(X_prior, b, si, f"full|{eid}|{model}|prior")

if out["post"].get("n_cells", 0) > 0 and out["prior"].get("crps_mean", 0) > 0:
    out["crps_improve"] = (out["prior"]["crps_mean"] - out["post"]["crps_mean"]) / \
                          out["prior"]["crps_mean"]
save_json(os.path.join(G9, f"full_{eid}_{model}.json"), out)
print(f"FULL {eid}/{model} done", flush=True)
