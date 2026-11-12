# -*- coding: utf-8 -*-
"""关口 9 补充 worker: p2i_worker_fullsub.py <eid> <model>
复现全站后验预测样本 (与 full worker 同种子) → 逐簇/边界站子集指标 (T4 空间迁移代价基准)"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import (G9, load_bundle, station_index, sample_theta_post,
                           profile_pick, predictive_samples, pred_metrics, save_json,
                           seed_of)

eid, model = sys.argv[1], sys.argv[2]
b = load_bundle(eid)
idx_of = station_index(b)
cl = json.load(open("E:/work/workBuddy/BlueHeartV2/output/p2/station_clusters.json"))

z = np.load(os.path.join(G9, f"full_{eid}_{model}.npz"))
grid, ll = z["grid"], z["ll"]
anchor = grid[0]
rng = np.random.default_rng(seed_of(f"full|{eid}|{model}|post"))
theta_post = sample_theta_post(grid, ll, rng)
s2_star, tau_star = profile_pick(b, model, theta_post, np.arange(b["n_st"]), anchor)
rngp = np.random.default_rng(seed_of(f"full|{eid}|{model}|postres"))
X_post = predictive_samples(b, model, theta_post, s2_star, tau_star, rngp)

out = {"eid": eid, "model": model, "clusters": {}}
for c, names in cl["clusters"].items():
    sh = np.array([idx_of[n] for n in names])
    out["clusters"][c] = pred_metrics(X_post, b, sh, f"fullsub|{eid}|{c}|{model}")
out["all"] = pred_metrics(X_post, b, np.arange(b["n_st"]), f"fullsub|{eid}|all|{model}")
save_json(os.path.join(G9, f"fullsub_{eid}_{model}.json"), out)
print(f"FULLSUB {eid}/{model} done", flush=True)
