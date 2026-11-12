# -*- coding: utf-8 -*-
"""关口 9 跨事件转移矩阵 worker: p2i_transfer.py <source_eid>
源事件全站后验 (full_*.npz) → 其余 10 事件预测 (A29-5 辅表)"""
import os
import sys
import numpy as np

sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
from p2i_gate9_lib import (G9, EVENTS, S_SAMPLES, load_bundle, sample_theta_post,
                           profile_pick, predictive_samples, pred_metrics, save_json,
                           seed_of)

src = sys.argv[1]
b_src = load_bundle(src)
out = {}
for model in ("E0", "E1", "E2"):
    z = np.load(os.path.join(G9, f"full_{src}_{model}.npz"))
    grid, ll = z["grid"], z["ll"]
    anchor = grid[0]
    # 源事件自身 profile s2*/tau* (源数据已校准, 目标事件未触碰 — 防泄漏)
    rng = np.random.default_rng(seed_of(f"transfer|{src}|{model}"))
    theta_s = sample_theta_post(grid, ll, rng)
    s2_star, tau_star = profile_pick(b_src, model, theta_s, np.arange(b_src["n_st"]), anchor)
    res = {}
    for tgt in EVENTS:
        if tgt == src:
            continue
        b_t = load_bundle(tgt)
        rngp = np.random.default_rng(seed_of(f"transfer|{src}|{tgt}|{model}|res"))
        X = predictive_samples(b_t, model, theta_s, s2_star, tau_star, rngp)
        m = pred_metrics(X, b_t, np.arange(b_t["n_st"]), f"transfer|{src}|{tgt}|{model}")
        # 改善率对照 = 目标事件的先验预测 CRPS (full_{tgt}_{model}.json 的 prior)
        try:
            prior_crps = pred_crps = None
            import json
            fj = json.load(open(os.path.join(G9, f"full_{tgt}_{model}.json")))
            prior_crps = fj["prior"]["crps_mean"]
            m["crps_improve_vs_prior"] = (prior_crps - m["crps_mean"]) / prior_crps
        except Exception:
            pass
        res[tgt] = m
    out[model] = res
save_json(os.path.join(G9, f"transfer_{src}.json"), out)
print(f"TRANSFER {src} done", flush=True)
