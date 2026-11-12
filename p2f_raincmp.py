# -*- coding: utf-8 -*-
"""关口 7 降雨对照分析 (prereg v1.8 A27): 修正臂 + 锚点扰动臂
用法:
  python p2f_raincmp.py --corr   # 修正臂: 7 事件 E0/E1/E2 + 锚点 dropout (A27-3)
  python p2f_raincmp.py --pert   # 扰动臂: 锚点 6 实现 E0/E1/E2 全后验 (A27-5 全量级)
前置: 第一波 1482 运行全部完成 (jobs_rain_wave1.json)
输出: xev/xev_raw_corr.json, xev/wmap_{eid}_corr.npy, xev/pert_raw.json,
      matrixA_stats_corr.json, ie_recompute_corr.csv (经 p2f0_ie_v3 --corr)
判级: A25 冻结阈值; s2* (E1 evidence 最大 profile, 供 wave-2 线性化) 一并提取
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_xev_posterior import load_event_runs
from p2b_realdata_v3 import (fit_gp, phi_from_residuals, build_wmap_event,
                             invert_real, ident_flags)

OUT = os.path.join(p2lib.P2, "xev")
CORR_ROOT = os.path.join(p2lib.P2, "runs_corr")
CORR_VAL = os.path.join(p2lib.P2, "xev", "val_runs_corr")
PERT_ROOT = os.path.join(p2lib.P2, "runs_pert")
PERT_VAL = os.path.join(p2lib.P2, "xev", "val_runs_pert")
CORR_EVENTS = ["ev00", "ev01", "ev04", "ev05", "ev06", "ev07", "ev08"]
EPS = [("rn1", -0.0225), ("rn2", 0.0225), ("rn3", -0.0450),
       ("ru1", -0.077), ("ru2", 0.077), ("ru3", 0.0385)]
TAU_GRID = [0.02, 0.05, 0.10, 0.20, 0.40]


def theta90():
    jobs = json.load(open(os.path.join(p2lib.P2, "jobs_lhs.json")))
    return np.array([j["rough"] for j in jobs], dtype=float)


def load_runs_from(root, tidx, cols, n_expect=90):
    dirs = sorted(d for d in os.listdir(root) if d.startswith("run_"))
    assert len(dirs) == n_expect, f"{root}: {len(dirs)} runs != {n_expect}"
    series = []
    for d in dirs:
        s = pd.read_csv(os.path.join(root, d, "series.csv"), index_col=0, parse_dates=True)
        try:
            s.index = pd.to_datetime(s.index, utc=True)
        except (TypeError, ValueError):
            pass
        series.append(s)
    Y = np.stack([s.reindex(index=tidx)[cols].to_numpy(dtype=np.float64) for s in series])
    if np.isnan(Y).any():
        raise ValueError(f"{root}: 对齐失败 (NaN)")
    return Y


def alignment(eid):
    """tidx/cols 数据层来源 (与 v2 管线一致)"""
    if eid == "ev00":
        from p2_emulator_gate import load_runs
        _, _, tidx, cols = load_runs()
    else:
        _, _, tidx, cols = load_event_runs(eid)
    return tidx, cols


def s2_star_e1(O, M, W, phi, Yhat, T, n_st):
    """E1 evidence 最大 profile 的 s2* (在给定 θ 残差处; 与 DiagLikV3.e1_ll 同法).
    O (n_st, T); Yhat (T, n_st)"""
    from p2_likelihoods_v3 import DiagLikV3
    lik = DiagLikV3(W, M, T, n_st, TAU_GRID)
    lik.set_phi(phi)
    N_eff = int(M.sum())
    e0 = (O - Yhat.T).astype(np.float64)          # (n_st, T)
    s2_hi = max((e0[M] ** 2).sum() / N_eff, 1e-6)
    grid = np.geomspace(s2_hi * 1e-4, s2_hi, 8)
    e_t = (O.T[None] - Yhat[None]).astype(np.float64)   # (1, T, n_st)
    e_t = np.where(M.T[None], e_t, 0.0)
    lls = [float(lik.e1_ll(e_t, np.array([s]))[0]) for s in grid]
    k = int(np.argmax(lls))
    return float(grid[k]), float(lls[k]), s2_hi


def run_corr():
    X = theta90()
    raw, rows = {}, []
    for eid in CORR_EVENTS:
        print(f"=== [corr] {eid} ===", flush=True)
        tidx, cols = alignment(eid)
        Y = load_runs_from(os.path.join(CORR_ROOT, eid), tidx, cols)
        n_runs, T, n_st = Y.shape
        gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
        xlim = (X.min(0), X.max(0))
        O, M = p2lib.load_obs_matrix(tidx, cols)
        M, qc_excl = p2lib.qc_stations(O, M, Y.mean(0).T)
        phi = phi_from_residuals(O, M, Y.mean(0).T)
        W, per_pt = build_wmap_event(CORR_VAL, eid, gps, Vt, Zmean, Ys, Ym,
                                     n_pc, T, n_st, tidx, cols, xlim)
        wm = W[M.T]
        np.save(os.path.join(OUT, f"wmap_{eid}_corr.npy"), W)
        print(f"  W@有效格 RMS={np.sqrt(wm.mean()):.4f} (per-pt {min(per_pt):.3f}-{max(per_pt):.3f})", flush=True)
        res = {m: invert_real(m, O, np.arange(n_st), M, W, phi, gps, Vt,
                              Zmean, Ys, Ym, n_pc, T, n_st, xlim)
               for m in ("E0", "E1", "E2")}
        # s2* 在 E1 后验均值 θ̂ 处 (供 wave-2 线性化; A27-5)
        Yhat = None
        from p2_posterior_matrixA import recon_series
        Yhat = recon_series(np.array([res["E1"]["mean"]]), gps, Vt, Zmean, Ys, Ym,
                            n_pc, T, n_st, xlim=xlim)[0].astype(np.float64)
        s2s, lle, s2hi = s2_star_e1(O, M, W, phi, Yhat, T, n_st)
        print(f"  E1 s2*={s2s:.2e} (s2_hi={s2hi:.2e})", flush=True)
        raw[eid] = {"n_runs": n_runs, "T": int(T), "n_st": int(n_st),
                    "n_obs": int(M.sum()), "n_valid_st": int(M.sum(1).clip(0, 1).sum()),
                    "qc_excluded": len(qc_excl), "phi_med": float(np.median(phi)),
                    "w_rms_valid": float(np.sqrt(wm.mean())),
                    "s2_star_E1": s2s, "s2_hi": s2hi,
                    "R": {m: res[m]["R_j"] for m in res},
                    "post_mean": {m: res[m]["mean"] for m in res},
                    "post_sd": {m: res[m]["sd"] for m in res},
                    "info": {m: res[m]["info"] for m in res},
                    "ident": {m: ident_flags(res[m]) for m in res}}
        if eid == "ev00":
            drop = {}
            for nsub in (15, 5):
                si = np.arange(min(nsub, n_st))
                for m in ("E0", "E1"):
                    drop[f"{m}_{nsub}"] = invert_real(m, O, si, M, W, phi, gps, Vt,
                                                      Zmean, Ys, Ym, n_pc, T, n_st, xlim)
            raw[eid]["dropout"] = {k: {"R_j": v["R_j"], "mean": v["mean"],
                                       "sd": v["sd"], "info": v["info"],
                                       "ident": ident_flags(v)}
                                   for k, v in drop.items()}
        for m in ("E0", "E1", "E2"):
            rows.append({"event_id": eid, "model": m,
                         "R_n1": res[m]["R_j"][0], "R_n2": res[m]["R_j"][1], "R_n3": res[m]["R_j"][2],
                         "mean_n1": res[m]["mean"][0], "mean_n2": res[m]["mean"][1], "mean_n3": res[m]["mean"][2],
                         "sd_n1": res[m]["sd"][0], "sd_n2": res[m]["sd"][1], "sd_n3": res[m]["sd"][2],
                         "ident_all": all(ident_flags(res[m]))})
        print({m: [round(x, 3) for x in res[m]["R_j"]] for m in ("E0", "E1", "E2")}, flush=True)
    json.dump(raw, open(os.path.join(OUT, "xev_raw_corr.json"), "w"), indent=1, default=float)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "xev_results_corr.csv"), index=False)
    # 锚点 stats (对照 matrixA_stats_v2.json)
    ma = raw["ev00"]
    stats = {"W_source": "runs_corr/ev00 24 val runs (corr forcing, A27)",
             "likelihood": "v3 diag(s2+W_n) + exact AR(1) + exact SM (A21/A26/A27)",
             "phi_med": ma["phi_med"], "n_valid_stations": ma["n_valid_st"],
             "qc_excluded": ma["qc_excluded"], "s2_star_E1": ma["s2_star_E1"],
             "R_j": ma["R"], "info": ma["info"], "post_sd": ma["post_sd"],
             "post_mean": ma["post_mean"], "ident": ma["ident"],
             "dropout": ma.get("dropout", {})}
    json.dump(stats, open(os.path.join(p2lib.P2, "matrixA_stats_corr.json"), "w"), indent=1, default=float)
    print("corr arm done:", len(raw), "events", flush=True)


def run_pert():
    X = theta90()
    tidx, cols = alignment("ev00")
    raw = {}
    for tag, eps in EPS:
        print(f"=== [pert] {tag} eps={eps:+.4f} ===", flush=True)
        Y = load_runs_from(os.path.join(PERT_ROOT, "ev00", tag), tidx, cols)
        n_runs, T, n_st = Y.shape
        gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
        xlim = (X.min(0), X.max(0))
        O, M = p2lib.load_obs_matrix(tidx, cols)
        M, qc_excl = p2lib.qc_stations(O, M, Y.mean(0).T)
        phi = phi_from_residuals(O, M, Y.mean(0).T)
        # build_wmap_event 拼接 root/eid/{vtag}; 扰动目录为 PERT_VAL/ev00/{tag}/{vtag}
        W, per_pt = build_wmap_event(PERT_VAL, os.path.join("ev00", tag),
                                     gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st, tidx, cols, xlim)
        wm = W[M.T]
        res = {m: invert_real(m, O, np.arange(n_st), M, W, phi, gps, Vt,
                              Zmean, Ys, Ym, n_pc, T, n_st, xlim)
               for m in ("E0", "E1", "E2")}
        raw[tag] = {"eps": eps, "w_rms_valid": float(np.sqrt(wm.mean())),
                    "phi_med": float(np.median(phi)), "n_obs": int(M.sum()),
                    "R": {m: res[m]["R_j"] for m in res},
                    "post_mean": {m: res[m]["mean"] for m in res},
                    "post_sd": {m: res[m]["sd"] for m in res},
                    "info": {m: res[m]["info"] for m in res},
                    "ident": {m: ident_flags(res[m]) for m in res}}
        print({m: [round(x, 3) for x in res[m]["R_j"]] for m in ("E0", "E1", "E2")}, flush=True)
    json.dump(raw, open(os.path.join(OUT, "pert_raw.json"), "w"), indent=1, default=float)
    print("pert arm done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", action="store_true")
    ap.add_argument("--pert", action="store_true")
    a = ap.parse_args()
    if a.corr:
        run_corr()
    if a.pert:
        run_pert()
    if not (a.corr or a.pert):
        print("usage: --corr / --pert")
