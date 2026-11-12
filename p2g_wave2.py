# -*- coding: utf-8 -*-
"""关口 7 wave-2 (prereg v1.8 A27-5/6): θ̂ 运行 + 线性化 Δn + 混淆表 + 总报告
用法:
  python p2g_wave2.py --make-jobs   # 生成 77 个 θ̂ 运行作业 (11 事件 × [1 corr + 6 pert])
  python p2g_wave2.py --report      # wave-2 运行完成后: 线性化 Δn, 混淆表, raincmp_report_v2.md
前置: p2f_raincmp.py --corr / --pert 已完成; wave-2 运行全部完成 (--report 时)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib
from p2_xev_posterior import load_event_runs
from p2b_realdata_v3 import fit_gp, phi_from_residuals
from p2f_raincmp import (theta90, alignment, load_runs_from, CORR_EVENTS, EPS,
                         CORR_ROOT, OUT, s2_star_e1)

ALL_EVENTS = ["ev00"] + [f"ev{i:02d}" for i in range(1, 11)]
W2_ROOT = os.path.join(p2lib.P2, "runs_w2_theta")
PRIOR_LO, PRIOR_HI = 0.009, 0.022
H_NORM = 1e-4


def event_windows():
    jobs = json.load(open(os.path.join(p2lib.P2, "jobs_xev.json")))
    win = {}
    for j in jobs:
        win.setdefault(j["event_id"], (j["start"], j["end"]))
    win["ev00"] = ("10/22/2025 15:15", "10/23/2025 17:00")
    return win


def make_jobs():
    raw = json.load(open(os.path.join(OUT, "xev_raw_corr.json")))
    raw_v2 = json.load(open(os.path.join(OUT, "xev_raw_v2.json")))
    # A27-2: 零偏差事件修正臂 ≡ 原始臂 → θ̂ 从 v2 复用
    raw_all = {**raw_v2, **raw}
    win = event_windows()
    jobs = []
    for eid in ALL_EVENTS:
        st, en = win[eid]
        th = raw_all[eid]["post_mean"]["E1"]
        jobs.append({"run_dir": f"{W2_ROOT}/{eid}/corr", "rough": [round(x, 6) for x in th],
                     "dwf_k": 1.0, "start": st, "end": en,
                     "event_id": eid, "rain_mode": "corr"})
        for tag, eps in EPS:
            jobs.append({"run_dir": f"{W2_ROOT}/{eid}/{tag}", "rough": [round(x, 6) for x in th],
                         "dwf_k": 1.0, "start": st, "end": en,
                         "event_id": eid, "rain_mode": "pert", "eps": eps})
    json.dump(jobs, open(os.path.join(p2lib.P2, "jobs_rain_wave2.json"), "w"), indent=1)
    print("wave2 jobs:", len(jobs))


def _linear_dn(eid, raw_all, gp_cache):
    """A27-5 冻结公式: Δn = (JᵀC⁻¹J + P⁻¹)⁻¹ JᵀC⁻¹ δy (raw-space diag(s2*+W))"""
    X = theta90()
    tidx, cols = alignment(eid)
    if eid in gp_cache:
        gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st, xlim, MT, O0, M0, Ymean = gp_cache[eid]
    else:
        # A27-2: 零偏差事件无 runs_corr → 用原始臂运行 (强迫逐位相同, GP 等价)
        croot = os.path.join(CORR_ROOT, eid)
        if not os.path.isdir(croot):
            croot = os.path.join(p2lib.P2, "runs_xev", eid)
        Y = load_runs_from(croot, tidx, cols)
        n_runs, T, n_st = Y.shape
        gps, Vt, Ym, Ys, n_pc, Zmean = fit_gp(X, Y)
        xlim = (X.min(0), X.max(0))
        O0, M0 = p2lib.load_obs_matrix(tidx, cols)
        M, _ = p2lib.qc_stations(O0, M0, Y.mean(0).T)
        MT = M.T.copy()                   # (T, n_st)
        Ymean = Y.mean(0).T               # (n_st, T) 集成均值 (phi/s2* 现算用)
        gp_cache[eid] = (gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st, xlim, MT,
                         O0, M0, Ymean)
    from p2_posterior_matrixA import recon_series
    # A27-2: 零偏差事件 W ≡ 原始臂 W (强迫逐位相同)
    wpath = os.path.join(OUT, f"wmap_{eid}_corr.npy")
    if not os.path.exists(wpath):
        wpath = os.path.join(OUT, f"wmap_{eid}_v2.npy")
    W = np.load(wpath)
    th_hat = np.array(raw_all[eid]["post_mean"]["E1"])
    if "s2_star_E1" in raw_all[eid]:
        s2s = raw_all[eid]["s2_star_E1"]
    else:
        # A27-2: 零偏差事件 s2* 现算 (与 p2f.s2_star_e1 同法, θ̂ 处 GP 重构; 修正臂≡原始臂)
        phi2 = phi_from_residuals(O0, M0, Ymean)
        Yhat = recon_series(np.array([th_hat]), gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st,
                            xlim=xlim)[0].astype(np.float64)
        s2s, _, _ = s2_star_e1(O0, M0, W, phi2, Yhat, T, n_st)

    # θ̂ 的模拟器响应 (wave-2 corr 运行)
    def sim_at(sub):
        s = pd.read_csv(os.path.join(W2_ROOT, eid, sub, "series.csv"),
                        index_col=0, parse_dates=True)
        try:
            s.index = pd.to_datetime(s.index, utc=True)
        except (TypeError, ValueError):
            pass
        return s.reindex(index=tidx)[cols].to_numpy(dtype=np.float64)   # (T, n_st)
    Yc = sim_at("corr")

    # J: 有限差分 (归一化空间 h=1e-4 → 物理空间)
    span = PRIOR_HI - PRIOR_LO
    thn = (th_hat - PRIOR_LO) / span
    J_cols = []
    for d in range(3):
        tp = th_hat.copy(); tm = th_hat.copy()
        tp[d] = PRIOR_LO + min(thn[d] + H_NORM, 1.0) * span
        tm[d] = PRIOR_LO + max(thn[d] - H_NORM, 0.0) * span
        Yp = recon_series(np.array([tp]), gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st, xlim=xlim)[0]
        Ym_ = recon_series(np.array([tm]), gps, Vt, Zmean, Ys, Ym, n_pc, T, n_st, xlim=xlim)[0]
        J_cols.append((Yp - Ym_) / (tp[d] - tm[d]))
    C_vec = (s2s + W)[MT]                 # (N,)
    J_mat = np.stack([Jc[MT] for Jc in J_cols], axis=1)   # (N, 3)
    Pinv = np.diag([1.0 / ((span ** 2) / 12.0)] * 3)

    out = {}
    for tag, eps in EPS:
        d = (sim_at(tag) - Yc)[MT]        # (N,)
        Cinv_J = J_mat / C_vec[:, None]
        A = J_mat.T @ Cinv_J + Pinv
        b = Cinv_J.T @ d
        dn = np.linalg.solve(A, b)
        out[tag] = {"eps": eps, "dn": [float(x) for x in dn]}
    return out


def report():
    raw_corr = json.load(open(os.path.join(OUT, "xev_raw_corr.json")))
    raw_v2 = json.load(open(os.path.join(OUT, "xev_raw_v2.json")))
    ma_corr = json.load(open(os.path.join(p2lib.P2, "matrixA_stats_corr.json")))
    ma_v2 = json.load(open(os.path.join(p2lib.P2, "matrixA_stats_v2.json")))
    pert_raw = json.load(open(os.path.join(OUT, "pert_raw.json")))
    ie_corr = pd.read_csv(os.path.join(p2lib.P2, "ie_recompute_corr.csv"))
    rep = ["# 关口 7 降雨对照报告 (prereg v1.8 A27; 判级 A25; caveat 随行)\n\n",
           "- 三臂: (a) 原始强迫 = 既有运行 (v2 产物); (b) 修正强迫 = runs_corr (A27-2); "
           "(c) 扰动臂 = 修正 × (1+ε) (A27-4 分层平衡 6 实现)\n",
           "- 零偏差事件 ev02/ev03/ev09/ev10: 修正臂 ≡ 原始臂 (逐位核验, I15)\n\n"]

    # ---- T1 强迫偏差 (I4 复核, 数据层) ----
    rep.append("## T1. 强迫口径偏差 (旧管线 − 修正口径, 窗口总雨深, I4 数据层复核)\n\n")
    rep.append("| 事件 | 旧 (mm) | 修正 (mm) | 偏差 | 扰动臂覆盖 |\n|---|---|---|---|---|\n")
    delta = {"ev00": 7.69, "ev01": 7.69, "ev04": 7.69, "ev05": 16.67, "ev06": 0.57,
             "ev07": 8.18, "ev08": 7.69, "ev02": 0.0, "ev03": 0.0, "ev09": 0.0, "ev10": 0.0}
    old_mm = {"ev00": 38.40, "ev01": 10.22, "ev02": 8.13, "ev03": 7.00, "ev04": 10.26,
              "ev05": 8.15, "ev06": 8.12, "ev07": 19.78, "ev08": 22.95, "ev09": 22.44, "ev10": 26.27}
    for eid in ALL_EVENTS:
        rep.append(f"| {eid} | {old_mm[eid]:.2f} | {old_mm[eid]/(1+delta[eid]/100):.2f} | "
                   f"{delta[eid]:+.2f}% | {'是' if eid in CORR_EVENTS else '≡原始臂'} |\n")
    rep.append("\n")

    # ---- T2 修正 vs 原始: 后验对照 ----
    rep.append("## T2. 修正臂 vs 原始臂后验对照 (Δmean 以原始臂 E1 σ 为单位; R_j 同一阈值)\n\n")
    rep.append("| 事件 | 模型 | R_j 原始 | R_j 修正 | ident 翻转 | Δn1/σ | Δn2/σ | Δn3/σ |\n|---|---|---|---|---|---|---|---|\n")
    flips = []
    for eid in ALL_EVENTS:
        for m in ("E0", "E1", "E2"):
            if eid == "ev00":
                r0, r1 = ma_v2["R_j"][m], ma_corr["R_j"][m]
                i0, i1 = ma_v2["ident"][m], ma_corr["ident"][m]
                m0, m1 = ma_v2["post_mean"][m], ma_corr["post_mean"][m]
                s1 = ma_v2["post_sd"][m]
            elif eid in CORR_EVENTS:
                r0, r1 = raw_v2[eid]["R"][m], raw_corr[eid]["R"][m]
                i0, i1 = raw_v2[eid]["ident"][m], raw_corr[eid]["ident"][m]
                m0, m1 = raw_v2[eid]["post_mean"][m], raw_corr[eid]["post_mean"][m]
                s1 = raw_v2[eid]["post_sd"][m]
            else:
                continue  # 零偏差事件: 修正 ≡ 原始
            flip = "—" if i0 == i1 else f"{sum(i0)}→{sum(i1)}"
            if i0 != i1:
                flips.append((eid, m, i0, i1))
            dn = [(m1[d] - m0[d]) / s1[d] for d in range(3)]
            fmt = lambda v: f"{v:.2f}"
            rep.append(f"| {eid} | {m} | {fmt(r0[0])}/{fmt(r0[1])}/{fmt(r0[2])} | "
                       f"{fmt(r1[0])}/{fmt(r1[1])}/{fmt(r1[2])} | {flip} | "
                       f"{dn[0]:+.1f} | {dn[1]:+.1f} | {dn[2]:+.1f} |\n")
    rep.append("\n")
    rep.append(f"- 判级翻转数: {len(flips)}" +
               ("; " + "; ".join(f"{e}/{m}: {sum(a)}→{sum(b)}" for e, m, a, b in flips) if flips else "") + "\n")
    # 强迫方向 vs Δn 方向混淆 (修正 vs 原始; δforcing = corr − old < 0 全部)
    rep.append("- 修正步方向: 全部事件 corr − old < 0 (强迫减小); Δn 方向逐 (事件, 参数) 见上表 → 混淆并入 §T4 汇总\n\n")

    # ---- T3 锚点扰动臂 (全量级, 非线性) ----
    rep.append("## T3. 锚点扰动臂全后验 (A27-5 全量级; Δn 相对修正臂, 以修正臂 E1 σ 为单位)\n\n")
    rep.append("| 实现 | ε | 模型 | Δn1/σ | Δn2/σ | Δn3/σ | sign 一致数 (/3) | R_j 修正→扰动 |\n|---|---|---|---|---|---|---|---|\n")
    conf_rows = []
    for tag, eps in EPS:
        r = pert_raw[tag]
        for m in ("E0", "E1", "E2"):
            m0 = ma_corr["post_mean"][m]
            s0 = ma_corr["post_sd"][m]
            m1 = r["post_mean"][m]
            dn = [(m1[d] - m0[d]) / s0[d] for d in range(3)]
            agree = sum(1 for d in range(3) if np.sign(eps) == np.sign(dn[d]) and dn[d] != 0)
            r0 = ma_corr["R_j"][m]; r1 = r["R"][m]
            conf_rows.append((tag, eps, m, dn, agree))
            rep.append(f"| {tag} | {eps:+.4f} | {m} | {dn[0]:+.1f} | {dn[1]:+.1f} | {dn[2]:+.1f} | "
                       f"{agree}/3 | {r0[0]:.2f}/{r0[1]:.2f}/{r0[2]:.2f}→{r1[0]:.2f}/{r1[1]:.2f}/{r1[2]:.2f} |\n")
    rep.append("\n")

    # ---- T4 事件级线性化 Δn + 混淆表汇总 ----
    rep.append("## T4. 事件级扰动传播 (A27-5 线性化; 11 事件 × 6 实现 × E1)\n\n")
    # A27-2: 零偏差事件 E1 后验从 v2 复用 (θ̂/σ/W 均同)
    raw_all = {**raw_v2, **raw_corr}
    gp_cache = {}
    lin = {}
    for eid in ALL_EVENTS:
        try:
            lin[eid] = _linear_dn(eid, raw_all, gp_cache)
            print(f"linear {eid} done", flush=True)
        except Exception as ex:
            print(f"linear {eid} FAIL: {ex}", flush=True)
    rep.append("| 事件 | 实现 | ε | Δn1/σ | Δn2/σ | Δn3/σ |\n|---|---|---|---|---|---|\n")
    sigagree = {"rn": [0, 0], "ru": [0, 0]}
    dn_all = []
    for eid in ALL_EVENTS:
        if eid not in lin:
            continue
        for tag, eps in EPS:
            dn = lin[eid][tag]["dn"]
            dn_s = [dn[d] / raw_all[eid]["post_sd"]["E1"][d] for d in range(3)]
            dn_all.extend(dn_s)
            arm = "rn" if tag.startswith("rn") else "ru"
            for d in range(3):
                sigagree[arm][0] += int(np.sign(eps) == np.sign(dn_s[d]) and dn_s[d] != 0)
                sigagree[arm][1] += 1
            rep.append(f"| {eid} | {tag} | {eps:+.4f} | {dn_s[0]:+.1f} | {dn_s[1]:+.1f} | {dn_s[2]:+.1f} |\n")
    rep.append("\n")
    tot0 = sum(v[0] for v in sigagree.values()); tot1 = sum(v[1] for v in sigagree.values())
    rep.append(f"- **混淆表汇总**: sign(ε) 与 sign(Δn/σ) 一致率 = {tot0}/{tot1} = {tot0/max(tot1,1):.2f} "
               f"(主臂 rn: {sigagree['rn'][0]}/{sigagree['rn'][1]}, 稳健臂 ru: {sigagree['ru'][0]}/{sigagree['ru'][1]})\n")
    if dn_all:
        rep.append(f"- |Δn|/σ 分布: 中位 {np.median(np.abs(dn_all)):.2f}, p90 {np.percentile(np.abs(dn_all), 90):.2f}, "
                   f"max {np.max(np.abs(dn_all)):.2f}\n")
    rep.append("- 锚点全量级 (T3) 为线性化的非线性校验基准 (A27-5)\n\n")

    # ---- T5 I_e 重算 ----
    rep.append("## T5. I_e 重算 (修正强迫; A27-7/I5 义务)\n\n")
    REG = {"ev00": 10.66, "ev08": 7.68, "ev10": 3.18, "ev07": 1.44, "ev09": 0.49}
    rep.append("| 事件 | I_e 登记 | I_e 修正 | 来源 |\n|---|---|---|---|\n")
    for _, r in ie_corr.iterrows():
        reg = REG.get(r["event_id"], "0.31–0.45 区间")
        rep.append(f"| {r['event_id']} | {reg} | {r['I_e']:.3f} | {r['source']} |\n")
    order_reg = ["ev00", "ev08", "ev10", "ev07", "ev09"]
    order_corr = ie_corr.sort_values("I_e", ascending=False)["event_id"].tolist()
    top5_stable = set(order_reg) == set(order_corr[:5])
    rep.append(f"\n- 排序稳定性: 登记前 5 = {order_reg}; 修正前 5 = {order_corr[:5]} → {'保持' if top5_stable else '变化'}\n")
    rep.append("- 措辞纪律: I_e 为描述性协变量 (A10/A14), 机制线索陈述非独立确认\n\n")

    # ---- caveats ----
    rep.append("## Caveats\n\n")
    rep.append("- 判级阈值校准域 = 锚点孪生 regime (小事件外推未校准, A25/I14);\n")
    rep.append("- E1/E2 n1/n2 反保守 (A25);\n")
    rep.append("- 线性化传播以锚点全量级校验 (T3 vs T4 ev00 行);\n")
    rep.append("- 扰动为事件内常数乘子 (保持时间结构, A9), 不代表空间非均匀误差;\n")
    rep.append("- 扰动实现的 W map 由该实现的 24 点验证运行构建 (I15)。\n")

    with open(os.path.join(p2lib.P2, "raincmp_report_v2.md"), "w", encoding="utf-8") as f:
        f.writelines(rep)
    json.dump({eid: {tag: lin[eid][tag] for tag in lin[eid]} for eid in lin},
              open(os.path.join(OUT, "linear_dn.json"), "w"), indent=1, default=float)
    print("raincmp_report_v2.md written")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-jobs", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.make_jobs:
        make_jobs()
    if a.report:
        report()
