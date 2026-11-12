# -*- coding: utf-8 -*-
"""I_e 重算 (prereg v1.8 A27-7 / I5 义务): A10 公式重新实现
步骤 1 (义务): 在既有运行 (v1 强迫) 上复现 A10 登记数值 → 验证实现
步骤 2: 修正强迫运行 (runs_corr, 7 事件) + 零偏差事件复用既有运行 → ie_recompute.csv
用法: python p2f0_ie_v3.py [--corr]
"""
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib

SIGMA_EFF = 0.050          # A10 噪声底
ACTIVE_MM = 0.001          # 活跃站阈值: 峰值先验离散度 > 1 mm
REGISTERED = {"ev00": 10.66, "ev08": 7.68, "ev10": 3.18, "ev07": 1.44,
              "ev09": 0.49, "ev01": 0.31, "ev02": 0.45, "ev03": 0.35,
              "ev04": 0.42, "ev05": 0.44, "ev06": 0.33}  # ev01-06 区间内登记值, 仅 ev00/08/10/07/09 为精确登记
EXACT = ["ev00", "ev07", "ev08", "ev09", "ev10"]
CORR_EVENTS = ["ev00", "ev01", "ev04", "ev05", "ev06", "ev07", "ev08"]


def ie_from_runs(Y):
    """Y: (n_runs, T, n_st) → (I_e, std_star 数组, n_active)"""
    std_t = Y.std(axis=0)            # (T, n_st)
    std_star = std_t.max(axis=0)     # (n_st,)
    act = std_star > ACTIVE_MM
    if act.sum() == 0:
        return 0.0, std_star, 0
    return float(np.percentile(std_star[act], 90) / SIGMA_EFF), std_star, int(act.sum())


def load_event(eid):
    if eid == "ev00":
        from p2_emulator_gate import load_runs
        return load_runs()
    from p2_xev_posterior import load_event_runs
    return load_event_runs(eid)


def main(do_corr=False):
    rows = []
    for eid in sorted(CORR_EVENTS + ["ev02", "ev03", "ev09", "ev10"]):
        # 数据源: 修正事件用 runs_corr (若 do_corr 且运行齐), 否则既有运行
        use_corr = do_corr and eid in CORR_EVENTS
        if use_corr:
            root = os.path.join(p2lib.P2, "runs_corr", eid)
            dirs = sorted(d for d in os.listdir(root) if d.startswith("run_"))
            run_dirs = [os.path.join(root, d) for d in dirs]
            ok = all(os.path.exists(os.path.join(d, "series.csv")) for d in run_dirs)
            if not ok or len(run_dirs) != 90:
                print(f"{eid}: runs_corr 不齐 ({len(run_dirs)}), 跳过修正重算")
                use_corr = False
        if not use_corr:
            X, Y, tidx, cols = load_event(eid)
            src = "v1runs"
        else:
            # 与 load_event 同法取对齐 (ev00 走锚点 loader), 再读 runs_corr
            if eid == "ev00":
                from p2_emulator_gate import load_runs
                _, _, tidx, cols = load_runs()
            else:
                from p2_xev_posterior import load_event_runs
                _, _, tidx, cols = load_event_runs(eid)
            series = []
            for d in run_dirs:
                s = pd.read_csv(os.path.join(d, "series.csv"), index_col=0, parse_dates=True)
                try:
                    s.index = pd.to_datetime(s.index, utc=True)
                except (TypeError, ValueError):
                    pass
                series.append(s)
            Y = np.stack([s.reindex(index=tidx)[cols].to_numpy(dtype=np.float64)
                          for s in series])
            if np.isnan(Y).any():
                raise ValueError(f"{eid}: runs_corr 对齐失败")
            src = "corr"
        ie, std_star, n_act = ie_from_runs(Y)
        reg = REGISTERED.get(eid, "")
        rows.append({"event_id": eid, "source": src, "I_e": round(ie, 3),
                     "registered": reg, "n_active": n_act,
                     "std_star_max": round(float(std_star.max()), 4)})
        print(f"{eid}: source={src} I_e={ie:.3f} (登记 {reg}) active={n_act}")
    df = pd.DataFrame(rows)
    # 复现校验 (步骤 1): 5 个精确登记事件
    if not do_corr:
        chk = df[df["event_id"].isin(EXACT)].copy()
        chk["abs_err"] = (chk["I_e"] - chk["registered"]).abs()
        worst = chk["abs_err"].max()
        print(f"\n复现校验 (5 精确登记事件): 最大绝对误差 = {worst:.3f}")
        assert worst < 0.25, f"I_e 复现失败 (max err {worst:.3f} >= 0.25), 不得用于修正运行"
        print("复现通过 ✓")
    out = os.path.join(p2lib.P2, "ie_recompute_corr.csv" if do_corr else "ie_reproduce_check.csv")
    df.to_csv(out, index=False)
    print("written:", out)


if __name__ == "__main__":
    main(do_corr="--corr" in sys.argv)
