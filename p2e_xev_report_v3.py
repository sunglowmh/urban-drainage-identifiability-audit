# -*- coding: utf-8 -*-
"""跨事件 v2 汇总报告: 翻转矩阵 + v1/v2 对照 + I_e 复核 (prereg v1.7 A26-8)
输入: xev/xev_raw_v2.json, xev/xev_results_v2.csv, matrixA_stats_v2.json
对照: *_v1_INVALID 存档 (I10 无效化, 仅作管线对照, 不作证据)
输出: xev/xev_report_v2.md
措辞纪律: I_e 对照只作机制线索陈述 (A14); 判级 caveat A25 随行
"""
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib

OUT = os.path.join(p2lib.P2, "xev")

# A10 登记 I_e (数据层, I10 无效化声明不含 I_e)
IE = {"ev00": 10.66, "ev08": 7.68, "ev10": 3.18, "ev07": 1.44, "ev09": 0.49}


def main():
    raw = json.load(open(os.path.join(OUT, "xev_raw_v2.json")))
    ma = json.load(open(os.path.join(p2lib.P2, "matrixA_stats_v2.json")))
    v1 = pd.read_csv(os.path.join(OUT, "xev_results_v1_INVALID.csv"))

    rep = ["# 跨事件 v2 汇总报告 (似然 v3, prereg v1.7 A26)\n\n",
           "- 判级: A25 冻结阈值 (R_j≥0.5 主 + info≥0.1 辅助); E1/E2 n1/n2 反保守 caveat 随行\n",
           "- W_n: 逐事件 24 冻结离设计点 (事件强迫); 锚点复用 A21 底图\n\n"]

    # ---- 翻转矩阵 ----
    rep.append("## 1. 可辨识性翻转矩阵 (v2)\n\n")
    rep.append("| 事件 | 雨深(mm) | E0 全可辨识 | E1 全可辨识 | E2 全可辨识 | E0 全非 | E1 全非 | E2 全非 | I_e(登记) |\n")
    rep.append("|---|---|---|---|---|---|---|---|---|\n")
    order = sorted(raw.keys(), key=lambda e: (IE.get(e, 0.45)), reverse=True)
    order = [e for e in ["ev00"] if e in ma] + order
    for eid in order:
        if eid == "ev00":
            ident = ma["ident"]; depth = "38.4"
        else:
            ident = raw[eid]["ident"]; depth = str(raw[eid].get("depth_mm", ""))
            # depth from results csv
        rows_csv = pd.read_csv(os.path.join(OUT, "xev_results_v2.csv"))
        sub = rows_csv[rows_csv["event_id"] == eid]
        if len(sub):
            depth = str(sub["depth_mm"].iloc[0])
        ie = IE.get(eid, "0.31–0.45" if eid != "ev00" else IE["ev00"])
        rep.append(f"| {eid} | {depth} | {'✅' if all(ident['E0']) else '—'} | "
                   f"{'✅' if all(ident['E1']) else '—'} | {'✅' if all(ident['E2']) else '—'} | "
                   f"{'✅' if not any(ident['E0']) else '—'} | {'✅' if not any(ident['E1']) else '—'} | "
                   f"{'✅' if not any(ident['E2']) else '—'} | {ie} |\n")
    rep.append("\n")

    # ---- R_j 跨事件 ----
    rep.append("## 2. R_j 跨事件分布 (v2, mean±sd / min–max)\n\n")
    rep.append("| 模型 | 参数 | mean | sd | min | max | 可辨识事件数 |\n|---|---|---|---|---|---|---|\n")
    for m in ("E0", "E1", "E2"):
        vals = {d: [] for d in range(3)}
        n_ident = 0; n_ev = 0
        if "ev00" in ma or True:
            for d in range(3):
                vals[d].append(ma["R_j"][m][d])
            if all(ma["ident"][m]): n_ident += 1
            n_ev += 1
        for eid in raw:
            for d in range(3):
                vals[d].append(raw[eid]["R"][m][d])
            if all(raw[eid]["ident"][m]): n_ident += 1
            n_ev += 1
        for d in range(3):
            a = np.array(vals[d])
            rep.append(f"| {m} | n{d+1} | {a.mean():.3f} | {a.std():.3f} | {a.min():.3f} | {a.max():.3f} | {n_ident}/{n_ev} |\n")
    rep.append("\n")

    # ---- v1/v2 对照 ----
    rep.append("## 3. v1 → v2 对照 (v1 已因 I10 单位错配整体无效; 对照仅示管线修正的数值影响)\n\n")
    rep.append("| 事件 | 模型 | R_j v1 (n1/n2/n3) | R_j v2 (n1/n2/n3) |\n|---|---|---|---|\n")
    for eid in list(raw.keys()) + ["ev00"]:
        if eid == "ev00":
            v2r = [f"{x:.2f}" for x in ma["R_j"]["E1"]]
            rep.append(f"| {eid} | E1 | (见 matrixA_report_v1_INVALID) | {'/'.join(v2r)} |\n")
            continue
        for m in ("E0", "E1", "E2"):
            sub = v1[(v1["event_id"] == eid) & (v1["model"] == m)]
            v1r = f"{sub['R_n1'].iloc[0]:.2f}/{sub['R_n2'].iloc[0]:.2f}/{sub['R_n3'].iloc[0]:.2f}" if len(sub) else "—"
            v2r = [raw[eid]["R"][m][d] for d in range(3)]
            rep.append(f"| {eid} | {m} | {v1r} | {v2r[0]:.2f}/{v2r[1]:.2f}/{v2r[2]:.2f} |\n")
    rep.append("\n")

    # ---- I_e 复核 (机制线索措辞, A14) ----
    rep.append("## 4. I_e 复核 (A26-8; 机制线索非独立确认, A14 措辞纪律)\n\n")
    rows_csv = pd.read_csv(os.path.join(OUT, "xev_results_v2.csv"))
    e1_rows = rows_csv[rows_csv["model"] == "E1"].set_index("event_id")
    n_ident_e1 = {eid: int(all(raw[eid]["ident"]["E1"])) for eid in raw}
    ma_e1 = bool(all(ma["ident"]["E1"]))
    rep.append(f"- E1 全可辨识事件 (v2): ev00={'是' if ma_e1 else '否'}; " +
               ", ".join(f"{e}={'是' if v else '否'}" for e, v in n_ident_e1.items()) + "\n")
    rep.append("- v1 时代的 I_e 排序一致性声明须以 v2 独立重建: 上表为 v2 判级与登记 I_e 的直接对照; "
               "任何一致性只作机制线索 (同一批运行的共同来源限制仍在, A14)\n\n")

    # ---- dropout 模式 ----
    rep.append("## 5. E1 dropout 阶梯 (v2; 15/5 站)\n\n")
    rep.append("| 事件 | 30 站 R_j | 15 站 R_j | 5 站 R_j | n2 先失守? |\n|---|---|---|---|---|\n")
    for eid in raw:
        r30 = raw[eid]["R"]["E1"]; r15 = raw[eid]["drop15_R"]; r5 = raw[eid]["drop5_R"]
        # n2 先失守: 从 30→15 或 15→5 中 n2 的跌幅最大
        losses = {f"n{d+1}": r30[d] - min(r15[d], r5[d]) for d in range(3)}
        first = max(losses, key=losses.get)
        rep.append(f"| {eid} | {r30[0]:.2f}/{r30[1]:.2f}/{r30[2]:.2f} | "
                   f"{r15[0]:.2f}/{r15[1]:.2f}/{r15[2]:.2f} | {r5[0]:.2f}/{r5[1]:.2f}/{r5[2]:.2f} | {first} |\n")
    rep.append("\n")

    # ---- ev08 异常复核 ----
    rep.append("## 6. ev08 异常复核 (v1 中 E2 全可辨识与 E1 矛盾)\n\n")
    if "ev08" in raw:
        r = raw["ev08"]
        rep.append(f"- v2: E0 ident={any(r['ident']['E0'])}, E1 ident={all(r['ident']['E1'])}, "
                   f"E2 ident={all(r['ident']['E2'])}; R_j E0={r['R']['E0']}, E1={r['R']['E1']}, E2={r['R']['E2']}\n")
        rep.append(f"- W@有效格 RMS={r['w_rms_valid']:.4f}, φ 中位={r['phi_med']:.2f}\n\n")

    with open(os.path.join(OUT, "xev_report_v2.md"), "w", encoding="utf-8") as f:
        f.writelines(rep)
    print("xev_report_v2.md written")


if __name__ == "__main__":
    main()
