# -*- coding: utf-8 -*-
"""
P2 阶段二 ①(下): 跨事件稳定性汇总报告 + 图
输入: output/p2/xev/xev_results.csv, xev_raw.json, 矩阵 A 锚点 (matrixA_stats.json, dropout_curve.csv)
输出: output/p2/xev/xev_report.md, fig_xev_heatmap.png, fig_xev_dropout.png
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import sys
sys.path.insert(0, "E:/work/workBuddy/BlueHeartV2")
import p2lib

OUT = os.path.join(p2lib.P2, "xev")
MAN = pd.read_csv(os.path.join(OUT, "events_manifest.csv"))
df = pd.read_csv(os.path.join(OUT, "xev_results.csv"))
raw = json.load(open(os.path.join(OUT, "xev_raw.json")))

# ---- 锚点 ev00 (矩阵 A 原事件) ----
mA = json.load(open(os.path.join(p2lib.P2, "matrixA_stats.json")))
anchor = {"event_id": "ev00", "stratum": "T3_Low(anchor)", "depth_mm": 38.35, "GW_pct": 0.370}
for model in ("E0", "E1", "E1w", "E2"):
    Rj = mA["R_j"][model]
    df = pd.concat([df, pd.DataFrame([{
        "event_id": "ev00", "stratum": "T3_Low(anchor)", "depth_mm": 38.35, "GW_pct": 0.370,
        "model": model, "R_n1": Rj[0], "R_n2": Rj[1], "R_n3": Rj[2],
        "ident_all": all(r >= 0.5 for r in Rj),
        "ident_none": all(r < 0.5 for r in Rj),
        "n_obs": np.nan, "n_valid_st": mA["n_valid_stations"], "phi_med": np.nan}])], ignore_index=True)

df = df.sort_values(["model", "event_id"]).reset_index(drop=True)
piv = df.pivot_table(index="event_id", columns="model",
                     values=["R_n1", "R_n2", "R_n3"], aggfunc="first")

order = (MAN.sort_values("event_id")["event_id"].tolist() + ["ev00"])
man = MAN.set_index("event_id")

rep = ["# P2 阶段二① 跨事件重复: 可辨识性结论的事件间稳定性\n\n",
       "- 10 场分层事件 (雨强三分位 T × 前期地下水 High/Low) + 锚点 ev00 (矩阵 A 原事件)\n",
       "- 每事件 90 组 LHS (与矩阵 A 同一设计) → 逐事件 PCA-GP + zoom 后验 (E0/E1/E1w/E2)\n",
       "- 判级 (预注册): R_j≥0.5 且 info≥0.1 nats; 表中 R_j 展示\n\n"]

# ---- 1. 翻转矩阵 ----
rep.append("## 1. 可辨识性翻转矩阵 (全可辨识 = 三参数均过)\n\n")
rep.append("| 事件 | 层 | 雨深(mm) | GW | E0 | E1 | E1w | E2 |\n|---|---|---|---|---|---|---|---|\n")
for eid in order:
    sub = df[df["event_id"] == eid].set_index("model")
    r = man.loc[eid] if eid != "ev00" else anchor
    def mark(m):
        if m not in sub.index:
            return "–"
        return "✅全" if sub.loc[m, "ident_all"] else ("❌全非" if sub.loc[m, "ident_none"] else "◐部分")
    rep.append(f"| {eid} | {r['stratum']} | {r['depth_mm']} | {r['GW_pct']} | "
               f"{mark('E0')} | {mark('E1')} | {mark('E1w')} | {mark('E2')} |\n")
n_e1 = df[(df.model == "E1")]["ident_all"].sum()
n_e0_none = df[(df.model == "E0")]["ident_none"].sum()
rep.append(f"\n- **E1 全可辨识: {n_e1}/{len(order)} 事件; E0 全不可辨识: {n_e0_none}/{len(order)} 事件**\n")
rep.append("- 翻转结论 (E0 不可辨识 → E1 可辨识) 若在全部事件成立, 则「结论随误差结构翻转」具有跨事件稳健性\n\n")

# ---- 2. R_j 跨事件分布 ----
rep.append("## 2. R_j 跨事件分布 (mean ± sd, min–max)\n\n")
rep.append("| 模型 | 参数 | mean±sd | min–max | 可辨识事件数 |\n|---|---|---|---|---|\n")
for model in ("E0", "E1w", "E1", "E2"):
    for j, p in enumerate(("n1", "n2", "n3")):
        v = df[df.model == model][f"R_{p}"].values
        nid = int((v >= 0.5).sum())
        rep.append(f"| {model} | {p} | {v.mean():.2f}±{v.std():.2f} | {v.min():.2f}–{v.max():.2f} | {nid}/{len(v)} |\n")
rep.append("\n")

# ---- 3. 事件间 Kendall τ (E1 R_j 向量) ----
e1 = df[df.model == "E1"].set_index("event_id")[["R_n1", "R_n2", "R_n3"]]
taus = []
eids = [e for e in order if e in e1.index]
for a in range(len(eids)):
    for b in range(a + 1, len(eids)):
        t = kendalltau(e1.loc[eids[a]], e1.loc[eids[b]]).statistic
        taus.append(t)
rep.append(f"## 3. 事件间排序一致性\n\n- E1 R_j 向量两两 Kendall τ: 中位 {np.median(taus):.2f}, "
           f"min {np.min(taus):.2f}, max {np.max(taus):.2f} ({len(eids)} 事件 {len(taus)} 对)\n\n")

# ---- 4. dropout 模式稳定性 ----
rep.append("## 4. E1 dropout 阶梯 (15/5 站) 是否事件间稳定\n\n")
rep.append("| 事件 | 30站 n1/n2/n3 | 15站 | 5站 | 15站损失主参数 |\n|---|---|---|---|---|\n")
for eid in order:
    if eid == "ev00":
        dc = pd.read_csv(os.path.join(p2lib.P2, "dropout_curve.csv"))
        r30 = dc[(dc.model == "E1") & (dc.n_stations == 30)][["R_j_n1", "R_j_n2", "R_j_n3"]].values[0]
        r15 = dc[(dc.model == "E1") & (dc.n_stations == 15)][["R_j_n1", "R_j_n2", "R_j_n3"]].values[0]
        r5 = dc[(dc.model == "E1") & (dc.n_stations == 5)][["R_j_n1", "R_j_n2", "R_j_n3"]].values[0]
    else:
        a = raw.get(eid)
        if a is None:
            continue
        r30 = a["R"]["E1"]
        r15, r5 = a["drop15"], a["drop5"]
    loss = np.array(r30) - np.array(r15)
    main_p = ["n1", "n2", "n3"][int(np.argmax(loss))] if loss.max() > 0.02 else "无集中损失"
    rep.append(f"| {eid} | {'/'.join(f'{x:.2f}' for x in r30)} | {'/'.join(f'{x:.2f}' for x in r15)} | "
               f"{'/'.join(f'{x:.2f}' for x in r5)} | {main_p} |\n")
rep.append("\n")

# ---- 5. 后验均值散布 ----
rep.append("## 5. E1 后验均值跨事件散布 (有效糙率的事件特异性)\n\n")
rep.append("| 事件 | n1 | n2 | n3 |\n|---|---|---|---|\n")
for eid in order:
    if eid == "ev00":
        continue
    a = raw.get(eid)
    if a is None:
        continue
    m = a["post_mean"]["E1"]
    rep.append(f"| {eid} | {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} |\n")
rep.append("\n")

# ---- 6. 完整性日志 ----
rep.append("## 完整性日志\n\n"
           "- 预注册偏差登记: 跨事件批次 dwf_k=1.0 (与矩阵 A 的 90 组 LHS 运行一致; 矩阵 A 的 LHS 在 P1.8 "
           "k=0 决策前已启动, 站点偏置由 E2/E1 去偏吸收), 保持与锚点严格可比\n"
           "- 每事件独立 QC (预注册规则)、独立 φ 标定; 窗口规则与 event_window 一致 (rain>0.1mm, 前3h/后9h)\n"
           "- 事件选择: 分层确定性 (层内按 depth 接近中位 + 间隔>14 天分散)\n")

with open(os.path.join(OUT, "xev_report.md"), "w", encoding="utf-8") as f:
    f.writelines(rep)

# ---- 图 1: R_j 热力图 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(16, 0.5 + 0.42 * len(order)), sharey=True)
for ax, model in zip(axes, ("E0", "E1w", "E1", "E2")):
    sub = df[df.model == model].set_index("event_id").reindex(order)
    arr = sub[["R_n1", "R_n2", "R_n3"]].values.astype(float)
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(3):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(3), ["n1", "n2", "n3"], fontsize=8)
    ax.set_title(model)
    ax.set_yticks(range(len(order)), order, fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.6, label="R_j")
fig.suptitle("Cross-event identifiability (R_j) by error model")
fig.savefig(os.path.join(OUT, "fig_xev_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- 图 2: dropout 阶梯跨事件 ----
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for j, (p, c) in enumerate((("n1", "#1f77b4"), ("n2", "#ff7f0e"), ("n3", "#2ca02c"))):
    for eid in order:
        if eid == "ev00":
            dc = pd.read_csv(os.path.join(p2lib.P2, "dropout_curve.csv"))
            d1 = dc[dc.model == "E1"]
            ys = [d1[d1.n_stations == k][f"R_j_{p}"].values[0] for k in (30, 15, 5)]
        else:
            a = raw.get(eid)
            if a is None:
                continue
            ys = [a["R"]["E1"][j], a["drop15"][j], a["drop5"][j]]
        axes[j].plot([30, 15, 5], ys, "o-", ms=3, alpha=0.6, label=eid)
    axes[j].axhline(0.5, ls="--", color="gray", lw=1)
    axes[j].set_title(f"Manning {p}")
    axes[j].set_xticks([30, 15, 5])
    axes[j].set_xlabel("n sensors")
axes[0].set_ylabel("R_j (E1)")
axes[2].legend(fontsize=6)
fig.suptitle("Dropout ladder stability across events (E1)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_xev_dropout.png"), dpi=150)
plt.close(fig)
print("xev report written")
