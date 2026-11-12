"""Figure 5. LOEO event-level paired CRPS change (posterior - prior; positive =
degradation). Panel a: 33 event x structure pairs. Panel b: per-structure and
pooled means with event-cluster bootstrap 95% CIs (B = 10,000, seed 20261111;
intervals as reported in §4.4)."""
import json, glob
import numpy as np
import matplotlib.pyplot as plt
from fig_style import C, STRUCT_COLOR, STRUCT_LABEL, panel_label, save

rows = []
for f in sorted(glob.glob("E:/work/workBuddy/BlueHeartV2/output/p2/gate9/loeo_ev*_E*.json")):
    j = json.load(open(f))
    d = j["post"]["crps_mean"] - j["prior"]["crps_mean"]
    rows.append((j["fold"], j["model"], d))
events = sorted(set(r[0] for r in rows))
MODELS = ("E0", "E1", "E2")
D = {(e, m): d for e, m, d in rows}

fig = plt.figure(figsize=(7.2, 3.2))
gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.26)

# ---------- panel a: event-level paired differences ----------
ax = fig.add_subplot(gs[0])
rng = np.random.default_rng(5)
for im, m in enumerate(MODELS):
    xs, ys = [], []
    for ie, e in enumerate(events):
        xs.append(ie + (im - 1) * 0.24 + rng.uniform(-0.03, 0.03))
        ys.append(D[(e, m)] * 1000.0)  # mm scale for readability? keep m
    ax.scatter(xs, np.array(ys), s=20, color=STRUCT_COLOR[m], alpha=0.85,
               edgecolor="white", lw=0.4, zorder=3,
               label=STRUCT_LABEL[m])
ax.axhline(0, color=C["black"], lw=1.0, zorder=2)
ax.set_xticks(range(len(events)))
ax.set_xticklabels([e.replace("ev", "ev ") for e in events], fontsize=7, rotation=45)
ax.set_xlabel("Held-out event (LOEO fold)")
ax.set_ylabel("Paired CRPS change\nposterior − prior (mm)")
ax.grid(axis="y", alpha=0.5)
panel_label(ax, "a", dx=-0.09)
ax.legend(frameon=False, fontsize=6.8, loc="lower left", bbox_to_anchor=(0.0, 1.01),
          ncol=3, handletextpad=0.3, columnspacing=0.9)
ax.text(0.01, 0.02, "positive = degradation; no fold reached the 5% improvement threshold",
        transform=ax.transAxes, fontsize=6.8, color="#555555")

# ---------- panel b: means with cluster-bootstrap CIs ----------
ax2 = fig.add_subplot(gs[1])
summ = [  # mean, lo, hi, p (Wilcoxon, §4.4)
    ("E0", 0.0002, -0.00002, 0.0005, "p = 0.12"),
    ("E1", 0.0004, 0.0001, 0.0008, "p = 0.019"),
    ("E2", 0.0012, 0.0005, 0.0019, "p = 0.007"),
]
for i, (m, mu, lo, hi, p) in enumerate(summ):
    ax2.errorbar(mu * 1000, i, xerr=[[(mu - lo) * 1000], [(hi - mu) * 1000]],
                 fmt="o", ms=6, mfc=STRUCT_COLOR[m], mec="white", mew=0.8,
                 ecolor=STRUCT_COLOR[m], elinewidth=1.4, capsize=3, capthick=1.4,
                 zorder=3)
    ax2.annotate(p, (hi * 1000 + 0.03, i), fontsize=6.5, va="center", color="#555555")
# pooled
ax2.errorbar(0.6, 3.0, xerr=[[0.2], [0.2]], fmt="D", ms=6, mfc=C["black"],
             mec="white", mew=0.8, ecolor=C["black"], elinewidth=1.4,
             capsize=3, capthick=1.4, zorder=3)
ax2.annotate("pooled: +0.0006 [+0.0004, +0.0008]", (0.88, 3.0), fontsize=6.5,
             va="center", color="#333333")
ax2.axvline(0, color=C["black"], lw=1.0)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(["E0", "E1", "E2", "Pooled"], fontsize=8)
ax2.set_xlim(-0.45, 3.3)
ax2.set_ylim(-0.6, 3.9)
ax2.set_xlabel("Mean CRPS change (mm)")
ax2.tick_params(axis="y", length=0)
ax2.grid(axis="x", alpha=0.5)
panel_label(ax2, "b", dx=-0.28)
ax2.text(0.20, 0.94, "error bars: 95% event-cluster\nbootstrap CI (B = 10 000)",
         transform=ax2.transAxes, fontsize=6.5, color="#555555", va="top")

save(fig, "figure5_loeo_paired")
