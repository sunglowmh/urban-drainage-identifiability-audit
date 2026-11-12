"""Figure 4. Unified effect-size comparison across the three design axes.
Median / 90th percentile / maximum posterior-mean shift as a percentage of the
prior width. Values verified: drift axis (33 cells) 68.3/96.8/98.1;
rainfall perturbation (54 units) 9.2/26.5/42.6; forcing convention (63 units)
0.88/9.3/26.0."""
import numpy as np
import matplotlib.pyplot as plt
from fig_style import C, panel_label, save

axes_lbl = ["Error structure\n(11 events × 3 structures)",
            "Rainfall perturbation\n(6 realizations, anchor only)",
            "Forcing convention\n(7 events)"]
median = [68.3, 9.2, 0.9]
p90 = [96.8, 26.5, 9.3]
mx = [98.1, 42.6, 26.0]
colors = [C["blue"], C["orange"], C["green"]]

x = np.arange(3)
w = 0.26
fig, ax = plt.subplots(figsize=(5.6, 3.2))
b1 = ax.bar(x - w, median, w, color=colors, alpha=0.95, edgecolor="white", lw=0.5,
            label="Median")
b2 = ax.bar(x, p90, w, color=colors, alpha=0.55, edgecolor="white", lw=0.5,
            label="90th percentile")
b3 = ax.bar(x + w, mx, w, color=colors, alpha=0.25, edgecolor="white", lw=0.5,
            label="Maximum")
for bars, vals in ((b1, median), (b2, p90), (b3, mx)):
    for r, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (r.get_x() + r.get_width() / 2, v * 1.07),
                    ha="center", fontsize=7, color="#333333")
ax.set_yscale("log")
ax.set_ylim(0.5, 400)
ax.set_xticks(x)
ax.set_xticklabels(axes_lbl, fontsize=7.5)
ax.set_ylabel("Posterior-mean shift\n(% of prior width, log scale)")
ax.yaxis.grid(True, alpha=0.5)
ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=3,
          handletextpad=0.4, columnspacing=1.0)
ax.text(0.01, 0.92, "Unified effect-size metric; the three designs differ in\n"
        "construction (events / realizations) — not a variance decomposition.",
        transform=ax.transAxes, fontsize=6.8, color="#555555", va="top")
panel_label(ax, "a", dx=-0.16)
save(fig, "figure4_axis_comparison")
