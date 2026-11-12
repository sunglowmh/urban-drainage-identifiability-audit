"""Figure 2. Twin-experiment acceptance matrix: 95% central posterior interval
coverage per (error structure x parameter) diagonal cell, n = 50 runs per cell.
Values are the verified Table 3 numbers (§4.1)."""
import matplotlib.pyplot as plt
import numpy as np
from fig_style import C, STRUCT_COLOR, STRUCT_LABEL, PARAM_LABEL, panel_label, save

# verified Table 3: hits/n, coverage, Wilson CI, status
cells = [
    # (model, param_idx, hits, n, cov, lo, hi, status)
    ("E0", 0, 46, 50, 0.920, 0.812, 0.968, "PASS"),
    ("E0", 1, 44, 50, 0.880, 0.762, 0.944, "PASS"),
    ("E0", 2, 49, 50, 0.980, 0.895, 0.996, "PASS"),
    ("E1", 0, 38, 50, 0.760, 0.626, 0.857, "PASS"),
    ("E1", 1, 34, 50, 0.680, 0.542, 0.792, "MARGINAL"),
    ("E1", 2, 48, 50, 0.960, 0.865, 0.989, "PASS"),
    ("E2", 0, 32, 50, 0.640, 0.501, 0.759, "MARGINAL"),
    ("E2", 1, 36, 50, 0.720, 0.583, 0.825, "PASS"),
    ("E2", 2, 48, 50, 0.960, 0.865, 0.989, "PASS"),
]

fig, ax = plt.subplots(figsize=(7.2, 3.0))

ypos, ylabels, ycolors = [], [], []
y = 0
for m in ("E0", "E1", "E2"):
    for j in range(3):
        cell = next(c for c in cells if c[0] == m and c[1] == j)
        _, _, hits, n, cov, lo, hi, status = cell
        col = C["blue"] if status == "PASS" else C["orange"]
        # acceptance band [0.80, 1.00]
        ax.axhspan(y - 0.38, y + 0.38, xmin=0, color="#EDF3FA", zorder=0) if False else None
        ax.errorbar(cov, y, xerr=[[cov - lo], [hi - cov]], fmt="o",
                    ms=6, mfc=col, mec="white", mew=0.8,
                    ecolor=col, elinewidth=1.4, capsize=3, capthick=1.4, zorder=3)
        ax.annotate(f"{hits}/{n}", (cov, y + 0.26), va="center", ha="center",
                    fontsize=6.8, color="#555555")
        ypos.append(y); ylabels.append(PARAM_LABEL[j]); ycolors.append(col)
        y += 1
    y += 0.6  # gap between structure groups

# acceptance / fail / nominal lines
ax.axvline(0.80, color=C["green"], lw=1.2, ls="--", zorder=1)
ax.axvline(0.70, color=C["red"], lw=1.0, ls=":", zorder=1)
ax.axvline(0.95, color="#AAAAAA", lw=0.8, ls="-.", zorder=1)
ax.text(0.80, len(ypos) + 0.15, "PASS if CI\nintersects 0.80", fontsize=7,
        color=C["green"], ha="center", va="bottom")
ax.text(0.70, len(ypos) + 0.15, "FAIL if CI\nupper < 0.70", fontsize=7,
        color=C["red"], ha="center", va="bottom")
ax.text(0.95, len(ypos) + 0.15, "nominal\n0.95", fontsize=7, color="#888888",
        ha="center", va="bottom")

# group brackets for structures
for gi, m in enumerate(("E0", "E1", "E2")):
    y0 = gi * 3 + gi * 0.6
    ax.text(-0.055, y0 + 1.0, STRUCT_LABEL[m], transform=ax.get_yaxis_transform(),
            rotation=90, va="center", ha="center", fontsize=7.5,
            color=STRUCT_COLOR[m], fontweight="bold")

ax.set_yticks(ypos)
ax.set_yticklabels(ylabels)
ax.set_xlim(0.45, 1.03)
ax.set_ylim(-0.6, len(ypos) + 1.05)
ax.set_xlabel("Coverage of central 95% credible interval (n = 50 runs per cell)")
ax.tick_params(axis="y", length=0)
for lbl, col in zip(ax.get_yticklabels(), ycolors):
    lbl.set_color("#222222")
panel_label(ax, "a", dx=-0.02, dy=1.02)

# status legend
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="o", ls="", mfc=C["blue"], mec="white", ms=6, label="PASS (7 cells)"),
    Line2D([], [], marker="o", ls="", mfc=C["orange"], mec="white", ms=6, label="MARGINAL (2 cells)"),
]
ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7,
          handletextpad=0.3, borderaxespad=0.2)

save(fig, "figure2_twin_acceptance")
