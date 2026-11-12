"""Figure 3. (a) R_j heat map: events x (parameters x error structures).
(b) Cross-structure posterior-mean drift as fraction of prior width,
baseline and LOEO calibrations.
All values recomputed from the registered output files (xev_raw_v2.json,
matrixA_stats_v2.json, gate9/loeo_*.json); baseline pooled median 68.3% and
LOEO pooled median 53.2% reproduced bit-exact."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fig_style import C, STRUCT_COLOR, STRUCT_LABEL, PARAM_LABEL, panel_label, save

P2 = "E:/work/workBuddy/BlueHeartV2/output/p2"
W = 0.013  # prior width (registered)
MODELS = ("E0", "E1", "E2")

ma = json.load(open(f"{P2}/matrixA_stats_v2.json"))
d = json.load(open(f"{P2}/xev/xev_raw_v2.json"))
events = ["ev00"] + sorted(d.keys())
srcs = {"ev00": ma}
srcs.update({e: d[e] for e in sorted(d.keys())})

R = np.zeros((len(events), 3, 3))          # event x model x param
drift = np.zeros((len(events), 3))         # event x param (cross-model range)
for ie, e in enumerate(events):
    s = srcs[e]
    for im, m in enumerate(MODELS):
        R[ie, im, :] = s["R"][m] if e != "ev00" else s["R_j"][m]
    M = np.stack([np.array(s["post_mean"][m]) for m in MODELS])
    drift[ie, :] = M.max(axis=0) - M.min(axis=0)
frac = drift / W * 100.0                    # percent of prior width

fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.56)

# ---------- panel a: R_j heat map ----------
ax = fig.add_subplot(gs[0])
sub = R.transpose(0, 2, 1).reshape(len(events), -1)   # rows=events, cols=param-major x model
# column order: for each param, three structures side by side
cmap = plt.cm.viridis.copy()
im = ax.imshow(sub, cmap=cmap, vmin=0, vmax=1, aspect="auto")
# mark grading threshold R_j = 0.5: cells below get a red frame
for ie in range(len(events)):
    for ic in range(9):
        if sub[ie, ic] < 0.5:
            ax.add_patch(plt.Rectangle((ic - 0.5, ie - 0.5), 1, 1, fill=False,
                                       edgecolor=C["red"], lw=1.0))
# structure separators (after each param block of 3)
for x in (2.5, 5.5):
    ax.axvline(x, color="white", lw=2)
ax.set_xticks(range(9))
ax.set_xticklabels([p for p in ("n$_1$", "n$_2$", "n$_3$") for _ in MODELS], fontsize=7.5)
ax.text(1.0, -1.25, STRUCT_LABEL["E0"], ha="center", fontsize=7, color=STRUCT_COLOR["E0"])
ax.text(4.0, -1.25, STRUCT_LABEL["E1"], ha="center", fontsize=7, color=STRUCT_COLOR["E1"])
ax.text(7.0, -1.25, STRUCT_LABEL["E2"], ha="center", fontsize=7, color=STRUCT_COLOR["E2"])
ax.set_yticks(range(len(events)))
ax.set_yticklabels([e.replace("ev", "ev ") for e in events], fontsize=7.5)
ax.set_ylabel("Event")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("R$_j$", fontsize=7.5, labelpad=4)
cb.ax.tick_params(labelsize=7)
ax.text(0.0, 1.02, "red frame: R$_j$ < 0.5 (not identified)", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=6.8, color=C["red"])
panel_label(ax, "a", dx=-0.10)

# ---------- panel b: drift distribution (baseline + LOEO) ----------
ax2 = fig.add_subplot(gs[1])
import glob, re as _re
loeo = {}
for f in glob.glob(f"{P2}/gate9/loeo_ev*_E?.json"):
    m = _re.search(r"loeo_(ev\d+)_(E\d)\.json", f)
    loeo[(m.group(1), m.group(2))] = np.array(json.load(open(f))["mean"])
loeo_frac = np.zeros_like(frac)
for ie, e in enumerate(events):
    M = np.stack([loeo[(e, m)] for m in MODELS])
    loeo_frac[ie, :] = (M.max(axis=0) - M.min(axis=0)) / W * 100

rng = np.random.default_rng(11)
flip = {("ev01", 0), ("ev02", 0), ("ev03", 0), ("ev04", 0), ("ev05", 1), ("ev06", 1)}
frag = {("ev05", 0), ("ev08", 0), ("ev08", 2)}
for j in range(3):
    xb = j - 0.17 + rng.uniform(-0.10, 0.10, len(events))
    xl = j + 0.17 + rng.uniform(-0.10, 0.10, len(events))
    ax2.scatter(xl, loeo_frac[:, j], s=20, marker="D", color=C["gray"], alpha=0.85,
                edgecolor="white", lw=0.4, zorder=3)
    ax2.scatter(xb, frac[:, j], s=22, color=STRUCT_COLOR["E1"], alpha=0.75,
                edgecolor="white", lw=0.5, zorder=3)
    for ie, e in enumerate(events):
        if (e, j) in flip:
            ax2.scatter(xb[ie], frac[ie, j], s=52, facecolor="none",
                        edgecolor=C["red"], lw=1.2, zorder=4)
        if (e, j) in frag:
            ax2.scatter(xb[ie], frac[ie, j], s=52, facecolor="none",
                        edgecolor=C["purple"], lw=1.2, ls=":", zorder=4)
    mb = np.median(frac[:, j])
    ml = np.median(loeo_frac[:, j])
    ax2.hlines(mb, j - 0.42, j - 0.02, color=C["black"], lw=1.6, zorder=4)
    ax2.hlines(ml, j + 0.02, j + 0.42, color=C["black"], lw=1.6,
               ls=(0, (3, 1.6)), zorder=4)
    ax2.annotate(f"{mb:.1f}%", (j - 0.46, mb + (3.5 if mb > 90 else 0)), fontsize=6.8,
                 va="center", ha="right", color="#333333",
                 bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
    ax2.annotate(f"{ml:.1f}%", (j + 0.46, ml + (3.5 if ml > 90 else 0)), fontsize=6.8,
                 va="center", ha="left", color="#333333",
                 bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
# pooled medians (baseline dashed grey, LOEO dotted), labelled at the left margin
pooled_b = np.median(frac)
pooled_l = np.median(loeo_frac)
ax2.axhline(pooled_b, color=C["gray"], lw=1.0, ls="--", zorder=1)
ax2.axhline(pooled_l, color="#8A8F98", lw=1.0, ls=":", zorder=1)
ax2.text(-0.90, pooled_b + 4.2, f"pooled {pooled_b:.1f}%", fontsize=6.8,
         color=C["gray"], ha="left")
ax2.text(-0.90, pooled_l - 5.0, f"LOEO pooled {pooled_l:.1f}%", fontsize=6.8,
         color="#8A8F98", ha="left")
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["n$_1$", "n$_2$", "n$_3$"])
ax2.set_xlim(-0.95, 2.95)
ax2.set_ylim(-6, 112)
ax2.set_ylabel("Cross-structure drift of posterior mean\n(% of prior width)")
ax2.grid(axis="y", alpha=0.5)
panel_label(ax2, "b", dx=-0.24)

from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="o", ls="", mfc=STRUCT_COLOR["E1"], mec="white", ms=6,
           label="baseline (n = 11)"),
    Line2D([], [], marker="D", ls="", mfc=C["gray"], mec="white", ms=5,
           label="LOEO (n = 11)"),
    Line2D([], [], marker="o", ls="", mfc="none", mec=C["red"], ms=7, label="grading flip (6)"),
    Line2D([], [], marker="o", ls="", mfc="none", mec=C["purple"], ms=7, label="fragile grading (3)"),
    Line2D([], [], color=C["black"], lw=1.6, label="median (solid base / dashed LOEO)"),
]
ax2.legend(handles=handles, loc="upper center", frameon=False, fontsize=6.8,
           handletextpad=0.4, borderaxespad=0.1, ncol=2,
           bbox_to_anchor=(0.5, -0.10))

save(fig, "figure3_heatmap_drift")
