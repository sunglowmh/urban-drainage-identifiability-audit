"""Graphical abstract for the EMS submission (Elsevier, min 531 x 1328 px h x w,
readable at 5 x 13 cm). Designed at 2x final physical size (10.24 x 4.10 in =
26 x 10.4 cm) so fonts are 2x print size; PNG exported at 300 dpi (3072 x 1230 px,
proportionally more than the minimum) plus vector PDF.

Three-stage banner:
  1. Pre-registered audit on the real pilot network (3 roughness zones x 3 error structures)
  2. Headline result: error-structure axis (68.3%) >> rainfall (9.2%) / forcing (0.9%)
  3. Verdict: 6/33 gradings flip across structures, 0/33 holdout tasks pass the
     pre-registered 5% CRPS bar -> grading gains do not demonstrate prediction skill.

Programmatic matplotlib rendering (no generative-AI artwork, per EMS policy).
All numbers are registered manuscript values (Abstract / S4.3 / S4.4)."""
import json, re
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11,
    "axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white",
})

ROOT = "E:/work/workBuddy/BlueHeartV2"
BLUE, ORANGE, GRAY, INK = "#0072B2", "#E69F00", "#8C8C8C", "#222222"
CONDUIT_C, STATION_C = "#7A8490", "#123B5C"
ZONE_TINT = ["#F1F6FA", "#E3EDF5", "#D5E4F0"]

# ---------- data (same pipeline as figure 1) ----------
nodes = gpd.read_file(f"{ROOT}/dataset/2_network_shape_files/nodes/Nodes.shp").to_crs(27700)
nxy = {r["id"]: (r.geometry.x, r.geometry.y) for _, r in nodes.iterrows()}
inp = open(f"{ROOT}/output/p1/blueheart_pilot.inp").read()

def sect(name):
    m = re.search(r"\[" + name + r"\](.*?)(\[|$)", inp, re.S)
    return [ln.split(";")[0].strip() for ln in m.group(1).splitlines() if ln.split(";")[0].strip()]

CD = [(ln.split()[1], ln.split()[2]) for ln in sect("CONDUITS")]
cl = json.load(open(f"{ROOT}/output/p2/station_clusters.json"))
stations = [n for v in cl["clusters"].values() for n in v]

sx = [nxy[n][0] for n in stations if n in nxy]
sy = [nxy[n][1] for n in stations if n in nxy]
zx0, zx1 = min(sx) - 900, max(sx) + 1300
zy0, zy1 = min(sy) - 900, max(sy) + 900

# zone boundaries: conduit midpoint easting terciles (S3.1 definition)
mids = [(nxy[a][0] + nxy[b][0]) / 2 for a, b in CD if a in nxy and b in nxy]
b1, b2 = np.percentile(mids, [33.3, 66.7])

# ---------- canvas ----------
W, H = 10.24, 4.10  # = 2 x (13 x 5 cm)
fig = plt.figure(figsize=(W, H))

def header(x, s, color=INK):
    fig.text(x, 0.945, s, fontsize=15, fontweight="bold", color=color,
             ha="left", va="center")

# ================= stage 1: design =================
header(0.018, "Pre-registered identifiability audit")

axm = fig.add_axes([0.020, 0.36, 0.315, 0.50])
axm.set_facecolor("#FBFCFD")
for a, b in CD:
    if a in nxy and b in nxy:
        (x0, y0), (x1, y1) = nxy[a], nxy[b]
        if zx0 <= x0 <= zx1 and zx0 <= x1 <= zx1 and zy0 <= y0 <= zy1 and zy0 <= y1 <= zy1:
            axm.plot([x0, x1], [y0, y1], color=CONDUIT_C, lw=0.22, zorder=2)

# zone bands + dividers (behind conduits: re-draw as axvspan under, via zorder)
for i, (x_lo, x_hi) in enumerate([(zx0, b1), (b1, b2), (b2, zx1)]):
    axm.axvspan(x_lo, x_hi, color=ZONE_TINT[i], zorder=0)
for bx in (b1, b2):
    axm.axvline(bx, color="#9FB4C8", lw=0.8, ls=(0, (4, 3)), zorder=1)

zc = axm.transData  # zone labels at top inside bands
axm.text((zx0 + b1) / 2, zy1 - 90, "n$_1$ upstream", ha="center", va="top",
         fontsize=10.5, color="#3A5A78")
axm.text((b1 + b2) / 2, zy1 - 90, "n$_2$", ha="center", va="top",
         fontsize=10.5, color="#3A5A78")
axm.text((b2 + zx1) / 2, zy1 - 90, "n$_3$ downstream", ha="center", va="top",
         fontsize=10.5, color="#3A5A78")

for n in stations:
    if n in nxy:
        axm.scatter(*nxy[n], s=16, color=STATION_C, edgecolor="white",
                    linewidth=0.5, zorder=5)

axm.set_xlim(zx0, zx1); axm.set_ylim(zy0, zy1)
axm.set_aspect("equal")
axm.set_xticks([]); axm.set_yticks([])
for sp in axm.spines.values():
    sp.set_color("#C9D2DA"); sp.set_linewidth(0.7)
axm.text(0.5, -0.045, "real drainage network\n30 level gauges  \u00b7  11 storm events",
         transform=axm.transAxes, ha="center", va="top", fontsize=10, color=INK,
         linespacing=1.3)
# theta + E-boxes row (figure coords, under the map)
fig.text(0.1775, 0.225, r"$\theta=(n_1,\,n_2,\,n_3)$  three roughness zones",
         fontsize=11.5, ha="center", va="center")
fig.add_artist(FancyArrowPatch((0.1775, 0.196), (0.1775, 0.168),
                               transform=fig.transFigure, arrowstyle="-|>",
                               mutation_scale=11, color=INK, lw=1.1))
boxes = [("E0  independent", GRAY, 0.026), ("E1  AR(1)", BLUE, 0.1315),
         ("E2  AR(1)+bias", ORANGE, 0.237)]
for lab, c, x in boxes:
    fig.add_artist(FancyBboxPatch(
        (x, 0.058), 0.092, 0.095, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor="white", edgecolor=c, linewidth=1.6))
    fig.text(x + 0.046, 0.1055, lab, fontsize=9, ha="center", va="center", color=INK)
fig.text(0.1775, 0.014, "same data \u00b7 same surrogate \u00b7 three likelihoods",
         fontsize=9, ha="center", va="center", color="#5A6570", style="italic")

# ================= stage 2: headline result =================
header(0.385, "Error structure dominates")
axb = fig.add_axes([0.400, 0.215, 0.215, 0.56])
cats = ["Forcing\nconvention", "Rainfall\nperturbation", "Error\nstructure"]
vals = [0.9, 9.2, 68.3]
cols = [GRAY, GRAY, BLUE]
bars = axb.barh(cats, vals, color=cols, edgecolor="white", height=0.58)
for r, v in zip(bars, vals):
    axb.text(v + 1.5, r.get_y() + r.get_height() / 2, f"{v}%",
             va="center", ha="left", fontsize=11.5,
             fontweight="bold" if v == 68.3 else "normal",
             color=BLUE if v == 68.3 else INK)
axb.set_xlim(0, 84)
axb.set_xlabel("median posterior-mean shift (% of prior width)", fontsize=10)
axb.tick_params(axis="y", labelsize=10.5, length=0)
axb.tick_params(axis="x", labelsize=9)
axb.text(0.0, -0.52, "descriptive comparison of three designs \u2014 not a variance decomposition",
         transform=axb.transAxes, fontsize=8.5, color="#5A6570", style="italic",
         ha="left", va="top")

# ================= stage 3: verdict =================
header(0.665, "but grading \u2260 prediction skill")
axc = fig.add_axes([0.672, 0.175, 0.315, 0.55])
axc.set_xlim(0, 1); axc.set_ylim(-0.35, 1.9)
axc.axis("off")

def cellstrip(y, filled, color, label, count, count_color):
    n = 33
    cw = 1.0 / n
    for i in range(n):
        axc.add_patch(Rectangle((i * cw + 0.003, y), cw - 0.006, 0.52,
                                facecolor=color if i < filled else "#E7EAEE",
                                edgecolor="none"))
    axc.text(0.0, y + 0.95, label, fontsize=10.5, va="center")
    axc.text(1.0, y + 0.95, count, fontsize=12, fontweight="bold",
             color=count_color, va="center", ha="right")

cellstrip(1.05, 6, ORANGE, "gradings flipping across structures", "6/33", ORANGE)
cellstrip(-0.15, 0, "#009E73", "holdout tasks passing the 5% CRPS bar", "0/33", BLUE)
axc.text(0.0, -0.33, "leave-one-event-out prediction, pre-registered threshold",
         fontsize=8.5, color="#5A6570", style="italic", va="top")

fig.text(0.672, 0.028,
         "Grading gains and posterior contraction\ndo not, by themselves,\ndemonstrate prediction quality.",
         fontsize=11, fontweight="bold", color=INK, ha="left", va="bottom")

# ================= inter-stage arrows =================
for x0, x1 in ((0.345, 0.378), (0.630, 0.663)):
    fig.add_artist(FancyArrowPatch((x0, 0.55), (x1, 0.55), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=16,
                                   color="#9AA5B0", lw=2.0))

fig.savefig(f"{ROOT}/research_route/manuscript/figures/graphical_abstract.pdf")
fig.savefig(f"{ROOT}/research_route/manuscript/figures/graphical_abstract.png", dpi=300)
from PIL import Image
im = Image.open(f"{ROOT}/research_route/manuscript/figures/graphical_abstract.png")
print("saved. PNG size:", im.size, "| >=1328x531:", im.size[0] >= 1328 and im.size[1] >= 531)
