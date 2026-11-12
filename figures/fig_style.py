"""Shared publication style for EMS manuscript figures (Elsevier EMS)."""
import matplotlib.pyplot as plt

# Elsevier EMS: double column 190 mm, single 90 mm; min font 7 pt at final size
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": False,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
})

# colorblind-safe
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "teal": "#56B4E9",
    "yellow": "#F0E442", "gray": "#8C8C8C", "black": "#222222",
}
# error structures (consistent across all figures)
STRUCT_COLOR = {"E0": C["gray"], "E1": C["blue"], "E2": C["orange"]}
STRUCT_LABEL = {
    "E0": "E0 independent",
    "E1": "E1 AR(1)",
    "E2": "E2 AR(1)+bias",
}
PARAM_LABEL = {0: "n$_1$", 1: "n$_2$", 2: "n$_3$"}


def panel_label(ax, s, dx=-0.12, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{name}.{ext}", format=ext)
    print("saved:", name + ".pdf/.png")
