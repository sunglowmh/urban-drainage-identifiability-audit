"""Figure 1. Study area and pilot monitoring network (Eastbourne-Pevensey,
south coast of England). Main panel: pilot SWMM network (2,290 conduits) with
30 candidate level stations coloured by effective cluster (complete-linkage,
K = 7); cluster 7 = river-boundary station TV60998101. Inset: study-area outline
with zoom extent. Basemap: simplified label-free vector rendering of OpenStreetMap
data (roads, coastline, green spaces; (c) OpenStreetMap contributors), cached in
osm_cache.json via Overpass API. Coordinates in British National Grid (EPSG:27700)."""
import json, re
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point, box
from shapely.ops import linemerge, unary_union
from pyproj import Transformer
from fig_style import C, panel_label, save

ROOT = "E:/work/workBuddy/BlueHeartV2"

# ---------- data ----------
nodes = gpd.read_file(f"{ROOT}/dataset/2_network_shape_files/nodes/Nodes.shp").to_crs(27700)
nxy = {r["id"]: (r.geometry.x, r.geometry.y) for _, r in nodes.iterrows()}
inp = open(f"{ROOT}/output/p1/blueheart_pilot.inp").read()


def sect(name):
    m = re.search(r"\[" + name + r"\](.*?)(\[|$)", inp, re.S)
    return [ln.split(";")[0].strip() for ln in m.group(1).splitlines() if ln.split(";")[0].strip()]


CD = [(ln.split()[1], ln.split()[2]) for ln in sect("CONDUITS")]
outline = gpd.read_file(f"{ROOT}/dataset/1_catchment_shape_files/catchment_outline/Study_area.shp").to_crs(27700)
cl = json.load(open(f"{ROOT}/output/p2/station_clusters.json"))
ck = {n: int(k) for k, v in cl["clusters"].items() for n in v}

palette = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7",
           5: "#56B4E9", 6: "#F0E442", 7: "#D55E00"}

# zoom extent (stations, tight)
sx = [nxy[n][0] for n in ck if n in nxy]
sy = [nxy[n][1] for n in ck if n in nxy]
zx0, zx1 = min(sx) - 900, max(sx) + 1300
zy0, zy1 = min(sy) - 900, max(sy) + 900

fig, ax = plt.subplots(figsize=(5.6, 5.2))

# conduits (darkened for contrast over the light basemap)
CONDUIT_C = "#5A6470"
for a, b in CD:
    if a in nxy and b in nxy:
        (x0, y0), (x1, y1) = nxy[a], nxy[b]
        if zx0 <= x0 <= zx1 and zx0 <= x1 <= zx1 and zy0 <= y0 <= zy1 and zy0 <= y1 <= zy1:
            ax.plot([x0, x1], [y0, y1], color=CONDUIT_C, lw=0.35, alpha=0.9, zorder=2)
# outfalls (within zoom)
for n in sect("OUTFALLS"):
    if n in nxy and zx0 <= nxy[n][0] <= zx1 and zy0 <= nxy[n][1] <= zy1:
        ax.plot(*nxy[n], marker=".", ms=1.6, color="#3D454F", zorder=2)

# ---------- basemap: label-free vector rendering of OSM data ----------
OSM = json.load(open("osm_cache.json"))
tr2 = Transformer.from_crs(4326, 27700, always_xy=True)
from shapely.geometry import Polygon as ShPolygon

def to_ls(el):
    return LineString([tr2.transform(g["lon"], g["lat"]) for g in el["geometry"]])

bbox = box(zx0, zy0, zx1, zy1)

# classify OSM features clipped to the map window
coasts, greens, roads_major, roads_minor = [], [], [], []
for el in OSM["elements"]:
    if el.get("type") != "way" or "geometry" not in el:
        continue
    t = el.get("tags", {})
    ls = to_ls(el)
    if not ls.intersects(bbox):
        continue
    ls = ls.intersection(bbox)
    parts = [ls] if ls.geom_type in ("LineString", "Polygon") else list(ls.geoms)
    for g in parts:
        t2 = t
        if t2.get("natural") == "coastline":
            coasts.append(g)
        elif t2.get("highway") in ("motorway", "trunk", "primary", "secondary", "tertiary"):
            roads_major.append(g)
        elif t2.get("highway"):
            roads_minor.append(g)
        elif g.geom_type == "Polygon":
            greens.append(g)
        elif g.geom_type == "LineString" and g.is_closed:
            greens.append(ShPolygon(g.coords))

merged = linemerge(unary_union(coasts))
if merged.geom_type == "MultiLineString":   # safety: take longest piece
    merged = max(merged.geoms, key=lambda g: g.length)

# close the coastline along the bbox boundary to form the sea polygon
per = [(zx0, zy0), (zx1, zy0), (zx1, zy1), (zx0, zy1)]  # corners, CCW from SW

def edge_of(p):
    if abs(p[0] - zx0) < 1e-6: return 3   # left edge (corner 3 -> 0)
    if abs(p[1] - zy0) < 1e-6: return 0   # bottom (0 -> 1)
    if abs(p[0] - zx1) < 1e-6: return 1   # right (1 -> 2)
    return 2                              # top (2 -> 3)

def arc(i_from, i_to):
    """corners strictly between index i_from (exclusive) and i_to (inclusive), CCW"""
    out, i = [], i_from
    while i != i_to:
        i = (i + 1) % 4
        out.append(per[i])
    return out

pts = list(merged.coords)
e0, e1 = edge_of(pts[0]), edge_of(pts[-1])
seaA = ShPolygon(pts + arc(e1, e0))                      # line then CCW boundary back
seaB = ShPolygon(list(reversed(pts)) + arc(e0, e1))      # reversed line, other way round
test = Point(zx1 - 5.0, (zy0 + zy1) / 2.0)               # eastern mid-edge is sea here
sea = seaA if seaA.contains(test) else seaB

# render order: sea, greens, roads, then network
ax.add_patch(MplPolygon(list(sea.exterior.coords), closed=True,
                        facecolor="#D9E8F2", edgecolor="none", zorder=0.5))
for g in greens:
    ax.add_patch(MplPolygon(list(g.exterior.coords), closed=True,
                            facecolor="#E2EFDA", edgecolor="none", zorder=0.6))
for ls in roads_minor:
    xs, ys = ls.xy
    ax.plot(xs, ys, color="#DADDE1", lw=0.32, zorder=1, solid_capstyle="round")
for ls in roads_major:
    xs, ys = ls.xy
    ax.plot(xs, ys, color="#C4C9CF", lw=0.55, zorder=1, solid_capstyle="round")

# stations by cluster
for n, k in sorted(ck.items(), key=lambda kv: kv[1]):
    x, y = nxy[n]
    if k == 7:
        ax.scatter(x, y, marker="*", s=200, color=palette[7], edgecolor="white",
                   lw=0.6, zorder=5)
    else:
        ax.scatter(x, y, s=40, color=palette[k], edgecolor="white", lw=0.6, zorder=4)

x, y = nxy["TV60998101"]
ax.annotate("TV60998101 (cluster 7)", (x, y), xytext=(x + 1050, y - 620),
            fontsize=6.8, color=palette[7],
            arrowprops=dict(arrowstyle="-", color=palette[7], lw=0.7))

# ---------- inset: whole study area ----------
axi = ax.inset_axes([0.02, 0.66, 0.34, 0.30])
outline.boundary.plot(ax=axi, color="#999999", lw=0.7)
axi.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False,
                        edgecolor=C["red"], lw=1.0))
axi.set_xticks([]); axi.set_yticks([])
axi.set_xlim(outline.total_bounds[0] - 300, outline.total_bounds[2] + 300)
axi.set_ylim(outline.total_bounds[1] - 300, outline.total_bounds[3] + 300)
axi.set_aspect("equal")
axi.set_title("Study area", fontsize=7, pad=2)
for s in axi.spines.values():
    s.set_color("#888888"); s.set_linewidth(0.6)

# ---------- main cosmetics ----------
ax.set_aspect("equal")
ax.set_xlim(zx0, zx1)
ax.set_ylim(zy0, zy1)
ax.set_xlabel("Easting (m, EPSG:27700)")
ax.set_ylabel("Northing (m)")
ax.ticklabel_format(style="plain")
ax.annotate("N", xy=(0.52, 0.96), xytext=(0.52, 0.885), xycoords="axes fraction",
            fontsize=9, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
x0, y0 = zx0 + 120, zy0 + 160
sb = 1000
ax.add_line(mlines.Line2D([x0, x0 + sb], [y0, y0], color="black", lw=1.6))
for xx in (x0, x0 + sb):
    ax.add_line(mlines.Line2D([xx, xx], [y0 - 40, y0 + 40], color="black", lw=1.2))
ax.text(x0 + sb / 2, y0 + 90, "1 km", ha="center", fontsize=7)
# OSM attribution: bottom-right, well clear of the scale bar (bottom-left)
ax.text(0.995, 0.008, "Basemap data © OpenStreetMap contributors",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
        color="#66707A")

handles = [Line2D([], [], color=CONDUIT_C, lw=1.0, label="Pilot conduit (n = 2 290)"),
           Line2D([], [], marker=".", ls="", color="#3D454F", ms=4, label="Outfall (n = 97)")]
for k, lab in ((1, "Cluster 1 (2)"), (2, "Cluster 2 (4)"), (3, "Cluster 3 (13)"),
               (4, "Cluster 4 (6)"), (5, "Cluster 5 (4)")):
    handles.append(Line2D([], [], marker="o", ls="", mfc=palette[k], mec="white", ms=6, label=lab))
handles.append(Line2D([], [], marker="*", ls="", mfc=palette[7], mec="white", ms=10,
                      label="Cluster 7: boundary station"))
ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92,
          edgecolor="#DDDDDD", fontsize=6.8, handletextpad=0.4)
save(fig, f"{ROOT}/research_route/manuscript/figures/figure1_study_area_network")
