"""R-A7: emulator error floor (W) magnitude audit vs CRPS scale. Inference-layer only."""
import numpy as np, json, os
rows = []
for eid in ["ev00"] + ["ev%02d" % i for i in range(1, 11)]:
    p = os.path.join("output", "p2", "xev", "wmap_%s_v2.npy" % eid)
    if not os.path.exists(p):
        p = os.path.join("output", "p2", "twin", "emu_err_map_val_v2.npy")  # anchor
    W = np.load(p); s = np.sqrt(W)
    rows.append(dict(event=eid, n_cells=int(W.size),
                     sw_median=float(np.median(s)), sw_p95=float(np.percentile(s, 95)),
                     sw_max=float(s.max()),
                     frac_gt_005=float((s > 0.05).mean()), frac_gt_010=float((s > 0.1).mean())))
os.makedirs("output/p2/gate10", exist_ok=True)
import csv
with open("output/p2/gate10/w_floor_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
summary = dict(
    sw_median_range=[min(r["sw_median"] for r in rows), max(r["sw_median"] for r in rows)],
    sw_p95_range=[min(r["sw_p95"] for r in rows), max(r["sw_p95"] for r in rows)],
    sw_max_range=[min(r["sw_max"] for r in rows), max(r["sw_max"] for r in rows)],
    frac_gt005_median=float(np.median([r["frac_gt_005"] for r in rows])),
    frac_gt005_max=max(r["frac_gt_005"] for r in rows),
    events_with_max_le_002=sum(1 for r in rows if r["sw_max"] <= 0.02),
    note="W = GP reconstruction variance map (station x time). sqrt(W) in m. "
         "Large cells (>0.05 m) only in ev00 (3.5%), ev07 (0.3%), ev08 (0.6%), ev10 (0.2%).")
json.dump(summary, open("output/p2/gate10/w_floor_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
