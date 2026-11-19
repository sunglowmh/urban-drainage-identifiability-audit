# Error-structure dependence of practical identifiability and the limited transfer of calibration gains to held-out events in urban drainage modeling

Code, simulation outputs, and pre-registration documents for the accompanying manuscript (submitted for publication).

The study audits practical identifiability of three roughness parameters in a real
urban drainage model (Eastbourne–Pevensey, UK) under three observation error
structures (E0: temporal independence; E1: AR(1) residual correlation; E2: E1 plus
event-persistent station offsets), using twin-calibrated grading thresholds and a
pre-registered leave-one-event-out CRPS holdout.

## Repository layout

```
p0_*.py, p1_*.py, p2*.py, p3_*.py, p4_*.py   analysis pipeline (run order = stage order)
p2lib.py                                     shared library (paths, QC, rainfall, likelihoods)
test_likelihoods_v2.py                       unit tests for the likelihood implementations
p0_audit/                                    event library, gauge index, feasibility report
output/p1/                                   SWMM network files (.inp), station tables, build report
output/p2/jobs_*.json                        frozen simulation job manifests
output/p2/emu_*.npz|json, matrixA_stats_v2.json   emulator design and statistics
output/p2/gate8|gate9|gate10/                posterior bundles, holdout and transfer scoring,
                                             classification (classification.csv), robustness audits
output/p2/*_report*.md                       stage reports
output/derived/aggregated_areal_rainfall_15min.csv  15-min areal rainfall series (>=60% station coverage)
figures/                                     figure scripts and rendered figures
                                             (manuscript Figs 1–3 = figure1–3_*; manuscript Fig 4 = figure5_loeo_paired_*;
                                             figure4_axis_comparison_* is an auxiliary analysis figure)
preregistration/preregistration.md           frozen pre-registration document
preregistration/integrity_log.md             per-run integrity log
supplementary_notes.md                       Supplementary Notes S1–S11 (Tables S1–S4), as submitted
```

## Environment

- Python 3.13.14; NumPy 2.4.4, SciPy 1.18.1, pandas 3.0.5, PySWMM 2.2.0 (bundles the
  EPA SWMM 5.2.4 engine), Matplotlib, Pillow.
- See `requirements.txt`.

## Raw telemetry data

The raw sewer water-level, rain-gauge, fluvial, and borehole telemetry used in this
study are part of the open Blue Heart dataset (497 gauges, 78 km² urban catchment,
Eastbourne and Southern Wealden, UK), published with the permission of Southern Water:

> Blue Heart (2026) *Dataset: High resolution hydrometric and sewer system monitoring of an urban catchment with complex flood risk* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.20699635

To re-run the pipeline from raw data, download the dataset and place the seven
sub-folders and `index_telemetry_data.csv` under `dataset/` (the path expected by
`p2lib.py`). `p0_event_library.py` rebuilds the event library from the rain-gauge
files; QC, rainfall aggregation, emulation, inference, and scoring then run in stage
order (`p0_*` → `p1_*` → `p2_*` → `p3_*` → `p4_*`), driven by the frozen manifests in
`output/p2/jobs_*.json`.

The archived `output/p2/gate9/` bundles (`bundle_ev*.pkl`) already contain, per
event, the QC-passed station set, aggregated 15-min observations, emulator design,
and emulator error map `W`, so all figure and table regeneration scripts
(`figures/*.py`) run without the raw dataset.

## Provenance and licences

- Code in this repository: MIT Licence (see `LICENSE`).
- The Blue Heart telemetry dataset is distributed under the terms stated in its
  Zenodo record; the SWMM network files in `output/p1/` are derived from the
  Blue Heart catchment and network shape files.

## Reference

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24162001.svg)](https://doi.org/10.5281/zenodo.24162001)

The manuscript (submitted for publication; pre-registration frozen before any
real-data analysis) documents the
full protocol: QC thresholds, error-structure definitions, twin-calibrated grading
thresholds, and the pre-registered CRPS holdout. This repository is mirrored as a
versioned Zenodo archive (concept DOI above;
version-specific DOIs are listed on the Zenodo record page).
