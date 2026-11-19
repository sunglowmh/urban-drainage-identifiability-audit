# Supplementary Notes

## Supplementary Note S1 — Event screening, stratification, and perturbation records (companion to §2.2–§2.3)

**Stratified event selection.** Per-stratum candidate lists and the selected events
for the stratified ten-event draw (§2.3) are recorded in the archived screening
manifest (p0_audit/ in the archived repository).

**Table S1.** Per-event station availability after quality control (companion table to §2.2).

| Event | Valid stations | Notes |
|---|---|---|
| ev00 (anchor) | 27 | 2,729 valid cells; φ median 0.61 |
| ev01 | 28 | |
| ev02 | 27 | |
| ev03 | 28 | |
| ev04 | 27 | |
| ev05 | **17** | 13 in-event exclusions (insufficient observations); low-coverage event |
| ev06 | 27 | |
| ev07 | 27 | |
| ev08 | 28 | |
| ev09 | 28 | |
| ev10 | 29 | |

*Fixed exclusions at all events: stations 14, 17, 20 (§2.2 text); per-event
station counts exclude these three stations. The zero rainfall-correction deviation of ev02, ev03, ev09 and ev10 is a rainfall-audit property and is reported with the rainfall
audit. Valid observation cells are listed for the anchor event
only; per-event cell counts, which enter the event-level CRPS aggregation, are provided in the archived repository.*

**Event information index J_e.** The full J_e value list for the 11 events and the
recomputation table under the corrected rainfall aggregation are provided as
`ie_recompute_corr.csv` and `ie_reproduce_check.csv` in the archived repository.

**Rainfall-perturbation comparison set.** The randomly drawn rainfall-perturbation
comparison set (registered before any posterior was computed; §3.4) is part of the
frozen job manifests (`jobs_*.json`) in the archived repository.

---

**Stratification cut-offs (companion to §2.3).** Intensity terciles: T1 ≤ 2.86 ≤ T2 ≤ 3.99 < T3 (mm/h). Groundwater class High: GW pct ≥ 0.5, Low otherwise.

**Rainfall re-aggregation, per-event window-total differences (companion to §2.2), expressed as (original − corrected)/corrected:** +7.69% for ev00, ev01, ev04 and ev08; +16.67% for ev05; +8.18% for ev07; +0.57% for ev06; ev02, ev03, ev09 and ev10 bit-identical.

## Supplementary Note S2 — Exclusion and low-coverage sensitivity audits

This note documents two rule-driven exclusion/coverage audits referenced in §2.2. Both are inference-layer recomputations on the archived posterior grids;
no simulation was re-run.

**S2.1 Station 20 re-inclusion (QC-boundary exclusion).** Station 20 (TQ61003301)
was excluded from every event by the fixed QC rules of §2.2 (anchor-event offset
+0.60 m, RMSE 0.67 m), the exclusion closest to the QC boundary. Because its
observations are present in the archive, we re-included it (unmasked its QC-valid
cells) and recomputed all 33 grading labels under each error structure:

- *Raw re-inclusion* (offset retained): all 33 grading labels are unchanged, and
  the six flip units are identical to the archived analysis (pass totals E0/E1/E2
  = 27/32/33 before and after).
- *Datum-corrected re-inclusion* (per-station offset removed, R-A8 protocol):
  one unit changes — the ev05–n₂ flip disappears (6 → 5 flips) and one E0 grading
  passes (27 → 28); the other 32 labels are unchanged.

Station 20's offsets on the other ten events are near-constant (−0.044 to −0.075 m,
median ≈ −0.05 m), so its anchor-event exclusion reflects a datum shift rather than
event-specific dynamics: the exclusion is conservative and not results-driven.

**S2.2 ev05 exclusion (low-coverage event).** Excluding ev05 (17 of 30 stations)
entirely leaves the sensitivity result of §4.2 intact; the full accounting is given
with the construction-sensitivity checks of Note S9.

**S2.3 Per-event station-20 offsets.** ev00 +0.605; ev01 −0.061; ev02 −0.051;
ev03 −0.053; ev04 −0.049; ev05 −0.075; ev06 −0.049; ev07 −0.051; ev08 −0.052;
ev09 −0.045; ev10 −0.044 (m).

---

## Supplementary Note S3 — Holdout implementation details (companion to §3.5)

**Prediction profiling grid.** The profiling grid for prediction uses 12 log-spaced
variance values spanning four orders of magnitude around the emulator-error scale, a
wider range than the inversion grid, so that the profile argmax cannot be truncated
at a grid boundary for θ values sampled from the prior; truncation would artificially
sharpen the prior predictive and inflate the apparent benefit of calibration.

**Event-specific E2 offsets.** The E2 offset b_i is event-specific: it is
re-marginalized analytically within each held-out event's likelihood, and no
cross-event joint estimation is attempted, as the s² and W floors are not stationary
across events and a hierarchical bias model lies beyond the registered scope. The
AR(1) coefficients of a held-out event are its own baseline-residual estimates, a
station-level property fixed across structures.

**Transfer of error-structure profiles.** The prediction-time s²(θ) and τ(θ) are not re-profiled against the held-out data; they are transferred from the calibration events as the median profile value at the same θ (over calibration stations for the station holdout). Profiling the error structure against held-out data is retained only as a sensitivity check (§4.4). The held-out event's QC mask and its baseline-residual φᵢ are likewise event-derived, conditioning the task on these fixed event properties.

**Station clustering construction.** Clusters come from complete-linkage hierarchical
clustering of the 30 candidate stations on a pipe-length-weighted shortest-path
distance matrix, computed once and frozen before any holdout computation. The frozen
clustering carries seven labels, six of which are populated: five in-sewer clusters
(labels 1–5) and the river-boundary gauge as a single-station cluster (label 7);
label 6 remained unpopulated because the boundary gauge, which has no conduit path to
the network, was assigned its own label before freezing.

**Station-subset ladder.** The station-subset ladder (30/15/5 stations) used in the
observation-density diagnostic at the anchor event follows the registered
nested-subset design: the 30-station full set, a 15-station quality-ranked subset,
and a 5-station subset, each defined before any holdout computation.

---

## Supplementary Note S4 — Twin construction details (companion to §3.3)

**Truth vectors.** Five truth vectors (t₁–t₅) span the prior box θ ∈ [0.009, 0.022]³
(prior width Δ = 0.013, uniform prior, §3.1): interior corners (all-low 0.010;
all-high 0.021), a mixed extreme placed exactly on the prior-support corners
(0.009, 0.022, 0.009), a gradient (0.012, 0.016, 0.020), and the center
(0.0155; t₅). The prior box is the shared frozen constant `PRIOR = (0.009, 0.022)`
in the archived analysis library, used identically by the twin and real-data
inversions.

**Structural-mismatch scenarios.** Two scenarios impose a discrepancy between data
generation and the inversion family, both at the central truth t₅: m1 — the
generation-side dry-weather-flow multiplier is 0.5 while the inversion assumes 1.0;
m2 — subcatchment widths are inflated by 10% on the generation side. Noise is
injected under E2 so that any degradation is attributable to structural mismatch
rather than noise-model mis-specification.

**Noise construction (common random numbers).** For each (truth run, seed), a single
innovation field ξ ~ N(0,1) and a single station-bias vector b ~ N(0, 0.10²) are
drawn; the three error structures are derived as exact nestings of the same
realization — E0: e = σᵢξ; E1: e = AR(1)(h·σᵢξ); E2: E1 + b — where h is a
depth-dependent heteroscedastic scale, h_i(t) = 0.1 + 0.9·clip(y_i(t)/d90_i, 0, 2),
with y_i(t) the true simulated depth and d90_i the 90th percentile of station i's
valid observed depths (h ≈ 0.1 at low water, h = 1 at d90_i, capped at 1.9). E0
noise is unscaled white noise σᵢξ and therefore carries no depth scaling, so
marginal noise variances are not exactly matched across structures — E0 noise is
relatively larger at low depths. Noise parameters are taken from the real residuals
of the anchor event (per-station σᵢ; median lag-1 autocorrelation), so the twins
inherit the actual error magnitude of the data.

**Run accounting.** 7 SWMM truth runs (5 truths + 2 mismatch generations) are reused
across generation models and seeds; 68 synthetic datasets (60 acceptance-matrix +
8 mismatch); 204 posterior evaluations (180 matrix + 24 mismatch); with the
pre-registered seed expansion, 270 acceptance-matrix and 294 total synthetic
inversions.


**Reporting history and secondary diagnostics (main-text v1, retained here for auditability).**

At the initial sample size (n = 20
runs per diagonal cell), the same rule returned eight PASS and one MARGINAL
(E1–n₂ then 0.75, PASS; E2–n₁ 0.60, MARGINAL). The pre-registered expansion to ten
seeds moved E1–n₂ from PASS to
MARGINAL (0.75 → 0.680) and left E2–n₁ MARGINAL (0.60 → 0.640). No cell changed to
FAIL. The acceptance decision and the frozen thresholds are based on the
expanded matrix.

Posterior intervals calibrated with one
error structure and evaluated on data generated with another cover the truth as
follows (pooled over the three parameters; n = 20 runs per off-diagonal cell): when
data generated under E1 or E2 are inverted with E0, coverage falls to 0.633 and 0.517;
when data generated under E0 are inverted with E1 or E2, coverage is 0.967 and 0.983;
the E1↔E2 cross cells lie at 0.800–0.817. The corresponding diagonal pooled values
are 0.927 (E0), 0.800 (E1), and 0.773 (E2). Off-diagonal cells were not extended by
the seed expansion and rest on n = 20 runs each; we report them as observed values
without interval adjudication.

Descriptive mismatch-scenario results from the
same stage (not part of the pre-registered acceptance battery; indicative at
n = 12 per scenario): under physically mismatched generation with E2 noise,
per-parameter coverage was 0.50/0.25/0.75 (m1) and 0.67/0.67/0.67 (m2), and the
parameter-recovery criterion was met in 3 of 8 checks.

Across all 810 (run × parameter)
combinations of the 3×3 matrix on truth cases, the grading statistic R_j has median
0.921 and per-run range 0.104–0.998; low values concentrate in isolated case–seed
combinations (e.g., n₁ under E1/E2 for one truth triplet) rather than in whole
matrix cells. The threshold
freeze was executed after the acceptance decision reported above and before
any real-data computation; the C1–C4 labels therefore carry the
diagnostic status stated in §3.3.

---

**First-round defect and audit trail (companion to §3.3).** Under the pre-registered invalidation rules, the twin gate caught one fatal defect: first-round twin posteriors were near-flat (posterior sd ≈ prior sd, R_j ≈ 0). Root-cause analysis exposed a grid-unit mismatch in the emulator reconstruction — physical θ values were passed directly into the normalized GP whose inputs are unit-scaled — so the emulator surface, and with it every first-round twin and real-data posterior, was invalid. The affected results were discarded and archived (audit trail in the archived repository), the reconstruction fixed and verified (emulator truth error RMS 0.006–0.012 m on the 24 validation runs), and the full twin and real-data sequence re-run from the corrected emulator. The near-flat signature itself is what the gate is designed to catch: a correct emulator must transmit some parameter information on the high-information anchor forcing, so near-zero R_j is evidence of a reconstruction fault rather than of genuine non-identifiability.

## Supplementary Note S5 — High-excitation twin replication (AMP = 2.5)

**Design.** The registered twin (§3.3 of the manuscript) inherits the anchor event's
forcing. This arm repeats the full pipeline with the anchor rainfall multiplied by a
factor of 2.5: peak 15-min intensity 7.1 mm, event total 96 mm. All other design
elements are held fixed: the same 90-point Latin-hypercube emulator design (27
parameters), the same 24-point validation set and W-map construction, the same noise
injection (per-station σᵢ from anchor residuals, median σᵢ = 0.0794; shared
innovation fields; φ_gen = 0.82; τ_b = 0.10), the same five truths t₁–t₅, the two
mismatch scenarios m1/m2 at t₅ (E2-generated), two seeds {0, 1}, and the frozen A25
grading thresholds (Rⱼ ≥ 0.5 and info ≥ 0.1). Total 121 SWMM runs (90 LHS + 7
truth/mismatch + 24 validation); all completed without failure.

**Results** (34 cases × 3 inversion structures = 102 case-by-parameter cells):

- Diagonal identification rate (generation model = inversion model, A25 flags):
  1.00 for all nine (structure × parameter) cells under the 2.5× forcing. Matched-seed
  comparison at the registered forcing: E0 0.90/1.00/1.00, E1 0.90/0.90/1.00,
  E2 0.857/0.929/1.00 (n₁/n₂/n₃).
- Cross-structure grading flips (a parameter's A25 flag differing across E0/E1/E2
  inversions of the same synthetic data): **0 of 102 cells**, versus grading
  disagreement observed on the real events at anchor forcing (6/33 cells, §4.2).
- Structural mismatch: all mismatched-inversion Rⱼ ≥ 0.98 (per-parameter minimum
  0.980), versus −0.173 to 0.998 at the registered forcing; the mismatch signature
  (Rⱼ degradation on the affected parameter, and two n₁ cells dropping below
  threshold) disappears at 2.5× excitation.

**Interpretation.** Raising the excitation level removes the cross-structure grading
disagreement entirely and simultaneously masks the structural-mismatch signature:
high-information regimes are permissive for grading and blunt for ranking-based
mismatch diagnostics alike. The registered anchor forcing is therefore not a worst
case for grading stability — it is the regime in which error-structure sensitivity
is expressible at all. All scripts and outputs of this arm are in the archived
repository.

---

## Supplementary Note S6 — Rainfall-perturbation design details (companion to §3.4)

**LOO calibration of perturbation magnitudes.** Dropping each gauge shifts
event-total areal depths by up to 7.69% (per-event standard deviation 0.72–3.48%,
median 2.25%), the same order as the audited correction at the anchor.

**Stratified realization set.** For each family, perturbations are at the central
scale of both signs plus one double-magnitude realization:
ε ∈ {−0.0225, +0.0225, −0.0450} under a normal family N(0, 0.0225²) centered on the
median LOO spread, and ε ∈ {−0.077, +0.077, +0.0385} under a uniform family spanning
the LOO envelope U(±0.077); signs are balanced in aggregate across the two families
but not within each.

**Event-level linearized propagation.** Across all 11 events,

$$\Delta \theta = (J^{\mathsf T} C^{-1} J + P^{-1})^{-1} J^{\mathsf T} C^{-1}\, \delta y,$$

where J is the GP Jacobian at the event's posterior mean (finite differences in
normalized space), C = diag(s²* + W) the local covariance, a common diagonal
approximation across the three structures (E1's AR(1) correlation and E2's rank-one
offset covariance are not carried into C) chosen so that the three sensitivity axes
share one prior-width scale, P the prior covariance, and δy the simulated depth
shift; the linearization's fidelity is checked against the anchor's full
re-inversions, and the marginalization sensitivity check (§4.2) bounds the effect of
the nuisance-treatment differences.

---

## Supplementary Note S7 — Computational reproducibility and model assets (companion to §2.1 and §3.1–§3.5)

All stochastic and discretization choices are fixed and archived. The profile grids
of §3.2 — the eight-point geometric s² grid and the five-point τ grid — are anchored
by the same rule in every analysis arm (full-network calibration, LOEO, LOCO, and
transfer), with the anchor always the residual scale of the design-mean simulation at
the current grid level, so no arm receives a sharper nuisance grid than another. GP
hyperparameters involve no gradient-based optimization: for each principal-component
score, the lengthscale is selected from a fixed five-point candidate grid by
leave-one-out predictive error, and the noise variance is set to that error's mean
square. The 90-point Latin hypercube design is archived in full as the run manifest,
so the emulator input set is exactly reproducible without re-sampling; prediction
sampling, event bootstrap (B = 10 000), and the Monte-Carlo error propagation of §3.6
each draw from a fixed seed recorded in the archived scripts. The bootstrap-based
point estimate carries a Monte-Carlo sampling error of 8.5 × 10⁻⁵ m (§4.4), which
bounds the reproducibility of the reported interval under seed changes.

---

**GP emulator hyperparameters (companion to §3.1; fixed, no gradient-based optimization; fixed candidate grid with leave-one-out selection).** Matérn 5/2 length scale 0.4 in the unit cube; signal variance equal to the empirical variance of the standardized training outputs; nugget (10⁻³ × output sd)² for factorization stability.

**Profile grids and fixed nuisance values (companion to §3.2).** E2 τ five-point grid: {0.02, 0.05, 0.10, 0.20, 0.40} m. φ_i clipped to [0, 0.95]; stations with fewer than 30 valid lag-1 pairs retain the default 0.5.

**Pilot model asset inventory (companion to §2.1).** The model file contains 2,290 conduits, of which 2,274 are circular conduits retained in the pilot subnetwork and 16 are dummy links at virtual confluence nodes; 82 free pseudo-outfalls are placed at dead-end nodes.

## Supplementary Note S8 — C1–C4 decision rules per analysis unit

This note gives the full classification rule table referenced in §3.7. Each
analysis unit is an event *e* × parameter *j* pair (11 × 3 = 33 units). Evaluation
is strictly sequential: the first satisfied row determines the class; a C2 unit
then receives exactly one sub-label, with grading non-unanimity (2a) taking
precedence over drift (2b). Grading status refers to the full-station calibration
posterior of each error structure against the twin-calibrated thresholds (§3.1);
drift is the cross-structure posterior-mean range expressed in prior widths (§3.6);
prediction improvement is the LOEO holdout median across structures (§3.5).

**Table S2.** C1–C4 decision rules per analysis unit (event *e* × parameter *j*).

| Step | Condition | Label | Interpretation |
|---|---|---|---|
| 1 | All three structures pass grading ∧ drift < 1 ∧ prediction improvement ≥ 5% | **C1** | Robustly identifiable and prediction-relevant |
| 2 | At least one structure passes grading (all other cases) | **C2** | Conditional — see sub-labels |
| 2a | └ Grading non-unanimous across structures | C2 · grading flip | Pass/fail depends on the error structure |
| 2b | └ Grading unanimous ∧ drift ≥ 1 | C2 · drift-dominated | Posterior location depends on the error structure |
| 2c | └ Grading unanimous ∧ drift < 1 | C2 · grading-stable, prediction-deficient | Grading passes but prediction improvement < 5% |
| 3 | No structure passes grading ∧ prediction improvement ≥ 5% | **C3** | Not identifiable yet prediction-relevant |
| 4 | No structure passes grading ∧ prediction improvement < 5% | **C4** | Neither identifiable nor prediction-relevant |

**Exhaustiveness check.** The rules are exhaustive and mutually exclusive: every
combination of (grading pattern × drift × prediction outcome) maps to exactly one
label, which was verified by enumeration before any unit was classified. An
independent re-implementation of Table S2 reproduced all 33 labels without
discrepancy.

**Mapping to registered artifacts.** The registered artifacts predate the
paper-facing sub-label names and carry filenames that are literal Chinese
translations of them; the one-to-one correspondence is recorded in the archived
integrity log, together with the pre-declared sub-label mechanism (A30-3:
partition within C2 without re-labeling). In the archived classification output,
6 units are grading flips, 24 drift-dominated, and 3 grading-stable/prediction-deficient;
classes C1, C3, and C4 are empty by outcome, not by construction (C1–C4 are
reachable in general).

---

## Supplementary Note S9 — Control-arm tail percentiles and construction-sensitivity checks (companion to §4.3)

Corrected forcing arm (63 cells): median 0.88%, 90th percentile 9.3%, maximum 26.0%
of the prior width. Rainfall perturbation arm (54 cells): median 9.2%, 90th
percentile 26.5%, maximum 42.6%.

Emulator error map W localization: 95th percentile of cell sd ≤ 0.032 m for every
event. Anchor-event cluster bootstrap, flexible structures: with all stations retained,
E1–n₂ P(R_j ≥ 0.5) = 0.640 (500 resamples; the value quoted in §4.3); with boundary
clusters excluded, 0.628 (500 resamples). ev08 cross-structure drift: per-parameter
maxima of 0.7–2.3 posterior sd (1.1–18.6% of the prior width), all means interior to
the prior.

**Construction-sensitivity checks (companion to §4.3).** Replacing the profile likelihood with full-grid marginalization over s² and τ (log-uniform weights; τ grid extended downwards to 0.005) changed no grading label across the 33 units, and boundary hits on the τ grid were absent (median mass 0.0%). Re-evaluating the six flipped units on a global 41³ grid without box shrinkage likewise changed no label (posterior mass outside the shrunken box ≤ 4.1%; posterior-mean shift ≤ 0.9% of the prior width). Varying the acceptance cut from R_j ≥ 0.5 to 0.45 or 0.55 leaves the flip count at 6 and 7 respectively, with all six registered flips present in both bands. A data-coverage check excludes the reduced-coverage event ev05 (17 of 30 stations) entirely: 5 of the remaining 30 units flip (the only lost flip being ev05–n₂'s own), and the median cross-structure drift is 66.2% of the prior width (90th percentile 96.3% vs 96.8% over all 33 units), so the sensitivity result is not driven by the least-covered event.

---

## Supplementary Note S10 — Literature lineage supporting the Introduction (companion to §1)

This note records the literature context moved out of the Introduction (which is
held to a single printed page); no analysis or number in the main text is affected.

**Practice and feasibility strand (companion to §1, sentence with Barco/Crowley/Razavi).**
The audit design presumes three operational substrates that the urban-drainage literature
has built independently: network-scale automatic calibration practice for SWMM
(Barco et al. 2008, large-catchment automatic calibration with multi-objective
refinements since), information-theoretic sensor placement that justifies dense
water-level sensing (Crowley et al. 2025), and surrogate modeling that makes
large simulation ensembles affordable (Razavi et al. 2012, the standard review of
surrogate methods across water resources; emulator-based inference of the kind used
here is one instance). None of these strands addresses error-structure sensitivity of
identifiability grading itself, which is the gap the main text takes up.

**Practical-identifiability methods.** Practical-identifiability methodology has
advanced considerably: tracer-based information analysis in transient storage models
(Bonanno et al. 2022), re-examination of how few parameters remain identifiable even
under data-rich conditions in a soil organic matter model (Van de Broek and Six 2026),
and screening-then-identification workflows for SWMM combining Morris screening with
GLUE (Zhong et al. 2023). These methods share the operational premise audited in the
main text: identifiability is assessed under a single assumed error model. The epistemic
alternative — behavioral reasoning under equifinality — originates with GLUE
(Beven and Binley 1992; Beven 2006) and remains in active methodological development,
for example in diagnostic-embedded samplers that calibrate and diagnose simultaneously
(Wu et al. 2025).

**The error model as a modeling decision.** In catchment hydrology, the
formal-likelihood lineage spans three decades: MCMC-based parameter uncertainty
assessment (Kuczera and Parent 1998), Bayesian total error analysis separating input,
model and observation error (Thyer et al. 2009), likelihoods for correlated,
heteroscedastic and non-Gaussian residuals (Schoups and Vrugt 2010), and analyses of
the pitfalls of jointly inferring heteroscedasticity and autocorrelation (Evin et al. 2013, 2014). Residual heteroscedasticity and correlation are widespread in practice
(Lombardo et al. 2025); accommodation remains contested between formal correlated
likelihoods (Herrera et al. 2022), GLUE-type behavioral reasoning (Beven and Binley 1992; Beven 2006, 2025), and diagnostic-embedded samplers (Wu et al. 2025).

**From calibration diagnostics to prediction.** Split-sample calibration–evaluation
remains the default transfer test (Guilpart et al. 2021; Hassan et al. 2026), with
probabilistic verification metrics such as the CRPS (Gneiting and Raftery 2007;
Huang and Zhao 2022) as its tooling. Surrogate-accelerated calibration (Tanim et al.
2024) and scalable variational inference coupled to differentiable hydrology
(Nguyen et al. 2025) inherit the same premise: gains are claimed at calibration
time under an assumed error model, and their transfer to unseen events is rarely
audited. Pool et al. (2026) show that additional calibration constraints can even
degrade predictive uncertainty; those tests, however, were run at catchment scale,
not in urban networks.

**Table S3.** Positioning of this study relative to the urban-drainage and nearest
catchment studies of error-model effects (companion to §1).

| Axis | Del Giudice et al. (2013; 2015) | Cedillo et al. (2024) | This study |
|---|---|---|---|
| Domain | urban drainage network | river reach | urban drainage network |
| Question | error-model choice vs predictive reliability; bias redistribution with model complexity | likelihood choice vs calibrated effective roughness | error structure vs identifiability grading and held-out skill |
| Practical-identifiability grading | not assessed | not assessed | per-cell grading under twin-calibrated thresholds |
| Held-out prediction test | validation periods (not pre-registered) | not performed | pre-registered leave-one-event-out with frozen CRPS criterion |
| Mechanism arm | — | — | per-station datum-offset correction |

---

## Supplementary Note S11 — Inference-layer robustness arms and secondary verification metrics (companion to §4)

This note documents five post hoc, pre-submission robustness arms (all inference-layer
recomputations on frozen archives; no new hydraulic simulations) and the secondary
verification metrics requested for complete reporting. All scripts and outputs are in
the archived repository (`p5_pc1`–`p5_pc7`; `output/p2/gate10/`).

### S11.1 Positive control for the RQ2 detection pipeline

The pre-registered ≥5% CRPS-improvement threshold is a detection criterion; a negative
result is informative only if the pipeline can detect an improvement of that size when
parameter information is genuinely present. We therefore ran the registered
prior-versus-posterior scoring (the §3.5 predictive pipeline, in-sample analogue of the
LOEO worker) on the registered twin with synthetic observations: 5 truth triplets × 3
generation structures × 2 innovation seeds (seeds shared with the high-excitation arm of
Note S5), each scored under all three structures (90 scored units; diagonal units
discussed here).

**Anchor forcing (AMP = 1).** Even though grading passes in every diagonal twin unit,
the median within-event CRPS improvement is 3.1%, and only 5 of 30 diagonal units
(16.7%) reach 5% (by truth triplet: t₁ 0/6, t₂ 3/6, t₃ 1/6, t₄ 1/6, t₅ 0/6). This
matches the range observed on the real events, whose in-sample posterior-versus-prior
improvements have median 0.2% and maximum 3.8% (11 events × 3 structures). At anchor
excitation the ≥5% threshold is thus rarely attainable even with identifiable
parameters — the threshold is demanding relative to the share of predictive variance
attributable to parameter uncertainty.

**High-excitation forcing (AMP = 2.5; the Note S5 arm).** The identical pipeline
detects the improvement: median 7.8%, 18 of 30 diagonal units ≥5%. All 18 units of
the three informative truth triplets reach the threshold (medians 8.9–9.4%), while
the moderately separated t₄ and the prior-center truth t₅ do not (0/6 each; medians
2.3% and 0.5%) — an embedded negative control that behaves as expected. Excitation
alone moves the median diagonal improvement from 3.1% to 7.8%.

Together these arms validate the detection pipeline (it reports ≥5% precisely when
forcing makes the parameter signal dominant in the predictive decomposition) and
support the reading that the 0/33 held-out result reflects the information content of
the observed events rather than insensitivity of the scoring pipeline.

### S11.2 W (emulator-error floor) sensitivity

The likelihood floor W = diag(σ²_emu) down-weights observations where the surrogate is
least trusted (§3.2); if it were severe enough it could create a floor effect on
identifiability independent of the error structure. We re-ran the anchor (ev00) and
ev08 inversions under all three structures with (i) W capped at its per-event median
over QC-valid cells (3.6 × 10⁻⁵ m for ev00, 2.8 × 10⁻⁷ m for ev08) and (ii) W removed
entirely (W = 0, the information upper bound); the two arms give near-identical
results. No grading label changes in either arm in either event, and every R_j is
unchanged or increases (by 0 to 0.17, largest for ev00 under E1), so the grading
sensitivities of §4.2 are not an artifact of observation down-weighting — removing the
floor only adds information.

### S11.3 Flow-regime stratification of the datum offsets

The datum-offset attribution (§4.5) treats the removed per-station offsets as
time-invariant sensor datum error; a flow-dependent structural bias would instead grow
with stage. Stratifying each event's valid timesteps by the baseline water level
(within-station median split, plus a lower-vs-upper quartile contrast) shows the
offsets to be essentially flow-independent: across the 11 events the median
|high-flow − low-flow offset| is 0.28–0.76 cm against median |offset| 3.7–7.2 cm
(5–10% of the offset magnitude; 57–93% of stations per event within half the offset),
and the quartile contrast stays below 1.9 cm. As a second check, the E2 structure's
per-station offset term has an analytic posterior mean at the maximum-likelihood
parameter (b̂ᵢ = τ²cᵢ/(1 + τ²aᵢ) at the profile-optimal τ, the conjugate update of the registered
Sherman–Morrison marginalization); at ev00 and ev08 these recovered biases match the
datum offsets closely (Spearman ρ = 0.83 and 0.98; Pearson r = 0.87 and 0.99; median
absolute difference 0.96 and 0.61 cm). The additive-offset reading of §4.5 is
therefore "consistent with" in a quantitative sense: the removed quantities behave as
datum shifts, and E2's absorbed offsets are the same quantities.

### S11.4 Common-random-number (CRN) paired differences

The registered Monte-Carlo error propagation (§4.4) drew posterior and prior predictive
residuals with independent seeds. A CRN variant that shares the standardized residual
fields (white-noise field for E0; AR(1) field and per-draw offset innovations for
E1/E2) between the posterior and prior predictive draws reproduces the pooled paired
change almost exactly (+0.000675 m against +0.000677 m independent-seed) while
reducing per-unit Monte-Carlo standard errors 10–20-fold (E0 4.0 × 10⁻⁵ → 2.0 × 10⁻⁶;
E1 3.9 × 10⁻⁵ → 3.8 × 10⁻⁶; E2 2.6 × 10⁻⁴ → 2.6 × 10⁻⁵ m). Under CRN the paired change
is positive (degradation) in 29 of 33 units and in 10 of 11 events under E2. The E2
degradation is therefore systematic rather than Monte-Carlo noise.

### S11.5 Acceptance-matrix robustness: by-truth clustering and a variance-matched E0 arm

The n = 50 runs per acceptance cell (§4.1) group into 5 truth triplets, so a simple
Wilson interval may understate uncertainty if truth triplets differ systematically. A
by-truth cluster bootstrap (resampling the 5 triplets, B = 20,000) widens several
intervals — most notably E2–n₁ from [0.501, 0.759] to [0.44, 0.84] — but no cell drops
status: E2–n₁, MARGINAL under Wilson, passes under the cluster interval (upper bound
0.84, above the 0.80 PASS boundary), leaving eight PASS and one MARGINAL (E1–n₂,
[0.62, 0.74]) and no FAIL; the E1–n₁ PASS is confirmed ([0.62, 0.92]). The by-truth
intra-class correlations range from 0.00 to 0.63. Because the cluster intervals are
equal or wider on the failing side, the two MARGINAL flags of §4.1 remain the
conservative reading, and we retain them in the main text.

Second, the registered E0 generation uses homoscedastic white noise, so E0's coverage
could in principle reflect a variance composition advantage rather than the absence of
memory and offset terms. Repeating the full 50-run design with a variance-matched E0
generation — white noise scaled by the same depth-dependent factor
(h = 0.1 + 0.9 d̂/d₉₀) used in the E1/E2 innovations, matching their pointwise
variance — reproduces the registered E0 coverage row almost exactly (hits 46/44/46 of
50, against 46/44/49; all three cells PASS under the registered rule; diagonal grading
rate 1.00, with E1/E2 inversions of the same observations at 0.96–1.00). E0's
acceptance therefore does not depend on the variance composition of its generation
arm, and the E1/E2 differences are attributable to the memory and offset components of
the error structure rather than to variance mismatch.

### S11.6 Secondary verification metrics (LOEO held-out predictions)

Table S4 assembles the registered secondary metrics for the 33 LOEO held-out
comparisons (per-fold values in the archived repository,
`secondary_metrics.csv`). The picture is consistent with §4.4: posterior predictive
distributions differ little from their priors in CRPS, with E2 purchasing its
near-nominal 90% coverage with visibly wider intervals.

**Table S4.** LOEO held-out secondary verification metrics (median across the 11
folds, with [minimum, maximum]; posterior predictive distributions).

| Metric | E0 | E1 | E2 |
|---|---|---|---|
| CRPS (m) | 0.0733 [0.0549, 0.1028] | 0.0737 [0.0540, 0.1042] | 0.0744 [0.0585, 0.1026] |
| Coverage 50% | 0.556 [0.497, 0.666] | 0.520 [0.466, 0.614] | 0.586 [0.532, 0.717] |
| Coverage 90% | 0.857 [0.782, 0.927] | 0.834 [0.730, 0.923] | 0.887 [0.820, 0.928] |
| Width 90% (m) | 0.371 [0.369, 0.378] | 0.324 [0.311, 0.334] | 0.493 [0.454, 0.503] |
| PIT mean | 0.640 [0.628, 0.683] | 0.639 [0.628, 0.687] | 0.628 [0.616, 0.658] |
| PIT outside [0.025, 0.975] | 0.124 [0.072, 0.184] | 0.148 [0.073, 0.233] | 0.064 [0.018, 0.161] |
| Brier (p90 threshold exceedance) | 0.176 [0.139, 0.201] | 0.184 [0.143, 0.207] | 0.173 [0.143, 0.198] |
| Peak error, median signed (m) | 0.183 [0.079, 0.219] | 0.150 [0.047, 0.191] | 0.053 [−0.112, 0.085] |
| Peak error, MAE (m) | 0.184 [0.160, 0.334] | 0.165 [0.136, 0.364] | 0.115 [0.073, 0.347] |
| Peak-timing error, median (steps) | +1 [−5, +3] | +1 [−5, +3] | +1 [−5, +2] |

Prior-predictive reference values (same folds, medians): CRPS 0.0729/0.0731/0.0737 m;
coverage 90% 0.858/0.836/0.876; width 90% 0.382/0.326/0.431 m; Brier
0.1835/0.1844/0.1845 (E0/E1/E2).

---

## Supplementary references

Beven KJ (2006) A manifesto for the equifinality thesis. J Hydrol 320:18-36. https://doi.org/10.1016/j.jhydrol.2005.07.007

Beven KJ, Binley A (1992) The future of distributed models: Model calibration and uncertainty prediction. Hydrol Process 6:279-298. https://doi.org/10.1002/hyp.3360060305

Bonanno E, Blöschl G, Klaus J (2022) Exploring tracer information in a small stream to improve parameter identifiability and enhance the process interpretation in transient storage models. Hydrol Earth Syst Sci 26:6003-6028. https://doi.org/10.5194/hess-26-6003-2022

Guilpart E, Espanmanesh V, Tilmant A, Anctil F (2021) Combining split-sample testing and hidden Markov modelling to assess the robustness of hydrological models. Hydrol Earth Syst Sci 25:4611-4629. https://doi.org/10.5194/hess-25-4611-2021

Hassan ZU, Su J, Wang D, Tang L, Hou Y, Huang W, Zhang J, Zhu Z (2026) Comparative evaluation of parallel optimization algorithms for urban drainage modeling using OSTRICH-SWMM. Front Water 8:1786203. https://doi.org/10.3389/frwa.2026.1786203

Huang Z, Zhao T (2022) Predictive performance of ensemble hydroclimatic forecasts: verification metrics, diagnostic plots and forecast attributes. WIREs Water 9:e1580. https://doi.org/10.1002/wat2.1580

Tanim AH, Smith-Lewis C, Downey AR, Imran J, Goharian E (2024) Bayes_Opt-SWMM: A Gaussian process-based Bayesian optimization tool for real-time flood modeling with SWMM. Environ Model Softw 179:106122. https://doi.org/10.1016/j.envsoft.2024.106122

Wu S, Tetzlaff D, Beven K, Soulsby C (2025) DREAM(LoAX): simultaneous calibration and diagnosis for tracer-aided ecohydrological models under the equifinality thesis. Water Resour Res 61:e2024WR038779. https://doi.org/10.1029/2024WR038779

Zhong B, Wang Z, Yang H, Xu H, Gao M, Liang Q (2023) Parameter optimization of SWMM model using integrated Morris and GLUE methods. Water 15:149. https://doi.org/10.3390/w15010149
