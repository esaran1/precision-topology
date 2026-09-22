# Registration: the 5D expressivity threshold at k = 1 (fresh registration)

**Written before any stage-1 run of this block.** Date: 2026-09-22, ~19:50 local
(EDT). Everything below — construction, network, depth, protocol, success
criterion, grid, seeds, onset definition, two-stage rule and predictions — is
fixed here and not revised.

## Why this registration exists

The original registration (`blockA5d_prediction.md`, k = 10 copies) ran and hit
its stop condition (`blockA5d_k10_results.md`, T72): no perfect run at any `a`
or budget, so A2 and A4 failed, A1 is uninformative, and the threshold was
unobservable because capacity dominated. **That result stands as reported.**
This is a **new** registration of the same experiment at **k = 1**, gated by a
capacity pilot, made after seeing the k = 10 failure.

## Depth: chosen by the pilot

Pilot (`blockA5d_k1_pilot.csv`, `src/blockA5d_k1.py pilot`): k = 1, targeted
thickening `ρ = 0.5`, 64,000 steps, seeds 0–9; positive controls GELU and
`a = 3.0`, barred control ReLU; depths 5 and 8. Pass rule: a positive control
reaches zero held-out errors in ≥ 1 seed at some depth; take the smallest
passing depth.

- **Depth 5 passes.** Zero uniform held-out errors at 64k: GELU seed 4; `a = 3.0`
  seeds 2, 4, 5, 8. ReLU: minimum 121 errors, no zero.
- **Re-checked on the stratified set before choosing** (`blockA5d_k1_stratcheck.csv`;
  the reruns reproduce the pilot's uniform error counts exactly, 32 of 32): all
  five passing seeds have **0 stratified errors and 0 errors on 200,000 fresh
  uniform points** at 64k.
- **Depth = 5**, chosen because it is the smallest pilot depth at which a
  positive control is perfect on both held-out sets. It equals the original
  registration's depth (Ren–Lim's five hidden blocks). The depth-8 pilot runs are
  reported but did not enter the choice.

## Construction, network, protocol

As in `blockA5d_prediction.md`, except **k = 1**: Ren–Lim's generator (their
`generate_multi_copy_dataset`, reimplemented call for call and byte-identical on
4 seeds), one linked pair `S² ⊔ S² ⊂ ℝ⁵`, targeted thickening `ρ = 0.5` (solids
disjoint for `ρ < 1`), 10,000 training points (their per-copy base), five hidden
blocks `Linear(5,5) → act`, one-logit BCE readout, full-batch Adam lr 1e-3,
float32, PyTorch default initialisation, one CPU thread per run, budgets
1,000 / 4,000 / 16,000 / 64,000 as checkpoints of one 64,000-step run.
Theorem 4.7's hypotheses hold exactly as checked there (`m + n + 1 = 5 = d`,
width 5, `f_a` non-decreasing iff `|a| ≤ 1`); **the expressivity threshold is
`a = 1`**.

## Held-out sets and success criterion

1. **Uniform**: 20,000 points from the generator, seed `s + 1,000,000`.
2. **Stratified (linking region)**: 20,000 points concentrated at the closest
   approach of the two cores, seed `s + 2,000,000`. An A-core point's distance to
   the B core depends only on `u₁` and is smallest at `u₁ = −1`
   (`2(√2 − 1) = 0.828427`); symmetrically for B at `v₁ = −1`. The set is the
   closest **10% of each core** (`u₁, v₁ ∈ [−1, −0.8]`, distance ≤ 0.8650),
   10,000 points per class, sampled exactly (`u₁` is uniform on `[−1, 1]` for a
   uniform point on `S²`; azimuth uniform), with the generator's targeted
   thickening at `ρ = 0.5`.
   **Size**: the uniform set places ~2,000 points in this band; the stratified set
   places 20,000 — **10× the density** where the cores are closest.
   **Calibration against the barred control (pilot ReLU, depth 5, seeds 0–2, all
   four budgets)**: every one of the 12 checkpoints shows ≥ 35 stratified errors
   (35 and 281 at 64k for the two ReLU runs that train; 10,145 for the stuck one),
   so a barred network with any error mass there shows it.
   *Post hoc observation, recorded for transparency* (`blockA5d_k1_errormap.csv`,
   `src/blockA5d_k1_errormap.py`): a barred network's errors are **not confined** to
   the closest-approach band — one ReLU run's errors lie mostly at `u₁ ∈ [0.2, 0.6]`
   on the far side of core A. This is why the 10× uniform check below is also run.
3. **10× uniform**: 200,000 points, seed `s + 3,000,000`, evaluated at every
   checkpoint with zero uniform errors; the count is reported.

**A run is perfect at budget `B` iff it makes zero errors on both the uniform and
the stratified set.** This criterion defines the onset and every prediction.
The 200,000-point count is reported for every such run; the pilot already shows
it can find errors the two 20,000-point sets miss (`a = 3.0` seed 4 at 1k: 0 / 0 /
2), so the fraction of perfect runs **also perfect on 200,000 points** is reported
as a secondary column. It changes no verdict.

**A1-violation rule.** If any `a ≤ 1` run has zero uniform errors, it is checked
against the stratified set and the 200,000-point sample before it is treated as
an A1 violation. It counts as a violation only if all three are zero.

## Grid and seeding

- 9 values of `a` (0.9, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0) + ReLU + GELU,
  **20 seeds each, seeds 100–119**, which are disjoint from the pilot's seeds 0–9,
  so no stage-1 outcome is known in advance.
- **Two-stage rule**: after stage 1, for each budget, the two grid values of `a`
  that bracket the onset each get **20 more seeds (120–139; 40 total)**, and the
  onset is recomputed on stage-2 counts. If no onset brackets at a budget, no
  seeds are added there.

**Onset**: `a_on(B)` = the smallest grid `a` at which at least half the runs are
perfect, **bracketed** only if a strictly smaller grid `a` has fewer than half.

## Registered predictions (unchanged from `blockA5d_prediction.md`)

- **A1.** For `a ≤ 1` (`a ∈ {0.9, 1.0}`): **zero perfect runs at every budget.**
- **A2.** At each budget a **bracketed onset `a_on(B) > 1`** exists.
- **A3.** `a_on(B)` is **non-increasing in `B`**, and strictly lower at 64,000 than
  at 1,000.
- **A4 (positive control).** At `a = 3.0` and `B = 64,000`, **at least one**
  perfect run.
- **A5 (exploratory, direction only).** With `ε_on = a_on − 1`, fit
  `ε_on ∝ B^{−γ}` over the bracketed budgets; `α_L` = slope of log median
  `‖W_out‖_F` against log `B` over the `a > 1` runs in the onset region.
  Registered **direction only**: `γ` and `α_L/β` (`β = 3/2`) have the same sign,
  both positive.

## Stop and report if

- any `a ≤ 1` run is perfect under the A1-violation rule;
- no onset brackets at any budget.

## Cutoff

If the result cannot be decided by **09:00 local time on 2026-09-23**, what exists
is reported as incomplete and the `paper-submitted-v2` tag is not delayed.

## Not part of this registration

The Appendix G.2 protocol comparison (private, k = 10) is not repeated here.
