# Registered predictions: does the weight-scale mechanism reach GELU? (Task A)

Written after the standard-arm measurement and **before** the scaled arms
run. The measurement matters: it changes what the mechanism itself
predicts, and the registration separates two readings that the outcome
must distinguish.

## The measurement (standard arm, 200 seeds, width 3, depth 5)

- Baseline dense-verified rate: **16/200 = 8.0%**, CI [4.6%, 12.7%].
- **Separators' weight scales are not in the tail**: final spectral-norm
  products of the 16 separators rank at percentiles 0.34–0.86 of the full
  200-run distribution (median 2,546 vs 1,700 for non-separators, heavily
  overlapping; per-layer norms differ by ≤30% at the last layers).
- Standard training grows the spectral product ~10,000× from
  initialization (0.26 → ~2,000); GELU's fold-depth demand is ~19×
  (dip depth 0.17). **Demand ≪ reach** — the opposite of `sin(1.02)`,
  where demand exceeded the reached maximum 7-fold.

## Two readings, registered separately

**R1 — naive extension ("larger scale always helps"):** scaled-up
initialization raises GELU's rate substantially, as it did for sin(1.02).
The task brief's falsification clause targets this reading.

**R2 — the mechanism as actually stated (rate is scale-limited only when
required amplification exceeds reached scale):** GELU is not in the
scale-limited regime, so the mechanism predicts **little or no change**
from scaled-up initialization. Under R2 an unchanged rate is not a
falsification of the weight-scale mechanism — it is the mechanism
correctly predicting its own scope: GELU's rate gap has a different
cause, and the mechanism's domain is the near-threshold regime where
demand outruns reach.

**Registered point expectations:** scaled-up arm 5–13% (unchanged within
noise); scaled-down arm (0.3× per layer) at or below baseline, direction
down if anything, with low confidence — training regrows norms ~10⁴×, so
a 0.3× start may be fully absorbed.

## Detectability, fixed in advance

With 200 seeds per arm against 16/200 baseline: a rise to ≥ ~18% or a
fall to ≤ ~2% is resolvable (two-proportion, 95%); anything in 3–17% is
indistinguishable from baseline. A null here therefore rules out only
R1-sized effects; it cannot detect small shifts, and will be reported
with that limit stated.

## What each outcome means, fixed in advance

- **Rate ≥ 18% under scaled-up:** scale helps even where demand < reach;
  the mechanism generalizes beyond its stated regime (stronger than R2).
- **Rate in 3–17%:** consistent with R2; the scope statement becomes
  "the weight-scale mechanism governs findability where required
  amplification exceeds reached scale — the near-threshold sin regime —
  and GELU's rate gap lies outside that regime." Not a falsification of
  the mechanism; a falsification of R1's universal form.
- **Rate ≤ 2% under scaled-up:** scaled initialization actively hurts;
  informative against both readings' framing and reported as such.
- **Scaled-down arm** is exploratory two-sidedness: a large drop supports
  scale mattering at the low end; no change is absorbed-by-training.

## The substantive difference from the sin intervention, noted as required

The sin intervention initialized at a **construction's** scale pattern;
GELU has no construction (its grid failed), so this intervention uses the
scale pattern of **found** solutions — trained endpoints with ordinary
basins. Initializing at a trained endpoint's scale is a weaker, more
diffuse intervention than initializing at a known solution family's
scale, and that asymmetry limits how directly the two interventions can
be compared.
