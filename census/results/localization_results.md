# Localizing the failure: outcomes (Part 3)

Predictions registered in `localization_prediction.md`. Data:
`localization_traces.csv` (8,192-point traces with 2,048-point convergence
check and per-layer determinants), `localization_errors.csv`,
`localization_distill.csv`.

## Probe 2: linking traces through the best monotonic runs

### A registered prediction was wrong, and the chase that found it

The registration predicted "linking stays at −1 through every layer in
every monotonic trace." The first traces showed signs flipping
(−1 → +1 → −1 → …) in the healthy-measurement regime, which triggered the
registered chase clause. The chase found two things:

1. **The prediction mis-stated the invariant.** A monotonic-activation
   layer is a homeomorphism, but a homeomorphism only preserves |lk|; an
   affine map with **negative determinant** reverses orientation and flips
   the sign. Recomputing per-layer determinants: **every reportable sign
   flip coincides exactly with a negative-determinant layer**, in every
   trace, including rejoining the predicted sign track after unreportable
   gaps. The sign sequence is bookkeeping; |lk| is the claim-bearing
   quantity.
2. **512-point traces under-resolve deep, strongly distorting networks.**
   At the default resolution some deep monotonic layers returned rounded
   values of 0 (and once 2) with apparently healthy minimum distances.
   Doubling and quadrupling the resolution revealed the true minimum
   distances there are below the artifact threshold (curves crushed to
   0.0005–0.02 apart), so those values are unreportable, not evidence.
   All traces below use n = 8,192 with a 2,048-point convergence
   requirement; unconverged or artifact-regime values are shown as `?`.

### The corrected result — prediction borne out in |lk| form

| Run (errors) | lk per layer (converged only) | neg dets | min distance over layers |
|---|---|---:|---:|
| gelu d3 s10 (0) | −1 **0** 0 0 | 2 | 0.147 |
| sin(0.95) d8 s2 (2) | −1 −1 −1 1 ? ? ? −1 ? | 2 | 0.011 |
| sin(0.9) d12 s0 (6) | −1 1 −1 1 ? … ? | 9 | 0.0005 |
| sin(0.95) d12 s19 (6) | −1 1 1 1 −1 −1 ? … ? | 5 | 0.0000 |
| sin(1.0) d3 s4 (6) | −1 1 −1 −1 | 2 | 0.103 |
| sin(1.0) d5 s14 (8) | −1 −1 1 −1 ? ? | 4 | 0.0015 |
| tanh d8 s8 (26) | −1 −1 ? … ? | 2 | 0.0001 |
| tanh d5 s12 (49) | −1 ? … ? | 3 | 0.0002 |

**Across every converged, reportable measurement in every monotonic trace,
|lk| = 1.** No monotonic layer changed the absolute linking number, ever.
The GELU comparison run takes |lk| from 1 to 0 at layer 1 and keeps
distance healthy.

What the monotonic near-misses do instead is now measurable: **they crush
the two components together.** The best runs drive minimum inter-core
distance to 10⁻³–10⁻⁴ — the sin(0.95) d8 s2 run (2 errors) has a
near-singular layer (det ≈ +0.002) at exactly the depth where distance
collapses. The monotonic strategy is compression toward contact: press the
tubes so close that the decision boundary threads between sample points.
That is one mechanism behind the 2-error SGD runs, the 0-train-error CMA
solutions, and the eval-perfect networks that failed dense verification —
sample-level success without regional separation, from a map that never
unlinks anything.

## Probe 1: layerwise distillation — layer 1 is irreplaceable, and MSE is the wrong lens

Teacher: the separating GELU d3 s10 network. Students: tanh, same prefix
depth, 3 seeds each, 4,000 Adam steps of MSE regression; classification
measured through the teacher's frozen remaining layers.

| Replaced prefix | student MSE | errors through teacher suffix |
|---|---:|---:|
| layer 1 | 0.19 | **1023, 1023, 1023** (chance) |
| layers 1–2 | 4.7 | 583–731 |
| layers 1–3 | 21 | **88, 89**, 1000 |

The registered prediction (failure localizes at layer 1) is borne out, with
a sharpening: the relationship between regression quality and functional
damage is **inverted**. The layer-1 student matches its target closest in
MSE and destroys classification completely — the suffix expects folded
coordinates, and the student delivers the unfolded best-approximation. The
full-depth student has 100× the MSE yet lands at 88 errors: the ordinary
monotonic near-miss band. A monotonic network can approximate *what the
GELU network ends with* to near-miss quality; it cannot substitute *the
fold that produced it* at all.

## Probe 3: error geography — partial support

Across the eight near-miss runs: misclassified points sit closer to the
other component's core than typical points (median 0.85–0.90 against
baseline 0.97–0.98, and the effect is in the predicted direction in all 8
runs), consistent with errors concentrating where the tubes approach. The
angular-concentration prediction is **mixed**: three runs concentrate hard
(concentration 0.87–1.00 against null p95 ≈ 0.6–0.7 — essentially all
errors at one angular location), three are above their null but modest,
two are below. Class balance of errors varies from all-A to all-B.
Registered verdict: the closest-approach clustering is supported; the
strong single-cluster picture holds only for some runs. Not a uniform
scatter, and not a single clean geometry.

## What Part 3 adds up to

The failure is localized in two independent senses. Architecturally: the
one thing a monotonic network cannot substitute is the layer-1 fold —
replace anything downstream and you land in the near-miss band, replace
the fold and you land at chance. Geometrically: monotonic networks never
change |lk|; their best runs instead compress the components toward
contact, spending near-singular layers to fake separation at sample level.
Both point at the same object — the fold — as the entirety of what
non-monotonicity buys at width 3.
