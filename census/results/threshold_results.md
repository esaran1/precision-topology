# Monotonicity threshold sweep: outcomes against the registered predictions

4,480 runs: two parametric activation families (13 + 11 parameter values) and
the four fixed activations, widths 3 and 4, depths 3/5/8/12, seeds 0–19,
baseline linked tori, our protocol. Predictions were registered in
`threshold_prediction.md` before implementation and are reported in the order
registered. Data: `threshold_sweep.csv` / `.parquet`.

## The headline

**Zero separations at width 3 anywhere on the monotonic side: 0 in 1,280
runs** — twelve parametric monotonic values (seven in Family A up to and
including the threshold `a = 1`, five in Family B down to and including ReLU
at `α = 0`, plus the two affine anchors) and the three fixed monotonic
activations, in the same sweep with the same seeds. Separations appear only
on the non-monotonic side, first at `a = 1.10` (Family A) and `α = −0.25`
(Family B).

With this sweep the width-3 monotonic zero stands at **0 separations in
4,610 runs** (3,330 prior + 1,280 here), and the monotonic category now
contains twelve parametrically distinct activations rather than three fixed
ones. Per cell here, 0/80 bounds the rate below 4.5% (exact two-sided 95%);
pooled bounds appear in the claims ledger rather than here, since pooling
across conditions needs stating carefully.

## Prediction 1: zero on the monotonic side, transition at the threshold — BORNE OUT at grid resolution

| Family | Last zero (from monotonic side) | First positive | Analytic threshold |
|---|---:|---:|---:|
| A (`x + a·sin x`) | `a = 1.05` | `a = 1.10` (3/80) | `a = 1` |
| B (neg-slope leaky) | `α = −0.10` | `α = −0.25` (3/80) | `α = 0` |

Both transitions lie strictly on the non-monotonic side of their analytic
thresholds, and no monotonic value separates. The transition intervals at
grid resolution are `(1.05, 1.10]` and `[−0.25, −0.10)`. No interpolation is
claimed. Note the CP intervals at the flanking values overlap (0/80 → [0,
0.045]; 3/80 → [0.008, 0.107]), so the location claim is the count-based one
— zero below, positive above — not a rate-difference claim between adjacent
grid points.

Width-3 separation counts, Family A (n = 80 per value):

| a | 0.0 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | **1.0** | 1.05 | 1.1 | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separations | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 3 | 5 | 16 | 22 | 23 |

Family B (n = 80 per value):

| α | −1.0 | −0.5 | −0.25 | −0.1 | −0.05 | **0.0** | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separations | 7 | 3 | 3 | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 |

The rate keeps rising with distance from the threshold, to 29% at `a = 3`
against GELU's 7.5% (6/80) in the same sweep — the depth of non-monotonicity
modulates how often SGD exploits it, not just whether.

## Prediction 2: asymmetric sharpness — BORNE OUT

The sharp edge is on the monotonic side, exactly as registered: the zero is
exact everywhere up to and including both thresholds, while the
non-monotonic side rises gradually — zeros persist at `a = 1.05` and at
`α = −0.05, −0.10` before the first positives. Per the registration, those
small-parameter zeros do not count against the barrier reading; a positive
on the monotonic side would have, and none occurred.

## Prediction 3: floor structure — PARTIALLY NOT BORNE OUT, and this matters

The registered prediction was that monotonic-side values show a hard floor
("no runs below ~9–16 errors on this link"). **That is not what happened.**
Width-3 error bands, Family A:

| a | 0 | 1–5 | 6–8 | 9–15 | 16–25 | >25 |
|---|---:|---:|---:|---:|---:|---:|
| 0.75 | 0 | 0 | 0 | 2 | 8 | 70 |
| 0.90 | 0 | 0 | 1 | 7 | 7 | 65 |
| 0.95 | 0 | **1** | 2 | 4 | 7 | 66 |
| **1.00** | 0 | 0 | 2 | 6 | 3 | 69 |
| 1.05 | 0 | 0 | 2 | 4 | 6 | 68 |
| 1.10 | **3** | 0 | 3 | 6 | 3 | 65 |

Monotonic-side width-3 runs below 9 errors (all of them):

| activation | parameter | depth | seed | errors |
|---|---:|---:|---:|---:|
| sin_family | 0.95 | 8 | 2 | **2** |
| sin_family | 0.90 | 12 | 0 | 6 |
| sin_family | 0.95 | 12 | 19 | 6 |
| sin_family | 1.00 | 3 | 4 | 6 |
| sin_family | 0.95 | 8 | 1 | 8 |
| sin_family | 1.00 | 5 | 14 | 8 |

A monotonic network reached **2 errors out of 2,000** — and still, no
monotonic run in this or any other sweep has reached 0. The ~9-error floor
observed for tanh/ReLU/leaky-ReLU (and 16 under corrugation) is a property
of those activations, not of monotonicity: near-threshold members of Family
A get far closer. **The barrier, on this evidence, sits at exactly zero
errors, not near it.** The band-versus-edge argument in
`corrugation_results.md` survives for the three fixed activations it was
stated about, but does not generalize to the monotonic category — a status
note has been added there.

This refinement cuts both ways and both directions are reported: it weakens
"monotonic networks cannot even get close," and it sharpens "the difference
is confined to exactly zero" — which is the count-based claim the project
leans on, and the form a topological obstruction would take. Whether these
near-miss networks fail on a geometrically meaningful set (the last points
of an incomplete fold) is exactly the Part 3 question, and the 2-error run
(`sin_family a=0.95, depth 8, seed 2`) is the natural specimen.

## Prediction 4: width 4 — BORNE OUT

At width 4 the threshold structure is gone: monotonic-side values separate
broadly (428/1,280 runs; e.g. `a = 1.0`: 59/80; `α = 0.5`: 37/80), rates
rise smoothly across both families with no feature at the thresholds, and
the affine anchors (`a = 0`, `α = 1`) separate **0 times in 320 runs** at
either width, with identical error distributions at widths 3 and 4 — as an
affine map must. ReLU (`α = 0`) separates 2/80 at width 4, matching its
known width-4 weakness (5.7% on the corrugated grid), which is a dying-unit
effect, not a monotonicity one.

## Prediction 5: the two families agree — BORNE OUT to within one grid step

In matched units — the most negative derivative the activation attains —
the onset lies in `(−0.05, −0.10]` for Family A and `(−0.10, −0.25]` for
Family B: adjacent intervals, not identical, and the grid cannot resolve
finer. The shared structure is that both families need a finite depth of
non-monotonicity, roughly 0.1–0.25 in derivative units, before SGD exploits
it at width 3.

A post-hoc observation, marked as such because it was not registered: GELU's
most negative derivative is **−0.129** (at `x ≈ −1.41`), inside that onset
window — consistent with GELU separating rarely (6–7.5% across sweeps)
compared with `a = 2` Family A members (28%) whose dips are much deeper.

## Confounds, as registered

**Family A is clean.** Median final loss and gradient norm vary smoothly
across `a = 1`; inactive-unit fraction is 0.0 at every value; no run
finishes at chance. The separation onset coincides with no optimization
discontinuity.

**Family B has a trainability singularity exactly at its threshold.** At
`α = 0` (ReLU): median final loss 0.693 (= log 2, chance), 49/80 runs at
chance, 50% of units inactive — the dying-ReLU collapse, with zero gradient
on the negative side. Flanking values (α = ±0.05) are healthy (≤3 runs at
chance, ≤2% inactive). This is the registered "transition coinciding with an
optimization discontinuity" condition, triggered for the threshold point
itself, so **Family B's evidence rests on its healthy monotonic flank
(α = 0.05–0.5, all zero separations, no pathology), not on α = 0.** Family
A, which has no such singularity anywhere, carries the within-family
threshold comparison on its own.

## Harness redundancies

- `pwl_family(0)` retrained the same configuration as the fixed ReLU
  baseline: 153/160 runs bit-identical in final eval accuracy. The 7 that
  differ (≤1% accuracy) trace to the autograd subgradient convention at
  exactly `x = 0` (`F.relu` uses 0, `torch.where` uses 1); neither version
  separates anywhere at width 3.
- The two identity anchors (`sin_family(0)` and `pwl_family(1)`) produced
  **exactly** equal accuracies run-for-run, as two implementations of the
  same affine network must.
- The fixed baselines reproduce their known behaviour: tanh width-3 minimum
  26, ReLU at-chance collapse, GELU 6/80 width-3 separations.

## What this sweep cannot distinguish

Everything observed is consistent with two readings: (i) monotonicity is the
operative property and SGD needs a finite margin past the threshold to find
the fold; (ii) the operative variable is the *depth* of non-monotonicity
with a practical critical value around 0.1–0.25, and the theorem's binary
threshold is the limiting boundary of that. Both predict exactly zero on the
monotonic side, which is where all the data is. Distinguishing them needs
evidence that separation is *representable* arbitrarily close above the
threshold even where SGD fails — which is the Part 2 (constructed witness /
harder search) question, not a sweep question.

---

> **Status note (2026-08-22, dense verification).** `dense_check.md` checked
> every width-3 separation in this sweep on 100,000 fresh points. All
> separations at the transition-defining values survive — `a = 1.10`: 3/3,
> `a = 1.25`: 5/5, all 13 pwl-family separations, all 6 GELU — so **the
> transition intervals reported above are unchanged under the dense
> criterion.** Ten sin-family separations at `a ≥ 1.5` (of 61) fail densely;
> per-value separation counts at those large parameters are sample-level.
