# Dense-sample verification of every separating run (Part 2c)

Every width-3 run recorded as separating (0 errors on the 2,000-point eval
set) was reconstructed deterministically — with a hard recovery check
against its recorded accuracy, all 163 passing — and evaluated on 100,000
fresh points from its own link. A 200-run stratified random sample of
width-4 separating runs got the same treatment. Data: `dense_check.csv`.

## The headline: a third of width-3 "separations" are not regional separations

| Group | checked | fail densely | rate |
|---|---:|---:|---:|
| width 3, all separating runs | 163 | **56** | **34.4%** |
| width 4, random sample | 200 | 37 | 18.5% |

For GELU at width 3: **46 of 81 fail** (57%). The failures are not
borderline as a class: 43 of the 56 have ≥4 errors on 100,000 points, with
medians around 20 and worst cases at 97, and margins as low as −25.

**The eval-0 criterion on 2,000 points systematically overstates regional
separation**, exactly as the Part 2b train-zero finding and the witness
selection (seed 14, eval-perfect, 2 dense errors) foreshadowed.

## Where it hits, and where it does not

GELU width-3 dense survivors by sweep:

| Sweep | recorded separations | survive densely |
|---|---:|---:|
| width_sweep (baseline link) | 6 | 5 |
| threshold_sweep (baseline link) | 6 | **6** |
| protocol_sweep (baseline link) | 2 | 1 |
| parametrization_sweep | 33 | 18 |
| **corrugation_sweep** | **34** | **5** |

- **Baseline-link results survive nearly intact** (12 of 14 GELU runs).
- **Corrugated links are the epicenter: 29 of 34 GELU separations fail
  densely, and 16 of the 21 corrugated configurations are left with zero
  dense-verified separations.** A plausible reading — an observation, not a
  measured mechanism — is that corrugation adds boundary detail that 2,000
  eval points under-resolve, so eval-0 was easiest to reach spuriously
  exactly where the geometry is most intricate. Status notes have been
  added to `corrugation_results.md` and `parametrization_sensitivity.md`.
- The parametric families in the threshold sweep: all dense failures sit at
  `a ≥ 1.5` (10 of 61 sin-family separations); **every separation at the
  transition-defining values survives** — `a = 1.10`: 3/3, `a = 1.25`: 5/5,
  and all 13 `pwl` separations. **The Part 1 transition intervals are
  unchanged under the dense criterion.**

Width-4 sample: sin-family 48/48 survive, tanh 45/51, GELU 55/70 — but
**ReLU 1/5 and leaky-ReLU 3/10**. The width-4 claim for the ReLU family was
already weak (≤6% separation rates); densely it is weaker still.

## What is unaffected

**The monotonic width-3 zero cannot be affected by this check**: it is a
zero at the sample level, and regional separation implies sample-level
separation, so 0 sample-level separations already implies 0 regional ones.
The dense check moves only the non-monotonic side downward. The categorical
asymmetry — some non-monotonic runs separate regionally, no monotonic run
separates at all — survives with the witness (0 errors on 2,000,000 points,
margin 0.28) as its strongest single exhibit.

## The soft boundary

The separated/not-separated boundary is itself soft at the 1-error-in-100k
scale. Re-checking the eight ≤2-error failures on two further independent
100k samples: three of them (width_sweep GELU d5 s1; corrugation B_f10 d12
s4; protocol GELU d12 s24 partially) pass 0/0 or flip between 0 and a few
errors; the rest fail consistently. Symmetrically, survivors with margins
near 0.02 would presumably flip on some draws. Counts above are quoted at
the primary protocol (one fixed dense seed per run, recorded in the CSV);
the flip-prone band is real and small, and the ≥4-error failures are stable
across samples.

## What follows

1. Every separation claim in this project now needs the qualifier
   *sample-level* or *dense-verified*; the claims ledger (Part 5) carries
   the authoritative counts of each.
2. Rates by activation at width 3, dense-verified, baseline link: the
   ordering of activations is unchanged; the levels drop.
3. The 34 corrugated fold-layer traces measured real maps (the traces are
   measurements, not classifications), but "fold layer of a separating run"
   now describes 5 corrugated runs, not 34. The layer-1 immediacy statement
   retains its full support on the baseline link (all dense survivors) and
   its corrugated support shrinks — noted in `corrugation_results.md`.
