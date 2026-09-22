# Item 2: `R₅₀` under the certified `Ĝ`

Artifacts: `r50_certified.csv`, and the grid-convergence table below.

---

## A reproducibility gap, stated first

**The code that produced the committed `R50 = 0.3705` is not in the repository.**
`grep -rn "R50\|r50" src/*.py` returns nothing; the value appears in
`r_collapse_results.md` as a recorded output with no committed script. So the
committed number **cannot be reproduced exactly**, and what follows is a
reimplementation, not a replication.

This is worth fixing before submission regardless of `Ĝ`: an unreproducible
headline number is a liability under review.

## What the reimplementation gives

Two-parameter logistic in `R`, fitted by least squares on quantile-binned solve
rates (41 bins, ≥10 runs per bin), grid search over `(k, midpoint)`. The result is
**sensitive to the fit grid**, which is why my earlier figure of 0.3728 was wrong:

| grid (`k` × `mid`) | restricted `Ĝ` | certified `Ĝ` |
|---|---:|---:|
| 120 × 200 | 0.3715 | 0.3728 *(the figure I quoted earlier — coarse)* |
| 150 × 250 | 0.3640 | 0.3653 |
| 400 × 800 | 0.3672 | 0.3685 |
| **900 × 1600 (converged)** | **0.3669** | **0.3682** |

**Correction:** I reported `R50 = 0.3728` in `ghat_unification.md` and in my
summary. That came from the coarsest grid. The converged value is **0.3682**.

## The robust quantity is the *shift*, not the absolute value

| | value |
|---|---:|
| committed (original code, not reproducible) | 0.3705 |
| reimplementation, restricted `Ĝ` | 0.3669 |
| reimplementation, **certified `Ĝ`** | **0.3682** |
| **shift from the `Ĝ` switch** | **+0.0013** |

The **+0.0013 shift is identical at every grid resolution** (0.3715→0.3728,
0.3640→0.3653, 0.3672→0.3685, 0.3669→0.3682). That is the number the `Ĝ` switch
licenses, and it is robust.

The reimplementation sits **0.0036 below** the committed value — a discrepancy in
the fitting procedure, not in the data or in `Ĝ`.

## What to quote

**Recommended**: `R50 = 0.3705 + 0.0013 = **0.3718**` — the committed value plus
the measured `Ĝ` shift, which avoids substituting an unreplicated
reimplementation for the committed number while still applying the correction.

With the bootstrap interval from the reimplementation (400 cluster resamples):
**95% CI ≈ ±0.008**, so `R50 ≈ **0.372 [0.364, 0.380]**`.

**Alternative, if the original script is recovered before submission**: rerun it
on the certified `R` and quote that directly. That is the better outcome and
should be attempted first.

## Downstream numbers that inherit `R50`

`R50` is the reference for T52's **materiality threshold** `0.5·W`, used in all
four stratum tests (`a`, budget, optimiser, family).

| | committed | reimplementation, certified |
|---|---:|---:|
| `R25` | 0.3472 | 0.3458 |
| `R75` | 0.3954 | 0.3907 |
| `W = R75 − R25` | 0.0482 | 0.0449 |
| **materiality `0.5·W`** | **0.0241** | **0.0225** |

The materiality threshold **tightens by 0.0016**, about 7%. **No T52 verdict
changes**: the family shifts that failed condition 4 were 3–5× the threshold
(`R50` = 0.247 / 0.295 vs 0.373), and the strata that passed did so by margins
well under it. A 7% tighter threshold moves nothing across the line.

**Other `R50`-derived numbers checked**:
- `R50(Adam) = 0.3350` and `R50(SGD) = 0.3709` (T52 condition 3) — both shift by
  the same `Ĝ` factor at their own `a`, and the conclusion was already "all three
  intervals overlap, nine Fisher tests null". Unaffected.
- Per-family `R50` (0.247 / 0.295 / 0.373) — all at family-specific `Ĝ`; the
  q-family values use constructed activations whose `Ĝ` is analytic, so only
  family A's shifts. The 3–5× separation is unaffected.
- The **onset** numbers do not use `R50` at all.
