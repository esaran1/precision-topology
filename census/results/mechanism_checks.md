# Items 4-5: two mechanism checks

Direct computation and reanalysis; Item 5 rebuilds 360 runs from seeds.

Date: 2026-09-12.

---

## Item 4: is `kappa` constant in `a`? — yes, to 3.2%, and the residual drift does not matter

**The gap in the chain.** The theorem is
`|w2| >= 2m/G*(a) >= 2m/(kappa * D(a))`. The empirical verification uses
`G*(a)` directly, but `beta` is derived from the exponent of `D(a)`. If
`kappa = G*/D` varied with `a`, then `G*` and `D` would have different
exponents and `beta` from `D` would be the wrong exponent for the bound that
actually constrains solutions. Scale-invariance of `kappa` under rescaling the
task windows was proved earlier; **constancy in `a` had never been checked.**

Computed directly across the `a` range used in the onset measurements:

| `a` | `D(a)` | `G*(a)` | `kappa` |
|---:|---:|---:|---:|
| 1.025 | 7.371e-03 | 2.251e-03 | 0.30544 |
| 1.05 | 2.062e-02 | 6.336e-03 | 0.30722 |
| 1.10 | 5.712e-02 | 1.755e-02 | 0.30735 |
| 1.25 | 2.130e-01 | 6.625e-02 | 0.31102 |
| 1.50 | 5.539e-01 | 1.745e-01 | 0.31498 |
| 1.60 | 7.067e-01 | 2.229e-01 | 0.31543 |
| 2.00 | 1.370e+00 | 4.402e-01 | 0.32142 |
| 3.00 | 3.195e+00 | 1.049e+00 | 0.32830 |

- Over the **onset range** (`a <= 1.60`): `kappa` in **[0.30544, 0.31543]**,
  a variation of **3.2%**.
- Over the full range to `a = 3.0`: **[0.30544, 0.32830]**, **7.3%**.

`kappa` **drifts upward slightly with `a`** rather than being exactly
constant, so `G*` and `D` do have marginally different exponents. Measuring
both local exponents over the onset range:

| quantity | local exponent | SE |
|---|---:|---:|
| `D(a)` | 1.4448 | 0.0055 |
| `G*(a)` | 1.4546 | 0.0045 |

**Difference: 0.0098.** Substituting `G*`'s own exponent for `D`'s in the
prediction changes the predicted onset exponent from **−0.7733** to
**−0.7680**, a shift of **0.0052** — **0.05x the grid resolution of 0.114**
that already dominates the comparison, against a measured −0.7340.

**Verdict: the gap in the chain closes.** `kappa` is not exactly constant, and
we report that rather than claiming it is, but the drift is far too small to
matter. Using `beta` derived from `D` in place of `G*`'s own exponent is
justified at our resolution, and the difference is two orders of magnitude
below the uncertainty on the quantity it feeds.

Note also that both measured exponents (1.4448, 1.4546) sit slightly **below**
the analytic 1.5, which is the expected finite-`eps` behaviour: the
`(a-1)^{3/2}` form is asymptotic as `a -> 1+`, and the measured range extends
to `a = 1.6`. This is the same effective-vs-asymptotic `beta` distinction §6
already carries.

---

## Item 5: is `|w2|` rate-limiting? — yes, and more sharply than expected

### What had to be checked

The joint criterion showed adequate `|w2|` is necessary but not sufficient
(signed mean +0.100, never negative). The law nevertheless predicts the onset
from `|w2|` crossing alone, which is only legitimate if the remaining
conditions — placing `w1`, `b1` in the fold's usable window — are satisfied
with roughly constant probability across `a`.

### A correction made during this measurement

A first pass defined "adequate `|w2|`" as `|w2| >= 2/G*(a)`, i.e. assuming a
unit logit margin. **That was wrong**, and it produced an apparent
contradiction: 81 runs at `a = 1.5` solved with `|w2|` between 3.99 and 6.38
against a supposed requirement of 11.46. The theorem's `m` is **each
solution's own achieved logit margin**, not a fixed constant
(`theorem_verification.csv` records margins from 0.014 to 1.97, median 0.68).
The contradiction was in my reading of the bound, not in the bound, which has
0 violations across 66 verified solvers.

The fix, and the right quantity to condition on: for each run define

> `R = |w2| * G*(a) / 2` — **the largest logit margin this run's terminal
> weight scale could support at the best possible `(w1, b1)`.**

`R` is the run's *margin capacity*. The theorem says a solution with margin
`m` requires `R >= m`. `R` is computable for every run, solving or not, which
is what the conditional test needs.

### Result: 360 runs rebuilt from seeds, 6 values of `a`, 60 seeds each

First, the fact that makes the question sharp — **terminal `|w2|` barely moves
with `a`, while the required `|w2|` diverges**:

| `a` | `G*(a)` | median `|w2|` | median `R` | P(solve) |
|---:|---:|---:|---:|---:|
| 1.25 | 0.0662 | 3.41 | 0.113 | 0.000 |
| 1.35 | 0.1068 | 3.76 | 0.201 | 0.000 |
| 1.45 | 0.1513 | 3.98 | 0.301 | 0.217 |
| 1.50 | 0.1745 | 4.05 | 0.354 | 0.400 |
| 2.00 | 0.4402 | 3.92 | 0.863 | 0.767 |
| 3.00 | 1.0489 | 3.13 | 1.644 | 0.733 |

Terminal `|w2|` spans only 3.13 to 4.05 across the whole range. **The solve
rate is not varying because `|w2|` varies; it is varying because the same
`|w2|` buys less margin as `G*(a)` shrinks.**

### `R` is very nearly a sufficient statistic

Bucketing all 360 runs by `R`, ignoring `a` entirely:

| regime | n | solved | rate | `a` values present |
|---|---:|---:|---:|---|
| `R < 0.30` | 191 | **0** | **0.000** | all six |
| `0.30 <= R <= 0.50` | 78 | 36 | 0.462 | five |
| `R > 0.50` | 91 | **91** | **1.000** | three |

**No run with `R < 0.30` solved, at any `a`, across 191 runs. Every run with
`R > 0.50` solved, across 91 runs.** Within those two regimes the solve rate
is constant in `a` — 0.000 at every one of the six `a` values below 0.30, and
1.000 at both `a` values above 0.50.

The predictive check: `P(R > 0.4)` against the marginal solve rate, by `a`:

| `a` | P(`R` > 0.4) | P(solve) |
|---:|---:|---:|
| 1.25 | 0.000 | 0.000 |
| 1.35 | 0.000 | 0.000 |
| 1.45 | 0.150 | 0.217 |
| 1.50 | 0.350 | 0.400 |
| 2.00 | 0.783 | 0.767 |
| 3.00 | 0.733 | 0.733 |

The two columns track across the full range, including the two cells at zero
and the two near 0.75.

### Verdict, against the registered options

The registration offered "roughly constant conditional rate → `|w2|` is
rate-limiting and the mechanism holds" versus "systematically varying → the
law predicts through a correlation rather than the stated mechanism."

**The first, and in a stronger form than the question anticipated.** The
conditional probability of solving given adequate margin capacity is not
merely roughly constant in `a` — it is **0 below `R = 0.30` and 1 above
`R = 0.50` at every `a` tested**. The residual `a`-dependence lives entirely
in the transition band `0.30 <= R <= 0.50`, where 78 runs sit and the rate is
0.46.

The honest qualification: because `R` is defined using `G*(a)`, it is not an
`a`-free quantity — it is `|w2|` measured in units of what the fold can
convert into margin. The finding is therefore that **`|w2|` is rate-limiting
once expressed in those units**, which is exactly the mechanism the theorem
states, and not that raw `|w2|` predicts solving (it does not: raw `|w2|` is
nearly constant across `a` while the solve rate runs from 0 to 0.77).

The earlier "necessary but not sufficient" finding stands and is now located:
the insufficiency is confined to a band of width about 0.2 in `R`, not spread
across the whole range.

## Artifacts

`results/item5_runs.csv` (360 rebuilt runs: `a`, seed, `|w2|`, achieved
margin, `G*`, solved), `results/item5_conditional.csv`. Item 4 uses
`src/fold1d_theorem.py`'s `dip_depth` and `maximum_gap` directly.
