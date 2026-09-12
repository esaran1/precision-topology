# Is R the controlling variable? Results

Registered in `results/r_collapse_prediction.md` before any R-collapse quantity
was computed. Reanalysis of committed artifacts plus recovery of terminal
weights from stored seeds.

Date: 2026-09-12.

---

## Headline

**R collapses far better than the budget-law variable `u` did, and the
optimizer test passes — but R is not a perfect sufficient statistic, and one
registered failure condition partially triggers.**

The honest verdict: **R is the controlling variable within family A, across
budgets and across both optimizers, with a residual `a`-dependence inside the
transition band that we can measure but not explain away.**

## Coverage, stated before any collapse number

| source | runs | what it contributes |
|---|---:|---|
| `fold1d_sweep` (sin, `a > 1`) | 1,600 | 8 values of `a`, Adam, 2k steps |
| `fold1d_refine` (sin, `a > 1`) | 800 | 4 more values of `a` |
| `alpha_composition` | 360 | **both optimizers**, 6 budgets, `a = 1.25` |
| `termination` (per-run rows) | 150 | 4 budgets; 12 aggregate rows excluded |
| **total** | **2,910** | 12 values of `a`, 6 budgets, 2 optimizers |

**Lost**: the constructed families (q4, q2, q1, q0.667) store only cell-level
rates, with no per-run `|w2|`. Those runs are **not recoverable from the
committed artifacts** and were regenerated from seeds for this task (q2 and q1
only; q4 and q0.667 were not regenerated for cost).

## 1a/1c: the pooled curve

Pooled over all `a`, all budgets, both optimizers:

| `R` | n | solved | rate | 95% CI |
|---|---:|---:|---:|---|
| < 0.10 | 1,428 | 0 | 0.000 | [0.000, 0.003] |
| 0.10-0.20 | 401 | 0 | 0.000 | [0.000, 0.009] |
| 0.20-0.30 | 331 | 0 | 0.000 | [0.000, 0.011] |
| 0.30-0.35 | 125 | 11 | 0.088 | [0.045, 0.152] |
| 0.35-0.40 | 108 | 65 | 0.602 | [0.503, 0.695] |
| 0.40-0.45 | 67 | 57 | 0.851 | [0.743, 0.926] |
| 0.45-0.50 | 47 | 44 | 0.936 | [0.825, 0.987] |
| 0.50-0.60 | 19 | 18 | 0.947 | [0.740, 0.999] |
| 0.60-0.80 | 59 | 59 | 1.000 | [0.939, 1.000] |
| 0.80-1.20 | 144 | 144 | 1.000 | [0.975, 1.000] |
| 1.20-2.00 | 138 | 138 | 1.000 | [0.974, 1.000] |
| > 2.00 | 43 | 43 | 1.000 | [0.918, 1.000] |

**0 of 2,160 runs solved below `R = 0.30`. 402 of 403 solved above
`R = 0.50`.** The entire transition occupies `0.30 < R < 0.50`.

Pooled logistic fit: **`R50 = 0.3705`**, `R25 = 0.3472`, `R75 = 0.3954`, so
the transition width is `W = 0.0481` and the registered materiality threshold
is **`0.5 W = 0.0241`**.

**Comparison with the `u`-collapse.** The budget-law collapse in
`u = (a-1) B^theta` gave location agreement and shape failure: per-budget
logistic slopes varied 7.6-fold and the residual minimum was only 1.62x below
the `theta = 0` null. R does better on the metric that matters here —
separation. Ranking runs by R separates solved from unsolved with
**AUC = 0.9940** and a best-threshold error rate of **3.75%**, against
**AUC = 0.8175** for raw `|w2|`.

## Failure condition 2 (budget): PASSES

At fixed `a = 1.25`, within R bins, across a 40x budget range:

| `R` bin | rates by budget | spread |
|---|---|---:|
| 0.0-0.2 | 0/60, 0/60, 0/42, 0/17, 0/14, 0/10 | 0.000 |
| 0.2-0.3 | 0/13, 0/18 | 0.000 |
| 0.3-0.4 | 2/5, 14/18 | 0.378 |
| 0.5-0.7 | 20/20, 21/21 | 0.000 |
| > 1.0 | 25/25, 28/28 | 0.000 |

Every bin outside the transition is perfectly homogeneous. **Budget enters
only by setting where a run lands on the R axis.**

## Failure condition 3 (optimizer): PASSES on the distribution-free test

This is the test that decides the scope claim. At `a = 1.25` fixed, SGD
reaches **5.0x lower terminal `|w2|`** than Adam at 40,000 steps (19.4 vs
97.5) and has a different onset exponent.

| `R` regime | Adam | SGD | Fisher two-sided |
|---|---|---|---:|
| `R < 0.3` | 0/95 | 0/141 | 1.000 |
| `0.3 <= R <= 0.5` | 7/10 | 14/18 | 0.674 |
| `R > 0.5` | 75/75 | 21/21 | 1.000 |

**No detectable difference in any regime.**

**A conflict between two ways of scoring this, reported rather than
resolved in our favour.** The parametric comparison gives
`R50(Adam) = 0.3350` [0.3203, 0.3933] and `R50(SGD) = 0.3709`
[0.3586, 0.3753]. The point estimates differ by **0.0359, which exceeds the
registered threshold of 0.0241** — by that criterion, taken literally,
condition 3 fails. But the two bootstrap intervals **overlap, with SGD's
interval nested entirely inside Adam's**, and the distribution-free binned
comparison above shows nothing at all. The reason is visible in the data:
Adam has only a handful of runs inside the transition band (its R values jump
from 0.35 to 0.5+), so its parametric `R50` is fitted with very little
information exactly where the fit is determined.

**Our reading**: the optimizer difference is **not established**. The
distribution-free test is the stronger evidence and it is null; the parametric
miss is attributable to Adam's sparse coverage of the band. We report both
numbers rather than only the favourable one, and we do **not** claim the
optimizer scope limitation is fully dissolved — we claim it is not detectable
at this coverage, which is weaker.

## Failure condition 1 (`a`): PARTIALLY TRIGGERS

| `a` | n | solved | `R50` | shift from pooled | material? |
|---:|---:|---:|---:|---:|---|
| 1.25 | 590 | 117 | 0.3546 | 0.0160 | no |
| 1.35 | 230 | 2 | 0.3550 | 0.0155 | no |
| 1.40 | 200 | 15 | 0.3609 | 0.0096 | no |
| 1.45 | 230 | 56 | 0.3703 | 0.0002 | no |
| 1.50 | 230 | 89 | 0.3795 | 0.0090 | no |
| 2.00 | 230 | 161 | 0.4844 | **0.1139** | **YES** |
| 3.00 | 200 | 139 | 0.6377 | **0.2672** | **YES** |

The five strata from `a = 1.25` to `a = 1.50` pass comfortably, with shifts of
0.0002 to 0.0160 against a threshold of 0.0241. **`a = 2.0` and `a = 3.0`
fail.**

**Diagnosis, which partially but not wholly excuses them**: those two strata
have almost no runs inside the transition band — 3 and 12 respectively, against
107 and 131 for `a = 1.45` and `a = 1.50`. Their R distributions are
concentrated above 0.5, where everything solves. A logistic `R50` fitted to a
stratum that barely spans the transition is poorly determined.

**But a residual `a` effect survives a direct test.** Restricting to narrow R
windows and regressing `solved ~ log R + a`:

| window | n | `a` coefficient | z |
|---|---:|---:|---:|
| `R` in [0.34, 0.40] | 139 | −6.99 (SE 2.76) | **−2.53** |
| `R` in [0.36, 0.42] | 128 | −5.96 (SE 3.10) | −1.92 |

At fixed R inside the band, larger `a` is associated with a *lower* solve rate
at nominal significance. The effect is not consistent across windows and the
`a = 3.0` stratum (0/12 in the band) cuts against a simple story, but it is
there. **R is not a perfect sufficient statistic.**

## Part 2: what R is

### 2a. The reading, checked against the definitions

`G*(a) = max_{w1,b1} G(w1,b1)` is the largest class gap any placement can
produce. The theorem gives `|w2| >= 2m/G*` for a solution of margin `m`, which
rearranges to **`R = |w2| G*/2 >= m`**. So R is **the largest logit margin a
run's terminal weight scale could support, under optimal placement of `w1` and
`b1`** — a *capacity*, fixed by `|w2|` and the activation, not by what the run
did with its placement.

**`G*` was verified to be a true maximum**, not an artifact of the committed
search window: an unrestricted scan over `w1` in [0.001, 12] and `b1` in
[−4π, 4π] reproduces the committed value to a ratio of 1.00 at every `a`
tested (1.25, 1.45, 1.5, 2.0).

### 2b. Achieved `G` does NOT tighten the collapse — it degrades it

Registered criterion: if `R_achieved = |w2| G/2` improves separation by more
than 25%, R as defined is a proxy. Measured on 160 runs with their own trained
`(w1, b1)`:

| variable | AUC | best-threshold error |
|---|---:|---:|
| **`R = |w2| G*/2`** | **0.9940** | **0.0375** |
| `R_achieved = |w2| G/2` | 0.8073 | 0.2250 |
| raw `|w2|` | 0.8175 | 0.2375 |

The achieved-gap version is **six times worse** by error rate and barely
better than raw `|w2|`. **R as defined is the right variable.**

The reason is structural: `G*` is a property of the *activation*, fixed before
training. The achieved `G` is a property of the *trained* `(w1, b1)`, hence
downstream of whether training went well — it inherits the outcome it is meant
to predict and adds noise. R measures the capacity a run **had**; achieved `G`
measures what it **used**.

**A sign error found and fixed during this check.** Achieved `G` came out
negative for roughly half of all runs, including solvers, and at `a = 1.25`
`|G|` read 262% of the supposed maximum. Neither is a theorem violation: `G`
as written is the gap for the `w2 > 0` orientation, and runs with `w2 < 0`
realize the mirrored arrangement (`min_I f - max_O f`), which the theorem's
"mirror the roles for `w2 < 0`" clause covers explicitly. Scoring each run in
its own orientation, **0 of 75 runs exceed `G*`** at any `a`. The 262% was my
error in reading the sign convention.

### 2c. The threshold has a margin interpretation, and it is consistent

`R >= m` means an empirical threshold `R* ~ 0.37` asserts: **solving requires
an achievable margin of at least about 0.37.**

Against the 66 verified solvers' measured margins (0.014-1.97, median 0.676):

- **0 of 66 have achieved margin exceeding their own R** — required by the
  theorem, and satisfied exactly.
- Median solver margin 0.676 is **1.82x** the threshold, consistent with R
  being an *upper* bound that runs approach but do not reach.
- 35% of solvers have margins below 0.37, which is expected: those runs have R
  above the threshold and simply settle at a smaller achieved margin.

So the threshold is not a bare empirical cut. It is a statement about margin
capacity, in the units the theorem already uses.

## Part 3: does the budget law follow?

### 3a. The onsets are predicted, not fitted

Using only (i) the pooled R threshold 0.3705, (ii) the measured median `|w2|`
growth at `a = 1.25` (`|w2| ~ B^{1.2627}`, fitted to weights alone, no onset
data), and (iii) `G*(a)`, the predicted onset is the smallest `a` with
`|w2|(B) G*(a) / 2 >= 0.3705`. **No parameter is tuned to the onsets.**

| budget | predicted `|w2|` | predicted onset | measured onset | error | `a`-grid |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 2.64 | 1.7175 | 1.60 | 0.1175 | 0.05 |
| 4,000 | 6.33 | 1.3750 | 1.32 | 0.0550 | 0.02 |
| 8,000 | 15.18 | 1.2025 | 1.18 | 0.0225 | 0.02 |
| 32,000 | 87.37 | 1.0625 | 1.06 | **0.0025** | 0.01 |
| 64,000 | 209.65 | 1.0350 | 1.04 | **0.0050** | 0.005 |
| 128,000 | 503.03 | 1.0200 | 1.03 | **0.0100** | 0.005 |

**At the four largest budgets the prediction is within one to two grid steps.**
The 2,000-step cell is the worst at 0.1175 (2.4 grid steps), which is also the
cell where the onset estimator is least stable (rates 0.550/0.450 straddling
the threshold, and an independent reimplementation placed it one grid step
away).

The predicted onset *exponent* is **−0.8583** against a measured **−0.7340**,
a difference of 0.1244 — larger than the grid resolution 0.114, so the
exponents are not equal. But the predicted onset *values* track the measured
ones closely at large budget, which is the stronger statement: the exponent
disagreement is dominated by the 2k cell.

### 3c. The extrapolation failure is inherited, not separate

The onset exponent is `−alpha/beta`, so a drifting `alpha` gives a drifting
onset exponent. `alpha` measured on `|w2|` alone, over sliding windows:

| window | `alpha` |
|---|---:|
| 1k-4k | 1.5109 |
| 2k-8k | 1.2828 |
| 4k-16k | 1.2108 |
| 8k-40k | 1.0670 |

**`alpha` drifts downward with the window**, reproducing the 1.51 → 0.72
pattern measured independently. **The direction matches the held-out failure
exactly**: short-budget `alpha` is larger, so a short-budget fit predicts a
steeper onset exponent, hence *too small* an `eps` at 128k. The held-out
prediction gave 0.0187 against a measured 0.030 — too small, as predicted.

**This converts a limitation into a consequence.** The law does not
extrapolate because `|w2|` growth is sub-power-law; the onset inherits that.
The failure is a property of `|w2|(B)`, not an independent defect of the onset
relation.

## Part 4: honest assessment

**Does R collapse?** Yes, strongly, within family A. 0 of 2,160 below
`R = 0.30`; 402 of 403 above `R = 0.50`; AUC 0.9940 against 0.8175 for raw
`|w2|`. The transition occupies a band of width 0.048 in R.

**Does it survive the optimizer test?** On the distribution-free test, yes —
no difference in any regime despite a 5x difference in terminal `|w2|`. The
parametric `R50` comparison exceeds the registered threshold by 0.012, which we
report; its cause is Adam's sparse coverage of the transition band, and the
intervals overlap with SGD's nested inside Adam's. **We claim the optimizer
difference is undetectable at this coverage, not that it is absent.**

**Does the budget law follow?** Largely. Onsets predicted from the R threshold
and `|w2|` growth alone match measured onsets to within one to two grid steps
at the four largest budgets, with no parameter tuned to onsets. The predicted
exponent (−0.8583) differs from the measured (−0.7340) by more than grid
resolution, driven by the 2k cell. And the extrapolation failure is explained
as inherited from `alpha`'s drift, with the direction confirmed.

**What R does not explain:**

1. **The residual `a`-dependence inside the transition band** (z = −2.53 in one
   window, −1.92 in another). At fixed R inside 0.3-0.5, `a` still carries
   information. R is not a sufficient statistic.
2. **The curve-sharpening result stands unchanged.** The `u`-collapse showed
   rate curves sharpen with budget (`k ~ B^{0.612}`); nothing here repairs
   that, because R is a per-run quantity and the sharpening is a statement
   about cell-level rate curves.
3. **The `a = 2.0` and `a = 3.0` strata** fail the registered materiality
   threshold on their fitted `R50`. Coverage explains this plausibly but not
   conclusively.
4. **Nothing outside family A is yet established** — see the family section.
5. **Family B** (positively homogeneous) is untested here; `G*` is defined for
   it but its `beta = 1` behaviour is a known counterexample to the budget law
   and nothing in this analysis addresses it.

**What this means for the paper.** R is a materially better organizing
variable than the budget-law parameterization: it collapses across budgets and
optimizers where the budget law is budget-specific and Adam-specific, and it
makes the onsets predictable rather than fitted. But the registered failure
conditions did not all pass cleanly, and restructuring the entire paper around
R would overstate what 2,910 runs from one activation family support.
