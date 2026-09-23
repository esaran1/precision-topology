# Every number the paper needs, recomputed from raw artifacts

> **Two `R₅₀` populations — they are not the same quantity.**
> (i) **Pooled `R₅₀`** = 0.3682 (binned logistic, `src/r50_fit.py`) / 0.3723 (MLE logistic, family A in T52):
> the pooled population, 3,150 untargeted runs over 12 values of a × 8 budgets × Adam/SGD/AdamW
> (`r_pooled.csv` + `r_adamw.csv`), certified `Ĝ(a)`.
> (ii) **Block C in-band `R₅₀`** = 0.3471 (Adam), 0.3479 (AdamW), 0.3442 (SGD): MLE logistic on new
> band-targeted runs at **a = 1.25 only** (1,200 / 960 / 840 runs, cells chosen to sit in the transition band).
> They are different populations of runs (different a, budget mix and sampling design), so the
> difference between (i) and (ii) is not a discrepancy and is not interpreted. Every `R₅₀` below is
> labelled with the population it belongs to.

> **0.3705 superseded 2026-09-22**: the script producing it is not in the repository (`results/r50_provenance.md`); replaced by the committed reimplementation **pooled `R50 = 0.3682` [0.3603, 0.3730]** (population (i)) (`src/r50_fit.py`), materiality `0.5W` **0.0224**. The old value lies inside the new interval, so no verdict changes. 


**Method**: each value below was recomputed in this pass from the raw CSV or by
re-deriving the quantity, **not** read from a ledger row or results document.
Where the record holds more than one value, all are named and the one to print
is stated with the reason.

Date: 2026-09-12. Regeneration: `PYTHONPATH=. python3 paper/make_figures.py`
re-verifies the three headline quantities on every run.

**Read §14 first if you are correcting a draft** — it lists what is wrong.

---

## 1. The theorem

**Statement, in the form to print:**

> Let `f_a(t) = t + a sin t` and let `N(x) = w2 * f_a(w1 x + b1) + b2` classify
> `I = [-0.8, 0.8]` as negative and `O = ±[1.2, 2.0]` as positive with logit
> margin `m > 0`. Write `G(w1, b1) = min_{x in O} f_a(w1 x + b1) - max_{x in I} f_a(w1 x + b1)`
> for the orientation `w2 > 0` (mirror the roles of `I` and `O` for `w2 < 0`).
> Then
>
>     |w2| >= 2m / G(w1, b1) >= 2m / G*(a),    G*(a) = max_{w1,b1} G(w1, b1).

**Both forms are correct and both should appear** — the first is the pointwise
bound at the network's own placement, the second the uniform bound. The chain
`>= 2m/(kappa D(a))` follows from `G* = kappa D`. Printing only `2m/G*` loses
the fact that the bound holds at each placement.

- `b2` cancels in the derivation, so the bound holds for the optimal `b2`.
- `w1` cannot substitute for `w2`: it enters only through `G <= kappa D(a)`.
- **Scope: width 1 only.** At width > 1 several units can split the two sign
  changes, leaving only `sum_i |w2_i| G_i >= 2m`.

**kappa = G*/D:**

| quantity | value |
|---|---|
| range over `a` in [1.02, 3.0] | **[0.30544, 0.32830]** |
| range over the onset range `a <= 1.60` | **[0.30544, 0.31543]**, variation **3.2%** |
| grid | 33 `b1` values x 600 `w1` values (`maximum_gap(resolution=600)`) |

Print **kappa in [0.305, 0.328]**, and state it is numerically obtained with
**no closed form derived**.

**Scale invariance**: scaling all three window edges by 2x
(0.8 / 1.2 / 2.0 -> 1.6 / 2.4 / 4.0) moves kappa from **0.3152 to 0.3147**, a
**0.2%** change. Changing the *ratio* moves it substantially (`r1 = 2.0` gives
**0.5166**). So `kappa = kappa(r1, r2)` with `r1 = 1.5`, `r2 = 2.5`.

**Verification** (`theorem_verification.csv`):

| quantity | value |
|---|---|
| solvers checked | **n = 66** (recorded weeks before the theorem, for another purpose) |
| violations | **0** |
| median slack `|w2|/bound` | **1.2740** |
| minimum slack | **1.0394** |
| margins of those solvers | min **0.0137**, median **0.6757**, max **1.9652** |
| `a` values | 1.45, 1.5, 2.0, 3.0 |

**The infimum result is stated correctly.** The infimum of `|w2|` over
solutions is **0**, because `m` can vanish: at `a = 1.02` a construction with
`|w2| = 1` solves with margin **2.7e-4**. So no bound of this form can explain
unfindability — the bound constrains solutions, not which solutions SGD
reaches.

---

## 2. Fold depth

### 2.1 The asymptotic constant: **4√2/3, not 8/3**

**This is an error in the current record.** Ledger T37 and
`results/fold1d_theorem.md` print `D(a) ~ (8/3)(a-1)^{3/2}`. Recomputed:

    D(a)/(a-1)^{3/2}  ->  1.885610 as a -> 1+
    4*sqrt(2)/3 = 1.885618     8/3 = 2.666667

**Print: `D(a) ~ (4√2/3)(a-1)^{3/2}` as `a -> 1+`.**

**Exact closed form** (verified to 1e-9 against `dip_depth`):

    D(a) = 2( sqrt(a^2 - 1) - arccos(1/a) )

**Error of the asymptotic, with each constant:**

| `a` | exact `D` | `(4√2/3)ε^{3/2}` | error | `(8/3)ε^{3/2}` | error |
|---|---:|---:|---:|---:|---:|
| 1.02 | 0.005286 | 0.005333 | **+0.9%** | 0.007542 | +42.7% |
| 1.10 | 0.057116 | 0.059628 | **+4.4%** | 0.084327 | +47.6% |
| 1.50 | 0.553931 | 0.666667 | **+20.4%** | 0.942809 | +70.2% |
| 3.00 | 3.194935 | 5.333333 | **+66.9%** | 7.542472 | +136.1% |

`results/asymptotic_finding.md` reports the 42.7–136% column. Those numbers are
right *for `8/3`* but they overstate the approximation error by roughly 40
percentage points, because the constant itself was wrong. **With the correct
constant the asymptotic is accurate to 0.9% at `a = 1.02`**, which is the
regime the theorem is about.

The conclusion of that document still stands and should still be followed:
**state the theorem in `D(a)`, give the asymptotic separately and label it as
the `a -> 1+` limit.** Every computation in the project calls `dip_depth()`
(exact), so **no computed result moves.**

### 2.2 Effective local exponent

| `ε` range | local exponent of exact `D` |
|---|---:|
| **[0.025, 0.6] (the measured onset range)** | **1.4269** |
| [0.02, 0.08] | 1.4809 |
| [0.3, 0.6] | 1.3605 |

Print **1.4269 over `ε` in [0.025, 0.6]** as the effective `beta`, against the
asymptotic **3/2**. Both belong in the text; §6 already carries the
sensitivity.

### 2.3 The `q`-family construction

    f'(x) = (1-a) + a|x|^q   near 0, giving   beta = 1 + 1/q   analytically
    f(x)  = (1-a)x + a sign(x)|x|^{q+1}/(q+1)   for |x| <= 1, slope 1 outside

| family | `q` | analytic `beta` | measured `beta` | error |
|---|---:|---:|---:|---:|
| q4 | 4 | 1.25 | 1.2500 | ≤0.7% |
| q2 | 2 | 1.50 | 1.4959 | ≤0.7% |
| q1 | 1 | 2.00 | 2.0000 | ≤0.7% |
| q0.667 | 2/3 | 2.50 | 2.4993 | ≤0.7% |

Source `depth_families_verification.csv`. Print "**verified numerically to
≤ 0.7%**".

### 2.4 The q2 / family A cross-route control

Both have `beta = 1.5`: q2 by construction (`q = 2`), family A analytically.
Measured `beta`: **1.4959 (q2)** against **1.4963 (family A)**. Prefactor ratio
**√2**, constant across three decades of `ε`. Measured onset exponents
**−0.6749 (q2)** and **−0.7340 (family A)**, agreeing to **0.059**.

This is the construction's **cross-route control** — same analytic exponent
reached through a different activation — and is why both points are kept in the
five-point fit.

---

## 3. Link setting

**Total: 0 separations in 5,580 distinct monotonic width-3 runs.**

One-sided exact 95% upper bound on the per-run rate:
**p ≤ 1 − 0.05^(1/5580) = 5.37e-4 = 0.0537%**. For any single 80-run cell the
same formula gives only **3.68%** — which is why the pooled statement carries
the weight.

**Stratum decomposition** (regenerated by `src/zero_decomposition.py`):

| stratum | runs | separations |
|---|---:|---:|
| width_sweep | 240 | 0 |
| threshold_sweep | 1,280 | 0 |
| parametrization_sweep | 1,440 | 0 |
| corrugation_sweep (distinct) | 1,620 | 0 |
| protocol_sweep | 360 | 0 |
| search_restarts | 400 | 0 |
| winding_sweep | 240 | 0 |
| **total** | **5,580** | **0** |

Corrugation is 1,890 rows minus 270 bit-identical Reading-B duplicates.

**Link types — "five link families" was withdrawn and is defined nowhere.**
The artifact-derived enumeration:

1. Hopf link, `|lk| = 1` (also `winding q=1`, corrugation `flat`)
2. corrugated Hopf, reading A (11 configurations)
3. corrugated Hopf, reading B (9 configurations)
4. winding link `|lk| = 2`
5. winding link `|lk| = 3`
6. winding link `|lk| = 4`

**Print: six distinct link types** (or four families if the three non-trivial
winding links are counted as one parametric family). **Neither count is five.**
Enumerate rather than quote a count.

**Parametrizations: 12** (`asymmetric_both`, `asymmetric_tube`, `baseline`,
`far_offset`, `generic`, `near_offset`, `oblique_offset`, `rotated_30`,
`rotated_generic`, `thick_tube`, `thin_tube`, `unequal_major`).
**Protocols: 2** (ours and the original authors').

**Four searches** (each reported separately, never pooled into the 5,580):

| method | attempts | separations |
|---|---|---|
| SGD/Adam restarts | 400 (200 tanh + 200 sin a=0.95) | 0 |
| swap-descent | 24 x 2 configurations | 0 |
| simulated annealing | 12 traces | 0 |
| CMA-ES | 120 restarts | 0 (positive control 12/20 at depth 3) |

**CMA-ES exact result**: monotonic networks reach **0 training errors on the
2,000-point sample** (3 restarts), while separate SGD runs come within
**2 eval errors**. The barrier is at exactly 0 eval errors, approached
arbitrarily closely without being reached. **Monotonic networks shatter the
sample and cannot separate the region.**

**Dense verification**: of 163 width-3 separations re-checked at 100,000
points, **56 failed = 34.4%**. For GELU specifically, **46 of 81 = 56.8%**.
Print "roughly a third overall, and more than half of GELU's".

**Two separate width-3 witnesses, both in this setting. Do not merge them.**

| | the `a = 1.02` offset exhibit | the absolute-value witness |
|---|---|---|
| module | `src/offset_witness.py` | `src/witness.py` |
| fold | frozen `f_{1.02}`, amplified 600x | `pwl_family(-1)`, i.e. `|x|` |
| points | **1,400,000** (4 samples x 175,000/class) | **2,000,000** (5 x 200k + 1 x 1M) |
| errors | **0** | **0** |
| margins | **0.54–0.85** (committed run) | worst **0.284**, per-sample 0.28–0.56 |
| linking | lk −1 → 0 at the `f_{1.02}` layer | lk −1 → 0 at the fold |
| artifact | `offset_witness_dense.csv` | `witness.md` |

Neither is a GELU witness — the second is an absolute-value (`pwl_family`)
construction.

**Margins for the 1.4M exhibit: print 0.54–0.85**, from
`offset_witness_dense.csv` (0.5406, 0.6205, 0.8455, 0.7068). The figure
**0.53–0.91** in ledger T25 is from the **original uncommitted run** whose
sample seeds were not recorded (`AUDIT.md` finding 8); the committed
reconstruction is the one to cite.

**The four searches at `a = 1.02`** (`offset_search.csv`, 476 rows, **0
separations**, best 3 eval errors) — all width-3 on linked tori, at the
**non-monotonic** value `a = 1.02`:

| family | attempts |
|---|---:|
| restarts | 400 |
| CMA-ES | 40 |
| fine-tunes | 24 |
| annealing | 12 |
| **total** | **476** |

**This is a different population from the monotonic searches** behind the
5,580-run zero. In particular the **120 CMA-ES restarts** reported above are
**monotonic** (positive control 12/20 at depth 3); the **40** here are at
`a = 1.02`, which is non-monotonic. Likewise `offset_search.csv`'s 400
restarts are at `a = 1.02`, while `search_restarts.csv`'s 400 are tanh and
`sin(0.95)`, both monotonic. **Never add the two 400s or the 40 and 120.**

**Width dependence** (`width_effect.csv`, depth 3, 100 seeds/cell,
dense-verified):

| width | GELU | tanh | ReLU | leaky-ReLU | sin(1.5) |
|---:|---:|---:|---:|---:|---:|
| 3 | 2/100 | **0/100** | 0/100 | 0/100 | 9/100 |
| 4 | 72 | 64 | 8 | 18 | 93 |
| 6 | 97 | 95 | 56 | 76 | 100 |
| 8 | 99 | **98** | **84** | 93 | 100 |
| 12 | 100 | 100 | 99 | 99 | 100 |
| 16–32 | 100 | 100 | 100 | 100 | 100 |

- Monotonic at width 3: **0/300** at depth 3, **0/120** at depth 6.
- **GELU vs tanh** (the monotonicity-specific comparison): n.s. **from width
  6** — p = **0.3605** at width 6 (one-sided Fisher), 0.5000 at width 8.
- **tanh vs ReLU at width 8: 98 vs 84, two-sided Fisher p = 7.91e-04.** Both
  monotonic, so this is a **ReLU-family optimisation deficit, not a
  monotonicity effect**. Print the p as **two-sided**.
- Registered **P-W1 failed at width 8 as written**.

**GELU dose-response** (width 3, dense-verified, n = 200 per arm):

| initialisation | separations | rate |
|---|---:|---:|
| 0.3x | 6/200 | **3.0%** |
| standard | 16/200 | **8.0%** |
| found-separator pattern | 34/200 | **17.0%** |

| comparison | one-sided | two-sided |
|---|---:|---:|
| up vs down | **1.45e-06** | 2.89e-06 |
| up vs standard | 0.0048 | 0.0096 |
| standard vs down | 0.0231 | 0.0461 |

**Print the one-sided p (1.5e-06), because the directions were registered in
advance.** Under Bonferroni x3 the down arm fails (3 x 0.0231 = **0.069**).
**Note: the ledger's 0.138 is 3 x the two-sided 0.0461** — a two-sided
correction applied to one-sided values. Either way the arm fails.

---

### 3b. Five-dimensional expressivity threshold (Block A5d, k = 1; T75, Fig 9)

Registered `blockA5d_k1_prediction.md`; recomputed by `src/blockA5d_analyze.py score k1`
from `blockA5d_k1_runs.csv`. Linked `S² ⊔ S² ⊂ ℝ⁵` (Ren–Lim generator, targeted
thickening ρ = 0.5, one copy), width 5, depth 5 (chosen by capacity pilot), 20 seeds
per cell. **Perfect = 0 errors on 20,000 uniform and 20,000 linking-region held-out points.**

| `a` | 0.9 | 1.0 | 1.05 | 1.1 | 1.2 | 1.35 | 1.5 | 2.0 | 3.0 | ReLU | GELU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perfect at 64k | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 2/20 | 1/20 | 3/20 | 7/20 | 0/20 | 6/20 |

- **A1 PASS**: 0 of 160 run-budgets perfect at `a ≤ 1`; minimum 22 uniform errors.
- **A4 PASS**: 7 of 20 at `a = 3.0`.
- **Stop condition**: no `a` reaches half perfect at any budget (max 0.35), so **no onset
  is measured** (A2 fails; A3, A5 undecidable). The k = 10 run (T72) found no perfect run anywhere.
- **Post hoc, descriptive** (`src/blockA5d_k1_trend.py`): smallest grid `a` with any perfect run
  is 3.0 at 1k and **1.35 at 4k, 16k and 64k**; largest fraction at any `a` is 0.05 / 0.25 / 0.30 / 0.35.
  At `a = 3.0` the fraction is 0.05 [0.001, 0.25] → 0.25 [0.09, 0.49] → 0.30 [0.12, 0.54] → 0.35 [0.15, 0.59]
  (exact 95%); GELU 0 → 0.05 → 0.25 → 0.30. The budgets are checkpoints of the same 20 runs. **The
  registered 50% onset was never reached at any budget.**

## 4. Exclusion table

`exclusion_table.csv`: **8 unreached** (constructed, `|w2| = 1`) and **4 found**
(Adam) configurations.

| population | `a` | `|w2|` | MEP med/max | linear | `λ_max η` | distance | margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| unreached | 1.02 | 1.000 | 0.000 / 0.000 | 0.0199 | 0.02718 | 4.671 | 0.000274 |
| unreached | 1.10 | 1.000 | 0.000 / 0.000 | 0.0316 | 0.02729 | 4.820 | 0.003142 |
| unreached | 1.25 | 1.000 | 0.000 / 0.000 | 0.0279 | 0.02775 | 4.943 | 0.012929 |
| unreached | 1.35 | 1.000 | 0.000 / 0.000 | 0.0216 | 0.02820 | 5.000 | 0.021893 |
| unreached | 1.45 | 1.000 | 0.000 / 0.000 | 0.0199 | 0.02884 | 5.051 | 0.032528 |
| unreached | 1.50 | 1.000 | 0.000 / 0.000 | 0.0214 | 0.02914 | 5.074 | 0.038420 |
| unreached | 2.00 | 1.000 | 0.000 / 0.000 | 0.0387 | 0.03021 | 5.257 | 0.114888 |
| unreached | 3.00 | 1.000 | 0.000 / 0.000 | 0.1027 | 0.02477 | 5.517 | 0.332924 |
| found | 1.45 | 5.467 | 0.000 / 0.000 | 0.6473 | 0.003549 | 17.691 | 0.103498 |
| found | 1.50 | 4.394 | 0.000 / 0.000 | 0.4880 | 0.002800 | 14.199 | 0.077061 |
| found | 2.00 | 4.237 | 0.000 / 0.000 | 0.3682 | 0.001473 | 13.440 | 0.898190 |
| found | 3.00 | 3.472 | 0.000 / 0.000 | 0.3722 | 0.000517 | 10.232 | 1.643642 |

- **MEP barriers are exactly 0.000, median and max, at every `a`, both
  populations.** The linear-interpolation proxy is non-zero but
  *anti*-correlates with findability once the endpoint is held fixed.
- **`λ_max η`**: unreached **0.0248–0.0302**, found **0.00052–0.00355**,
  against the Ahn–Zhang–Sra threshold of **2** — inside by **66x to 3,868x**.
  Print "at least 66x inside the stable regime". Registered **P-EoS-1b
  falsified**.
- **Distance from initialisation** (Euclidean): unreached **4.67–5.52**, found
  **10.23–17.69**. **Unreached points are NEARER, not further** — opposite to
  the registered prediction. Under **Adam-preconditioned** distance the
  reversal does not survive; the finding is metric-dependent and must be
  reported as "nearer **in parameter norm**".
- **Margin.** In the table's representative runs, found margin at `a = 1.5` is
  **0.0771** against the construction's **0.0384** — i.e. *larger*. The claim
  that found solutions can have *smaller* margins is a **population**
  statement, verified separately: at `a = 1.5`, **6 of 24 found solvers
  (25%) have margin below 0.0384**, minimum **0.0054**. Print the population
  form with its n; do not attach "smaller" to the single tabulated run.
- **Gradient norms** (`criticality.csv`): unreached **0.0192–0.2724**, found
  **0.0057–0.0137**; training's own terminal gradient norm is
  **0.00033–0.0211**. Unreached points sit at **20–227x** training's terminal
  gradient norm, so **they are not critical points**.
- **`λ_min` < 0 at 6 of 12** rows overall; **6 of 8** unreached configurations.
  Print "**λ_min < 0 at six of eight unreached points**", which is what makes
  Ahn–Zhang–Sra's *first* condition applicable and was our error to have
  checked only the second.

**What the 1D unreachability claim rests on — corrected 2026-09-13.** An
earlier version of this section placed the `a = 1.02` exhibit and the "476
attempts" figure here. **Both belong to the link setting (§3), not to the 1D
task**, and they have been moved there. See §14.13.

**There is no dedicated four-family search at any 1D parameter value.**
`src/cmaes.py` is imported only by `src/offset_search.py` and
`src/search_direct.py`, both of which operate on linked tori at width 3. No 1D
module runs CMA-ES, annealing, or swap-descent.

The 1D claim rests on **sweep failure, not on a targeted search**, and should
be stated that way:

| evidence | count |
|---|---|
| `fold1d_sweep.csv` | 4,800 per-run rows (6 activations x 12 parameters, 200 seeds/cell at width 1) |
| `fold1d_refine.csv` | 800 per-run rows |
| onset sweep cells | 36 cells x 40 seeds = **1,440 runs** |
| **at `a = 1.02` specifically** | **0 of 200 solved; best run 100 eval errors** |

The honest sentence: *at `a = 1.02` in the 1D task, 200 Adam runs at the
standard budget produce no solution and the best reaches 100 eval errors,
while a solution provably exists* (the `|w2| = 1` construction, margin
2.7e-4). This is weaker than a targeted multi-method search and must not be
written as one.

---

## 5. The R variable

**Definition**: `R = |w2| * G*(a) / 2` — the largest logit margin a run's
terminal weight scale could support under optimal placement. By the theorem
`R >= m`, so R is the quantity the bound constrains.

**THREE populations exist and must never be conflated.** All three are
correct; each is a different set of runs.

| population | what it is | n | below R=0.30 | above R=0.50 |
|---|---|---:|---|---|
| **pooled, all runs (CERTIFIED `Ĝ`)** | `r_pooled.csv` + `r_adamw.csv`; 12 values of `a`, 8 budgets, three optimisers | **3,150** | **0 of 2,281** | **461 of 463** |
| *pooled, all runs (restricted `Ĝ`, superseded 2026-09-22)* | same data, pre-`Ĝ`-unification | *3,150* | *0 of 2,285* | *460 of 461* |
| *pooled, pre-AdamW* | `r_pooled.csv` alone; 12 values of `a`, 6 budgets, Adam+SGD | *2,910* | *0 of 2,160* | *402 of 403* |
| **three-optimiser comparison** | `a = 1.25` only, Adam/AdamW/SGD | **600** | **0 of 361** | **154 of 154** |

**Print the 3,150 row for any pooled claim** (abstract, Figure 3a). The 2,910
row is **superseded and stale** — it is the same dataset before AdamW was
added, not a different quantity. **Print the 600 row only where the text is
explicitly about the optimiser comparison** (§5.5, Figure 3b).

**Note on the file layout**: AdamW lives in `results/r_adamw.csv`, *not* inside
`r_pooled.csv`, which still holds 2,910 Adam+SGD rows. The 3,150 figure is the
union, computed at render time by `make_figures.py`. Any script quoting a
pooled figure must read **both** files.

**The TWO non-solving runs above R = 0.50, identified (certified `Ĝ`, T65).**
Both are `a = 3.0`, Adam, budget 2,000, and they are **different failure modes** —
do **not** describe them as two near-misses.

| | seed 13 | seed 82 |
|---|---|---|
| `\|w₂\|` | 1.099398 | 0.951985 |
| `R` (certified) | 0.578447 | **0.500886** |
| `R` (restricted) | 0.576580 | *0.499269 — below the cutoff* |
| placement `G` | **+0.768428, OK** | **−0.216202, FAILS** |
| `w₁` | −1.452066 | **+0.019358** |
| failure | **bias**: `b₂` misses `(2.342373, 3.187182)` by **0.010958** | **placement**: degenerate `w₁ ≈ 0` constant-predictor region |
| dense violations | **11 of 4,000 outer**, 0 inner, at `x ∈ [1.2000, 1.2040]` | **3,847**: 1,847 of 4,001 inner, 2,000 of 4,000 outer |
| sample errors | **1** | **185** |

**Seed 13** is a genuine near-miss, unchanged across all three pooled counts
(`402 of 403`, `460 of 461`, `461 of 463`). AdamW contributed 58 runs above
R = 0.50, all solving.

**Seed 82 is a new entrant** under the certified `Ĝ`, clearing the cutoff by
**0.0009**. It can reach `R > 0.5` with `\|w₂\| < 1` because `Ĝ(3.0) ≈ 1.05`, so
the geometry factor alone nearly clears it — a known limitation of `R` as a single
product at large `a`, already visible in T52's finding that `a = 2.0/3.0` fail the
budget-homogeneity condition. Full diagnosis: `results/exceptions_above_050.md`.

| | |
|---|---|
| run | `a = 3.0`, seed **13**, Adam, 2,000 steps (`fold1d_sweep.csv`) |
| weights | `w1 = -1.4521`, `b1 = -1.9446`, `w2 = 1.0994`, `b2 = 2.3314` |
| `R` | **0.5766** (capacity is adequate) |
| 200-point sample | **0 errors** — it looks solved at sample level |
| dense check (4,001 pts) | **11 outer violations**, `x` in [1.2000, 1.2040], worst logit **-0.011** |

**It is a sample-level-only apparent solution**: correct on all 200 sampled
points, violating the region on a sliver at the very edge of the outer window
that the sample missed. This is the failure mode the dense regional criterion
exists to catch — the same one that removed 56 of 163 claimed separations in
the link setting (§3).

**Ready answer for a reviewer**: R measures *capacity* at optimal placement,
not achieved placement. This run has the capacity and did not use it; its
`(w1, b1)` is slightly off, and the shortfall shows up only in a 4-thousandth
of the outer window. Two such runs in 463 — one a bias near-miss, one a large-`a` placement failure — are consistent with R being necessary
and very nearly sufficient, which is what we claim.

**AdamW adds no new `a` value** — all 240 of its runs are at `a = 1.25`, which
is already among the 12 — so the activation-value count stays 12 while the
budget count rises 6 -> 8 (5,000 and 6,000 were added to populate the
transition band).

| R bin | n | solved | rate |
|---|---:|---:|---:|
| (0, 0.1] | 1,428 | 0 | 0.000 |
| (0.1, 0.2] | 401 | 0 | 0.000 |
| (0.2, 0.25] | 179 | 0 | 0.000 |
| (0.25, 0.3] | 152 | 0 | 0.000 |
| (0.3, 0.32] | 41 | 1 | 0.024 |
| (0.32, 0.34] | 53 | 4 | 0.075 |
| (0.34, 0.36] | 51 | 13 | 0.255 |
| (0.36, 0.38] | 45 | 26 | 0.578 |
| (0.38, 0.4] | 43 | 32 | 0.744 |
| (0.4, 0.42] | 40 | 33 | 0.825 |
| (0.42, 0.44] | 20 | 17 | 0.850 |
| (0.44, 0.46] | 17 | 15 | 0.882 |
| (0.46, 0.48] | 27 | 27 | 1.000 |
| (0.48, 0.5] | 10 | 9 | 0.900 |
| (0.5, 0.6] | 19 | 18 | 0.947 |
| (0.6, 0.8] | 59 | 59 | 1.000 |
| (0.8, 1.2] | 144 | 144 | 1.000 |
| (1.2, 2.0] | 138 | 138 | 1.000 |
| > 2.0 | 43 | 43 | 1.000 |

**Pooled `R50` = 0.3682** (population (i): 3,150 untargeted runs, all a, all optimizers), `R25 = 0.3472`, `R75 = 0.3954`, **transition width
W = 0.0481**. Registered materiality threshold **0.5W = 0.0241**.

**AUC** (1,200-resample bootstrap 95% CI; transition-band n = runs where the
binned rate is strictly between 5% and 95%):

| family | n | solved | band n | R | `|w2|` | `G*` | margin |
|---|---:|---:|---:|---|---|---|---:|
| A (Adam+SGD only) | 2,910 | 579 | 415 | **0.9975** [0.9965, 0.9984] | 0.8393 [0.8246, 0.8543] | 0.8009 [0.7794, 0.8219] | **+0.1583** |
| q2 | 360 | 213 | 26 | **0.9998** [0.9994, 1.0000] | 0.9439 [0.9205, 0.9637] | 0.6281 [0.5667, 0.6869] | **+0.0559** |
| q1 | 360 | 213 | 25 | **0.9995** [0.9982, 1.0000] | 0.9250 [0.8960, 0.9488] | 0.6330 [0.5735, 0.6911] | **+0.0745** |
| **B** | 1,000 | 578 | 286 | 0.9804 [0.9729, 0.9873] | **0.9901** [0.9821, 0.9965] | 0.3837 [0.3501, 0.4192] | **−0.0097** |

R dominates in A, q2, q1 (registered criterion: margin ≥ 0.05 with
non-overlapping intervals). **It fails in family B**, where it is *worse* than
raw `|w2|` and `G*` is **below chance**.

**Exponent scan** (`|w2|^p G*^q`; ranking depends only on `p/q`, so the
theorem's product form is `p/q = 1`):

| family | R (`p/q=1`) | best | best `p/q` | gain |
|---|---:|---:|---:|---:|
| A | 0.9975 | 0.9984 | 1.18 | **+0.0008** |
| q2 | 0.9998 | 0.9998 | 1.00 | **+0.0000** |
| q1 | 0.9995 | 0.9999 | 1.17 | **+0.0004** |
| B | 0.9804 | 0.9987 | 1.54 | +0.0183 |

**Print: optimal ratio 1.00–1.18 with a maximum gain of 0.0008 AUC, in the
three families where the structure holds.** Family B's 1.54 is part of its
failure, not part of the claim.

**Fixed-`a` consistency check**: at fixed `a`, `G*` is constant so R must be a
monotone rescaling of `|w2|` and the AUCs must agree by construction. Checked
in **24 populated cells**; maximum discrepancy **0.00e+00** (exact). This is
the check that would have caught a misaligned `G*` lookup or a wrong
orientation.

**Optimiser comparison** (`a = 1.25` fixed, n = 180 per optimiser):

| R regime | Adam | SGD | Fisher two-sided |
|---|---|---|---:|
| R < 0.3 | 0/95 | 0/141 | **1.000** |
| 0.3 ≤ R ≤ 0.5 | 7/10 | 14/18 | **0.674** |
| R > 0.5 | 75/75 | 21/21 | **1.000** |

**Three optimisers, 2026-09-14** (`r_adamw.csv`, registered in
`third_optimizer_prediction.md`). AdamW was added because decoupled weight
decay **opposes** the mechanism -- it shrinks `|w2|` at every step -- so
agreement under it is informative in a way a third adaptive method would not
be. Budgets 5k/6k were chosen after an 8-seed pilot **to populate the
transition band**; the pilot measured where runs land, not whether they solve.

| optimiser | n | solved | in-band | `R50` | 95% CI |
|---|---:|---:|---:|---:|---|
| Adam | 180 | 82 | 10 | 0.3350 | [0.3203, 0.3939] |
| AdamW (wd 0.01) | 240 | 99 | **57** | 0.3454 | [0.3342, 0.3597] |
| SGD | 180 | 35 | 18 | 0.3709 | [0.3587, 0.3753] |

**All three intervals overlap** on [0.3587, 0.3597]. Regime-wise Fisher exact,
all nine pairwise tests: `R < 0.3` all p = 1.0000 (0/95, 0/125, 0/141);
`0.3 <= R <= 0.5` p = 1.0000 / 0.6744 / 0.7647 (7/10, 41/57, 14/18);
`R > 0.5` all p = 1.0000 (75/75, 58/58, 21/21). **Minimum p = 0.6744.**

Across **361 runs below R = 0.3 spanning three optimisers, none solved**;
across **154 runs above R = 0.5, all solved**.

`alpha` on the shared 1k-40k window: **Adam 1.2627 +- 0.0593**, **AdamW
0.9802 +- 0.1269**, **SGD 0.7188 +- 0.0238**; terminal `|w2|` at 40k is 97.48,
30.55, 19.38 (a 5x spread). Different scales, different growth exponents, one
curve. The registered directional sub-prediction `alpha_adamw < alpha_adam` is
**confirmed**.

**Print: the optimiser enters only through where on the R axis its runs land.**
The earlier hedge ("undetectable at this coverage, not absent") is **replaced**.

**SUPERSEDED FOR THE EQUIVALENCE CLAIM, 2026-09-22 (Block C, T73).** Overlapping
intervals and null Fisher tests show no *detected* difference, not equivalence.
A registered two-one-sided test (`blockC_equivalence_prediction.md`), margin
**δ = 0.024** (half the 0.048 transition width), on **new band-targeted runs**
(cells chosen by an 8-seed placement pilot; ≥ 500 in-band runs per optimizer,
the registered TOST sample size), MLE logistic `R₅₀` with certified
`Ĝ(1.25) = 0.0665056`, 4,000 cell-level bootstrap resamples:

| optimiser | n | in-band | cells | Block C in-band `R₅₀` (population (ii), a = 1.25) |
|---|---:|---:|---:|---:|
| Adam | 1,200 | 708 | 10 | **0.3471** |
| AdamW (wd 0.01) | 960 | 687 | 8 | **0.3479** |
| SGD (lr 0.3) | 840 | 653 | 7 | **0.3442** |

| pair | difference | 90% interval | verdict |
|---|---:|---|---|
| Adam − AdamW | −0.0008 | [−0.0052, +0.0043] | **equivalent** |
| Adam − SGD | +0.0029 | [−0.0040, +0.0086] | **equivalent** |
| AdamW − SGD | +0.0037 | [−0.0038, +0.0092] | **equivalent** |

**Print: the solve threshold `R₅₀` is equivalent across Adam, AdamW and SGD
within δ = 0.024, for all three pairs.** The untargeted 600-run table above gives
**inconclusive** for all three pairs under the same test (`blockC_existing_context.csv`),
so the targeted design is what establishes equivalence. The old SGD estimate
(0.3709/0.3723, from the 600-run untargeted three-optimiser table at a = 1.25) came from 18 in-band runs;
the Block C in-band estimate (population (ii)), from 653 in-band runs, is 0.3442.
Note that on point estimates alone AdamW's gap (0.0251) and Adam's (0.0355)
both exceed 0.0241; the registered criterion required interval separation **and**
a significant Fisher test, neither of which occurs. Scope: one task, `a = 1.25`,
width 1 -- a statement about optimisers, not settings. R remains
family-specific (T52, T54).

**Residual `a` effect** (logistic `solved ~ log R + a` inside narrow R windows):

| window | n | `a` coefficient | z |
|---|---:|---:|---:|
| R in [0.34, 0.40] | 139 | −6.99 (SE 2.76) | **−2.53** |
| R in [0.36, 0.42] | 128 | −5.96 (SE 3.10) | −1.92 |

**R is not a sufficient statistic.** Report both windows; the effect is not
consistent across them.

**Family B compensation**: median `|w2| ~ G*^{-0.4747}`; product varies
**4.93x** against **19.5x** for `G*`.

---

## 6. Metric-artifact test

2,400 runs with both a continuous eval-error count (0–2,000) and computable R.

| quantity | value |
|---|---|
| mean errors at R ≤ 0.05 | **153.6** (n = 995) |
| mean errors at R in (0.25, 0.30] | **36.3** (n = 128), **0 solved** |
| mean errors at R > 0.6 | **0.0** |
| error improvement occurring at **zero** solve rate | **76%** (117.3 of 153.6) |
| 10–90% crossing, continuous | **R in [0.055, 0.307]** |
| 10–90% crossing, binary | **R in [0.332, 0.452]** |
| the two intervals | **disjoint** |
| steepest continuous slope | **−1,649 at R ≈ 0.263** |
| slope at the binary threshold (R ≈ 0.362) | **−136**, a fifth of peak |
| registered width ratio `W_cont / W_binary` | **2.09** |
| registered bands | ≤ 2 genuine, ≥ 5 artifact, between = intermediate |
| **verdict** | **intermediate**, marginally outside "genuine" |

**Scope qualifier that must travel**: the ratio 2.09 flatters the result; the
locations are the finding. Print **"R predicts where the binary criterion
flips"**, not "R identifies a sharp capability threshold" — that framing is
**withdrawn**.

Achieved margin was computed and **refused as evidence**: `margin > 0` is
definitionally separating, so its zero-crossing must coincide with the binary
transition. Using it would be circular.

---

## 7. Budget law

**`alpha` (terminal `|w2|` growth), `a = 1.25`, 30 seeds per cell:**

> **alpha = 1.1172 ± 0.0605, R² = 0.9827, n = 8 cells, budgets 1k–160k.**

**Drift across every contiguous sub-range (≥ 3 cells):**

| range | cells | alpha | R² |
|---|---:|---:|---:|
| 1k–4k | 3 | **1.5109** | 0.9912 |
| 1k–8k | 4 | 1.4234 | 0.9935 |
| 1k–16k | 5 | 1.3452 | 0.9930 |
| 1k–40k | 6 | 1.2627 | 0.9913 |
| 1k–80k | 7 | 1.1903 | 0.9882 |
| 1k–160k | 8 | 1.1172 | 0.9827 |
| 2k–8k | 3 | 1.2828 | 0.9999 |
| 2k–16k | 4 | 1.2358 | 0.9990 |
| 2k–40k | 5 | 1.1726 | 0.9972 |
| 2k–80k | 6 | 1.1086 | 0.9940 |
| 2k–160k | 7 | 1.0389 | 0.9881 |
| 4k–16k | 3 | 1.2108 | 0.9981 |
| 4k–40k | 4 | 1.1334 | 0.9970 |
| 4k–80k | 5 | 1.0621 | 0.9938 |
| 4k–160k | 6 | 0.9859 | 0.9874 |
| 8k–40k | 3 | 1.0670 | 0.9994 |
| 8k–80k | 4 | 0.9960 | 0.9960 |
| 8k–160k | 5 | 0.9160 | 0.9890 |
| 16k–80k | 3 | 0.9397 | 0.9956 |
| 16k–160k | 4 | 0.8464 | 0.9878 |
| 40k–160k | 3 | **0.7193** | 0.9949 |

**Print: 1.5109 (1k–4k) falling monotonically to 0.7193 (40k–160k), R² ≥ 0.983
throughout** — local power-law behaviour with a drifting exponent.

**Onsets** (40 seeds per cell, all 6 bracketed):

| budget | onset `a` | `ε` |
|---:|---:|---:|
| 2,000 | 1.60 | 0.6000 |
| 4,000 | 1.32 | 0.3200 |
| 8,000 | 1.18 | 0.1800 |
| 32,000 | 1.06 | 0.0600 |
| 64,000 | 1.04 | 0.0400 |
| 128,000 | 1.03 | 0.0300 |

> **Measured onset exponent −0.7340 ± 0.0370, R² = 0.9900, 6 of 6 bracketed.**
> Registered prediction **−0.7448**, band **[−0.895, −0.595]** — inside.
> Bootstrap over seeds (redetermining every onset through the bracketing rule):
> **95% interval [−0.761, −0.707], half-width 0.027**, which is 0.24x the grid
> term 0.114 that dominates it.

**`a = 1.25` solve counts** (the statement requiring no fit):

| budget | 1k | 2k | 4k | 8k | 16k | 40k | 80k | 160k |
|---|---|---|---|---|---|---|---|---|
| solved | 0/30 | 0/30 | 2/30 | 25/30 | 27/30 | 28/30 | 30/30 | 30/30 |

**Print the cleanest pair: 0 of 200 at the standard 2,000-step budget, 40 of 40
at 80,000 and at 160,000** (those are from the larger onset sweep, n = 200/40;
the table above is the 30-seed `budget_alpha` arm — do not mix the two n's).

**Held-out test:**

| quantity | value |
|---|---|
| fit range | 4 smallest budgets: 2k, 4k, 8k, 32k |
| fitted exponent | **−0.8261** (vs −0.7340 on all six) |
| predicted `ε` at 128k | **0.01872** |
| 95% prediction interval | **[0.01403, 0.02497]** |
| measured | **0.030** |
| verdict | **OUTSIDE, +60.3%** |

**Compute cost:**

> **B(ε) ~ ε^{−1.36}, interval [1.151, 1.670]**; halving `ε` costs **2.57x**,
> interval **2.22x–3.18x**. **Valid only over B in [2,000, 160,000].**

The interval inverts the onset exponent's statistical+grid interval
**[−0.8691, −0.5989]** (95% CI ±0.0725 combined in quadrature with grid
resolution ±0.1140), *not* the wider registered acceptance band. Verified:
1/0.8691 = 1.1506, 1/0.5989 = 1.6697.

**Onsets predicted from the R threshold** (no parameter tuned to onsets):

| budget | predicted | measured | error | `a`-grid |
|---:|---:|---:|---:|---:|
| 2,000 | 1.7175 | 1.60 | 0.1175 | 0.05 |
| 4,000 | 1.3750 | 1.32 | 0.0550 | 0.02 |
| 8,000 | 1.2025 | 1.18 | **0.0225** | 0.02 |
| 32,000 | 1.0625 | 1.06 | **0.0025** | 0.01 |
| 64,000 | 1.0350 | 1.04 | **0.0050** | 0.005 |
| 128,000 | 1.0200 | 1.03 | **0.0100** | 0.005 |

**The four errors at the largest budgets: 0.0225, 0.0025, 0.0050, 0.0100** —
within one to two grid steps.

**Collapse:**

| quantity | value |
|---|---|
| minimising `theta` | **0.7250** |
| bootstrap 95% CI over cells (300 resamples) | **[0.500, 0.888]**, contains −(−0.7340) |
| residual at minimum | 0.1780 |
| residual at `theta = 0` / `1.5` | 0.2885 / 0.2634 (ratios **1.62x / 1.48x**) |
| registered pass condition | factor 2 — **NOT MET** |
| sharpening | **k ~ B^{0.612}, r = 0.974**, slopes 3.94 → 29.82 (7.6-fold) |
| midpoint collapse | spread 0.330 in log `u` = **1.39x** across 32x of budget |
| q2 / q1 shape | q2 sharpens `B^{+0.428}` (r = 0.993); **q1 broadens `B^{-0.718}`** (r = −0.906) |

**Threshold independence**: 50% and 75% give **−0.7325** (agreeing to four
decimals); 25% gives **−0.5037** and is bracketed at only **3 of 6** budgets.

---

## 8. Four-family relationship

| family | `beta` | `1/beta` | measured | predicted | bracketed cells | resolution |
|---|---:|---:|---:|---:|---:|---:|
| q0.667 | 2.4993 | 0.4001 | **−0.5000** | −0.4470 | 3 | ±0.171 |
| q1 | 2.0000 | 0.5000 | **−0.6521** | −0.5586 | 4 | ±0.114 |
| q2 | 1.5000 | 0.6667 | **−0.6749** | −0.7448 | 4 | ±0.114 |
| family A | 1.5000 | 0.6667 | **−0.7340** | −0.7448 | 6 | ±0.114 |
| q4 | 1.2500 | 0.8000 | **−0.8305** | −0.8938 | 3 | ±0.171 |

**Do not use a uniform error bar.** The two extremes carrying the ordering test
(q0.667 and q4) are the three-cell families at **±0.171**.

> **Through-origin slope: 1.0984, 95% interval [0.958, 1.239], five-point fit
> including q2 and family A at matched `beta = 1.5`.**
> Against independently measured **alpha = 1.1173 [0.999, 1.236]** — intervals
> overlap; point estimates differ by **1.7%**.

**Alternatives in the record, all withdrawn**: 1.1240 (four points, q2 merged
into family A — a point-set error), 1.0969, 1.0878 (three families).
**Effective-`beta` alternative: 1.0547**, which is **5.6%** from alpha. Print
both the asymptotic (1.0984, 1.7%) and the effective (1.0547, 5.6%) figures.

**Ratio test (the primary registered test)**: extreme-pair ratio **measured
1.661** against **predicted 2.000**, deviation **16.9%**, inside the registered
**±25%** band — **PASS**.

R² about the *predicted* line falls to **0.684** with four constructed
families; diagnosed as resolution, not departure — every deviation lies inside
its own family's grid resolution and the mean signed deviation is **−0.0033**.

**Scope that must travel: Adam only.**

---

## 9. Cross-optimiser

| quantity | Adam | SGD |
|---|---:|---:|
| `alpha` (a = 1.25, 6 budgets 1k–40k, n=30/cell) | **1.2627 ± 0.0593** | **0.7188 ± 0.0238** |
| median `|w2|` at 40k | 97.48 | 19.38 (**5.0x lower**) |
| p10 `|w2|` at 40k | — | 0.73 (see below) |

**Note the window**: `alpha_adam = 1.2627` here is the **1k–40k** window, not
the 1k–160k headline **1.1172**. Both are correct; state the window.

**SGD family-A onsets** (`sgd_onsets.csv`, 4 of 4 bracketed, 2k–128k = 64x):

| budget | 2,000 | 8,000 | 32,000 | 128,000 |
|---|---|---|---|---|
| onset `a` | 1.60 | 1.38 | 1.22 | 1.16 |

> **SGD onset exponent −0.3255 ± 0.0226.**

**Registered P-1c** predicted `alpha_SGD = 1/(1 + 1.085) = 0.479`, band
**[0.38, 0.58]**; measured **0.719** — **FALSIFIED**.

**The chain does not close cross-optimiser:**

| ratio | value |
|---|---:|
| onset exponent SGD/Adam | **0.4434** |
| `alpha` SGD/Adam | **0.5693** |

If the law held across optimisers these would match; they differ by 28%.

**q4 under both**: Adam exponent **−0.8305** (3 bracketed cells). **SGD: onset
`ε` = 0.25 at every budget** (2k, 8k, 32k, 128k) — **exactly flat across 64x**,
so the SGD exponent is **0.0000** and the bound is `|exponent|` below grid
resolution. **q0.667 under SGD: unbracketed at every budget** (0 of 4), so no
exponent exists for it.

**SGD terminal weight distribution**: the p10 of `|w2|` is nearly **flat in
budget** (0.51, 0.39, 0.52, 0.57, 0.64, 0.73 across 1k–40k) while the median
rises 1.36 → 19.38. **A substantial lower tail does not move with budget at
all** — SGD's population splits rather than translating.

---

## 10. Family B

`f(t) = max(t, alpha t)`, non-monotone iff `alpha < 0`, threshold **exactly
`alpha = 0`**. **Positively homogeneous.**

| quantity | value |
|---|---|
| analytic `beta` | **1** (homogeneity forces it) |
| measured `beta` | **`G* ~ |alpha|^{0.9935}`** |
| `G*` values | 3.1196 (−1.0), 1.6000 (−0.5), 0.8000 (−0.25), 0.3200 (−0.1), 0.1600 (−0.05) |
| predicted onset exponent | **−1.1172** (the largest in magnitude, since `beta` is smallest) |
| measured | **+0.25 over a matched 64x range** — the **opposite sign** |
| onsets | `ε = 0.05` at **2,000** and at **16,000** (both bracketed); **64,000 unbracketed** |
| earlier 0.0000 figure | from an 8x range, **superseded** by the 64x measurement |
| solve rate across a 20x `G*` range | **0.565, 0.555, 0.580, 0.595, 0.595** — near-constant |
| compensation | median **`|w2| ~ G*^{-0.4747}`** |
| product variation | **4.93x** against **19.5x** for `G*` alone |
| required-scale overshoot | training overshoots the requirement by **900–1,200x and still fails** |

**One mechanism, two symptoms**: homogeneity gives `beta = 1` *and* full
compensation, which is why family B breaks **both** the budget law and the
product structure. These are not independent failures.

---

## 11. CIFAR

**Convergence run** (depth-8 CNN, CIFAR-10, Adam, fresh seeds,
`cifar_convergence.csv`):

| arm | n | mean test errors | SD | median epochs | plateaued |
|---|---:|---:|---:|---:|---:|
| GELU | 10 | **2,047.0** | 47.0 | 22.0 | 9/10 |
| ReLU | 10 | **2,179.7** | 45.0 | 31.5 | 9/10 |
| tanh | 5 | **2,591.8** | 36.0 | 21.0 | 5/5 |

**Plateau count: 23 of 25, not 24 of 25.** 9 + 9 + 5 = 23, and the source
document's own "two censored runs" (GELU seed 10, ReLU seed 4, both at the
40-epoch cap) agrees. The "24 of 25" figure propagated to four documents and is
wrong. The criterion passes either way.

| comparison | difference | p |
|---|---:|---|
| GELU vs ReLU | **132.7** | **4e-05** (two-sided permutation, 100k resamples) |
| tanh behind GELU | **544.8** | **3.5e-04** |
| tanh behind ReLU | **412.1** | **3.5e-04** |
| plateaued-only GELU vs ReLU | 2,049 vs 2,183 | 5e-05 |

**Training errors per 10,000**: GELU 173.0, ReLU 184.3, **tanh 744.4** — a
**4.0x** ratio. tanh is **underfitting**, not generalising differently.

**Margin shift** (`budget_flip.csv`, n = 8 per cell, seeds 100–107, **single
experiment**):

| budget | 2 epochs | 5 epochs | 12 epochs |
|---|---:|---:|---:|
| GELU-over-ReLU advantage | **750.4** | **418.4** | **150.6** |

> **Within-experiment ratio 750.4 / 150.6 = 4.98x.** Print this.

**5.65x is the cross-experiment figure** — it pairs the 2-epoch cell from this
sweep with a converged endpoint (132.7) from a *different* run with n = 10 and
a different seed set. The two agree where they overlap (150.6 vs 132.7, within
13%), but the ratio spans two seed populations. **Do not print 5.65x** except
to explain why it is not used.

**Note the contrast**: these are **GELU-over-ReLU** numbers. The GELU-over-tanh
advantage in the same experiment runs the *other* way with budget (251.4 →
486.9 → 478.4).

**The flip** (tanh vs ReLU):

| budget | tanh | ReLU | ordering | p |
|---|---:|---:|---|---:|
| 2 epochs | 3,424.5 ± 149.8 | 3,923.5 ± 180.2 | **tanh ahead by 499.0** | **3.8e-04** |
| converged (30–40 ep) | 2,591.8 ± 36.0 | 2,179.7 ± 45.0 | **ReLU ahead by 412.1** | **3.9e-04** |

**Crossing bound: between 2 and 30 epochs.** Status **undetermined** — the
registered rule requires the crossing at 5 epochs or later to carry the paper,
and present data cannot place it. **Report as the extreme case, not the
headline.**

**P-no-flip-GELU borne out**: GELU leads ReLU at 2 epochs (3,173.1 vs 3,923.5,
p = 2.0e-04) and at convergence, so the flip is specific to one pair.

**Test error rates**: 2 epochs **31.7–39.2%**; 5 epochs GELU 21.7%; 12 epochs
GELU 20.7%.

---

## 12. MNIST transfer

Architecture `784 → 256 → [w] → 128 → 10`, `f_a` at the bottleneck.

| quantity | value |
|---|---|
| **width-1 reversal** | monotonic **0.8658** vs non-monotonic **0.7715** |
| difference | **+0.0943 in favour of MONOTONIC** |
| p | **0.0002** (two-sided permutation, 20,000 resamples), n = 10 and 15 |
| widths tested | **1, 2, 3, 4, 6, 8** |
| thresholds swept | **0.80 to 0.97** — **no threshold produces the registered pattern** |
| fold activation | **10.2–34.7%** of bottleneck units past `|x| > arccos(−1/a)` — the test is **not vacuous** |
| budget control | 4x budget (8,000 steps) leaves the reversal intact: **0.9041 vs 0.8487** |
| ID estimators | MLE k=10 **8.04**, MLE k=20 **8.10**, TwoNN **13.67**; spread **8.03–13.83 = 1.72x** |

**1.72x is below the factor-2 threshold** the registration set, and far tighter
than the **1.99x** systematic disagreement that closed the earlier CIFAR
width-axis attempt.

**Verdict: registered outcome 4 — the setting does not exhibit the
phenomenon.** Control 2a failed, so Parts 3–5 were not run. This does **not**
show the law is false outside the toy task; it shows MNIST-MLP cannot pose the
question.

---

## 13. Registrations

| outcome | count | share |
|---|---:|---:|
| PASS | 21 | 45% |
| **FAIL** | **18** | **38%** |
| PARTIAL | 1 | 2% |
| UNRESOLVED | 7 | 15% |
| **total** | **47** | |

**v2 additions (2026-09-22)**, counted separately so the table above is unchanged:

| block | PASS | FAIL | UNRESOLVED |
|---|---:|---:|---:|
| A5d, k = 10 (A1–A5) | 1 (A1, uninformative) | 2 (A2, A4) | 2 (A3, A5) |
| A5d, k = 1 (A1–A5) | 2 (A1, A4) | 1 (A2) | 2 (A3, A5) |
| C, optimiser equivalence (3 pairs) | 1 | 2 (failed in the favourable direction) | 0 |
| cross-family X1, X2 | 1 (X2) | 1 (X1) | 0 |
| cross-family follow-up Y1 | 1 | 0 | 0 |
| **v2 total (16)** | **6** | **6** | **4** |

**Combined, 63 registered predictions: 27 PASS, 24 FAIL, 1 PARTIAL, 11 UNRESOLVED.**
The v2 unresolved outcomes are all onset-dependent predictions made undecidable by a
registered stop condition (no onset exists), not predictions left unscored.

Dates are **git-verified** from the commit that first added each registration
document.

**The seven unresolved were not abandoned after seeing results.** They divide
into three kinds, each checkable:

1. **Superseded before scoring** — P-1b, whose chain ceased to exist once P-1a
   was falsified.
2. **Demoted before running, on resolution grounds** — P-cost, whose
   measurement resolution **±0.431** exceeds the band half-width **0.26**, so
   it cannot adjudicate. **This was recorded before the data came in.**
3. **Gated on a failed control** — P-mnist-2b, never run because P-mnist-2a
   failed.

**A high failure rate reported openly is evidence of genuine pre-registration.**
Four failures overturned committed claims: P-barrier took the Arrhenius
programme with it, P-1a and P-1c cost the derivation of `alpha`, and
P-4a/4b/4c removed the basin account.

**Remaining trust requirement, stated plainly**: our timestamps are our own
commits. An external registration would remove it; we did not do that.

---

## 14. What is wrong in the draft

Ordered by severity.

### 14.1 The asymptotic constant is wrong in the record — **8/3 should be 4√2/3**

`D(a) ~ (8/3)(a−1)^{3/2}` appears in ledger T37 and
`results/fold1d_theorem.md`. The correct constant is **4√2/3 ≈ 1.8856**;
`D/(a−1)^{3/2} → 1.885610`. **No computed result moves** (everything calls the
exact `dip_depth`), but the printed form is wrong and a reviewer substituting it
gets a number 41% too large.

Knock-on: `results/asymptotic_finding.md`'s error table (42.7% / 47.6% / 70.2%
/ 136.1%) is computed against `8/3`. With the correct constant the errors are
**0.9% / 4.4% / 20.4% / 66.9%**. That document's *conclusion* — state the
theorem in `D(a)` and label the asymptotic — still stands.

### 14.2 "Found solutions have smaller margins" — right claim, wrong support

The exclusion table's `a = 1.5` row shows found margin **0.0771** *above* the
construction's **0.0384**. The real claim is a **population** statement: **6 of
24 found solvers (25%) at `a = 1.5` have margin below 0.0384**, minimum
**0.0054**. Attach it to the population with its n, not to the tabulated run.

### 14.3 "24 of 25 runs plateaued" — arithmetically impossible

9 + 9 + 5 = **23**, and the same document names **two** censored runs. Correct
to **23 of 25**. Already corrected in four documents; check the draft.

### 14.4 The Bonferroni figure mixes one- and two-sided

The ledger's **0.138** is 3 × the **two-sided** 0.0461, applied to a set of
**one-sided** p-values. The one-sided correction is **3 × 0.0231 = 0.069**. The
arm fails either way; print **0.069** with the one-sided values.

### 14.5 −0.8305 is q4's exponent, not a family-A window variant

`src/figures.py`'s watch list recorded "−0.8305 (3 cells, 2k–32k)" as if it
were a family-A fit. It is **q4's measured onset exponent** — a different
family. The family-A 4-cell fit is **−0.8261**. Figure *data* was always
correct; the comment was wrong and is now fixed.

### 14.6 "Five link families" is defined nowhere

Withdrawn. The enumeration gives **six link types** or **four families**;
neither is five. Enumerate rather than count.

### 14.7 Two cost-interval figures, both correct, easily swapped

**[1.151, 1.670]** inverts the onset exponent's statistical+grid interval
[−0.8691, −0.5989]. **[1.117, 1.681]** would invert the registered *acceptance*
band [−0.895, −0.595]. Print **[1.151, 1.670]** and say which interval it
inverts.

### 14.8 `alpha` has two correct values with different windows

**1.1172** (1k–160k, n = 8) is the headline. **1.2627** is the **1k–40k**
window used in the cross-optimiser comparison, where it is paired against
`alpha_SGD = 0.7188` on the same window. Never quote 1.2627 as the headline or
compare it against a different window's SGD value.

### 14.9 `a = 1.25` solve counts come from two arms

The 30-seed `budget_alpha` arm gives 0/30, 0/30, 2/30, 25/30, 27/30, 28/30,
30/30, 30/30. The headline "0 of 200 at 2,000 steps, 40 of 40 at 80,000 and
160,000" is from the larger onset sweep. **Both are right; do not mix the n's
in one sentence.**

### 14.10 Two witness exhibits, different point counts — and neither is GELU

**1,400,000 points at 0 errors** is the `a = 1.02` offset exhibit
(`src/offset_witness.py`), margins **0.54–0.85** from the committed run.
**2,000,000 points at 0 errors, worst margin 0.284** is the **absolute-value**
witness (`src/witness.py`, `pwl_family(-1)`), **not GELU**. An earlier version
of this document called the second a "GELU width-3 witness"; that was wrong.
Both are width-3 link-setting objects. Do not merge them.

### 14.11 The 4.98x contrast is GELU-over-ReLU

Not GELU-over-tanh. The GELU-over-tanh advantage in the same experiment
*increases* with budget (251.4 → 478.4), which is the flip seen from the other
side.

### 14.13 The `a = 1.02` exhibit and the 476 attempts belong to §3, not §4

Traced in source: `src/offset_witness.py` builds `nn.Linear(3,3)` layers on
`linked_tori` data with `tube_radius = 0.2`, evaluates 175,000 points per class
across four samples in `R^3`, and reports a linking trace `lk −1 → 0`. It is a
**width-3 link-setting object**. `src/offset_search.py` (the 476) imports
`cmaes` and runs on the same data at depths 3, 5 and 8.

An earlier version of this document placed both in §4, which is the **1D**
exclusion table. That was a mis-scoping in this document; **the paper draft did
not make the error** — `results_draft.md` and `link_section_draft.md` never
cite the 1.4M exhibit or the 476 figure. Corrected above.

**Consequence for §4 of the paper**: it has no targeted search to cite, and
must rest on sweep failure (0 of 200 at `a = 1.02`, best 100 eval errors)
stated as such.

### 14.12 Things listed in the brief that do not exist

- **"Adam-preconditioned distance"** exists as a check but **does not reverse
  the Euclidean finding**; it shows the reversal is metric-dependent. There is
  no separate preconditioned column in `exclusion_table.csv`.
- **q0.667 under SGD**: no exponent exists — **unbracketed at all four
  budgets**. Any SGD value for it in the draft is fabricated.
- **Family B's "four onsets"**: only **three budgets** were run (2k, 16k, 64k)
  and **two bracketed**. There is no fourth.
