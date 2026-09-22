# Item 1: both exceptions above `R = 0.50`, identified and diagnosed

Under the certified `Ĝ` (T65) the pooled union is **461 of 463 solved above
`R = 0.50`**. Both exceptions are at **`a = 3.0`, Adam, budget 2,000**, from
`fold1d_sweep.csv` via `r_pooled.csv`. Artifact: `exceptions_above_050.csv`.

**They are different failure modes and must not be described as one kind of
near-miss.**

---

## Exception 1 — `a = 3.0`, seed 13, `|w₂| = 1.099398`, `R = 0.578447`

The already-known one, unchanged from the restricted-`Ĝ` count.

| | |
|---|---|
| placement `G(w₁,b₁)` | **+0.768428 — OK** |
| bias interval for `b₂` | `(+2.342373, +3.187182)`, width 0.844809 |
| actual `b₂` | **+2.331415 — FAILS, misses the interval by 0.010958** |
| dense violations | **11 of 4,000 outer**, 0 inner |
| where | `x ∈ [1.2000, 1.2040]` — the **inner edge of the outer window** |
| worst outer logit | **−0.010958** |
| 200-point sample errors | **1** |
| parameters | `w₁ = −1.452066`, `b₁ = −1.944588`, `w₂ = +1.099398`, `b₂ = +2.331415` |

**A genuine near-miss**: the placement is good, the fold is correctly positioned,
and the run simply lands `b₂` about 1% of the bias interval's width outside it.
The violations are confined to a 0.004-wide strip at the window edge.

## Exception 2 — `a = 3.0`, seed 82, `|w₂| = 0.951985`, `R = 0.500886`

**New**, and not a near-miss at all.

| | |
|---|---|
| placement `G(w₁,b₁)` | **−0.216202 — FAILS** |
| bias | not defined (placement must hold first) |
| dense violations | **1,847 of 4,001 inner and 2,000 of 4,000 outer — 3,847 total** |
| where | inner `x ∈ [0.0616, 0.8000]`, outer **all of** `x ∈ [−2.0, −1.2]` |
| 200-point sample errors | **185 of 400** |
| parameters | `w₁ = **+0.019358**`, `b₁ = +0.073460`, `w₂ = +0.951985`, `b₂ = −0.284067` |

`w₁ ≈ 0.019` places this in the **constant-predictor region** — the degenerate
`w₁ → 0` configuration that `is_degenerate` excludes from Block B, and that the
Block E stall analysis identified as an ill-conditioned flat valley. The network
is nearly constant across both windows; it fails on roughly half the domain.

### Why it appears above `R = 0.50` at all

| | `R` |
|---|---:|
| under the **restricted** `Ĝ` (1.048901) | **0.499269 — below 0.50** |
| under the **certified** `Ĝ` (1.052298) | **0.500886 — above 0.50** |

It is a **new entrant**, admitted by the +0.324% rise in `Ĝ(3.0)`, and it clears
the cutoff by **0.0009**.

The structural reason it can get there with `|w₂| < 1` is that `a = 3.0` has
`Ĝ ≈ 1.05`, so `R = |w₂|Ĝ/2 > 0.5` needs only `|w₂| > 0.95` — far below what any
smaller `a` requires. **`R` is a product, and at large `a` the geometry term
nearly clears the cutoff on its own.** That is a known limitation of `R` as a
single scalar, already visible in T52's finding that `a = 2.0/3.0` fail the
budget-homogeneity condition on near-zero band coverage.

## What to state

> Above `R = 0.50`, **461 of 463** pooled runs solve. Both exceptions are at
> `a = 3.0`, Adam, budget 2,000. **Seed 13** is a bias near-miss: placement is
> correct (`G = +0.768`), `b₂` misses its interval by 0.011, and the 11 dense
> violations lie in `x ∈ [1.2000, 1.2040]` with 1 sample error. **Seed 82** is a
> placement failure in the degenerate `w₁ ≈ 0.019` constant-predictor region, with
> 3,847 dense violations and 185 sample errors; it sits at `R = 0.5009`, clearing
> the cutoff by 0.0009 only because `a = 3.0`'s large `Ĝ ≈ 1.05` lets `|w₂| < 1`
> produce `R > 0.5`.

The honest summary is that **one exception is a near-miss and one is a
large-`a` artifact of `R` being a product** — not that there are "two near-misses".
