# Cross-family test of the scaling reduction: results

Scored against `crossfamily_prediction.md` (registered `a649e31` before any
threshold was computed). Artifacts: `crossfamily_thresholds.csv`,
`crossfamily_scores.csv`; producer `src/crossfamily_landscape.py` (the frozen
Block B procedure with only the activation substituted; steps scaled so the
`R`-resolution equals family A's at each `a`).

## Verdicts

| prediction | result |
|---|---|
| **X1**: q2 within `tol` of family A at all six `a`, for both `R_glob` and `R_solve` | **FAIL** — holds for `R_glob` (6 of 6), fails for `R_solve` (0 of 6) |
| **X2**: q1 outside `tol` at ≥ 4 of 6 `a` for some threshold | **PASS** — `R_glob` outside at 6 of 6, `R_solve` at 5 of 6 |

## Thresholds

| `a` | `R_glob` A | `R_glob` q2 | `R_glob` q1 | tol | `R_solve` A | `R_solve` q2 | `R_solve` q1 | tol |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.30 | 0.2153 | **0.2050** | 0.2782 | 0.0153 | 0.3078 | 0.2954 | 0.3191 | 0.0043 |
| 1.35 | 0.2165 | **0.2157** | 0.2799 | 0.0166 | 0.3074 | 0.2879 | 0.3200 | 0.0053 |
| 1.40 | 0.2189 | **0.2238** | 0.2786 | 0.0190 | 0.3058 | 0.2754 | 0.3204 | 0.0064 |
| 1.45 | 0.2197 | **0.2319** | 0.2773 | 0.0198 | 0.3069 | 0.2622 | 0.3190 | 0.0076 |
| 1.50 | 0.2233 | **0.2383** | 0.2733 | 0.0234 | 0.3065 | 0.2558 | 0.3171 | 0.0088 |
| 1.60 | 0.2300 | **0.2519** | 0.2688 | 0.0301 | 0.3085 | 0.2575 | 0.3024 | 0.0112 |

- **`R_glob`: q2 matches family A** at every `a` (differences −0.010 to +0.022,
  all within tolerance), and **q1 does not** (+0.039 to +0.063). The placement
  threshold separates the families exactly as the reduction says.
- **`R_solve`: q2 does not match.** It sits below family A by 0.012 at `a = 1.30`,
  and the gap grows to 0.051 at `a = 1.50–1.60`. Family A's `R_solve` is flat in
  `a` (0.306–0.309); q2's falls from 0.295 to 0.256.

## What this means, per the registration

X1 failed, so the landscape thresholds differ too. The registered reading
applies: **the reduction misses something at these ε**, and that goes in the
math note. **The q2–A difference in pooled `R₅₀` (0.3032 vs 0.3723) cannot be
attributed to dynamical lag on the strength of this test**, because the two
families' `R_solve` already differ on the landscape.

## Diagnosis — POST HOC

The q-families are polynomial only for `|t| ≤ 1` and continue linearly with
slope 1 outside (`depth_families.make_family`). Their reduction
`f = ε^{1+1/q}[h_q + ε g_q]` is exact only while `ε^{1/q}|σ| ≤ 1` — a condition
already stated in T70. At the conditional minimisers:

| `a` | threshold | range of `t = w₁x + b₁` on the windows | share of points with `|t| > 1` |
|---:|---|---|---:|
| 1.30 | glob | [−0.670, +1.599] | 25.0% |
| 1.30 | solve | [−0.642, +1.516] | 25.0% |
| 1.45 | glob | [−0.781, +1.865] | 28.4% |
| 1.45 | solve | [−0.769, +1.828] | 27.4% |
| 1.60 | glob | [−0.869, +2.109] | 34.1% |
| 1.60 | solve | [−0.867, +2.101] | 33.9% |

**At every tested `a`, a quarter to a third of the window lies on the q2
family's linear continuation, where it is not a rescaled sin family.** The test
was run outside the domain of the q2 reduction, which requires roughly
`ε ≲ 0.05` for the placements used (`max|σ| ≈ 4.4`). That was a design error in
the registration: the `a` values were copied from family A's Block B grid
without checking the q-family domain condition.

Two further points, stated as observations:

- The domain violation is equally present at `R_glob` and `R_solve`, so it does
  not by itself explain why one agrees and the other does not. A plausible
  mechanism is that `R_solve` depends on the loss minimiser's choice of `b₂`,
  which weighs every window point, tails included, while `R_glob` depends on the
  placement's edge-limited class gap, set near the fold (T58: the active
  constraints sit at `x = −0.8` and `x = −1.2`). **Not tested.**
- q2's `R_solve` (0.256–0.295) lies close to its pooled `R₅₀` (0.3032). Family A's
  `R_solve` (≈ 0.307) lies 21% below its pooled `R₅₀` (0.3723). So part of the
  pooled difference may be landscape and part dynamics. This test does not
  separate them.

## A valid version of the test

Run the same comparison at `a` where `ε^{1/2}·max|σ| ≤ 1` for q2, i.e.
`a ∈ {1.01, 1.02, 1.03, 1.05}`, computing family A's thresholds there as well
(Block B has none below 1.30). **Proposed, not run.** It needs registering, and
family A's thresholds at those `a` need computing.
