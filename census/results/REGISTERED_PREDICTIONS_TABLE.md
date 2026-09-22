# Registered predictions across Blocks E, F, G, H, K — prediction, measurement, verdict

Every row is a prediction registered **before** its data. Values re-derived from
the committed artifacts, all checked by `verify_ledger.py` (0 findings).

---

## Block E — intervening on `|w₂|` (`a = 1.30`, 40 seeds/arm, 12,000 steps)

| # | registered prediction | measured | verdict |
|---|---|---|---|
| **E-1** | `hold_low` (`R = 0.85 R_glob`): **placement LOST**, ≥ 9/10 | **0 of 37 kept** — lost within **50 steps** (first checkpoint) in all 37; CP95 [0.0000, 0.0949] | **PASS** |
| **E-2** | `hold_high` (`R = 1.15 R_glob`): **placement KEPT**, ≥ 9/10 | **33 of 37 kept**; CP95 [0.7458, 0.9697]. Fisher vs `hold_low` **p = 1.2e-16** | **PASS** |
| **E-3** | `noise_floor` (×1+1e-6): indistinguishable from control | **37 of 37 kept**; Fisher vs `hold_high` p = 0.115 (null) | **PASS** |
| **E-4** | `cold_high`: placement **achieved** within the relaxation time | **2 of 40** (pilot at 1.50 `R_glob`, 20,000 steps: **0 of 9**) | **FAIL** |
| **E-5** | `jump`: places **well before** control | **9 of 40** vs control **37 of 40**, Fisher **p = 1.3e-10** — worse, not better | **FAIL (opposite direction)** |

### `hold_high`'s `R` against `R_solve` — the second threshold, confirmed causally

| quantity | value |
|---|---:|
| `R_glob` (placement switch) | 0.21446 |
| **`hold_high` held `R`** | **0.24663** |
| `R_solve` (solve switch) | 0.30667 |
| held `R` strictly between? | **YES** (+15.0% above `R_glob`, −19.6% below `R_solve`) |
| measured | **33/40 placed, 0/40 solved** |

Verified against the landscape itself: Block B's conditional minimiser at exactly
`|w₂| = 5.750` has gap **+0.022543** and **does not solve**. Solving is
**impossible** at that `R`, so 0/40 is the prediction met, not a shortfall.
**Block E therefore tests both switch points.**

### Saturation vs trapping (`blockE_stall_prediction.md`) — registered expectation was TRAPPING

| measurement | registered signature | measured | points to |
|---|---|---|---|
| gradient norm at step 500 | saturation if **< 10%** of control | control **1.57e-02**, jump **1.13e-05** (**0.07%**), cold_high **4.97e-06** | **saturation** |
| loss above the conditional minimum | trapping if **> 0.005** | jump **+0.341834**, cold_high **+0.367765** | trapping |
| 5× budget (60,000 steps) | saturation if placement rate **> 25%** | jump **7/15 = 46.7%** (from 2/15), cold_high 3/15 | **saturation** |

**Verdict: SATURATION.** Two of three signatures, and the budget arm is decisive —
a trapped run cannot be rescued by more steps, and these are. The large loss
excess is explained by the endpoint: an ill-conditioned flat valley near the
constant predictor (`w₁ ≈ −0.003`, Hessian condition number > 10⁷), not a wrong
basin. **Registered expectation was wrong.**

**Consequence: the annealing framing is NOT adopted** — its precondition was
trapping.

## Block H — growth-rate dose-response (`a = 1.30`, 40 seeds/arm)

| # | registered prediction | measured | verdict |
|---|---|---|---|
| **H-3** | **validity gate**: `m = 1` within **10 pp** of unconstrained control | `m=1` **20/40 = 0.500** vs control **37/40 = 0.925** — **42.5 pp**, Fisher **p = 4.3e-5** | **FAIL** |
| **H-1** | placement falls monotonically in `m`; endpoints separated by ≥ 0.25, Fisher p < 0.01 | **0.550 → 0.500 → 0.375 → 0.000**, separation **0.550**, **p = 8.4e-09** | *observed, but **not a valid test*** |
| **H-2** | solve rate follows the same ordering | **0.550 → 0.500 → 0.000 → 0.000** | *observed, same caveat* |
| — | (`m = 0.5` better than `m = 1`?) | 22/40 vs 20/40, Fisher **p = 0.823 — not separated** | no "slower is better" |

**H-3's failure makes H-1/H-2 uninterpretable as registered.** Cause: the natural
`|w₂|` trajectory is **non-monotone** (it dips before rising — seed 0 falls
0.371 → 0.162 before reaching 9.45), so a linear schedule is never a null. The
decline is real but confounded with departure from the natural trajectory.

**Second precondition for annealing fails independently.**

## Block F — three-optimiser lag predictions (`a = 1.25`, 60 seeds each, 20,000 steps)

`R_glob(1.25) = 0.21066`. Growth rates span **3.92×**.

| optimiser | crossing `R` | local rate | offset | predicted offset | error |
|---|---:|---:|---:|---:|---:|
| Adam | 0.23094 | 0.002471 | 9.627% | 9.627% | — |
| AdamW | 0.22980 | 0.001908 | 9.085% | 7.433% | 1.65 pp |
| SGD | 0.22888 | 0.000631 | **8.647%** | **2.458%** | **6.19 pp** |

| # | registered prediction | measured | verdict |
|---|---|---|---|
| **F-1** | `R(Adam) > R(AdamW) > R(SGD)` | ordering **holds exactly** — but bootstrap CIs overlap almost completely and **P(full ordering) = 0.290** vs 0.167 by chance | **no power** |
| **F-2** | offsets scale with the rate ratio, each within **±4 pp** | SGD misses by **6.19 pp** | **FAIL** |
| **F-3** | Spearman(rate, offset) **> 0** across six `a` | **+0.7827** | PASS, but **confounded** — rate and offset both rise with `a` while rates vary only 9% |

**A 3.92× change in growth rate moves the offset by 0.98 pp.** Per the
registration's own falsifier, **the relaxation-lag account is a label, not a
mechanism**. What survives: the offset is **robust and one-signed**, 8.6–9.6%
across three optimisers and 8.6–14.4% across six `a`.

## Block G — out-of-distribution windows against training-free predictions

| # | registered prediction | measured | verdict |
|---|---|---|---|
| **G-1** | crossing within **15%** of that window's own `R_glob` | **6 of 10**; all ten ratios **one-signed above** prediction, 1.082–1.191 | **FAIL as written** |
| **G-2** | `CV(R) < CV(\|w₂\|)/2` across windows at fixed `a` | a=1.30: **0.2189 vs 0.7358 (3.36×)**; a=1.50: **0.2304 vs 0.6925 (3.01×)** | **PASS both** |
| **G-3** | `Ĝ` ordering `G2_narrow < base < G1_wide` | holds at both `a` | **PASS** |
| **G-3b** | `G3_far_outer ≥ base` | **wrong by a provable inequality** (`O' ⊃ O` ⟹ `Ĝ(O',I) ≤ Ĝ(O,I)`); corrected **before** any crossing was measured | **FAIL (my error)** |
| **G-4** | each window's `κ₀` from `h(σ)` within **3%** of measured `κ(1.02)` | **5 of 5**, worst **0.808%**, across `κ₀` spanning **8.1×** | **PASS** |

### The G-1 table in full

| window | `a` | predicted `R_glob` | measured | ratio |
|---|---:|---:|---:|---:|
| base | 1.30 | 0.21290 | 0.23690 | 1.113 |
| G1_wide_gap | 1.30 | 0.22300 | 0.24139 | 1.082 |
| G2_narrow_gap | 1.30 | 0.12868 | 0.14367 | 1.116 |
| G3_far_outer | 1.30 | 0.20386 | 0.22637 | 1.110 |
| G4_shifted | 1.30 | 0.15313 | 0.16791 | 1.097 |
| base | 1.50 | 0.22107 | 0.25709 | **1.163** |
| G1_wide_gap | 1.50 | 0.23790 | 0.27953 | **1.175** |
| G2_narrow_gap | 1.50 | 0.13192 | 0.15127 | 1.147 |
| G3_far_outer | 1.50 | 0.20905 | 0.24527 | **1.173** |
| G4_shifted | 1.50 | 0.16314 | 0.19424 | **1.191** |

**All four misses are at `a = 1.50`, and the base task misses too (1.163)** — so
the offset is a property of `a`, not of the window: sd **0.014** at `a = 1.30`,
**0.016** at `a = 1.50`. Normalised by the base task's own offset at the same `a`,
**all ten land within 2.72% of 1.0**. The registered flat 15% tolerance did not
allow for the per-`a` lag the calibrated task itself shows.

## Block K — family B under the forced normalisation

| # | registered prediction | measured | verdict |
|---|---|---|---|
| **K-1** | `β_B = 1`, exponent within 0.01 | **1.000000** | **PASS** |
| **K-2** | overshoot falls by **exactly 8×**, stays **> 10×** | **7.99–8.00× in every row**; range **11.8×–145×** | **PASS** |
| **K-2b** | *"roughly 115–145×"* | **11.8×–145×** — I omitted the 2,000-budget row when registering | **FAIL (my arithmetic)** |
| **K-3** | `AUC(R_B) ≥ 0.90` **and** margin **≥ 0.10** over `\|w₂\|` | AUC **0.9990** (passes) but margin **0.0089** (fails) | **FAIL → claim removed** |
| **K-4** | onset exponent unchanged | structurally cannot move — no `Ĝ` in its inputs | **PASS** |

**K-3 could never have passed**: within fixed `α`, `Ĝ_norm` is a constant so
`R_B` and `|w₁w₂|` have **identical AUC by construction**, and **4 of 5 α cells
are perfectly separable** (band runs **2, 2, 2, 2, 69** against T54's own ≥ 20
requirement). The box was a *second* defect. Claim removed entirely.

---

## Summary

| block | passes | fails | notes |
|---|---:|---:|---|
| **E** | 3 | 2 | reversibility pair passes at **p = 1.2e-16**; both fails are informative — `R > R_glob` is necessary for placement to **persist**, not sufficient for it to be **found** |
| **E-stall** | — | — | **saturation**, not the registered trapping |
| **H** | 0 | 1 (gate) | dose-response observed but not a valid rate test |
| **F** | 1 (confounded) | 1 | lag account dropped as a mechanism |
| **G** | 3 | 2 | G-2 and G-4 are the strong results; G-1's misses are the documented per-`a` lag |
| **K** | 3 | 2 | AUC claim removed; `β` and overshoot restored |

**Nine registered predictions failed.** Every one is logged with its explanation,
and two of them (E-5, K-3) changed what the paper claims.
