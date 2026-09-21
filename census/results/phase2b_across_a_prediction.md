# Registration: does geometry carry the placement threshold?

**Written before any crossing value outside `a = 1.30` was read.** The sweep has
produced `a = 1.35` cells on disk, but no crossing statistic from them has been
computed or inspected. The only crossing numbers in hand are from `a = 1.30`:
median `R` at crossing **0.2330** (CV 0.067), median `|w2|` at crossing
**5.43**.

Date: 2026-09-21.

---

## 1. The across-`a` discriminator

At fixed `a`, `Ĝ(a)` is a constant, so "`R` constant at crossing" and "`|w2|`
constant at crossing" are the same statement. **They separate only across `a`**,
because `Ĝ` varies by a factor of ~5 over the sweep range:

| `a` | 1.30 | 1.35 | 1.40 | 1.45 | 1.50 | 1.60 |
|---|---|---|---|---|---|---|
| `Ĝ(a)` | 0.0894 | 0.1068 | 0.1240 | 0.1513 | 0.1745 | 0.2229 |

So if `R = |w2| Ĝ / 2` is constant at crossing, `|w2|` at crossing must fall by
~2.5x across the range, and vice versa. The two hypotheses are cleanly
separable.

### Registered outcomes

**O1 — geometry carries the threshold.** `R` at crossing is roughly constant
across `a`; `|w2|` at crossing falls with `a` roughly as `1/Ĝ(a)`.

**O2 — geometry is irrelevant.** `|w2|` at crossing is roughly constant across
`a`; `R` at crossing rises with `a` roughly as `Ĝ(a)`.

**O3 — neither.** Both vary materially.

### Tolerance, fixed now

The deciding statistic is the **coefficient of variation across the six `a`
values** of the per-cell medians.

- **O1** if `CV(R_cross) <= 0.15` **and** `CV(R_cross) < CV(|w2|_cross) / 2`.
- **O2** if `CV(|w2|_cross) <= 0.15` **and** `CV(|w2|_cross) < CV(R_cross) / 2`.
- **O3** otherwise.

0.15 is chosen as roughly twice the within-cell CV already measured at
`a = 1.30` (0.067): a quantity that is no more variable across `a` than it is
within a single cell, to within a factor of two, is constant for our purposes.

**Registered prediction: O1.** Grounds: the crossing is a statement about
whether the unit's image separates the windows, and `Ĝ(a)` is precisely the
best separation the activation can produce. `|w2|` enters only by scaling. If
this fails, the geometric factor is decoration and the paper's central quantity
is just `|w2|`.

## 2. A training-free prediction of both thresholds

**Hypothesis (slow-variable / adiabatic).** `w2` is a slow variable; at any
moment `(w1, b1, b2)` sit near the minimizer of the training loss *conditional
on the current* `w2`.

If true, both thresholds are computable **without training**:

- Sweep `w2` on a fine grid. At each value, minimize the same
  cross-entropy on the same 400 training points over `(w1, b1, b2)` from many
  restarts.
- **`w2*_cross`**: the smallest `|w2|` at which the conditional minimizer (or
  its dominant basin) has `G > 0`. Predict
  **`R_cross(a) = |w2*_cross| Ĝ(a) / 2`**.
- **`w2*_solve`**: the smallest `|w2|` at which the conditional minimizer is
  sign-correct on the dense grid. Predict **`R_solve(a)`** likewise.

**Committed before looking at the sweep.** Both predictions are computed and
written to `results/phase2b_conditional.csv` before any further crossing
statistic is read, and scored against the measured `0.2330` at `a = 1.30` and
against the `0.30–0.50` solve band.

**Adiabatic check.** On the existing trajectories, measure the distance between
`(w1, b1, b2)` at each checkpoint and the conditional minimizer at that
checkpoint's `w2`. Close tracking makes the mechanism **established** rather
than fitted. Registered: median relative distance **< 0.2** counts as tracking.

**If the conditional minimizer never switches, or switches at a materially
different `w2`, that is reported as a failure of the adiabatic hypothesis**, and
the measured threshold stands as empirical.

## 3. What governs the second stage

70 runs crossed at `a = 1.30`; 54 solved; the 16 failures all have
`placement_ok = 1.00` and `bias_ok = 0.00`. The admissible bias interval has
width `|w2| G(w1,b1)`, which is **near zero at the crossing** because `G`
crosses zero there.

So the second stage may be waiting for **placement quality** (`rho`) to improve
rather than for `|w2|` to grow further.

**Registered**: for the runs that crossed and solved, report at the solve step
both `R` and the run's own attainable margin `R * rho`. **Whichever is less
variable across runs and cells is the quantity governing the second stage.**
Deciding statistic is again CV, same 0.15 tolerance.

**Registered prediction**: `R * rho` is the less variable, because the bias
condition depends on `|w2| G` — which is exactly `2 R rho` — and not on `|w2|`
alone.

## 4. Phase 3, sharpened with these numbers

Registered before Phase 3 runs:

1. Runs scaled from below `R ≈ 0.233` to above it **achieve placement within
   the fast relaxation time** (registered as within 500 steps, the observed
   scale of post-intervention transients).
2. The **freeze arm held below `R = 0.233` never achieves placement**.
3. Runs **frozen between 0.233 and the solve threshold reach placement but
   fail on bias** — placement rate high, solve rate near zero.

Prediction 2 is the sharpest: it is a claim that a specific intervention
produces exactly zero of an outcome that is otherwise common.
