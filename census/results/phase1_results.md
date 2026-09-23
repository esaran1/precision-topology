# Phase 1: what determines sign correctness

Registered in `results/phase1_prediction.md` (commit `ab7556d`; *corrected 2026-09-23 from `7db80cf`, which is the decomposition commit*) before the
decomposition was computed. Data: `phase1_runs.csv`, `phase1_decomposition.csv`
— 12 values of `a` × 200 seeds × both precisions, Adam, lr 1e-2, 2,000 steps.

---

## Question

Sign correctness is equivalent to two conditions, **placement**
(`G(w1,b1) > 0`) and **bias** (`b2` inside an interval of width `|w2| G`).
`|w2|` does not appear in the placement condition. Which condition do failures
violate, and does `|w2|` relate to placement at all?

## Validity: the identity holds exactly

**0 disagreements in 4,800 runs** between the placement-and-bias classification
and `solves()`. The decomposition is an algebraic identity, so this is the
strongest available check that the implementation is right — and it found a
real bug first (see *Orientation*, below).

Grid-vs-continuous placement sign differs on **0 runs**, so no conclusion here
is a grid effect.

## Result: failures are placement failures

| | n | share |
|---|---:|---:|
| **placement** (`G ≤ 0`) | **1,664** | **84.5%** |
| **bias** (`G > 0`, `b2` outside) | 306 | 15.5% |

**P1-a confirmed.** Neither stop-and-report condition triggered: bias failures
do not dominate, and low-`|w2|` runs with good placement are not the story
(below).

**Sign balance** (registered check for residual orientation error):

| | n | placement | bias |
|---|---:|---:|---:|
| `w2 > 0` | 1,051 | 84.0% | 16.0% |
| `w2 < 0` | 919 | 85.0% | 15.0% |

**1.0 pp apart** — no residual orientation error.

By `a`, placement's share of failures is 100% at `a ≤ 1.15`, falls to ~51% in
the transition band (`a` = 1.40–1.50), and returns to 95–98% at `a` = 2.0–3.0,
where the few remaining failures are runs that never left the origin.

## The mechanism result: within a cell, placement tracks `|w2|`

Pooling over `a` is misleading — the low-`|w2|` quartile is dominated by
large-`a` runs where `Ĝ` is large. **Within each cell** (`a` fixed, `B` = 2,000),
`P(G > 0)` against terminal `|w2|` quartile:

| `a` | Q1 | Q2 | Q3 | Q4 | corr(quartile, P) |
|---|---:|---:|---:|---:|---:|
| 1.25 | 0.000 | 0.000 | 0.000 | 0.020 | +0.775 |
| 1.30 | 0.000 | 0.000 | 0.020 | **0.480** | +0.796 |
| 1.35 | 0.000 | 0.000 | 0.360 | **0.940** | +0.925 |
| 1.40 | 0.000 | 0.080 | **0.960** | **1.000** | +0.921 |
| 1.45 | 0.000 | 0.460 | **1.000** | **1.000** | +0.947 |
| 1.50 | 0.000 | 0.680 | **1.000** | **1.000** | +0.909 |
| 2.00 | 0.000 | 0.920 | **1.000** | **1.000** | +0.815 |
| 3.00 | 0.000 | 0.800 | **1.000** | **1.000** | +0.868 |

**Within every cell, `P(G > 0)` rises monotonically with terminal `|w2|`**,
correlation **+0.78 to +0.95**. The lowest quartile (`|w2| ≈ 0.4–0.5`, runs that
barely moved) has `P(G > 0) = 0.000` **in every cell without exception**.

This is the answer to the question the decomposition raises. `|w2|` is absent
from the placement condition, yet within a fixed condition it **predicts
placement**. So `|w2|` is not acting as a threshold on the criterion; it is a
marker of how far training travelled, and travelling far is what produces
placement. **Correlational**: Phases 2b and 3 test the ordering and the
intervention.

## Placement is close to all-or-nothing

Among runs grouped by `rho = G / Ĝ`:

| `rho` | runs | solved |
|---|---:|---:|
| `G ≤ 0` | 1,664 | 0 |
| 0 < `rho` ≤ 0.5 | 221 | 3 |
| 0.5 < `rho` ≤ 0.9 | 207 | 119 |
| **`rho` > 0.9** | **308** | **308** |

**Every run with `rho > 0.9` solves — 308 of 308.** Good placement is almost
always accompanied by an admissible bias, so once the unit is placed well the
bias condition is nearly automatic.

*(The `G ≤ 0` row solving 0 times is an identity, not evidence: `G > 0` is
necessary for sign correctness. It is listed as a consistency check.)*

## P1-b: FAILED as registered

Registered: median `rho` among solved runs in **[0.5, 0.95]**.
Measured: **0.9674** (IQR [0.871, 0.974]).

**Why the registration was wrong.** I inferred `rho ≈ 1/1.274 = 0.785` from the
theorem verification's median slack. That inference does not hold: slack is
`|w2| / (2m/Ĝ)` where `m` is the run's **achieved margin**, which depends on
where `b2` sits. `rho` is `G(own placement)/Ĝ`, which does not involve `b2` at
all. The two quantities differ by exactly the margin the run chose to take.

The measured result is **stronger** than what I registered: solved runs place
near-optimally rather than partially. `rho` rises with `a` (0.68 at `a` = 1.35,
0.97 at `a` = 2.0–3.0), so near the transition solved runs do accept
meaningfully sub-optimal placement.

## The four near-misses

All four runs with 0 sample errors that fail the dense region are **bias**
failures, with `b2` just outside the admissible interval:

| `a` | seed | gap | `b2` | admissible interval | shortfall |
|---|---:|---:|---:|---|---:|
| 1.5 | 132 | +0.1231 | 11.8248 | (11.8456, 12.3387) | 0.021 |
| 1.5 | 145 | +0.1381 | 13.9575 | (13.9602, 14.6117) | 0.003 |
| 1.5 | 178 | +0.0885 | 13.3901 | (13.3982, 13.7966) | 0.008 |
| 3.0 | 13 | +0.7684 | 2.3314 | (2.3424, 3.1872) | 0.011 |

Placement is fine in every case. They miss on `b2` by 0.003–0.021, which is why
the violation is a sliver at a window edge invisible to a 200-point sample.

## Certification: all 870 solved runs are certified region-wide

A dense grid is finite. Since `|N'(x)| ≤ |w2||w1| sup|f_a'|`, a run whose
minimum grid margin exceeds `L h / 2` is sign-correct **between** grid points
too, hence on the whole window.

- At the production grid (`h = 4.0e-4`): **866 of 870** certified.
- The remaining 4 have margins of 2.8e-4 to 2.1e-3. A **local** Lipschitz
  constant (sup of `|1 + a cos z|` over the actual pre-activation range rather
  than the global `1 + a`) shaves only 5–12% and does not certify them.
- Refining to `h = 4.0e-5` certifies **all four**.

**All 870 solved runs are certified sign-correct on the full windows.** The
solve criterion is *certified*, not sampled, and the finite-sample caveat is
answered.

## Orientation: a bug the registration caught

The registered validity condition — the classification must agree with
`solves()` run for run, because the relation is an identity — fired on the first
test case. For `w2 < 0` the admissible bias interval endpoints were swapped
(dividing by a negative flips the order), making the interval **empty always**.
Every `w2 < 0` run would have been reported a bias failure: **the registered
competing hypothesis P1-c, manufactured**.

Five other orientation-dependent sites were audited; **none changed**. Logged as
instrument-artifact instance 6, with the new per-placement theorem check (T56)
produced by that audit.

## Precision

`float32` and `float64` **agree run-for-run given a shared initialisation** —
180 paired runs, 0 flips, relative parameter distance 0.0000. An earlier
apparent disagreement was an RNG artifact (`uniform_` consumes the stream
differently per dtype), now corrected in `phase1_relog.py` and retracted in
`precision_prediction.md`. Phase 1 therefore describes the **same population**
as the float32 headlines; numbers above are float32.
