# The harder search for a monotonic solution: outcomes (Part 2b)

Predictions were registered in `search_prediction.md` before implementation.
Reported in registered order. Data: `search_restarts.csv`, `search_swap.csv`,
`search_anneal.csv`, `search_direct.csv`.

**No search found a monotonic width-3 network with 0 eval errors.** But one
of them found something the sweeps could not have: monotonic networks with
**0 train errors** and nonzero eval errors, which relocates what the barrier
is about. Both results are reported below.

## Search 1: mass restarts — 0 separations in 400 runs

| Configuration | n | separations | min | p25 | median | p75 | runs < 9 errors |
|---|---:|---:|---:|---:|---:|---:|---|
| `sin(0.95)`, depth 8 | 200 | **0** | 2 | 23 | 60 | 74 | 2,3,4,5,5,5,8,8 |
| tanh, depth 8 | 200 | **0** | 26 | 64 | 72 | 83 | none |

Ten times the standard seed count at the best-performing monotonic
configuration: the 200-seed best equals the 20-seed best (2 errors, the same
seed-2 run), seven more runs enter the sub-9 band, and none reaches 0. Seeds
0–19 duplicate the earlier sweeps by design and reproduced them exactly
(bit-identical eval accuracies, both configurations) — a recovery check.

## Search 2: descend from a separating GELU solution — the swap always breaks

All six separating width-3 GELU networks were reconstructed exactly
(deterministic retrain, verified at eval accuracy 1.0). Swapping the
activation with weights frozen at the GELU solution:

| Target | best errors after swap | best after tune (lr 1e-3) | best after tune (lr 1e-2) |
|---|---:|---:|---:|
| leaky-ReLU | 351 | 71 | 74 |
| tanh | 883 | 89 | **52** |
| `sin(1.0)` | 681 | 158 | 92 |
| `sin(0.95)` | 681 | 124 | 97 |

The swap is catastrophic — 351 to 1,000 errors immediately (1,000 is
chance), across all 24 source × target combinations. Fine-tuning from the
GELU solution recovers only into the ordinary monotonic band (best 52,
tanh), no better than training from scratch. **The registered strongest
overturn — the fold surviving the swap — did not occur, in the strongest
possible way:** the GELU weights are not merely suboptimal for monotonic
activations; they are near-chance, so the separating solution does not sit
in monotonic parameter space at or near those weights.

## Search 3: annealing through the threshold — all 12 traces fail above it

All twelve `sin(a=3)` separating networks were annealed down in steps of
0.05 with 200 adaptation steps per value. The largest `a` still holding 0
eval errors, per trace:

1.1, 1.15, 1.2, 1.2, 1.25, 1.25, 1.3, 1.3, 1.3, 1.4, 1.65, 1.8
(median 1.275)

- **Every trace loses separation strictly above the threshold** — none held
  0 at any `a ≤ 1.0` (registered overturn condition: not triggered).
- The registered against-the-reading condition (`a* > 2`, which would have
  said the annealing path is the obstacle) was also not triggered: 10 of 12
  traces fail at or below 1.4.
- Minimum errors at `a ≤ 1` per trace: 25–93. Once lost, separation is
  never regained on the way down.
- The failure region `[1.1, 1.8]` overlaps the sweep's cold-start onset
  (first separations at `a = 1.10`) from the other side: warm-started
  networks hold separation down to roughly where cold-started networks
  begin to find it. The practical barrier sits just above the analytic
  threshold and is approached consistently from both directions.

## Search 4: derivative-free direct search — 0 of 120 monotonic restarts, passing positive control

**Objective deviation from the registration, documented.** The registered
error-count objective failed its own positive control: on `sin(1.5)`,
count-driven CMA-ES stalled at 11–57 errors across budgets to 6,000
generations. The count landscape is a staircase and the search cannot
descend it near zero. The objective was changed to smooth full-batch
cross-entropy (counts used for verification only), after which the positive
control passes. Probes of a misclassified-focused tie-breaker were worse
than the cross-entropy tie-break. All results below use the smooth
objective at matched budget (population 15, ≤4,000 generations, 20
restarts per cell).

| Target | depth 3 | depth 5 | best train errors | best eval errors |
|---|---:|---:|---:|---:|
| `sin(1.5)` — **positive control** | **12/20 separated** | 5/20 | 0 | **0** |
| `sin(1.0)` | 0/20 | 0/20 | **0** | 9 |
| `sin(0.95)` | 0/20 | 0/20 | **0** | 13 |
| tanh | 0/20 | 0/20 | 43 | 60 |

### The train-zero finding

Three monotonic restarts reached **exactly 0 training errors on 2,000
points** (`sin(1.0)` depth 3, restarts 2 and 3, final loss down to 9.9e-5;
`sin(0.95)` depth 5, restart 8) — and their eval errors were 19, 16, 18.
Across all 120 monotonic restarts the best eval error was 9, never 0.

This is the registered 0-train/>0-eval outcome, and it matters beyond this
search: **a monotonic width-3 network can shatter the 2,000-point sample.
What it has not done, under any method tried, is generalize that to 0 on a
fresh sample from the same solid tori.** The barrier is not about fitting
capacity on finite point sets — it is about carrying the separation to the
region the points were drawn from, which is what a topological obstruction
would require and what mere interpolation cannot deliver. This finding
makes the Part 2c dense-sample verification necessary for the GELU side
too: eval-0 on 2,000 points is itself a sample criterion, and every
separating network must now be checked densely.

## The upgraded claim

As registered in advance for this outcome: no separating monotonic width-3
network was found by —

- SGD from default initialization: **0 in 4,970 runs** (4,610 prior + 360
  new restart seeds),
- descent from a known-good non-monotonic solution: **0 in 24 swaps × 2
  fine-tune schedules**, all swaps landing at 351–1,000 errors,
- activation annealing across the threshold: **0 of 12 traces** held
  separation at any monotonic parameter,
- derivative-free direct search: **0 of 120 CMA-ES restarts**, under a
  machinery whose positive control separates 12/20 at depth 3.

Each count stated separately; they are different search families and are
not pooled. The monotonic zero survives its own strongest attack so far,
and the attack surfaced the sharpest characterization yet of what the zero
means: monotonic networks can reach 2 errors by SGD and 0 *train* errors by
direct search, and the eval zero remains empty.

> **Count correction (2026-08-22, ledger audit).** "4,970 runs" above has the
> same omission as earlier totals (width-sweep and protocol-sweep monotonic
> strata excluded). Authoritative total: **5,570**, `CLAIMS.md` (T1).

---

> **Audit correction (2026-08-25).** The annealing localization above
> quoted "every trace loses separation at a\* ∈ [1.1, 1.8] (median
> 1.275)". Recomputation (`AUDIT.md` finding 2) shows the median 1.275
> corresponds to the midpoint-of-first-loss definition, whose values
> span **[1.075, 2.275]**; the quoted interval matches no definition of
> the statistic. Two of the 12 traces (depth 5, seeds 10 and 16) are
> **re-entrant** — they fail, regain separation at lower `a`, and fail
> again (final losses at 1.375 and 1.125); the remaining ten are
> single-transition. No trace fails below the analytic threshold
> (min 1.075). Ledger row T19 carries the corrected statement.
