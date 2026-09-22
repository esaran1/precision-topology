# Block A5d at k = 10: results as registered

Scored against `blockA5d_prediction.md` (committed `3a75e54` before any run).
Artifacts: `blockA5d_runs.csv` (220 runs × 4 budgets), `blockA5d_scores.csv`,
`blockA5d_perfect_fraction.csv`, `blockA5d_controls.csv`; producers
`src/blockA5d.py`, `src/blockA5d_analyze.py`.

**Registered stop condition triggered: no onset brackets at any budget.** There
was **not one perfect run** in the whole grid — 11 activations × 20 seeds × 4
budgets, 880 run-budgets. Stage 2 was not run (the registered rule adds seeds
only where an onset brackets), and neither was the protocol cell.

## Registered predictions

| id | prediction | measured | verdict |
|---|---|---|---|
| A1 | zero perfect runs for a ≤ 1 | 0 of 160 run-budgets (fewest held-out errors 555) | passes, **but carries no information**: nothing is perfect anywhere |
| A2 | a bracketed onset `a_on(B) > 1` at every budget | no onset at any budget | **FAIL** |
| A3 | `a_on` non-increasing in B | no onsets to compare | undecidable |
| A4 | at a = 3.0, B = 64k, at least one perfect run | 0 of 20 | **FAIL** |
| A5 | γ and α_L/β both positive | γ undefined (no onsets); α_L = 0.4748 | undecidable |

## What the runs show

At 64k steps (held-out n = 20,000):

| arm | median held-out acc | best | fewest errors | best train acc |
|---|---:|---:|---:|---:|
| a = 0.9 | 0.9216 | 0.9628 | 743 | 0.9696 |
| a = 1.0 | 0.9210 | 0.9722 | 555 | 0.9770 |
| a = 1.05 | 0.9208 | 0.9760 | 480 | 0.9813 |
| a = 1.1 | 0.9175 | 0.9813 | 374 | 0.9863 |
| a = 1.2 | 0.9185 | 0.9807 | 386 | 0.9879 |
| a = 1.35 | 0.9339 | 0.9695 | 610 | 0.9753 |
| a = 1.5 | 0.9320 | 0.9708 | 584 | 0.9764 |
| a = 2.0 | 0.8953 | 0.9423 | 1,154 | 0.9525 |
| a = 3.0 | 0.8806 | 0.9090 | 1,821 | 0.9223 |
| ReLU | 0.7919 | 0.8980 | 2,039 | 0.8953 |
| GELU | 0.8698 | 0.9210 | 1,579 | 0.9277 |

- **No network fits even its training set**; the best training accuracy anywhere
  is 0.988.
- **The expressivity threshold is unobservable at this k.** Networks at a ≤ 1,
  which Theorem 4.7 bars from perfect separation, score like those at
  a = 1.05–1.2. Capacity limits at k = 10 swamp the monotonicity barrier.
- **Consistent with Ren & Lim's own numbers**: their Table 4 reports 91.1% best for
  GELU at k = 10; our GELU's best is 0.921. Their multicopy experiment exists to
  show errors growing with k at fixed width.

## The design error

The success criterion (zero held-out errors) was fixed without a pilot checking
that any arm could meet it at k = 10, width 5. It could not. The rerun at k = 1 is
gated by exactly that pilot (`blockA5d_k1_prediction.md`).
