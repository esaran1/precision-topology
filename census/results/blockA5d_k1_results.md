# Block A5d at k = 1: results — registered stop condition

Scored against `blockA5d_k1_prediction.md` (registered at `68c4e5d`, before any
stage-1 run). Artifacts: `blockA5d_k1_runs.csv` (220 runs × 4 budgets, seeds
100–119), `blockA5d_k1_scores.csv`, `blockA5d_k1_perfect_fraction.csv`,
`blockA5d_k1_controls.csv`; pilot `blockA5d_k1_pilot.csv`,
`blockA5d_k1_stratcheck.csv`; producers `src/blockA5d_k1.py`,
`src/blockA5d_analyze.py` (`score k1`). Decided 2026-09-22 21:11 local, before the
09:00 cutoff. **Perfect = zero errors on the 20,000-point uniform set and the
20,000-point stratified (linking-region) set.**

## Registered verdicts

| prediction | result |
|---|---|
| **A1**: zero perfect runs at `a ≤ 1`, every budget | **PASS**: 0 of 160 run-budgets. No `a ≤ 1` run reached zero uniform errors, so the violation check was never triggered. Minimum uniform errors 22 (`a = 1.0`) and 30 (`a = 0.9`) at 64k |
| **A2**: a bracketed onset `a_on(B) > 1` at every budget | **FAIL**: no grid `a` reaches ≥ half perfect at any budget (maximum 0.35, `a = 3.0` at 64k) |
| **A3**: `a_on` non-increasing, strictly lower at 64k than 1k | **undecidable** (no onset exists at any budget; recorded as at k = 10) |
| **A4**: ≥ 1 perfect run at `a = 3.0`, 64k | **PASS**: 7 of 20 |
| **A5** (exploratory): `γ`, `α_L/β` both positive | **not evaluable**: `γ` undefined with no bracketed onset (`α_L = 0.404`) |

**Stop condition met: no onset brackets at any budget.** Per the registration,
no stage-2 seeds are added.

## Fraction perfect (20 seeds per cell)

| `a` | 1k | 4k | 16k | 64k | 64k, also 0 on 200k |
|---:|---:|---:|---:|---:|---:|
| 0.9 | 0 | 0 | 0 | 0 | 0 |
| 1.0 | 0 | 0 | 0 | 0 | 0 |
| 1.05 | 0 | 0 | 0 | 0 | 0 |
| 1.1 | 0 | 0 | 0 | 0 | 0 |
| 1.2 | 0 | 0 | 0 | 0 | 0 |
| 1.35 | 0 | 0.05 | 0.10 | 0.10 | 1 of 20 |
| 1.5 | 0 | 0.05 | 0.05 | 0.05 | 1 of 20 |
| 2.0 | 0 | 0.10 | 0.15 | 0.15 | 3 of 20 |
| 3.0 | 0.05 | 0.25 | 0.30 | 0.35 | 7 of 20 |
| ReLU | 0 | 0 | 0 | 0 | 0 |
| GELU | 0 | 0.05 | 0.25 | 0.30 | 3 of 20 |

## What this does and does not show

- **The onset, as registered, is not observable at this capacity.** The "≥ half
  perfect" criterion is never met, even by the positive controls (GELU 0.30,
  `a = 3.0` 0.35). A2 fails as registered; A3 and A5 are undecidable. The capacity
  pilot selected depth on *any* perfect seed, which made A4 likely but said
  nothing about reaching a ≥ half rate. That was a weakness of the pilot rule.
- **Descriptive, not a registered test:** perfect runs occur only for `a ≥ 1.35`,
  and in none of the 100 runs per budget at `a ≤ 1.2`, including the three
  grid values just above the threshold. The pattern respects the barrier and
  suggests an effective capacity threshold above `a = 1`, but it is not a
  measured onset.
- **Post hoc observation:** at `a ≤ 1`, median training accuracy is 0.9999–1.0000
  while held-out uniform errors stay at 22 or more. Barred networks fit the finite
  training sample and fail off it, as the theorem (a statement about the cores)
  allows.
- **Stratified set, post hoc:** some `a ≤ 1` runs have zero stratified errors
  while failing on the uniform set (median stratified errors 7–13). Barred
  networks' errors do not concentrate at the cores' closest approach
  (`blockA5d_k1_errormap.csv`). The criterion's discriminating power came from
  the uniform set.
- The k = 10 result (T72) stands as reported.
