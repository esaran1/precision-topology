# Track A exploration: causal Adam per-run forecast

**EXPLORATORY. This is not a registration and not a pilot.**
- **Seeds:** 9,771,000–9,771,119 (exploration seeds only). The prefix 9771 was unused in src/, tests/, results/,
  paper/ and notes/.
- **Activation:** a = 1.85, which is Test 2A's value and already has data. It uses 2A's landscape,
  `results/track2a/landscape.json`.
- **Pipeline:** free Adam with lr 0.01, `make_data(200, seed)`, the local-Generator `phase1a_pilot.w1_train`, a budget
  of 32,000 steps, and every-step `w1_observe`.
- **Cutoff:** 1C's W1 nested cutoff on s\*_run, at f ∈ {0.90, 0.95}.
- **What used it:** every choice marked ‡ on `../trackA_causal_adam_design.md` was made after these data were seen.

## Files

| file | what it holds |
|---|---|
| `explore.py` | Step 1: P predictors × {1C extrapolation, actual path}. Writes `explore_runs.jsonl` and `summary.json` |
| `explore2.py` | Step 2: Adam-specific extrapolation families × P ∈ {cut, coupled_sw}. Writes `explore2_runs.jsonl` and `summary2.json` |
| `memory_gate.log` | The gate checks. Each job was preceded by a separate gate step. |
| `explore.log` | The run log. |

The P predictors read rows before t_c only, through GuardedArray views. `max_index_read < t_c` is asserted for every
causal forecast.

| predictor | how v̂ (and P) is set |
|---|---|
| `cut` | v̂ at t_c − 1 |
| `decay` | g² = 0 after the cutoff |
| `gwin` | g² held at its window mean, with g recovered from M |
| `logquad` | log v̂ extrapolated quadratically |
| `coupled` | v̂ rolled forward with g = Hδ, giving a time-varying P |
| `coupled_sw` | P frozen at the coupled v̂ at t_sw,fc |
| `oracle_sw` | v̂ at the actual t_sw. **Not causal**; it is 2A's rule. |

## Machine use

- One process at a time, nice 15, one thread, under 0.2 GB RSS (the stop limit is 1 GB).
- About 3.5 s per seed.
- Step 1 took 6.5 min and step 2 took 7.5 min.
- Track L's job ran alongside both steps. It was light; the memory gate was ≥ 33% free.

## Key numbers (f = 0.90; 119 crossing runs, all with the cutoff before the crossing)

Columns:
- **P error:** median \|P̂/P(t_sw) − 1\| for (w₁, b₁, b₂).
- **C1, C2:** at W1's tolerances 10 / 15.

| path | P | P error | misses | C1 | C2 | median r_obs/r_fc |
|---|---|---|---|---|---|---|
| 1C quad-log, 5% | cut | 7.7%, 7.7%, 3.3% | 32 | 0.71 | 0.70 | 1.27 |
| 1C quad-log, 5% | coupled_sw | 0.28%, 0.28%, 0.32% | 32 | 0.72 | 0.70 | 1.30 |
| 1C quad-log, 5% | oracle_sw | 0 | 32 | 0.72 | 0.72 | 1.30 |
| actual path (not causal) | cut | — | 4 | 0.91 | 0.92 | 1.04 |
| actual path (not causal) | coupled_sw | — | 4 | 0.91 | 0.92 | 1.06 |
| `slin_20` | cut | — | 5 | 0.91 | 0.92 | 0.95 |
| `slin_20` | coupled_sw | — | 5 | 0.92 | 0.92 | 0.94 |

**Detail for `slin_20` with coupled_sw:**
- q90 \|t_fc − t_obs\| = 6 steps; q90 \|lag error\| = 4 steps.
- C4 upper end −47.
- The 5 misses: 4 R4-unstable runs and 1 with no predicted crossing.
- Tolerances from the first-half split rule: 10 / 10 / 0.20. The second half passes.
- At f = 0.95: 4 misses, C1 0.93, median ratio 0.97.

**The 1C extrapolation at f = 0.95:** 22 misses, C1 0.80, median ratio 1.19, with any P.

Full tables are in `summary.json` and `summary2.json`.
