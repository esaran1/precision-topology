# 36-arm agreement: causal re-score (POST HOC), result 2026-11-14

LABEL: POST HOC. No registration, and no registered verdict changes.

## What the submitted paper says
- §5.1, p6: "integrating the linearized dynamics along measured output-scale paths brings every defined arm median within 5%".
- Fig. 5, grey crosses, p7: observed/predicted arm medians span 1.00–1.05, over the 1,708 of 1,750 runs that have a prediction.
- App. V.3, Table 36, p49: the "Full, from 0.7s*" and "from 0.5s*" columns.
- Table 39, p51: "Integrated, tracked branch 1,708 runs, 99.9% within ±10%".

The producer is `src/linear_response.py`, and the data are in `results/linear_response/` (`compare_arms.csv`, `compare_runs.csv`). The recursion is integrated along each run's own measured s_t up to the predicted crossing. That makes the result **path-conditioned, not a forecast**.

## Feasibility
- **Stored paths:** none. 0 of the 1,750 runs have a saved per-run path; `results/linear_response/` holds only predictions and comparisons.
- **Reproduction:** every run can be replayed deterministically by `linear_response.replay_path`, unchanged. The lag-2 and 4b runs also assert bit-equality with the 192 checkpoints in `results/lag_test2_states/`.
- **What the replay is checked against:** no stored path hashes exist for these runs. Each replay is instead checked against the committed `n_steps` and `s_max` (`compare_runs.csv`), and against the committed crossing |w2| at the committed crossing step.
- **Forecaster:** the frozen forecaster `src/causal_forecast.py` applies through its width-1 adapter (`phase1b.W1Case` / `forecast_one`, guarded views, NaN recomputation), using Phase 1C's W1 configuration with f = 0.90 and f = 0.95.
- **Limitation:** in the 4b reset arms, the forecaster's bias correction uses the step index. It does not see the reset step count.
- **Run time:** about 2–3 h at 1 worker, well before the Sun 20:00 ET deadline.

## Result (run 2026-11-14 17:15–20:59 ET, 1 worker; all 1,750 runs; summary 20:59)

The primary setting was fixed in the code before any outcome was computed: f = 0.90 (Phase 1C's W1 rule), with Adam's
preconditioner frozen at the causal rule point. The forecaster is the frozen Phase 1C forecaster, unchanged.
Numbers come from `arm36_causal.json` / `arm36_causal.csv` (`python -m src.posthoc_36arm summarize`) and are checked
in the ledger (RB-18).

**Primary (f = 0.90, P at the rule point): the agreement does not hold as a forecast.**
- **Arms:** 5 of 36 arm medians lie within 5% (observed over forecast). The arm ratios span 0.690–1.324. The submitted,
  path-conditioned figures had every arm within 5%, spanning 1.000–1.046.
- **Runs:** 11.4% of the 1,750 scored runs lie within 10%; 5 have no forecast.
- **Crossing step:** within 10 steps in at least 80% of runs in only 2 of 36 arms, the two SGD arms.
- **Guard:** the NaN recomputation differs in 0 runs.

**The other settings (DESCRIPTIVE, no verdict):**

| f | Adam P | arms within 5% | arm ratio range | runs within 10% | no forecast |
|---|---|---|---|---|---|
| 0.90 | rule point (primary) | 5 / 36 | 0.690–1.324 | 11.4% | 5 |
| 0.90 | one step before the cutoff | 15 / 36 | 0.947–1.280 | 62.7% | 8 |
| 0.95 | rule point | 9 / 36 | 0.671–1.190 | 12.9% | 3 |
| 0.95 | one step before the cutoff | 19 / 36 | 0.972–1.185 | 81.5% | 10 |

**Facts that bear on the reading (stated, not used to rescue anything):**
- **Forecaster.** 34 of the 36 arms are Adam runs. The forecaster was built and tuned for SGD at width one. On the
  registered Adam runs at a = 1.77 it also misses (median ratio 1.252; input blocks, block 3).
- **Interventions.** In 768 of the 1,750 scored runs (the Block 4b arms and the lag-2 primary-rule arms), an
  intervention (a teleport, a moment reset or a slowdown) happens after the causal rule point. The forecaster is not
  told about it.
- **Replay check.** Every run was replayed with `linear_response.replay_path` unchanged. n_steps and the observed
  crossing index match the committed values in all 1,750 runs. s_max is not bit-equal in 45 runs, by at most
  2.2·10⁻¹⁶ relative (the last floating-point bit); the check used exact equality.

**Consequence for the paper.** The submitted 36-arm agreement (§5.1 p6, the Fig. 5 grey crosses, Table 39
"Integrated", App. V.3 / Table 36) is **path-conditioned, not a forecast**. Under the frozen causal forecaster at its
primary setting it does not hold (5 of 36 arms within 5%). This re-score is post hoc.

## Reproduce

    U2_GATE_DIR=results/rebuttal/arm36 python -m src.u2_gate gate "arm36 run" --disk-gb 10     # its own step
    U2_GATE_DIR=results/rebuttal/arm36 python -m src.posthoc_36arm run 1 0
    U2_GATE_DIR=results/rebuttal/arm36 python -m src.posthoc_36arm summarize
