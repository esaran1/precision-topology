# 36-arm agreement: causal re-score (POST HOC), status 2026-11-14

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

## Status: code and tests ready, job NOT run
- Module: `src/posthoc_36arm.py`. Tests: `tests/test_posthoc_36arm.py`, 9 passed, exit 0.
- The coordinator relayed that the author had lowered the disk threshold to 10 GB. The gate call with `--disk-gb 10` was refused by the session's permission check, so the job was not launched.
- Free disk is about 10–11 GB, below the standing 15 GB gate.

## Fallback
If no gated run finishes by **Sun 2026-11-15 20:00 ET**, the 36-arm trajectory-integrated agreement is labelled **"path-conditioned, not a forecast; moved to the supplement"**. That covers:
- the §5.1 p6 sentence;
- the Fig. 5 grey crosses;
- the Table 39 "Integrated" row;
- App. V.3 / Table 36.
