# Figure: forecast horizon (post hoc)

**Message.** How early a strictly causal forecast can be made depends on the setting. For width-one SGD, the crossing-step forecast is within tolerance in every run once the cutoff is at least 0.84 of the switch scale. Outside width-one sine, it fails at every cutoff up to 0.92 and reaches 80% only from 0.94 (GELU; width two with slowed output) or 0.95 (width two from a branch point). Under Adam it depends on the forecaster: the Adam-specific forecaster (a = 1.77) stays at or above 77 of 98 runs from a cutoff of 0.80, while the SGD forecaster applied to Adam runs (a = 1.65) never exceeds 39 of 74.

**Population and n.** Existing runs only; no new training. Each set's stored, registered or re-scored runs are re-forecast with the frozen forecaster at cutoff fractions f = 0.80, 0.82, …, 0.98. The cutoff is the first step at which the output scale reaches f times the switch scale. The forecast reads only the run before that step, and the NaN recomputation (every later value replaced by NaN) gives identical forecasts in every run at every f.
- (a) Width-one sine. SGD at a = 1.58 (n = 84; registered at f = 0.90, chosen after the estimation seeds). SGD and Adam at a = 1.65 (n = 63 and 74; post hoc re-scores of a path-conditioned test, SGD forecaster). Adam at a = 1.77 (n = 98; registered at f = 0.90, Adam-specific forecaster).
- (b) Outside width-one sine, all registered at f = 0.95, chosen after the pilot. GELU, SGD, warm start (n = 79). Width-two sine with slowed output (n = 101). Width-two sine from a branch point (n = 120).

The large open marker is the f each set used. Tolerance: the set's own crossing tolerance, 10 steps (width one and GELU), 15 steps (width two, slowed output) or 30 steps (width two, branch-point start). Dashed line: the 80% required by every registered criterion. At each set's own f, the forecasts reproduce the stored ones in every run, and the scores reproduce the committed verdict numbers.

**Uncertainty.** 95% Wilson intervals for the fraction of scored runs within tolerance; a run with no forecast counts as not within. The f grid and the scoring were applied after all of these runs existed: post hoc.
