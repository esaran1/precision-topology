# Writer inputs: registered test 2B, the validity boundary of the lag law in κχ (redesigned), 2026-09-29

## Provenance

- **Producer:** `src/track2b.py` writes to `results/track2b/`.
- **Registration:** `results/track2b_registration.md`.
  - Registration commit `0708ae9` (01:19:48), with the SHA-256 manifest `results/track2b/registration.sha256`.
  - OpenTimestamps proof in `349911f`.
- **Design:** `results/designs/2B_boundary_design.md`, approved as designed on 2026-09-29.
- **Order of events:**
  - registration `0708ae9` at 01:19:48;
  - training from 01:20:06;
  - predictions committed in `08057a9` at 01:33:24 (`predictions.csv`, SHA-256 `9bff0519…`), with no gap evaluated
    during training;
  - observation from 01:33:31.
- **Scores:** `results/track2b/scores.json` and `results/track2b/observed_runs.csv` (480 runs).
- **Post hoc (labelled):** `src/track2b_posthoc.py` writes `results/track2b/posthoc_stability.json` and `.csv`.
- **Ledger:** `verify_ledger.track2b_checks`.

## Registered verdicts: C1 PASS, C2 UNRESOLVED

- **C1 PASS.** All six cells with κχ ≤ 0.1 are valid, and their median obs/pred lies in [0.75, 1.25]:
  - a = 1.30: 1.046, 1.102, 1.209 (κχ = 0.02, 0.05, 0.1);
  - a = 1.50: 0.990, 1.046, 1.139.
- **C2 UNRESOLVED.** All four cells with κχ ≥ 0.3 are invalid.
  - Each fails the realised stability check: η_cell·λ_max > 1 on the realised trajectory in 39, 40, 17 and 40 of 40
    runs.
  - Two also have fewer than 30 crossings: a = 1.30, κχ = 0.3 (27) and a = 1.50, κχ = 0.4 (6).
- **κχ\* (descriptive) = 0.2 at both a.** This is the first target whose median ratio leaves the band: 1.557 at a = 1.30
  (an invalid cell) and 1.410 at a = 1.50 (a valid cell).

| a | κχ | η_cell | crossings / 40 | median obs/pred | IQR | median χ_own / target | max realised η·λ_max | runs violating | valid |
|---|---|---|---|---|---|---|---|---|---|
| 1.30 | 0.02 | 0.3 | 36 | 1.046 | 1.003–1.104 | 1.006 | 0.853 | 0 | yes |
| 1.30 | 0.05 | 0.3 | 40 | 1.102 | 1.048–1.207 | 1.007 | 0.889 | 0 | yes |
| 1.30 | 0.1 | 0.3 | 40 | 1.209 | 1.146–1.344 | 1.007 | 0.970 | 0 | yes |
| 1.30 | 0.2 | 0.2766 | 40 | 1.557 | 1.466–1.805 | 1.007 | 1.135 | 21 | no (V3) |
| 1.30 | 0.3 | 0.2630 | 27 | 2.482 | 2.185–2.817 | 0.987 | 5.90 | 39 | no (V1, V3) |
| 1.30 | 0.4 | 0.2578 | 33 | 5.101 | 4.748–5.492 | 1.003 | 22.3 | 40 | no (V3) |
| 1.50 | 0.02 | 0.3 | 32 | 0.990 | 0.941–1.051 | 0.985 | 0.573 | 0 | yes |
| 1.50 | 0.05 | 0.3 | 40 | 1.046 | 0.984–1.103 | 0.998 | 0.621 | 0 | yes |
| 1.50 | 0.1 | 0.3 | 40 | 1.139 | 1.053–1.202 | 0.998 | 0.686 | 0 | yes |
| 1.50 | 0.2 | 0.3 | 40 | 1.410 | 1.265–1.508 | 0.998 | 0.912 | 0 | yes |
| 1.50 | 0.3 | 0.3 | 39 | 1.975 | 1.692–2.280 | 0.997 | 1.95 | 17 | no (V3) |
| 1.50 | 0.4 | 0.3 | 6 | 2.676 | 2.298–3.147 | 0.952 | 2.79 | 40 | no (V1, V3) |

**Counts:**
- 480 runs, of which 413 crossed. No run was placed in the held phase, and no crossing run changed winding.
- At κχ = 0.02, the non-crossers (4 at a = 1.30, 8 at a = 1.50) are exactly the seeds that the registration's a-priori
  risk (§8) ruled out: their predicted crossing lay beyond s_end.
  - So the a = 1.50, κχ = 0.02 cell had exactly the 32 possible crossings, against the 30 required.
  - Its median is over the 32 seeds with the smaller switches (a truncation stated before the run).

**Stability, a priori and realised:**
- η_cell was 0.3 in 9 of the 12 cells; it was 0.2766, 0.2630 and 0.2578 at a = 1.30, κχ = 0.2, 0.3 and 0.4.
- In the six C1 cells the realised η·λ_max stayed at or below 0.970.

## Descriptive and post hoc readings (beside the verdicts, labelled)

- **Descriptive (not a verdict):** in every C2 cell the median ratio is outside the band: 2.48, 5.10, 1.98 and 2.68.
  Those cells are invalid, so C2 is UNRESOLVED, not PASS.
- **Descriptive:** in the valid cells the median ratio rises with κχ: 0.99 → 1.05 → 1.14 → 1.41 at a = 1.50, and
  1.05 → 1.10 → 1.21 at a = 1.30. The law under-predicts increasingly as κχ grows.
- **POST HOC (`posthoc_stability.json`): where V3 failed.**
  - Every violation occurred after the ramp scale had passed the run's branch switch, during the lag. The median scale
    at the first violation was 1.20–1.26·s_branch at a = 1.30 and 1.53·s_branch at a = 1.50. In every violating run it
    was after the switch.
  - At the first violation, the realised own-sample λ_max was a median 1.23–1.56 times the population-branch λ_max at
    the same scale.
  - So the a-priori rule, which bounds λ_max on the branch, does not bound it off the branch, where large-lag runs sit.
  - The fraction of realised steps in violation was 0.7%, 6.0% and 9.4% at a = 1.30, κχ = 0.2, 0.3 and 0.4, and 0.5%
    and 3.6% at a = 1.50, κχ = 0.3 and 0.4.

## Replacement sentence for the paper's boundary statement

> "In a registered redesign of the boundary test (width-1 SGD forced ramps with the learning rate set a priori for linear
> stability along the ramp, a = 1.30 and 1.50, 40 seeds per cell), the lag stayed within 25% of the predicted κχ in every
> cell with κχ ≤ 0.1 (cell medians 0.99–1.21); at κχ = 0.2 it exceeded the prediction by 41–56%, and the cells at
> κχ ≥ 0.3 could not be scored because the realised dynamics left the stability range the design required."

## Say / Do not say, for every possible outcome (the one that occurred is marked)

**C1**
- **PASS (occurred).**
  - Say: "the lag law held within ±25% (registered) in all six cells with κχ ≤ 0.1, at both a".
  - Say: "the registered test confirms the validity range κχ ≲ 0.1 that the first boundary test suggested
    descriptively".
- **FAIL (did not occur).**
  - Would have said: "the law degrades already at κχ ≤ 0.1 in cell …".
  - Would not have said: "the law holds for κχ ≲ 0.1".
- **UNRESOLVED (did not occur).**
  - Would have said: "the registered test could not score the small-κχ range".

**C2**
- **PASS (did not occur).**
  - Would have said: "the law fails by κχ = 0.3 (registered)".
- **FAIL (did not occur).**
  - Would have said: "the law holds within ±25% at κχ ≥ 0.3 in cell …".
- **UNRESOLVED (occurred).**
  - Say: "the κχ ≥ 0.3 cells were invalid, because the realised dynamics left the design's stability range (η·λ_max > 1
    on the run's own Hessian in 17–40 of 40 runs per cell), so where the law fails above 0.1 is not established by a
    registered criterion".
  - Say: "descriptively, the lag exceeded the prediction by more than 25% in every cell from κχ = 0.2 upward (medians
    1.41–5.10), with κχ\* = 0.2 at both a".

**κχ\*:**
- Say "κχ\* = 0.2 at both a (descriptive)".
- Do not present κχ\* as a registered verdict.

**Do not say:**
- "the boundary test passed", or "the registered test locates the boundary between 0.1 and 0.3". C2 is UNRESOLVED; only
  the lower side (C1) is registered.
- "the law fails at κχ ≥ 0.3" as a registered result. That reading is descriptive, from invalid cells.
- "the redesign kept the dynamics stable" without qualification. It did so in the C1 cells and at a = 1.50, κχ = 0.2. In
  the fast cells the realised own-sample λ_max exceeded 1/η_cell during the lag (post hoc).
- "the law is accurate to 25% up to κχ = 0.2". The κχ = 0.2 medians are 1.41 and 1.56.
- That the a = 1.50, κχ = 0.02 cell used all 40 seeds. It is 32 crossing runs, the seeds with the smaller switches.
