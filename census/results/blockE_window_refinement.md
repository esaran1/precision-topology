# The `a = 1.60` window at step 0.01: it is a grid artifact

Registered secondary test in `blockF_lag_prediction.md`: refine the `|w2|` grid
to 0.01 at `a = 1.60`, and add a freeze-mid arm **only if the window is still at
least five grid steps wide**. Reversibility stays the primary Block E result.

---

## Result: one grid step, not five. The arm is NOT added.

| | step 0.05 (Block B) | **step 0.01** |
|---|---:|---:|
| `R_fold` | 0.2229 | **0.22291** (`|w2| = 2.00`) |
| `R_glob` | 0.2285 | **0.22402** (`|w2| = 2.01`) |
| `R_spin` | 0.2508 | **0.22402** (`|w2| = 2.01`) |
| window `R_spin - R_fold` | 0.0279 | **0.00111** |
| width in grid steps | — | **1.0** |

Registered threshold `>= 5 steps`: **does not qualify**. No freeze-mid arm.
Block E runs the reduced arm set with reversibility as the primary test, exactly
as registered.

## The stronger consequence: no hysteresis at the largest `a` either

`a = 1.60` was where Block B's coarse grid reported the **widest** window
(0.0279 in `R`). At step 0.01 the three switch points collapse onto **two
adjacent grid points**:

    R_fold = 0.22291  |  R_glob = R_spin = 0.22402

`R_glob` and `R_spin` are **identical** — the same coincidence already found at
`a = 1.30`, now at the opposite end of the range. The `G <= 0` branch followed
upward is lost at exactly the `|w2|` where the global minimiser first has
`G > 0`, and the `G > 0` branch followed downward is lost one step below. That is
resolution, not bistability.

**The no-hysteresis falsifier, recorded at `a = 1.30`, now holds at `a = 1.60`.**
The apparent windows at `a >= 1.35` were the 0.05 grid. An audit of
`a = 1.35, 1.40, 1.45, 1.50` at step 0.01 is running
(`results/blockB_fine_windows.csv`).

## What this changes

- **Nothing in the bifurcation framing**: it was already dropped on the `a = 1.30`
  falsifier. This strengthens that decision rather than reversing anything.
- **`R_spin` values at `a >= 1.35` in `blockB_switches.csv` are grid-limited
  upper bounds**, not resolved spinodals. Every claim that used them must be read
  that way, including:
  - the "crossings track `R_spin`" test (already recorded as **not
    discriminating**, in `blockB_results_final.md`);
  - the `R_spin` drift figure of **+16.9%** across `a`, which is now suspect
    because its endpoint at `a = 1.60` moves from 0.2508 to 0.22402. Recomputed
    below once the audit lands.
  - **`R_glob` is unaffected**: it comes from a global search at each grid point,
    and at `a = 1.60` the 0.05 and 0.01 grids agree to 0.0045 (0.2285 vs 0.22402),
    which is within one 0.05 step.
- **D-1 (in `blockF_lag_prediction.md`) is affected.** It compares `M-spin`
  against `M-glob`. If `R_spin = R_glob` at every `a` once resolved, the two
  models are **identical**, and D-1's registered expected outcome — "neither will
  be distinguished" — holds for a stronger reason than the one registered: not
  that the data lack leverage, but that **the two hypotheses coincide**. Recorded
  as such rather than scored as a pass.

## Provenance

Same frozen Block B procedure (sha256 `9f1b10741d8bf48c`) with only the grid step
changed from 0.05 to 0.01. Per the freeze rule, this change is justified by
**resolution, not by agreement with any measured crossing**: the registration
itself called for the refinement, and the direction of the result (window
shrinks) is the opposite of what agreement-seeking would produce, since a wider
window would have supported the spinodal reading.
