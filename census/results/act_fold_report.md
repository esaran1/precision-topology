# POST HOC: fold or switch of the occupied branch, Track 3A crossings (GELU; SiLU partial)

**Label: POST HOC** (coordinator / author request, 2026-09-28). Nothing here is registered. The registered Track 3A
verdicts stand as scored (`results/act_summary.md`).

- **Producer:** `src/act_fold.py`.
- **Tests:** `tests/test_act_fold.py` (9 tests). The fold detector is tested on a constructed saddle-node: fold at s = 2
  found to 1e−8, λ_min ≈ 0 there. It is also tested on a fold-free branch, where it must report no fold, and on
  switch detection, step halving, the independent fold check (pass and fail cases), the memory watchdog and the CSV
  reader.
- **Outputs:** `results/act_fold/runs_gelu_pre95.csv`, `runs_gelu_pre85.csv`, `runs_silu_pre95.csv` (partial) and
  `summary.json`. The run log is `results/act_fold_run.log`.

**Stopped early.** The coordinator stopped Track 1 work on 2026-09-28, after the Track 1A gate failed on the v3
setting. GELU is complete. SiLU is partial and Mish was not run; see "Not done".

## Question and method

The author asked: "Is there an occupied branch whose fold or switch matches the crossings?"

The hypothesis comes from Track 1A. A run stays on the branch it occupies past the global switch while that branch
remains a local minimum. It leaves at the branch's fold, where λ_min of the reduced Hessian goes to 0 and the branch
turns back in s.

The definitions below were fixed in the module docstring and committed (9b41abc) before any run on the data.

- **Runs.** The registered 200-seed arm: crossing runs that were not placed at initialisation.
  - A run is **non-early** if s_cross ≥ 0.5·s_glob.
  - GELU has 132 crossing runs: 114 non-early and 18 early (the tail-placed crossings, not analysed).
- **Replay.** Each run is replayed with act_general.run_one's protocol. The recorded |w₂| at the crossing step was
  reproduced bit for bit in all 114 GELU runs and all 34 SiLU runs analysed.
- **Pre-crossing state.** The last step with |w₂| ≤ 0.95·s_cross; for GELU this is a median of 51 steps before the
  crossing. The sensitivity version uses 0.85·s_cross.
- **Occupied branch.** Damped Newton in (w₁, b₁, b₂) at w₂ = σ·s_pre on the run's own 400-point sample, accepted only
  if max|∇| < 1e−8 and the Hessian is positive definite. Every run was accepted. For GELU the median distance from the
  training state to the branch point is 0.009.
- **Continuation.** Pseudo-arclength continuation in (z, s).
  - It runs upward to 2·s_cross, or until a fold.
  - If the branch is already placed at s_pre, it also runs downward to find that branch's switch.
- **Fold.** A turning point in s (the tangent's s-component changes sign), bisected in arclength to 1e−10, with λ_min
  recorded.
- **Switch.** A sign change of G₊ (exact-extrema enclosure on the continuous windows) along the stable branch.
- **Validation.**
  - Step halving: the same events at the same locations within 1e−6 relative.
  - An independent BFGS search near any fold.
  - The λ_min path must be positive on the stable segment, with no λ_min sign change (no branch point).

## Result: GELU (114 non-early crossers, all analysed)

| quantity | value |
|---|---|
| occupied branch unplaced at the pre-crossing state | 114 / 114 |
| **fold on the occupied branch before 2·s_cross** | **0 / 114** (every continuation reached s = 2·s_cross; no turning point, no branch point) |
| switch found upward (unplaced → placed) | 114 / 114 |
| s_cross / s_switch: median (10–90%) | **1.0017** (1.0004–1.0043) |
| within 15% of the switch | **100%** |
| within 1% of the switch | 98.2% (2 runs cross 1.4% / 2.0% below their branch's switch) |
| crossing below its branch's switch | 4.4% |
| λ_min on the continued stable segment | ≥ 0.076 everywhere (at the switch: 0.089–0.133, median 0.119) |
| step halving agrees (events and locations to 1e−6) | 114 / 114 |
| sensitivity: pre-crossing state at 0.85·s_cross | the same events at the same locations, 114 / 114 |

**Answer for GELU:** the occupied branch has **no fold** anywhere between the pre-crossing scale and 2·s_cross. The
crossings match the occupied branch's **switch**, a continuous G = 0 crossing on a branch that stays a local minimum.

The median lag is +0.17%. That is consistent with posthoc2 (cce777d), which found a median r_b of +0.0017 from the
crossing state. Here the branch is identified *before* the crossing, from the state 5% lower in s, so no information
from the crossing is used.

The fold mechanism plays no role for GELU's non-early crossers. None of the 114 continuations passes a turning point up
to s = 2·s_cross. There is therefore **no fold to report λ_min near**, for §14's |m′c′| = ¼·d(λ_min²)/ds. What was
recorded is the λ_min path's minimum on the stable segment and its value at the switch (table above; per-run columns
`min_lam_stable_up` and `switch_lam`).

## SiLU (descriptive, partial: 34 of 134 non-early crossers, seeds 850,000–850,053)

- No fold on the occupied branch up to 2·s_cross in any of the 34 runs.
- 7 of 34 branches were already placed at the pre-crossing state. Their switch lies below s_pre (downward
  continuation), so those runs lag their branch.
- s_cross/s_switch has median 1.0066 (10–90%: 0.988–1.066). 100% of runs are within 15% and 38% within 1%; 38% cross
  below their branch's switch.
- λ_min on the stable segment is ≥ 0.016.
- Step halving agrees in 34 of 34 runs.
- The switch matches loosely. SiLU's χ at crossing (median 0.22) is outside the range in which the lag law was
  verified (posthoc2), and no fold is involved.

## Not done (stopped by the coordinator, 2026-09-28)

- **SiLU:** 100 of 134 non-early runs were not analysed, and its 0.85 sensitivity pass was not run.
- **Mish:** not run.
- **Independent fold check:** never exercised on the data, because no fold was found. It was tested only on the
  constructed saddle-node.
- **λ_min near a fold** for §14: none exists on these branches up to 2·s_cross, so there is nothing to record.
- The continuation stops at 2·s_cross. A fold beyond that is not excluded.

## Process notes

- The computation ran as one process at nice 15 (RSS about 0.5 GB). The memory watchdog was checked before every run
  (memory_pressure free ≥ 25%, swap free ≥ 500 MB) and never triggered.
- The run was aborted with SIGTERM between runs; each run's row is appended only after the run completes, so no row is
  partial.
- The per-run CSV had rows with two different field sets: two extra fields for branches placed at s_pre. They are
  parsed by field count (`read_rows`, tested). The writer now keeps one fixed column set per file.
- Nothing in these modules or tests changes global torch state. All dtypes are set per tensor, and no default dtype is
  set.
