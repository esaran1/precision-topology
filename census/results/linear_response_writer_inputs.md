# Track 1 (final round): exact linear response along the trajectory — writer inputs

**Label: POST HOC throughout.** Existing runs only. The 1A shortfall (observed lag 0.82–0.91 of κ(a)χ) was known
before this analysis. Every per-run prediction was nevertheless computed and committed with its SHA-256 before any
observed crossing was read. Producer: `src/linear_response.py`. Outputs: `results/linear_response/`. Every number below
is in `results/linear_response/summary.json`, or in the CSV/markdown tables it points to.

## 1. What was committed before the comparison

| item | value |
|---|---|
| predictions commit | `60cb084` (code, tests, `predictions.csv`, `predictions.sha256`, `predict.log`) |
| `predictions.csv` SHA-256 | `e86ca102bcf87f6ec44565d189d4c50fd0e4204db338309d6240527b1c39faa2` |
| contents | all 1,750 runs of the 1A set (36 arms), status ok for 1,750 of 1,750. Full model from 0.7·s\* and 0.5·s\*; ablations a, b, c, c_path, d, a_nomom, full_m0H (each from both start points); slaved own-coefficient formula |
| gate | `tests/test_linear_response.py` (8 tests: constant coefficients plus a linear θ\* path reproduce the analytic slaved lag κχ; a start at the slaved δ keeps δ constant; momentum with β₁ = 0 equals the no-momentum recursion; momentum leaves the steady lag unchanged; events; analytic derivatives against autograd; the exact gap against `lag_law.gap`; branch interpolation). These ran before any prediction. Suite at the predictions commit: 469 passed (tests minus certificates) and 27 passed (certificates) |
| compare | `python -m src.linear_response compare` first asserts that the hash matches and that the file is committed and unmodified. Observed crossings are read only inside `observed()` / `observed_1A()` |

**Disclosures about the pre-commit phase.**
- While developing, before the commit, the exact-gradient reconstruction (§4.4) was run on two runs: a = 1.30, seed
  300000, the φ = 1 trajectory and the 4b teleport arm. That reconstruction reproduces the run's own crossing, so these
  two crossing scales were effectively seen. The reconstruction was then removed from the committed variants. No setting
  was tuned on it.
- The stability rule (§2) and the reading of the ablations (momentum kept in a–c) were fixed after prediction-side
  development output on a handful of runs. No observed data was involved.

## 2. Method (the `src/linear_response.py` docstring is the authoritative statement)

- **Runs.** The 1,750 runs of `results/lag_law/predictions.csv`. Only their identifiers and the committed κχ were read.
  All runs were computed, with no subsample, in one worker process at nice 15 with one thread (41 min).
- **Replay.**
  - Each run is replayed deterministically from its initialisation under its own protocol:
    - Adam: torch defaults, lr 0.01.
    - SGD: plain torch SGD, lr 0.3, no momentum.
    - The φ rule on w₂.
    - The Block 4b interventions at t\*, exactly as run, including the AS-RUN sharing of optimiser state.
  - At every step the replay records θ, |w₂|, Adam's first moment, v̂ and the step count.
  - It stops at a fixed horizon: |w₂| ≥ 1.6·s\*_pop(a) or the run's budget.
  - The gap of the actual trajectory is never evaluated.
  - For the lag2 and 4b arms, the pre-t\* segment is replayed from initialisation and checked bitwise against the saved
    t\* checkpoint (θ and both Adam moments).
- **Branch and s\*.**
  - Objective: the run's own sample (6,400 points for lag1, lag2 and 4b; 400 for SGD and Task B) with mean BCE.
  - Orientation: canonical w₂ > 0. When w₂ < 0, (w₁, b₁) → (−w₁, −b₁), and Adam's first moment likewise.
  - θ = (w₁, b₁, b₂), with b₂ trained, as in 1A.
  - Tracked branch: Newton at fixed s from the run's state at the first step with s ≥ s\*_pop(a). Continuation then
    uses a tangent predictor on the grid s ∈ [0.3, 1.7]·s\*_pop with spacing 0.002·s\*_pop.
  - s\* is the root of G(θ\*(s)) nearest the anchor, found by bisection on the Newton-refined branch. G uses exact extrema
    on the continuous windows.
  - Refinement: Newton from the state at the first step with s ≥ s\*. This never changed the branch (0 of 1,750).
  - Interpolation:
    - θ\*(s) uses cubic Hermite interpolation with the exact tangents. The maximum error at the switch is 8.3e-13.
    - H(s) is interpolated linearly. The maximum error at the switch is 6.8e-7.
- **Full model.** It starts at t₀, the step at which the run last passes c·s\* before the switch (c = 0.7 and 0.5).
  - Initial state: δ₀ = θ_{t₀} − θ\*(s_{t₀}), and m₀ = the run's actual Adam first moment at t₀. The variant full_m0H
    uses H δ₀ instead.
  - Recursion:
    - m_{t+1} = β₁m_t + (1 − β₁)H(s_t)δ_t
    - δ_{t+1} = δ_t − η P_{t+1} m_{t+1}/(1 − β₁^{k_{t+1}}) − [θ\*(s_{t+1}) − θ\*(s_t)]
    - P is the run's own 1/(√v̂ + ε), using torch's ordering and bias corrections.
    - SGD: δ_{t+1} = (I − ηH)δ_t − Δθ\*.
  - Prediction: the first step with G(θ\*(s_t) + δ_t) > 0. The prediction is the actual s_t at that step.
  - 4b interventions inside the window: the actual θ jump is added to δ, and m is replaced by the loaded first moment,
    which is zero for reset.
- **Ablations.** The author's (a)–(c) change the coefficients of the full model and keep its momentum state; (d) removes
  that state.
  - (a) All coefficients are frozen at the switch: H, P, θ\*′ and ∇G at s\*, with the linear branch and linear G exactly
    as in 1A. This is the 1A model iterated along the actual s path, without the steady-state assumption.
  - (b) Only P varies in time.
  - (c) Only H and θ\*′ vary: the exact θ\*(s) path with exact G; P frozen.
  - c_path: the exact path, with H and P frozen.
  - (d) The full model without the momentum state.
  - a_nomom: the literal 1A dynamics in discrete time, i.e. (a) without momentum.
  - slaved: the closed-form 1A formula with each run's own coefficients at its own switch, and ṡ averaged over the 100
    steps before the switch.
- **Stability rule** (committed). A variant gets no prediction if its iterated one-step linear map has spectral radius
  > 1 (checked every 10th step), or if sup|δ| > 1.
- **Different branch at the start.** Newton from the actual state at s_{t₀} converges to a point at sup distance
  > 1e-6 from θ\*(s_{t₀}), or does not converge. The model is run for these runs anyway: none is excluded.
- **Lag.** r_obs = s_obs/s\* − 1 and r_pred = s_pred/s\* − 1, both against the run's tracked-branch switch s\*. The
  observed crossing uses the runs' own dense-grid gap (4,001 points); the prediction uses the exact gap. §4.4 shows the
  difference never changes the crossing step.

## 3. Results

### 3.1 Branch mismatch (the author's addition)

- **0 of 1,750** runs are on a different branch at 0.7·s\*, and **0 of 1,750** at 0.5·s\*. No runs were excluded.
  - The maximum distance from the start state's Newton point to the tracked branch is 6.5e-12 at 0.7·s\* and 8.7e-12 at
    0.5·s\*.
  - No run changes the sign of w₂ inside the window.
  - Per-arm counts are 0 / 0 in every arm (`tables.md`, last column).

### 3.2 The full model, per a (pooled over optimisers, the statistic behind 1A's 0.82–0.91)

Through-origin slope of r_obs on r_pred, with the median per-run ratio r_obs/r_pred in brackets (`attribution.csv`):

| a | n | 1A as published: residual vs global own threshold, on κχ | **full, from 0.7·s\*** | **full, from 0.5·s\*** |
|---|---|---|---|---|
| 1.30 | 795 | 0.89 (1.02) | **1.02 (1.02)** | **1.02 (1.02)** |
| 1.45 | 80 | 0.83 (1.01) | **1.03 (1.03)** | **1.03 (1.03)** |
| 1.50 | 795 | 0.91 (1.00) | **1.04 (1.04)** | **1.04 (1.04)** |
| 1.60 | 80 | 0.82 (1.01) | **1.05 (1.05)** | **1.05 (1.05)** |

- The 1A column reproduces the published 0.89 / 0.83 / 0.91 / 0.82.
- Per arm, the full model's observed/predicted ratio of medians is **1.00–1.05 in all 36 arms**, from either start
  point.
- By κχ tercile within each a, the full model's median per-run ratio is 1.015–1.049. It is flat in χ, whereas κχ falls
  to 0.79–0.93 in the top tercile (§3.4).
- **The two starting points give the identical predicted crossing in every run** (1,706 of 1,706 runs with both
  defined). The transient from δ₀ has died out long before the crossing: the median Adam relaxation time at the switch
  is 8.7 steps, and the start points lie 428 and 672 steps before the switch (medians). **The conclusion does not
  change between 0.7·s\* and 0.5·s\*.**
- 42 runs have no full-model prediction from 0.7·s\*, and 23 from 0.5·s\*. The stability rule removed them.
  - 24 of the 42 are 4b reset runs. Right after the moment reset, Adam's bias-corrected v̂ is tiny, so P is huge and the
    iterated map transiently has radius > 1.
  - The rest are a few recurring seeds spread across arms.

### 3.3 Which simplification accounts for the 0.82–0.91

The table removes one approximation at a time. Entries are the per-a through-origin slope of r_obs on r_pred. The 96
4b-reset runs are left out of the ablation rows only; the slope of every row including them is in `attribution.csv`.

| step | 1.30 | 1.45 | 1.50 | 1.60 |
|---|---|---|---|---|
| 1A: residual vs **global own-sample threshold**, on κχ | 0.89 | 0.83 | 0.91 | 0.82 |
| κχ, residual vs the run's **tracked-branch switch** | 1.02 | 1.01 | 0.98 | 0.90 |
| slaved formula with **own coefficients** at own switch | 1.01 | 0.99 | 0.96 | 0.90 |
| (a) frozen coefficients, **no steady-state assumption** (recursion along the actual s path) | 1.01 | 1.02 | 1.03 | 1.04 |
| (b) + time-varying P | 1.02 | 1.03 | 1.04 | 1.04 |
| (c) + time-varying H and θ\* path (P frozen) | 1.01 | 1.02 | 1.03 | 1.04 |
| (d) all time-varying, no momentum (only where stable) | 1.02 | 1.03 | 1.03 | 1.03 |
| full (all time-varying, momentum) | 1.02 | 1.03 | 1.04 | 1.05 |

Median per-run ratios of observed to predicted lag, by κχ tercile at the two largest a (`attribution_tercile.csv`):

| a, tercile (median κχ) | vs global threshold, κχ | vs branch switch, κχ | slaved own | (a) | full |
|---|---|---|---|---|---|
| 1.50, low (0.046) | 1.03 | 1.03 | 1.01 | 1.00 | 1.03 |
| 1.50, high (0.072) | 0.91 | 0.93 | 0.92 | 1.06 | 1.04 |
| 1.60, low (0.070) | 1.07 | 1.06 | 1.01 | 1.02 | 1.05 |
| 1.60, high (0.099) | 0.79 | 0.80 | 0.83 | 1.07 | 1.05 |

**Attribution.**
- **The reference against which 1A measured the lag.** 1A used each run's global own-sample threshold. Here the lag is
  measured against the switch of the branch the run actually tracks. In 22.8% of runs the tracked switch lies more than
  1% from the global threshold (17.5–24.2% per a and optimiser).
  - At a = 1.30 and 1.45 this change alone takes the slope from 0.89 to 1.02 and from 0.83 to 1.01.
  - This is the same effect §13.6 of the math note found for the ramps.
- **The steady-state (slaved) displacement.** This part matters at a = 1.50 and 1.60, where χ is largest.
  - With own coefficients the slaved formula still gives 0.96 and 0.90, and 0.83–0.92 in the top κχ tercile.
  - Iterating the same frozen-coefficient linear model along the actual s path, which is ablation (a), gives 1.03 and
    1.04, and 1.06–1.07 in the top tercile.
  - The start point does not matter (§3.2), so this is not the initial transient. It is the variation of ṡ within the
    relaxation memory, which the slaved formula replaces by ṡ at the switch. That is an interpretation; the tercile
    pattern is the measured fact.
- **Coefficients frozen at the switch.** Not responsible. Adding time-varying P, or time-varying H and θ\*(s), moves the
  slope by at most 0.01 from (a).
  - c_path, which uses the exact θ\*(s) path with H and P frozen, alone gives 0.98 at every a. The exact path and the
    time variation of H partly cancel.
- **Momentum.** Not responsible for the lag. Where the no-momentum model is defined, full and (d) give the same
  prediction: median relative difference 0, maximum 9%, over 510 runs.
  - The momentum state is, however, needed for the linearised Adam dynamics to exist at all. At the switch the
    no-momentum map I − ηPH has spectral radius > 1 in 71% of Adam runs (median 2.9, maximum 37.8).
  - Along the path it is unstable in 1,240 of 1,690 Adam runs. So (d) and a_nomom have no prediction for most Adam runs
    (18 of 36 arms have none).
  - With the momentum state the map's radius is √β₁ = 0.949 at the switch in every Adam run but one.
- **Ablation (a) reproduces κχ, as the machinery check requires.**
  - Median (a) prediction / median κχ is 0.88–1.05 per arm, excluding the two 4b-reset arms. The lower end is the
    high-χ arms, where the non-steady recursion predicts less than the slaved formula.
  - Own κ over the committed κ has median 0.995. Own χ at the switch over 1A's χ at the crossing has median 1.016.
  - In the reset arms, (a) and (b) from 0.7·s\* give a spurious early crossing: arm ratios −0.11 and −0.23. The moment
    reset lies inside the window, and Adam's post-reset step (about η per coordinate on w₂) drives the frozen linear
    branch.
- **What remains, +2% to +5%.** The full model under-predicts the observed lag by 2% (a = 1.30) to 5% (a = 1.60). The
  excess grows with χ. See §4.4.

### 3.4 Per-arm table

`results/linear_response/tables.md` gives:
- all 36 arms with observed/predicted for κχ, slaved, full from both start points, and ablations (a)–(d) from both;
- the per-a table split by optimiser;
- the prediction-only table of each variant over κχ.

Arm-level ranges of the observed/predicted ratio of medians:

| model | range |
|---|---|
| κχ vs tracked-branch switch | 0.91–1.09 |
| slaved own | 0.94–1.03 |
| full, 0.7·s\* | 1.00–1.05 |
| full, 0.5·s\* | 1.00–1.05 |
| (a), excluding reset | 0.98–1.05 |
| (b), excluding reset | 1.00–1.05 |
| (c) | 0.98–1.05 |
| (d), 18 arms with predictions | 1.00–1.05 |

### 3.5 SGD separately (P = I, no momentum; the full model is (c) and (d))

Full model / observed, with slope and median ratio:

| a | slope | median ratio |
|---|---|---|
| 1.30 | 1.02 | 1.02 |
| 1.50 | 1.03 | 1.04 |

κχ against the tracked-branch switch gives 1.05 and 1.08 (medians 1.07, 1.08). The slaved own formula gives 1.00 and
1.00.

## 4. Checks

1. **Interpolation.**
   - θ\*(s) at the switch: maximum error 8.3e-13, Hermite against Newton.
   - H at the switch: maximum error 6.8e-7.
   - Tests check θ to 1e-7 and H to 1e-3 at interior points.
2. **Replays.**
   - Every lag2 and 4b replay reproduced its t\* checkpoint bitwise.
   - Every run completed: status ok for 1,750 of 1,750.
3. **Start points.**
   - s₀/s\* is at most 0.716 for the 0.7 start and at most 0.511 for the 0.5 start.
   - Interventions fall inside the window in 90 runs (0.7 start) and 288 runs (0.5 start). They are handled as events.
4. **Exact-gradient reconstruction** (POST HOC, **run after the comparison**; `diagnose`, the first 12 seeds of every
   arm; `diagnostic_exact_grad*.csv`).
   - Replacing Hδ by the exact gradient ∇L(θ\*(s_t) + δ_t; s_t) in the full model, with everything else unchanged,
     reproduces the **observed crossing step exactly in 426 of 426 runs (0.7 start) and 430 of 430 (0.5 start)**. The
     other 6 and 2 of the 432 fall under the committed stability rule.
   - So the machinery is exact, including P_t, the bias corrections, the event handling and the continuous-vs-grid gap.
     The **only** difference between the full model and the actual run is the linearisation of the gradient in δ.
   - On these runs the observed-to-full slope is 1.034 overall. Per a it is 1.02 (1.30 Adam), 1.01 (1.30 SGD),
     1.03 (1.45), 1.04 (1.50 Adam), 1.03 (1.50 SGD) and 1.05 (1.60).
   - The +2–5% excess is therefore the second-order (nonlinear-in-δ) response.

## 5. Writer section

**Outcome: the full model matches closely.**
- The observed/predicted lag ratio is **1.02–1.05 per a** by both the through-origin slope and the median per-run
  ratio, and **1.00–1.05 in each of the 36 arms**.
- This holds from either start point, with predictions identical in every run, and for Adam and SGD alike.

**Say:**
- "Post hoc, we integrated the linearised tracking dynamics along each run's own trajectory. The inputs were the
  actual output-scale path, the run's own Adam preconditioner and momentum, and the exact branch path and gap. The
  crossing predicted this way was committed before comparison. It gives observed/predicted lag ratios of 1.02–1.05 at
  every a (1.00–1.05 in each of the 36 arms). The lag is therefore quantitatively first-order tracking."
- "The 0.82–0.91 shortfall of the closed-form law κχ has two sources, neither of them frozen coefficients or the
  omitted momentum.
  (i) The reference. Measured against the switch of the branch each run actually tracks, rather than its sample's
  global threshold, the ratio becomes 1.02 and 1.01 at a = 1.30 and 1.45. About one run in five tracks a branch whose
  own switch lies more than 1% from the global threshold.
  (ii) The steady-state (slaved) approximation, which over-predicts the largest lags. At a = 1.50 and 1.60 the slaved
  formula gives 0.96 and 0.90, and 0.83–0.92 in the top third of κχ. Iterating the same frozen-coefficient model along
  the actual path gives 1.03 and 1.04."
- "Time variation of the Hessian, branch tangent and preconditioner changes the predicted lag by at most about 1%.
  Momentum leaves it unchanged, but it is required for the linearised Adam dynamics to be stable at all."
- "The remaining 2–5% under-prediction is second order in the displacement. Replacing the linearised gradient by the
  exact one reproduces every observed crossing step exactly (856 run-starts)."
- "No run was on a different branch at the starting point; none was excluded."
- Always with the label: post hoc; derived and compared after the 1A shortfall was known.

**Do not say:**
- "The lag law r = κχ is exact" or "κχ matches to 2%". The closed form, which is steady-state, over-predicts the
  largest lags by up to 20% (top κχ tercile at a = 1.60: 0.80).
- "The shortfall was caused by frozen coefficients" or "by momentum". Ablations (b)/(c) move the slope by ≤ 0.01, and
  (d) equals full where it exists.
- "The linear model is exact". It under-predicts by 2–5%, growing with χ.
- That the 0.82–0.91 was a uniform per-run shortfall. The median per-run ratio against the 1A reference was already
  1.00–1.02. The through-origin slope was pulled down by the largest-κχ runs and by runs whose tracked switch is off the
  global threshold.
- That this analysis is prospective or registered. It is post hoc on existing runs, with predictions committed before
  comparison.
- Any statement about widths other than 1, or about activations other than f_a.

**If the full model had not matched** (for completeness, not the outcome): "the linear-tracking picture is incomplete
by X%, and the paper says so." Not applicable: X is 2–5%, and §4.4 locates it in the second-order response.

## 6. Files

- `src/linear_response.py`: replay, branch, recursion, predict/finalize/compare/attribution/extras/tables/diagnose.
- `tests/test_linear_response.py`
- `results/linear_response/`:
  - `predictions.csv` and `predictions.sha256` (committed at 60cb084)
  - `compare_runs.csv`, `compare_arms.csv`, `compare_per_a.csv`
  - `attribution.csv`, `attribution_tercile.csv`
  - `diagnostic_exact_grad.csv`, `diagnostic_exact_grad_per_a.csv`, `diagnostic_exact_grad.jsonl`
  - `tables.md`, `summary.json`, `predict.log`, `diagnose.log`
- Not committed: `predictions_parts.jsonl`, the 10 MB resumable raw form of `predictions.csv`; same content.
