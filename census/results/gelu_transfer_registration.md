# GELU-T registration: the lag law at GELU under free SGD from a declared start

**Status: written before any registered run.** No registered seed (876,000–876,079) has been trained. No gap, placement
or crossing of any training trajectory after release has been evaluated for any seed. This file becomes the
registration when it is committed together with the frozen inputs of §6 and their SHA-256 hashes
(`results/gelu_transfer/registration.sha256`). That commit is timestamped with OpenTimestamps
(`results/gelu_transfer/registration_stamp.txt` and its `.ots` proof).

## 1. Design reference

- Approved design: `results/designs/GELU_transfer_design.md` (commit 4ecc97d; approved by the author 2026-09-29). The
  draft with the exploratory feasibility numbers is commit 5e6323f; its scripts are in
  `results/designs/GELU_transfer_explore/`.
- The author's binding changes to the draft: Adam arm dropped; the random start is the primary arm; the branch-point
  start is a mechanism control; η = 0.03 with the justification of §8; the follows-branch condition V7; L5; L4 against
  s_glob; s₀ = 0.5·s_pop; branch copy by seed parity in the control arm.
- Code: `src/gelu_transfer.py`. Tests of every decision rule on constructed pass, fail and unresolved cases:
  `tests/test_gelu_transfer.py`.
- Where the approved page leaves a detail open, this file fixes it (§10). Nothing else differs from the page.

## 2. Setting (Track 3A) and seeds

- Width 1, z = w₂·GELU(w₁x + b₁) + b₂ with the exact (erf) GELU; I = [−0.8, 0.8] class 0, O = ±[1.2, 2.0] class 1; each
  seed's own 400-point sample `fold1d.make_data(200, seed)`; float64; s = |w₂|.
- Placement and the observed crossing: `phase2b_ordering.state` on the dense 4,001-point windows, G > 0 in w₂'s
  orientation, evaluated at every step after release. The observed crossing is the first placed step; s_obs = |w₂| there.
- **s_pop = 6.645633** (`act_general/kappa_gelu_frozen.json`, `s_star`; SHA-256 9a4848d2…, registered in Track 3A) is
  used only to set **s₀ = 0.5·s_pop = 3.322817** and the branch-point start. **s_glob = 6.641492** (3A's registered
  switch, the geometric midpoint of the validated bracket) is L4's comparator.
- **Seeds: 876,000–876,079** (80; the same seeds and own samples in both arms). **Pilot seeds: 876,900–876,909.**
  Checked on 2026-09-29: no integer in either range appears as a token in any text file under `src/`, `tests/`,
  `results/` or `paper/` (other than the design page), and no `*seed*` column of any `results/**/*.parquet` file takes
  a value in 876,000–876,999.

## 3. The two arms

Both arms, per run:
- **Hold.** w₂ is held at exactly +s₀. (w₁, b₁, b₂) take full-batch GD steps at lr 0.3 for W steps. G is evaluated at
  the start state and after every hold step (`hold_G_positive` if G > 0 at any of them).
- **W.** For each copy c of the seed's own-sample branch, W_c = max(4000, ⌈25/(0.3·λ_min,own,c(s₀))⌉). The control arm
  uses the assigned copy's W_c. The primary arm uses max(W₊, W₋), because its copy is not known before release.
- **Release.** Fresh `torch.optim.SGD` (no momentum, no weight decay), lr η = 0.03 for (w₁, b₁, b₂), and ρ·η for w₂
  (w₂'s gradient is multiplied by ρ before each step). Budget: 40,000 steps after release.
- **Classification at release** (release information only). Damped Newton at s₀ from the release state
  (`act_general.branch_point`) is accepted if max|∇| < 1e−8 and the Hessian is positive definite. The run is **on copy
  c** if the accepted Newton point is within 1e−6 (sup norm) of the frozen θ\*_own,c(s₀) and the release state is within
  1e−3 of it. Otherwise it is on **neither** copy.

| arm | role | start before the hold | on-branch at release | gate |
|---|---|---|---|---|
| `random` | **primary** (the headline verdict) | (w₁, b₁, b₂) = coordinates 0, 1 and 3 of U(−1, 1)⁴, drawn in float32 by a torch Generator seeded with the seed (the same draw as 3A's `torch.manual_seed(seed)`), cast to double; w₂ = +s₀ | on the target copy (w₁ > 0) or the mirror copy; the run gets that copy's prediction | ≥ 80% of the 80 runs on-branch, and no run with G > 0 at any hold step |
| `branch` | **mechanism control** | θ\*_pop(s₀); even seeds the w₁ > 0 copy, odd seeds the mirror (w₁ → −w₁) | on the assigned copy | ≥ 90% of the 80 runs on target, and no run with G > 0 at any hold step |

- A run on neither copy (primary) or off target (control) gets no prediction. It is counted and reported, never
  replaced. This is the paper's "reaches a branch" condition.
- **Gate failure:** the arm is UNRESOLVED and stops. Its crossings are not evaluated.
- The headline verdict is the primary arm's. The control tests the mechanism from a start on the branch.

## 4. Predictions (per on-branch run; the release state and the s path only)

- **r_traj (primary prediction; L1, L3, L4, L5).** Track A's R4 recursion (`linear_response.simulate`, SGD form:
  P = I, no momentum, lr η), started at release.
  - δ₀ = θ_release − θ\*(s₀) on the occupied copy's branch.
  - θ\*(s_t) (cubic Hermite with exact tangents) and H(s_t) (linear) come from that branch on a grid of spacing
    0.002·s_pop over [0.95·s₀, 1.6·s_pop], built by natural-parameter continuation from θ\*_own(s₀), along the run's
    own s_t.
  - Predicted crossing: the first step with G(θ\*(s_t) + δ_t) > 0 (exact-extrema enclosure midpoint on the continuous
    windows). s_pred = s_traj = s_t there, and r_traj = s_traj/s_switch,branch − 1.
  - No prediction if the one-step map has spectral radius > 1 (checked every 10th step) or sup|δ| > 1.
- **r_cf (closed form; L2).** r_cf = κ_seed·χ, with χ = (ṡ/s_switch,branch)/(η·λ_min(H(s_switch,branch))).
  ṡ = (s_{t_sw} − s_{t_sw−w})/w with w = min(100, t_sw). t_sw is the first step with s_t ≥ s_switch,branch.
- **Observed lag:** r_obs = s_obs/s_switch,branch − 1.
- **Competing predictions.** (a) No lag: crossing at the occupied branch's switch, s_switch,branch (L5's comparator).
  (b) The population switch: s_glob for every run (L4's comparator).

## 5. Scored runs

A run is **scored** if it is on-branch at release (primary: either copy; control: the assigned copy), crosses within
the budget, and has finite r_traj, r_cf and r_obs. Runs with κ_seed < 0 are predicted to cross early and are scored.

## 6. Frozen before any registered training (the registration commit)

- `results/gelu_transfer/landscape.json` (`gelu_transfer.landscape`). θ\*_pop(s₀) by act_fold continuation of the
  tracked population branch from (z\*, s_pop) down to s₀, then damped Newton at s₀. Validated: the continuation
  reaches s₀ with no turning point and no branch point, λ_min > 0 along it, step halving and direct Newton from z\*
  agree to 1e−6, G(θ\*_pop(s₀)) < 0 and dL/ds < 0 at s₀. If this fails, GELU-T stops before registration.
- `results/gelu_transfer/frozen_seeds.json` (`gelu_transfer.freeze`), per seed (pilot and registered) and per copy:
  - θ\*_own(s₀), by Newton from θ\*_pop(s₀) or its mirror (accepted as in §3), and λ_min(s₀) and W_c;
  - **s_switch,branch**: act_fold pseudo-arclength continuation from s₀ to 1.6·s_pop (h0 = 0.01, hmax = 0.05). It is the
    first G sign change, bisected to 1e−10 in arclength. Validated: step halving gives the same switch to 1e−6
    relative; no turning point and no branch point before it; λ_min > 0 on [s₀, s_switch]; G(s₀) < 0; the switch goes
    to placed; the exact-extrema enclosure is decided at (1 ± 1e−3)·s_switch (upper end < 0 below, lower end > 0 above);
  - **κ_seed** = λ_min·[∇G·H⁻¹θ\*′]/[∇G·θ\*′] (P = I) from the own H, θ\*′ and ∇G at s_switch, and λ_min(H(s_switch)).
    ∇G is by central differences (step 1e−6) of the enclosure midpoint; the one-sided differences agree to ≤ 1e−5
    relative in both components.
  - A copy failing any check is invalid. A run on an invalid copy gets no prediction (counted).
- `results/gelu_transfer/pilot.json` (`gelu_transfer.pilot`): ρ and the pilot medians of χ at t_sw (§7).
- The code and its hash. `registration.sha256` lists: `src/gelu_transfer.py`, `tests/test_gelu_transfer.py`, this
  file, the design page, the three frozen files above, `act_general/kappa_gelu_frozen.json`, and the modules the
  pipeline calls (`act_general`, `act_fold`, `linear_response`, `lag_law`, `track_a`, `phase2b_ordering`, `fold1d`,
  `width2_geometry`, `width2_conditional`). `train` and `observe` assert every hash before running.

## 7. Pilot rule (pilot seeds only; before registered training)

- Both arms are run on the 10 pilot seeds up to t_sw only. No gap is evaluated after release, and no crossing or
  residual is computed.
- The values are κ_seed·χ at t_sw of the on-branch pilot runs, pooled over both arms. q90 is the 90th percentile of
  their absolute values (numpy linear interpolation).
- ρ = min(1, 2^⌊log₂(0.1/q90)⌋). If ρ < 1, the pilot is rerun at ρ, and ρ is kept only if q90 ≤ 0.1. Otherwise ρ is
  halved and the pilot rerun, up to three halvings, and then GELU-T STOPS. With no pilot value at all, it STOPS.
- Each arm's pilot median of χ at t_sw (at the chosen ρ) is frozen for V6.

## 8. Why η = 0.03 (registered evidence and theorem, stated separately)

- **Theorem (proved; math note §15.4, Corollary L3).** For fixed-P gradient descent without momentum, which includes
  SGD, with the scale itself trained, χ_t = p_sσ_t/(s_tλ) holds identically, so η cancels. The lag tends to κχ̄ + O(χ̄²)
  as η → 0; the only η-dependence is the discreteness τ ≤ ṡ/s\*, at most one step.
- **Registered evidence (math note §13.6, R4).** Free Adam at η = 0.01, 0.005 and 0.0025, width-1 sine family: PASS at
  both a (1.30 and 1.50).
- **Not claimed.** There is no registered result on learning-rate invariance for SGD or for GELU. The theorem covers
  SGD; R4 is Adam on the sine family.
- η = 0.03 is chosen only so that the lag spans enough steps to resolve (Track A's η = 0.3 gives 3–4 steps).

## 9. Registered criteria and validity (`gelu_transfer.score_arm`, per arm)

**Gate first** (§3). If it fails: every criterion UNRESOLVED, outcome "UNRESOLVED (gate)".

**Validity** (all seven required; otherwise L1–L5 are UNRESOLVED, outcome "UNRESOLVED (validity)"):

| | condition |
|---|---|
| V1 | at least 60 of the 80 runs are scored (on-branch, crossed, with predictions). For the primary arm the count is out of 80; the on-branch count is reported separately. |
| V2 | t_sw strictly precedes the crossing in ≥ 90% of on-branch crossing runs with κ_seed > 0 (undefined t_sw counts as not preceding; with no such run the condition holds). |
| V3 | regime: η·λ_min(H(s_switch)) ≤ 0.5 in ≥ 80% of scored runs. |
| V4 | resolution: the median over scored runs of the predicted lag in steps, κ_seed/(η·λ_min(H(s_switch))), is ≥ 10. |
| V5 | the q90 over scored runs of \|κ_seed·χ\| at t_sw is ≤ 0.1. |
| V6 | \|median χ at t_sw over scored runs / the arm's pilot median − 1\| ≤ 0.30. |
| V7 | follows the branch: the q90 over scored runs of max_{0 ≤ t < t_sw} χ_t is ≤ 0.25, with χ_t = ((s_{t+1} − s_t)/s_t)/(η·λ_min(H(s_t))) on the occupied branch (Corollary L3's χ_t with the one-step ṡ). |

**Criteria** (scored runs only):

| | statistic | PASS if |
|---|---|---|
| **L1** | median of r_obs/r_traj | ∈ [0.90, 1.10] |
| **L2** | median of r_obs/r_cf | ∈ [0.80, 1.20] |
| **L3** | Spearman(r_traj, r_obs), average ranks | ≥ 0.5 |
| **L4** | D = \|log(s_obs/s_pred)\| − \|log(s_obs/s_glob)\|, s_pred = s_traj, s_glob = 6.641492 | upper end of the 95% percentile bootstrap interval of mean D < 0 |
| **L5** | D = \|log(s_obs/s_pred)\| − \|log(s_obs/s_switch,branch)\| (tests the lag itself) | the same rule |

- Bootstrap: 10,000 resamples of the scored runs with replacement, `numpy.random.default_rng(876000)` (the same seed
  for L4 and L5); the interval is the 2.5th and 97.5th percentiles of the resampled means.
- A statistic that cannot be computed (Spearman NaN, e.g. all predictions tied; fewer than 2 runs for the bootstrap) is
  UNRESOLVED.
- **Outcome per arm:** PASS if L1–L5 all pass; UNRESOLVED if any is UNRESOLVED; otherwise FAIL, naming each failing
  criterion. Numbers are compared in float64 without rounding.

## 10. Details the approved page leaves open, fixed now

- The primary arm's W: the larger of the two copies' W (its copy is unknown before release).
- The pilot with two arms: both arms on the pilot seeds; κχ values pooled for ρ; each arm's own pilot median χ for V6.
- κχ enters the pilot rule and V5 by absolute value (a negative-κ copy's lag is as large in magnitude).
- s_pred in L4 and L5 is the primary prediction s_traj.
- "Predicted lag in steps" (V4) is the slaved lag κ_seed/(η·λ_min), the quantity the design used to choose η.
- χ_t in V7 is defined in §9; the start state of the hold counts as a hold step for the gate.
- The random draw uses a local torch Generator (bit-identical to 3A's `torch.manual_seed` draw; tested), so no global
  RNG state changes.

## 11. Competing outcomes and falsifiers

| outcome | reading |
|---|---|
| PASS (primary) | The lag law transfers to GELU under free SGD from a random start, with no fit; the occupied branch's switch sets the lag. |
| L1 passes, L2 fails | The slaved closed form misses the early transient; the recursion does not. |
| L3 fails | The magnitude is right but the per-run ordering is not. |
| L4 fails | The own-branch switch does not carry the sample variation (contrary to 3A's post hoc result). |
| L5 fails | The predicted lag is not resolved against a crossing at the branch switch itself. |
| gate fails | Random starts (primary) or branch-point starts (control) do not reach a branch at release, or the hold places a run. |
| UNRESOLVED (validity) | Too few crossings, t_sw at or after the crossing, outside the regime, lag unresolved, κχ too large, χ off the pilot, or the run does not follow its branch. The test says nothing about the law. |

A failure stays a failure. Post hoc readings may be placed beside it, labelled POST HOC.

## 12. A-priori risks, stated now

- Exploratory (`designs/GELU_transfer_explore/gelu_explore_f.py`; population only, numpy seeds; no candidate seed):
  7.2% of 4,000 random (w₁, b₁) draws are placed at w₂ = +s₀. Of 150 random hidden starts held as in §3, 16 are placed
  at the start, 11 of them are still placed after the first hold step, and 5 unplaced starts become placed during the
  hold: 21 of 150 have G > 0 at the start or at some hold step (16 of 150 after a hold step). The primary gate requires
  no such run among 80. At these rates the chance that none of 80 runs has one is below 1e−3 (0.86⁸⁰ ≈ 6e−6 counting
  the start; 0.893⁸⁰ ≈ 1e−4 without it).
- V7: at release λ_min(s₀) is small (0.035 on the population), so χ_t is largest just after release; its size on
  own samples is first measured by the pilot (§7; `pilot.json`).
- On the population, t_sw at η = 0.03 from θ\*_pop(s₀) is 5,684 steps, well inside the 40,000-step budget.

## 13. Frozen values

Filled by the registration commit: see `results/gelu_transfer/landscape.json`, `frozen_seeds.json` and `pilot.json`.
