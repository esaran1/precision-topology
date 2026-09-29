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
- **The author's decisions of 2026-09-29, after the pre-registration pilot and exploration (commits 09e2fea, 29b0a7e;
  pilot and population only, no registered seed):**
  1. the primary gate counts on-branch runs at release regardless of G in the hold (§3), with a descriptive
     sensitivity analysis excluding runs with G > 0 in the hold (§9);
  2. the control gate is unchanged;
  3. V7 is taken over the part of the path with s_t ≥ 0.8·s_switch,branch, and a follow check at 0.8·s_switch,branch
     is added (§5, §9). **This window was chosen after the pilot** (§12);
  4. κχ is signed everywhere; the sign of κ is shown to be physical (§10a);
  5. the other §10 details are approved as written.

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
| `random` | **primary** (the headline verdict) | (w₁, b₁, b₂) = coordinates 0, 1 and 3 of U(−1, 1)⁴, drawn in float32 by a torch Generator seeded with the seed (the same draw as 3A's `torch.manual_seed(seed)`), cast to double; w₂ = +s₀ | on the target copy (w₁ > 0) or the mirror copy; the run gets that copy's prediction | ≥ 80% of the 80 runs on-branch at release, **regardless of G during the hold** |
| `branch` | **mechanism control** | θ\*_pop(s₀); even seeds the w₁ > 0 copy, odd seeds the mirror (w₁ → −w₁) | on the assigned copy | ≥ 90% of the 80 runs on target, and no run with G > 0 at any hold step |

- A run on neither copy (primary) or off target (control) gets no prediction. It is counted and reported, never
  replaced. This is the paper's "reaches a branch" condition.
- Primary arm: a run with G > 0 at the start state or at any hold step that is on-branch at release **is scored**. It
  is flagged (`hold_G_positive`) and counted, and the descriptive sensitivity analysis of §9 excludes it.
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

A run is **scored** if it is on-branch at release (primary: either copy; control: the assigned copy), **follows its
branch** (below), crosses within the budget, and has finite r_traj, r_cf and r_obs. Runs with κ_seed < 0 are predicted
to cross early and are scored (§10a).

**Follow check** (author's decision). At t₀.₈, the first step with s_t ≥ 0.8·s_switch,branch of the release copy:
damped Newton at s = s_{t₀.₈} from the run's (w₁, b₁, b₂), accepted as at release. It is classified against both
copies' frozen branch points at 0.8·s_switch,branch (§6), each carried to s_{t₀.₈} by one tangent predictor and damped
Newton, with the release tolerances: Newton point within 1e−6 (sup) of that copy's branch point; the state condition is
`FOLLOW_STATE_TOL` (§13, OPEN). A run whose copy there differs from its release copy, or is neither, is not scored; it
is counted and reported. The check uses the state at t₀.₈ only (no gap); the number of runs whose crossing comes at or
before t₀.₈ is reported.

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
  - **Follow points:** for each copy c, both copies' branch points at s = 0.8·s_switch,branch,c, by act_fold
    continuation from their θ\*_own(s₀) and damped Newton, with θ\*′ there; validated by step halving (1e−6 sup),
    no turning point on the way and Newton acceptance. All 360 are valid.
  - **κ-sign checks** (§10a): κ under the three reparametrisations, and κ recomputed on the mirrored sample.
- `results/gelu_transfer/pilot.json` (`gelu_transfer.pilot`): ρ and the pilot medians of χ at t_sw (§7).
- The code and its hash. `registration.sha256` lists: `src/gelu_transfer.py`, `tests/test_gelu_transfer.py`, this
  file, the design page, the three frozen files above, `act_general/kappa_gelu_frozen.json`, and the modules the
  pipeline calls (`act_general`, `act_fold`, `linear_response`, `lag_law`, `track_a`, `phase2b_ordering`, `fold1d`,
  `width2_geometry`, `width2_conditional`). `train` and `observe` assert every hash before running.

## 7. Pilot rule (pilot seeds only; before registered training)

- Both arms are run on the 10 pilot seeds up to t_sw only. No gap is evaluated after release, and no crossing or
  residual is computed.
- The values are the SIGNED κ_seed·χ at t_sw of the on-branch pilot runs, pooled over both arms. q90 is their 90th
  percentile (numpy linear interpolation).
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
| V2 | t_sw strictly precedes the crossing in ≥ 90% of on-branch crossing runs with κ_seed > 0 (predicted-late runs; undefined t_sw counts as not preceding; with no such run the condition holds). Predicted-early runs (κ_seed ≤ 0) are not in V2. |
| V3 | regime: η·λ_min(H(s_switch)) ≤ 0.5 in ≥ 80% of scored runs. |
| V4 | resolution: the median over scored runs of the SIGNED predicted lag in steps, κ_seed/(η·λ_min(H(s_switch))), is ≥ 10. |
| V5 | the q90 over scored runs of the SIGNED κ_seed·χ at t_sw is ≤ 0.1. |
| V6 | \|median χ at t_sw over scored runs / the arm's pilot median − 1\| ≤ 0.30. |
| V7 | follows the branch: the q90 over scored runs of max χ_t over 0 ≤ t < t_sw **with s_t ≥ 0.8·s_switch,branch** is ≤ 0.25, with χ_t = ((s_{t+1} − s_t)/s_t)/(η·λ_min(H(s_t))) on the occupied branch (Corollary L3's χ_t with the one-step ṡ). An empty window or a non-finite χ_t in it fails. The maximum over the whole path is reported, not scored. |

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
- **Signed lags.** Every lag is signed: r_traj, r_cf and r_obs are negative for a crossing before the switch. L1 and L2
  are medians of ratios of signed lags (a run observed on the wrong side of its switch has a negative ratio). L3 ranks
  the signed values. L4 and L5 use the signed s_pred = s_traj; the registered closed-form prediction is
  s_cross = s_switch,branch·(1 + κ_seed·χ), reported per run (`pred_signed_s_cross_cf`).
- **Predicted-early runs (κ_seed < 0).** Scored like every other run. For them t_sw normally comes after the crossing,
  so χ in r_cf uses ṡ at t_sw from the s path after the crossing (the s path only; no gap and no crossing information).
- **Descriptive sensitivity analysis (primary arm; registered; NOT a criterion).** The whole scoring (validity and
  L1–L5) is repeated with the runs that had G > 0 at the start or in the hold excluded. It is reported beside the
  registered verdicts and never replaces them.
- **Outcome per arm:** PASS if L1–L5 all pass; UNRESOLVED if any is UNRESOLVED; otherwise FAIL, naming each failing
  criterion. Numbers are compared in float64 without rounding.

## 10. Details the approved page leaves open, fixed now

- The primary arm's W: the larger of the two copies' W (its copy is unknown before release).
- The pilot with two arms: both arms on the pilot seeds; κχ values pooled for ρ; each arm's own pilot median χ for V6.
- s_pred in L4 and L5 is the primary prediction s_traj.
- "Predicted lag in steps" (V4) is the slaved lag κ_seed/(η·λ_min), the quantity the design used to choose η.
- χ_t in V7 is defined in §9; the start state of the hold counts as a hold step for the gate.
- The random draw uses a local torch Generator (bit-identical to 3A's `torch.manual_seed` draw; tested), so no global
  RNG state changes.

## 10a. The sign of κ is physical, not a convention

- κ = λ_min·[∇G·H⁻¹θ\*′]/[∇G·θ\*′] (P = I). Its numerator and denominator are both bilinear in (∇G, θ\*′):
  - **Mirror coordinates** (w₁ → −w₁, i.e. choosing to describe a copy through its mirror image): H → DHD, θ\*′ → Dθ\*′,
    ∇G → D∇G with D = diag(−1, 1, 1) = D⁻¹. Then ∇G·H⁻¹θ\*′ → ∇G·D(DHD)⁻¹Dθ\*′ = ∇G·H⁻¹θ\*′, and ∇G·θ\*′ is unchanged.
  - **Sign convention of G** (G → −G): both factors change sign; κ is unchanged.
  - **Direction of s** (θ\*′ → −θ\*′): both factors change sign; κ is unchanged.
  - λ_min is invariant under all three.
- So κ has no orientation freedom. Its sign has a direct meaning: at a switch to placed, g′ = ∇G·θ\*′ > 0, and the
  slaved displacement δ = −(ηH)⁻¹θ\*′ṡ gives ∇G·δ = −(ṡ/η)∇G·H⁻¹θ\*′. κ < 0 ⇔ ∇G·δ > 0: the lag raises G, and the run
  is predicted to cross **before** its branch's switch. **The sign is part of what is tested.**
- The orientation that remains fixed by construction: w₂ > 0 (σ = +1, the placing orientation; w₂ is held at +s₀ and
  grows), and G is `phase2b_ordering.state`'s gap in w₂'s orientation.
- **Numerical check (frozen, all 180 copies):** κ recomputed from scratch on the mirrored sample (x → −x) at the mirror
  image of each frozen switch point (Newton, H, θ\*′, ∇G of the mirrored problem) equals the frozen κ exactly in all
  180 (the mirror map is exact in floating point), including the 13 negative ones (κ from −0.168 to −0.004: 4 target
  copies, 9 mirror copies). κ under the three transforms equals κ to 0.0 in all 180. Tests:
  `test_kappa_invariant_under_mirror_coordinates_G_sign_and_s_direction`,
  `test_kappa_recomputed_on_the_mirrored_sample_equals_the_original`.

## 11. Competing outcomes and falsifiers

| outcome | reading |
|---|---|
| PASS (primary) | The lag law transfers to GELU under free SGD from a random start, with no fit; the occupied branch's switch sets the lag. |
| L1 passes, L2 fails | The slaved closed form misses the early transient; the recursion does not. |
| L3 fails | The magnitude is right but the per-run ordering is not. |
| L4 fails | The own-branch switch does not carry the sample variation (contrary to 3A's post hoc result). |
| L5 fails | The predicted lag is not resolved against a crossing at the branch switch itself. |
| negative-κ runs on the wrong side | Counted in L1–L3 through negative ratios and ranks; the sign of the lag is part of the test (§10a). |
| gate fails | Random starts (primary) or branch-point starts (control) do not reach a branch at release, or the hold places a run. |
| UNRESOLVED (validity) | Too few crossings, t_sw at or after the crossing, outside the regime, lag unresolved, κχ too large, χ off the pilot, or the run does not follow its branch. The test says nothing about the law. |

A failure stays a failure. Post hoc readings may be placed beside it, labelled POST HOC.

## 12. A-priori risks, stated now

- Exploratory (`designs/GELU_transfer_explore/gelu_explore_f.py`; population only, numpy seeds; no candidate seed):
  7.2% of 4,000 random (w₁, b₁) draws are placed at w₂ = +s₀. Of 150 random hidden starts held as in §3, 16 are placed
  at the start, 11 of them are still placed after the first hold step, and 5 unplaced starts become placed during the
  hold: 21 of 150 have G > 0 at the start or at some hold step (16 of 150 after a hold step). Under the gate before the
  author's decision (no such run among 80) the chance of passing was below 1e−3 (0.86⁸⁰ ≈ 6e−6 counting the start;
  0.893⁸⁰ ≈ 1e−4 without it); this is why the primary gate now ignores G in the hold (§1, §3).
- **V7's window was chosen after the pilot** (pilot seeds 876,900–876,909 only, disjoint from the registered seeds;
  `designs/GELU_transfer_explore/gelu_explore_g.py`, reproduced by `pilot.json`). Over the whole path (the text before
  the decision) the q90 / median of max χ_t were 0.274 / 0.199 (random 0.251 / 0.187, branch 0.278 / 0.203; 3 of 20
  above 0.25), with the maximum at step 0, the first step after release, in 20 of 20 runs (s/s_switch 0.42–0.57, where
  λ_min is smallest). With s_t ≥ 0.8·s_switch,branch: 0.046 / 0.036 (random 0.046 / 0.035, branch 0.044 / 0.037;
  none above 0.25). Over the last 5 relaxation times before t_sw: 0.037 / 0.029. χ at t_sw: 0.026 / 0.023.
- **Primary gate, population (exploratory, 600 random starts, `gelu_explore_g.py`):** 563 of 600 (93.8%) on-branch at
  release regardless of G (the registered gate's quantity); 81 of them had G > 0 at the start or in the hold.
- **Follow check, pilot:** at t₀.₈ the state lags its branch point by 0.012–0.028 (sup) in all 20 pilot runs; Newton from
  the state lands on the release copy's branch point in 20 of 20.
- On the population, t_sw at η = 0.03 from θ\*_pop(s₀) is 5,684 steps, well inside the 40,000-step budget.

## 13. OPEN before registration: the state condition of the follow check

The author's rule says "registered tolerances". At release these are: Newton point within 1e−6 of the branch point
AND state within 1e−3 of it. In free training the state at t₀.₈ lags its branch by the tracking displacement: 0.012–0.028
(sup) in 20 of 20 pilot runs. With the 1e−3 state condition every run would be "neither" at t₀.₈, no run would be scored,
and both arms would be UNRESOLVED (V1). The code has the constant `FOLLOW_STATE_TOL`; it is set to None (Newton
identity only: the Newton point from the state lies within 1e−6 of the release copy's branch point, which held in 20
of 20 pilot runs) pending the author's decision. Nothing is registered until it is decided.

## 14. Frozen values

See ``results/gelu_transfer/landscape.json`, `frozen_seeds.json` and `pilot.json` (recomputed with the final code;
identical to the 09e2fea values to 3e−16 in every frozen scalar, and the pilot identical).
