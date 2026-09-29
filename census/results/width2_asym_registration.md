# W2-A registration: the lag law at width 2 on the asymmetric windows (arms T, D, T′)

**NOT YET REGISTERED.** This is the registration text, committed before any pilot run. The registration commit will
add the frozen inputs (§6), the pilot result (§8) and the manifest `results/width2_asym/registration.sha256`. It is
made only after the author has decided the open items in §0. No registered seed (884,000–884,119, 884,200–884,319) has
been held, trained or observed. No pilot seed (884,900–884,909) has been held or trained. No gap, placement or crossing
of any training state after release has been evaluated for any of these seeds.

## 0. Open items for the author (the standing rule: STOP before the registration commit)

These details are not fixed by the approved page, and each one touches a criterion, a validity condition or the
scoring set. Nothing is decided here. The code implements the literal reading shown and is tested on it.

1. **V4 for T′.** GELU-T's V4 requires the median SIGNED predicted lag κ/(η·λ_min) to be ≥ 10 steps. T′'s lag is
   negative by design (population: κ = −0.0564, −47.7 steps). So under the literal rule T′ is always UNRESOLVED
   (validity), whatever the data show. The test `test_negative_lags_t_prime_pass_l1_to_l5_with_signed_ratios_and_the_literal_v4`
   shows this. One option, by analogy with clarification (2): V4 on the median of |κ|/(η·λ_min).
2. **V5 for T′.** GELU-T's V5 (q90 of the SIGNED κχ at t_sw ≤ 0.1) holds trivially when every κ < 0. By analogy with
   (2) it could use |κχ|. The population value is |κχ| ≈ 0.0040 at ρ = 2⁻¹⁰, so either reading passes there.
3. **L4's comparator for D′ runs.** The page freezes the population switches of D, T and T′ (5.080, 0.482, 0.328). L4
   is "vs the branch's population switch", and D′ is scored in arm D, but D′'s population switch is not on the page.
   D′ is not the mirror image of D on these windows. Its recomputed population switch is 4.97059 (D: 5.07964). The
   code uses D′'s own value.
4. **T's κ by winding.** The page's table is indexed by k₁ + k₂ (−3..3). That is exact for D, where v′ = (½, ½), but
   not for T or T′, where κ_k = κ₀ + a₁k₁ + a₂k₂ with a₁ ≠ a₂. At the population T switch, a = (−0.139, −0.127), so
   κ(1, 0) = −0.090 and κ(0, 1) = −0.078, against κ(0, 0) = +0.049. One winding reverses the predicted sign. The code
   computes κ for the run's actual (k₁, k₂) by the page's formula (exact, tested against a direct computation). It
   admits a winding to the table iff −3 ≤ k₁ + k₂ ≤ 3, which is the page's range. All 400 exploratory random-start
   landings were at winding (0, 0).
5. **For information** (on the page, no decision needed unless the author wants one):
   - The recomputed population switches, validated and bisected, are T 0.481586 and T′ 0.327167. The page's 0.482 and
     0.328 came from the exploration's adiabatic path, which was not bisected: it reported the first 0.5% step past
     the sign change. L4 uses the recomputed values.
   - V2 for T′ is vacuous. GELU-T's V2 excludes predicted-early runs (κ ≤ 0), and every T′ run is predicted early.

## 1. Design reference

- **Approved design:** `results/designs/width2_asym_design.md`, as committed in 907e225. The author approved it on
  2026-09-29 with changes A–D, including the NEW items A and D, and two clarifications (below). The page's NEW markers
  now read "approved 2026-09-29". Clarifications (1) and (2) are added to it, one line each.
- **Draft with the width-2 background:** ccce4fa. The background is in §3 here.
- **Exploratory producers:** `results/designs/width2_asym_explore/` (`w2core.py` and `w2_explore_a`–`i`). They used
  the population and the already-used T2-3 seeds 600,000–600,011 only.
- **Code:** `src/width2_asym.py`. Its numerics are those of `w2core.py`, checked again in the tests.
- **Tests:** `tests/test_width2_asym.py`. They cover every decision rule on constructed pass, fail and unresolved
  cases, the gradient and Hessian against finite differences, and integration checks on the population.
- **Inherited from GELU-T** (`results/gelu_transfer_registration.md`, registered in 736b6bf) unless stated otherwise:
  gates, Newton classification, the follow check (Newton condition only, §13 there), V1–V7, L1–L5, the bootstrap,
  signed ratios, the pilot rule for D, the stamp, and predictions committed before observation.
- **Arms and roles:** T is the headline. D is the control, with D′ scored as its mirror copy. T′ is its own arm with
  its own verdict. Each arm gets a separate verdict.

**Clarifications (author, 2026-09-29)**
1. The negative-κ explanation of T′'s early crossings came from exploratory runs on used seeds only (the population
   and T2-3 seeds 600,000–600,011), not from any registered seed.
2. N_min applies to the ABSOLUTE predicted lag. A run enters L5 only if |t_pred − t_sw| ≥ 6 steps, so T′'s negative
   lags are treated exactly as T's and D's. L5 is UNRESOLVED if fewer than half the scored runs, or fewer than 2,
   remain.

**Disclosures (approved page, changes B and C)**
- **B.** The total-derivative κ was adopted after exploratory population runs gave the wrong sign with GELU-T's
  formula. The denominator dG/ds = ∇_zG·θ\*′ + ∂_vG·v′ replaced GELU-T's ∇_zG·θ\*′: with the old denominator, T had
  κ ≈ −1.4 with a positive observed lag. With the total derivative, the exploration already showed obs/pred of
  0.997 (T), 1.005 (D) and 1.02 (T′). So the formula was chosen after seeing exploratory lags. It was fixed before
  any registered or pilot seed.
- **C.** T's and T′'s lag-free switch is read from each run's own v path (§5), not frozen, because their G depends on
  the direction of v and not on s alone. The procedure (`width2_asym.own_path_switch` with `OwnBranch`) reads the v
  path and the frozen copy only. It uses no gap of the training state and is computed with the predictions, before
  any gap is read. It is frozen and hashed with the code. **T's and T′'s predictions are conditional on that path.**

## 2. Setting and seeds

- a = 1.30 and Δ = 0.4, as T2-1 to T2-3 (`results/asym_registration.md`). I = [−0.8, 0.8] is class 0; O = [−2.0, −1.2]
  ∪ [1.2, 2.4] is class 1.
- Width 2: N = v₁f(α₁x + β₁) + v₂f(α₂x + β₂) + b, with f(t) = t + a·sin t. The hidden coordinates are z = (α₁, β₁, α₂,
  β₂, b) and the output coordinates are v = (v₁, v₂), with s = ‖v‖₁.
- Each seed has its own 400-point sample `asym_register.training_set(seed)`. The loss is mean BCE with logits, in
  float64 (numpy, with the analytic gradient and Hessian).
- **G₊** = min_O φ − max_I φ, with φ = v₁f(t₁) + v₂f(t₂). It is the exact-extrema enclosure (`width2_geometry.extrema`
  on explicit windows, ε_G = 1e−9), screened by a dense grid whose value bounds G₊ from above. It is evaluated at
  every step.
- **s₀** = 0.5·s_pop2 = 0.2225397, where s_pop2 = √(s_lo·s_hi) = 0.4450794 is T2-1's validated bracket
  (`results/asym_scores.json`, hashed).
- **Seeds.** Arms T and D use 884,000–884,119, with the same seeds and the same own samples in both. Arm T′ uses
  884,200–884,319. The pilot uses 884,900–884,909 for all arms.
- **Unused-seed check (2026-09-29).** No integer in these ranges appears as a token in any text file under `src/`,
  `tests/`, `results/` or `paper/`, other than this test's own files. No `*seed*` column of any
  `results/**/*.parquet` file (14 files) takes a value in 884,000–884,999.

## 3. Width-2 background (from the draft, ccce4fa; exploratory, recomputed where frozen)

- **Branches at s₀ with v held at equal shares.** Random hidden starts reach:
  - the unplaced **duplicate** branch D in 72% of starts (units equal mod 2π: the width-1 function; own samples pooled
    63%, range 13–83%);
  - a two-unit branch that is already placed in 27%;
  - a duplicate D′ in 1%.
- **Branches at s₀ with v = s₀·(0.1, 0.9).** An unplaced **two-unit** branch T in 59%, T′ in 8%, and D in 33%.
- At s₀ = 2.54 (half of D's switch, the GELU-T analogue) every landing point is placed. This is why s₀ = 0.5·s_pop2.
- **Symmetries.**
  - Sign flip per unit: fixed by v > 0.
  - Unit swap: gives no second copy. D is invariant under it; T's v₀ is not.
  - A 2π winding per unit (β_i + 2πk_i, b − 2πk·v): the same function up to the bias and the same switch, but a
    different κ. For D, κ = 7.67, 0.18 and −7.31 at k₁ + k₂ = 0, 1 and 2.
  - T′ is not T's unit-swap image (exploratory, `w2_explore_h`).
- **Feasibility, D (population, exploratory).** At η 0.3 and ρ 1: lag 203 steps (slaved 200), r_obs 0.0313 against
  κχ 0.0311, window max χ_t 0.006. At η = 1 there is no tracking.
- **Feasibility, T (population, exploratory).**
  - At ρ = 1 the run jumps to s = 0.48 in one step.
  - At η 0.03 and ρ 0.001: lag 28 steps (slaved 28.1), r_obs 0.00128 against κχ 0.00129, κ 0.048.
  - T's switch is 0.482 on the adiabatic path (own samples 0.42–0.54) and 1.92 on the fixed ray. T2-3's occupied
    branches switched near there (median 0.486).
- **Feasibility, T′ (exploratory).** T′ crosses early, as its negative κ predicts: −49 steps against −47.9 slaved at
  η 0.03 and ρ 0.001 (`w2_explore_c_Tprime.log`).
- **Relation to T2-3.** T2-3 registered a crossing at the width-2 threshold. It failed: Adam crossings sat far above
  it. The design's reading is that T2-3's runs occupied T-type branches at a speed where the lag is large. PASS on T
  reads as "T2-3's branch type is crossed at its switch plus the lag". PASS on D reads as "the law holds on the branch
  that T2-3b–d's slow runs followed".

## 4. The three arms

**Common to all arms**
- **Hold.** v is held at v₀. z takes full-batch GD steps at lr 1.0 for W steps. G is evaluated at the start state and
  after every hold step; `hold_G_positive` is set if G > 0 at any of them (placed = enclosure lower end > 0).
- **W.** W_c = max(4000, ⌈25/λ_min,own,c(s₀)⌉), with λ_min the full z-Hessian's λ_min at the seed's own copy c.
  - Arm D uses max(W_D, W_D′), because the copy is not known before release.
  - Arm T uses W_T. Arm T′ uses W_T′.
  - A copy with no accepted own point does not count. The floor is 4000.
- **Release.** Full-batch SGD without momentum: η on z, and ρ·η on v.
  - Budgets: D 40,000 steps; T and T′ 100,000·2⁻¹⁰/ρ steps.
  - ρ comes from each arm's pilot (§8).
- **Classification at release** (release information only). Damped Newton at v₀ from the release state is accepted if
  max|∇| < 1e−8 and the z-Hessian is positive definite. The run is **on copy c at winding k** if:
  - the accepted Newton point is within 1e−6 (sup) of shift(θ\*_own,c(s₀), v₀, k), and
  - the release state is within 1e−3 of it, and
  - for D and D′, the Newton point is a duplicate.
  Otherwise the run is on **neither**. An unconverged Newton counts as neither. Such runs are counted and never
  replaced.

| arm | role | v₀ | start before the hold | η | seeds | classified against | scored copy | gate |
|---|---|---|---|---|---|---|---|---|
| **T** | headline | s₀·(0.1, 0.9) | coordinates 0, 1, 3, 4, 6 of the seed's U(−1, 1)⁷ (width2_train's draw, local generator) | 0.03 | 884,000–884,119 | T, T′ | T | ≥ 60/120 on T, regardless of G in the hold |
| **D** | control | s₀·(½, ½) | the same | 0.3 | 884,000–884,119 | D, D′ | D, D′ | ≥ 60/120 on D or D′, regardless of G in the hold |
| **T′** | own arm | s₀·(0.1, 0.9) | θ\*_T′,pop(s₀) | 0.03 | 884,200–884,319 | T′, T | T′ | ≥ 108/120 on T′ **and** no run with G > 0 at the start or any hold step |

- **G > 0 in a T or D hold.** The run is scored and flagged. GELU-T's registered descriptive sensitivity analysis
  (§10) excludes it.
- **Gate failure.** The arm is UNRESOLVED and stops. Its crossings are not evaluated.
- **Winding table.** A run on a copy at a winding with k₁ + k₂ outside −3..3 gets no prediction (counted).

## 5. Predictions (per on-branch run; the release classification, the frozen inputs and the run's v path only)

**The switch (the lag-free comparator of L5)**
- **D, D′.** The frozen s_switch. D's branch is stationary for every v with the same s, and its G depends on s only
  (tested). t_sw is the first step with s_t ≥ s_switch.
- **T, T′ (change C): `own_path_switch`.**
  - z\*(v_t) is followed along the run's own v path by warm-started damped Newton. It starts from the release copy at
    its winding, with the predictor z\*(v_{t−1}) + ∂z\*/∂v·(v_t − v_{t−1}).
  - The branch is lost at the first step whose Newton is not accepted or whose correction exceeds 0.2 (sup).
  - t_sw is the first step t ≥ 1 at which z\*(v_t) is placed (enclosure midpoint > 0).
  - s_switch is ‖v‖₁ at the root of the branch gap on the segment v_{t_sw−1} → v_{t_sw}, bisected to 1e−12 in the
    segment parameter.
  - Only v is read. No gap of the training state is read.

**The predictions**
- **r_traj** (the primary prediction; L1, L3, L4, L5). Track A's R4 recursion, SGD form (P = I):
  - δ₀ = z_release − z\*(v₀);
  - δ_{t+1} = (I − ηH_t)δ_t − (z\*(v_{t+1}) − z\*(v_t)), with H_t the z-Hessian at z\*(v_t) on the run's own v path;
  - the predicted crossing t_traj is the first t ≥ 1 with z\*(v_t) + δ_t placed (enclosure midpoint > 0);
  - s_traj = s_{t_traj} and r_traj = s_traj/s_switch − 1.
  - There is no prediction if the one-step map's spectral radius exceeds 1 (checked every 10th step), if sup|δ| > 1,
    or if the branch is lost first.
- **r_cf** (L2) = κ_k·χ, with χ = (ṡ/s_switch)/(η·λ_min).
  - ṡ = (s_{t_sw} − s_{t_sw−w})/w, with w = min(100, t_sw).
  - κ_k and λ_min are the frozen values at the copy's reference switch (§6), at the run's winding.
  - The closed-form crossing is s_cross = s_switch·(1 + κ_kχ).
- **Observed.** The observed crossing is the first step t ≥ 1 whose state is placed (enclosure lower end > 0). Then
  s_obs = ‖v‖₁ there and r_obs = s_obs/s_switch − 1. All lags are signed.
- **Competing predictions.** (a) No lag: the crossing at s_switch (L5). (b) The branch's population switch (L4): D
  5.07964, D′ 4.97059, T 0.481586, T′ 0.327167 (recomputed; see §0 item 3 for D′).

## 6. Frozen before any registered training (the registration commit)

**`landscape.json` (`width2_asym.landscape`).** The four population copies at v₀. Each is found by damped Newton on
the population from the exploratory class point, and checked to be accepted, of its type and unplaced. Each switch is
validated as below with s_max = 30. The landscape STOPs if a copy is invalid, or if its switch is more than 0.5% from
the page's value (one exploratory step, §0 item 5).

**`frozen_seeds.json` (`width2_asym.freeze`), per seed and copy**
- **θ\*_own(s₀):** Newton from the population copy on the own sample, accepted and of its type. With it: λ_min(s₀)
  and W.
- **s_switch by validated continuation.**
  - D and D′ go along the diagonal (log-s steps 0.02). T and T′ go along the adiabatic reference: the reduced gradient
    flow of v, integrated in s by RK4 in steps of 0.5% of s, with z\* by Newton at every stage.
  - The switch is the first sign change of the branch gap (the enclosure midpoint), bisected.
  - Validation:
    - step halving agrees to 1e−6 relative;
    - no turning point (s increasing along the flow) and every Newton accepted;
    - reduced λ_min > 0 along the path;
    - D and D′: the split block is positive definite and the point is duplicate throughout;
    - G(s₀) < 0 and the switch goes to placed;
    - the enclosure is decided at (1 ± 1e−3)·s_switch;
    - Newton is accepted at the switch;
    - ∇G's forward and backward differences agree to ≤ 1e−5, relative to the sup norm of the five derivatives;
    - κ is finite.
- **κ at the switch,** as in §7: κ₀, the winding coefficients (a₁, a₂), the table over k₁ + k₂ = −3..3, and the
  reduced λ_min.
- **Follow point:** the copy's branch point at 0.8·s_switch on its reference path, validated by step halving (1e−6
  sup), with ∂z\*/∂v there.
- **Frozen in full or as points only.**
  - Registered seeds 884,000–884,119: T, D and D′ in full; T′ as a point only (T's classification).
  - 884,200–884,319: T′ in full; T as a point only.
  - Pilot seeds: all four in full.
  - A copy that fails any check is invalid. A run on it gets no prediction (counted).

**Also frozen**
- `pilot.json` (`width2_asym.pilot`): ρ per arm, and each arm's pilot median of χ at t_sw.
- **The code and its hash**, including the own-path switch procedure (change C). `registration.sha256` lists: this
  file, the design page, the code, the tests, the three frozen files, `results/asym_scores.json`, and the modules the
  pipeline calls (`width2_geometry`, `asym_register`, `asym_pilot`, `width2_train`, `track_a`, `act_fold`). `train`
  and `observe` assert every hash before running.

## 7. κ (P = I; the total derivative, disclosure B)

- **Definition.** κ = λ_min·[∇_zG·H⁻¹θ\*′]/[dG/ds], with dG/ds = ∇_zG·θ\*′ + ∂_vG·v′. Here H is the z-Hessian at fixed
  v, θ\*′ = −H⁻¹H_zv·v′ is the branch tangent, and v′ = dv/ds is the reference path's direction: (½, ½) on the
  diagonal, the flow direction on the adiabatic path.
  - ∇_zG uses central differences (step 1e−6) of the enclosure midpoint; b does not enter G.
  - ∂_vG·v′ is central along v′.
  - The ∂_vG term is 0 for D at its switch, because G is homogeneous in v there. For T it dominates: at the population
    switch, ∇_zG·θ\*′ = −0.042 and ∂_vG·v′ = 1.317.
- **Reduced Hessian (D, D′).** The split directions δ(α₁, β₁) = v₂e, δ(α₂, β₂) = −v₁e are excluded. They are
  first-order function-preserving, orthogonal to ∇_zG and θ\*′ at v₁ = v₂, and positive definite. λ_min is taken on
  their orthogonal complement. T and T′ use the full z-Hessian.
- **Winding.** A winding k leaves H, ∇_zG and ∂_vG unchanged and moves θ\*′ by −2π(k·v′) in b. So κ_k = κ₀ + a₁k₁ +
  a₂k₂ with a_i = −2π·λ_min·(H⁻¹∇_zG)_b·v′_i/[dG/ds] (exact; tested against a direct computation at the shifted
  point). For D, a₁ = a₂.
- **Sign.** κ is signed. It is invariant under G → −G and s → −s (tested); the argument is GELU-T's §10a. κ < 0
  predicts a crossing before the switch. **The sign is part of what is tested,** and it is T′'s point.

## 8. Pilot rules (pilot seeds only; up to t_sw; before registered training)

- Each arm is run on the 10 pilot seeds up to t_sw only. No gap of the training state is evaluated after release, and
  no crossing or residual is computed.
- The values come from the on-branch pilot runs with a prediction set-up: a valid copy and a winding in the table.
- **D** (GELU-T's rule). The values are the SIGNED κ·χ at t_sw.
  - At ρ = 1: ρ = min(1, 2^⌊log₂(0.1/q90)⌋).
  - If ρ < 1, rerun at ρ. Keep ρ if q90 ≤ 0.1; otherwise halve it, up to three halvings, and then STOP.
  - No value: STOP.
- **T and T′, separately** (the page's rule). Start from ρ = 2⁻¹⁰. The value is V7's window max χ_t (s_t ≥
  0.8·s_switch, t < t_sw).
  - Keep ρ if q90 ≤ 0.1; otherwise halve it and rerun, at most three halvings, and then STOP.
  - No value: STOP.
- Each arm's pilot median of χ at t_sw, at its chosen ρ, is frozen for V6.
- A STOP halts W2-A before registration, and the author is asked.
- **Result:** §14, after the pilot.

## 9. Scored runs and the follow check

**Scored runs.** A run is scored if all of these hold:
- it is on its arm's scored copy at release, at a winding in the table, on a valid copy;
- it follows its branch;
- it crosses within the budget;
- it has finite r_traj, r_cf, r_obs and s_traj.

Runs with κ ≤ 0 are scored (predicted early).

**Follow check** (GELU-T's §5 and §13).
- **Where.** At t₀.₈, the first step with s_t ≥ 0.8·s_switch of the copy's frozen switch (where its follow point is).
- **The run's point.** Damped Newton at v_{t₀.₈} from the run's z.
- **The copy's point.** Newton at v_{t₀.₈} from the frozen follow point, carried by one tangent predictor and shifted
  to the release winding.
- **Follows** iff both Newton points are accepted, they agree within 1e−6 (sup; no state-distance condition), and,
  for D and D′, the run's Newton point is still a duplicate.
- A run that does not follow is not scored; it is counted and reported. The check uses the state at t₀.₈ only, with no
  gap.

## 10. Registered criteria and validity (`width2_asym.score_arm`, per arm; GELU-T's with the changes stated)

**Gate first** (§4). If it fails, every criterion is UNRESOLVED and the outcome is "UNRESOLVED (gate)".

**Validity.** All seven conditions are required. Otherwise L1–L5 are UNRESOLVED and the outcome is "UNRESOLVED
(validity)".

| | condition |
|---|---|
| V1 | ≥ 60 of the arm's 120 runs are scored |
| V2 | t_sw strictly precedes the crossing in ≥ 90% of on-branch crossing runs with κ > 0 (undefined t_sw counts as not preceding; with no such run it holds) |
| V3 | η·λ_min(switch) ≤ 0.5 in ≥ 80% of scored runs |
| V4 | median over scored runs of the SIGNED predicted lag κ/(η·λ_min) ≥ 10 steps (**§0 item 1 for T′**) |
| V5 | q90 over scored runs of the SIGNED κχ at t_sw ≤ 0.1 (**§0 item 2 for T′**) |
| V6 | \|median χ at t_sw over scored runs / the arm's pilot median − 1\| ≤ 0.30 |
| V7 | q90 over scored runs of max χ_t over 0 ≤ t < t_sw with s_t ≥ 0.8·s_switch ≤ 0.25. Here χ_t = ((s_{t+1} − s_t)/s_t)/(η·λ_min,t), with λ_min,t the reduced λ_min at z\*(v_t). An empty window or a non-finite χ_t fails. |

**Criteria** (scored runs only; every lag signed):

| | statistic | PASS if |
|---|---|---|
| **L1** | median of r_obs/r_traj | ∈ [0.90, 1.10] |
| **L2** | median of r_obs/r_cf | ∈ [0.80, 1.20] |
| **L3** | Spearman(r_traj, r_obs), average ranks | ≥ 0.5 |
| **L4** | D = \|log(s_obs/s_traj)\| − \|log(s_obs/s_pop,branch)\| | upper end of the 95% percentile bootstrap interval of mean D < 0 |
| **L5** | D = \|log(s_obs/s_traj)\| − \|log(s_obs/s_switch)\|, over the scored runs with \|t_traj − t_sw\| ≥ 6 (change D, clarification 2) | the same rule; UNRESOLVED if fewer than half the scored runs, or fewer than 2, remain |

- **Bootstrap.** 10,000 resamples with replacement, `numpy.random.default_rng(884000)`; the interval is the 2.5th and
  97.5th percentiles of the resampled means.
- **Not computable.** A statistic that cannot be computed is UNRESOLVED: a NaN Spearman (e.g. all predictions tied),
  or fewer than 2 runs for the bootstrap.
- **L5 resolution (change D).** N_min = 2(g + τ + d) = 6 steps:
  - g = 1: first-step detection;
  - τ = 1: Corollary L3's discreteness;
  - d = 1: one exact enclosure, ε_G = 1e−9. Observation and prediction use the same enclosure: the observation needs
    its lower end > 0, the branch and prediction rule its midpoint > 0.
- **Outcome per arm.** PASS if L1–L5 all pass; UNRESOLVED if any is UNRESOLVED; otherwise FAIL, naming each failing
  criterion. Numbers are compared in float64 without rounding.
- **Descriptive sensitivity analysis (arms T and D; GELU-T's; NOT a criterion).** The whole scoring is repeated
  without the runs that had G > 0 at the start or in the hold. It is reported beside the registered verdicts and
  never replaces them.

## 11. Details the approved page leaves open, fixed here (implementation only)

1. **Population copies.** Found by damped Newton from the exploratory class points (`POP_START`, 6 decimals), not by
   a new multistart. They are checked for type and unplaced status, and their switches against the page's numbers to
   0.5% (§0 item 5).
2. **Observation and prediction rules.** Observation: enclosure lower end > 0. Branch, switch and prediction:
   enclosure midpoint > 0. Both are screened by the dense grid, which bounds G₊ from above.
3. **W per arm** as in §4.
4. **Classification sets.** Arm T also classifies against T′, and arm T′ against T, for counting only. D-landings at
   T's v₀ (D is stationary there) are counted as neither.
5. **Frozen in full or as points** as in §6.
6. **κ at a winding** by the exact affine formula (§7; §0 item 4).
7. **T's and T′'s r_cf.** It uses the frozen κ_k and λ_min at the adiabatic reference switch, with the own-path
   s_switch and ṡ.
8. **Two different switches.** t₀.₈ uses the copy's frozen switch, where its follow point is. V7's window uses the
   prediction's s_switch (the own path for T and T′).
9. **The own-path procedure's numerics** as in §5 (jump limit 0.2 sup; segment bisection to 1e−12).
10. **R4's predicted crossing** is counted from t ≥ 1, as the observation is. The spectral radius is checked every
    10th step.
11. **The one-sided ∇G check** is relative to the sup norm of the five derivatives, because a component can be near 0.
12. **Bootstrap seed:** 884,000.
13. **Pilot.** ρ per arm (the page's "ρ per arm"). T′ is piloted from the branch-point start on the pilot seeds.
14. **Continuation steps.** Diagonal: log-s steps 0.02, halved 0.01. Adiabatic: RK4 at 0.5% of s, halved 0.25%.
    Decided at (1 ± 1e−3)·s_switch along the reference path.
15. **Lost branch.** A run whose own-path branch is lost before its switch or its predicted crossing gets no
    prediction (counted).
16. **Paths.** Stored as float64 `.npy` in `results/width2_asym/paths/` (untracked). Each is hashed in
    `predictions.csv`, and `observe` asserts every hash before reading any path.
17. **Machine rules.** One process, nice 15, one thread. A memory gate before every job and between seeds (logged in
    `results/width2_asym/memory_gate.log`). The job stops above 3 GB RSS.

## 12. Competing outcomes and falsifiers (per arm)

| outcome | reading |
|---|---|
| PASS (T, headline) | T2-3's branch type is crossed at its own switch plus the predicted lag, with no fit; T2-3's large overshoot was speed, not a different law. |
| PASS (D) | The law holds on the duplicate (width-1) branch that T2-3b–d's slow runs followed, at width 2. |
| PASS (T′) | A predicted-early crossing (κ < 0) happens early by the predicted amount: the sign of the lag is predicted. |
| L1 fails | The recursion misses the typical lag (for T′: its sign or size). |
| L2 fails, L1 passes | The slaved closed form misses what the recursion gets. |
| L3 fails | The magnitude is right but not which runs lag more. |
| L4 fails | The per-run prediction is no closer than the branch's population switch. |
| L5 fails | The lag is not resolved against a lag-free crossing at the run's own switch. |
| L5 UNRESOLVED | Fewer than half the scored runs (or fewer than 2) have \|t_traj − t_sw\| ≥ 6 steps. |
| follow check fails often (V1) | Runs leave D for the placed branch, or T splits: the branch identity is not kept. |
| gate fails | T: random holds do not reach T; D: they do not reach D or D′; T′: the branch-point start does not stay on T′, or the hold places a run. |
| UNRESOLVED (validity) | The test says nothing about the law. |

A failure stays a failure. Post hoc readings may be placed beside it, labelled POST HOC.

## 13. A-priori risks (exploratory; population and used seeds only)

- **The T gate is at risk.**
  - The draft's chance of passing 60/120 was about 0.998 (`w2_explore_g`). It used the pooled own-sample rate of all
    unplaced two-unit landings, 0.625, which counts T and T′ together.
  - On the population at shares 0.1/0.9, T-type landings are 62 of 100 and T′-type 12 of 100 (`w2_explore_h`). At
    that proportion T's own rate would be about 0.52, and the binomial chance of ≥ 60/120 about 0.73.
  - Unconverged holds (up to 35% in one own sample) and landings on other two-unit points also count as neither.
  - D's pooled rate (0.633) is D-type only, so its chance of about 0.999 is unaffected.
- **Windings.** For T, one winding reverses κ's sign (§0 item 4). All 400 exploratory landings were at (0, 0).
- **T′'s pilot.** Its window max χ_t on the population is 0.176 at ρ = 0.001 (sampled every 25 steps), so the pilot
  may halve ρ once or twice. At ρ = 0.0003 it is 0.053.
- **T's lag.** It is about 28 steps. Its resolution r_min is 3e−4 against a lag of 1.3e−3 (change D's numbers).

## 14. Frozen values

Added with the registration commit: the landscape, the freeze counts and the pilot.
