# Design for review: the width-2 extension (Route A)

**Design only. Nothing here has been computed or trained. It supersedes `block6_width2_design.md`.**

- Target: the rebuttal revision. If it finishes and verifies cleanly before 25 September, it can go into
  the submission.
- Compute priority: the registered items in flight keep priority (the c₁ test, the certificates, the lag
  test, the prospective test).
- Every validity check and stop condition below is exercised on constructed pass and fail cases
  **before** registration.

## 0. Model and notation

- **Network**: N(x) = v₁u(α₁x + β₁) + v₂u(α₂x + β₂) + b.
  - Hidden parameters θ = (α₁, β₁, α₂, β₂); output weights w₂ = v = (v₁, v₂); bias b.
  - The activations u ∈ {f₁.₃₀, f₁.₅₀, tanh}, with f_a(t) = t + a sin t.
- **Task**: I = [−0.8, 0.8] is class 0, O = ±[1.2, 2.0] is class 1. Logistic loss, mean reduction,
  training sets `fold1d.make_data(200, seed)` (400 points), as at width 1.
- **Sign-correct**: N < 0 on I and N > 0 on O, decided with exact extrema.
  - For f_a, the extrema are the window endpoints plus the interior critical points, which are known in
    closed form.
  - For tanh, which is monotone, each unit's extrema are at the endpoints. The extrema of the sum use the
    critical points of the two-unit sum, found by root isolation.
- **Output function without bias**: φ_v(x) = Σᵢ vᵢu(αᵢx + βᵢ), with direction ṽ = v/‖v‖₁.
- **Oriented gaps**: G₊(φ) = min_O φ − max_I φ and G₋(φ) = min_I φ − max_O φ, with G = max(G₊, G₋).

## 1. The geometric quantity Γ₂ and the margin bound

**Definition.** Γ₂(u) = sup over ‖ṽ‖₁ = 1 and all θ of G(Σᵢṽᵢu(αᵢx + βᵢ)).

**Lemma 1 (homogeneity, bias invariance).** For c > 0, G±(cφ) = c·G±(φ). And G±(φ + b) = G±(φ).
- *Proof*: min and max commute with positive scaling and with adding a constant.

**Theorem 1 (margin bound).** If N is sign-correct with margin m > 0, i.e. N ≤ −m on I and N ≥ m on O,
then **m ≤ ½‖w₂‖₁Γ₂**.
- *Proof*:
  - With φ = N − b, we have G₊(φ) = min_O N − max_I N ≥ 2m.
  - By Lemma 1, G₊(φ) = ‖v‖₁·G₊(φ_ṽ) ≤ ‖v‖₁·Γ₂. ∎
- **Definition**: **R₂ = ‖w₂‖₁Γ̂₂/2.** Theorem 1 then says a margin-m network has R₂ ≥ m·Γ̂₂/Γ₂.

**Lemma 2 (Γ₂ ≥ G, the width-1 value).** Take ṽ = (±1, 0). Then φ is a single unit, and the supremum over
(α₁, β₁) is the width-1 Ĝ(u). So Γ₂ ≥ Ĝ. ∎

**Do not sum per-unit gaps (Lemma 3).**
- G₊ is **superadditive**: G₊(φ₁ + φ₂) ≥ G₊(φ₁) + G₊(φ₂), because the minimum of a sum is at least the sum
  of the minima, and likewise for the maximum.
- So per-unit gaps give only a lower bound on the combined gap, and they can be badly wrong. Every tanh
  unit has G ≤ 0: a monotone function cannot lie below or above both outer sides at once. Yet a pair of
  them has G → 1.
- Γ₂ is therefore computed directly on the sum.

**Theorem 2 (tanh, exact).** Γ₂(tanh) = 1. It is a supremum and is not attained.
- *Upper bound*: take any x_I ∈ I, x_L = −1.2 and x_R = 1.2.
  - Then G₋(φ) ≤ φ(x_I) − ½(φ(x_L) + φ(x_R)) = Σᵢṽᵢ[uᵢ(x_I) − ½(uᵢ(x_L) + uᵢ(x_R))].
  - Each uᵢ is monotone, so uᵢ(x_I) lies between uᵢ(x_L) and uᵢ(x_R). Each bracketed term therefore has
    modulus at most ½|uᵢ(x_R) − uᵢ(x_L)|, which is < 1 because tanh takes values in (−1, 1).
  - Hence G₋ < Σ|ṽᵢ| = 1. G₊ is bounded the same way.
- *Lower bound*: ṽ = (½, −½), with φ = ½tanh(α(x + 1)) − ½tanh(α(x − 1)). As α → ∞, G₋(φ) → 1. ∎
- So R₂(tanh) = ‖w₂‖₁/2.
- The design verifies Theorem 2 numerically: the search must approach 1 and never exceed it.

**Γ̂₂ for f_a (a = 1.30 and 1.50): a validated search, plus a certificate where it closes.**
- **Symmetries** used to reduce the domain (each checked numerically, §3):
  - unit permutation;
  - per-unit orientation: (αᵢ, βᵢ, ṽᵢ) → (−αᵢ, −βᵢ, −ṽᵢ), since u is odd;
  - global sign φ → −φ, which swaps G₊ and G₋;
  - the mirror x → −x, a symmetry because the windows are symmetric;
  - βᵢ mod 2π for f_a: a shift adds the constant ṽᵢ·2π, which G ignores.
- **Compactness (Lemma 4, to be proved before registration).** Write φ = c·x + d + aΣṽᵢ sin(αᵢx + βᵢ),
  with c = Σṽᵢαᵢ.
  - Then G₊(φ) ≤ −2.8|c| + 2a‖ṽ‖₁, so any configuration with positive gap has |c| < a/1.4.
  - The individual |αᵢ| are not bounded by this alone: the units can cancel. The proof to be completed
    shows that when |αᵢ| ≥ 2π/1.6, unit i's oscillation covers its full range within I and within each
    side of O. Then its oscillating part cannot raise G, and the configuration is dominated by one with
    that unit's α reduced.
  - **If the proof does not close**, Γ̂₂ is reported as the certified supremum over a stated box
    (|αᵢ| ≤ 10, βᵢ ∈ [0, 2π)), labelled as a box result.
- **Search**:
  - multistart (4,000 starts over the reduced domain; Nelder–Mead on the non-smooth G, then local
    refinement on the active-set pieces);
  - an **independent** search: differential evolution with a different parametrisation of ṽ;
  - agreement to 1e−6 relative is required.
- **Certificate**: branch and bound over the 5-dimensional reduced box, with a Lipschitz step from
  |φ′| ≤ Σ|ṽᵢ||αᵢ|(1 + a), giving an enclosure [Γ̂₂, Γ̂₂ + slack].
  - If the 5D certificate is too costly on this machine (memory is capped, §6), Γ̂₂ is reported as a
    validated lower bound with the stated uncertified gap.
- **Numerical verification of Theorem 1**: on every sign-correct network produced in the calibration runs
  (§4), m ≤ ½‖w₂‖₁Γ̂₂. Report the slack distribution. **A violation is a stop condition.**

## 2. The decomposition at width 2

- **Exact identity** (the width-1 split generalised). N is sign-correct ⟺ G₊(φ_v) > 0 **and**
  −min_O φ_v < b < −max_I φ_v.
  - *Proof*: sign-correct means max_I N < 0 < min_O N, i.e. max_I φ_v + b < 0 < min_O φ_v + b. That
    interval for b is non-empty iff min_O φ_v > max_I φ_v. ∎
  - The orientation must be G₊, because class O is the positive class.
- **Two placement notions**, both reported:
  - **Hidden placement** P₊(θ) = max over ‖ṽ‖₁ = 1 of G₊(Σṽᵢuᵢ). This is the gap the hidden configuration
    can attain under the optimal output direction.
  - **Directional placement** G₊(φ_ṽ), the gap under the actual direction.
  - Always G₊(φ_ṽ) ≤ P₊(θ). Correctness needs the directional one: a hidden configuration with P₊ > 0 can
    still be unplaced if ṽ points the wrong way.
  - At width 1, ṽ = ±1 and the two coincide in the orientation w₂ > 0.
- **Verification before relying on it**:
  - check the identity on 10⁵ random networks per activation, comparing exact-extrema sign correctness
    with the identity (**any mismatch stops**);
  - check it on every checkpoint of the calibration runs.

## 3. Conditional minimisation at fixed ‖w₂‖₁ = s

- **Problem**: minimise L*(θ, ṽ; s) = min_b mean ℓ(s·Σṽᵢuᵢ + b, y) over θ ∈ ℝ⁴ and ṽ on the ℓ₁ circle.
  - ṽ = (t, σ(1 − |t|)), with t ∈ [−1, 1] and σ ∈ {±1}: 5 continuous dimensions plus a sign.
  - b is profiled exactly (strictly convex; the safeguarded Newton of Block 1b). The gradient in (θ, t)
    follows from the envelope theorem.
- **Global search per s**:
  - 2,000 restarts over the reduced compact domain (a stratified design plus uniform draws);
  - L-BFGS on the profiled loss to a gradient norm ≤ 1e−8.
- **Validation, as Block 1 did.** This is validation, not certification: a 5D exhaustive certificate is
  not attempted.
  1. **Retain every restart**: endpoint, loss, gradient norm, G₊, directional and hidden placement, and
     degeneracy flags. The **constant predictor** (loss log 2) is included explicitly as a candidate.
     - **Audit**: no discarded candidate (degenerate, not carried, not lowest) has lower loss than the
       retained one at any s. **A violation stops.**
  2. **Restart convergence**: the retained minimum is unchanged, to 1e−9, from 500 → 1,000 → 2,000 →
     4,000 restarts at every s within ±0.1 of a threshold.
  3. **A stricter, independent search**: 10× the steps with a gradient-norm stop at 1e−10, **and** a
     second search with a different parametrisation (v unconstrained and normalised afterwards) and a
     different optimiser (CMA-ES). **Stop** if either finds a lower loss beyond 1e−9.
  4. **Local certificates** at the retained minimiser near each threshold:
     - an interval-arithmetic Hessian positive-definiteness check on a small box (a strict local
       minimum);
     - slice certificates: the Block 1c two-dimensional branch and bound over one unit's (α, β), with
       the other unit and ṽ fixed.
- **Symmetries and branches**:
  - **Canonical form**: for each unit, orientation such that αᵢ ≥ 0; βᵢ mod 2π for f_a; units sorted by
    (αᵢ, βᵢ); the mirror x → −x, i.e. (αᵢ, βᵢ) → (−αᵢ, βᵢ) for both units, taken to the lexicographically
    smaller form.
  - A **branch** is an equivalence class under this group. The branch distance is the minimum over group
    elements of the Euclidean distance in (θ, t), with β taken mod 2π.
  - **Tests**: loss and G₊ invariant to 1e−12 under every generator. Canonicalisation idempotent.
    Distance zero between group images.
- **Thresholds** per activation:
  - **R₂,glob** is the smallest s at which the global conditional minimiser has G₊ > 0 (directional). It
    is found on a grid, then by bisection to one grid step, and expressed as R₂ = s·Γ̂₂/2.
  - R₂,solve is defined likewise, for sign correctness.
  - A second sign change, or none at all, is reported. **If there is no sign change for an activation,
    its threshold is undefined, and its registered predictions are recorded as not applicable, not
    passed.**

## 4. Unconstrained training and registered predictions

**Everything in this section is registered before any unconstrained width-2 training.** A separate set of
40 **calibration** runs per activation (seeds 500,000+) is used only for the Theorem 1 check, the identity
check and timing. It is never used for predictions.

- **Protocol**:
  - width 2, float64, full batch, Adam lr 1e−2;
  - torch.manual_seed(seed), then (α₁, β₁, v₁, α₂, β₂, v₂, b) ~ U(−1, 1)⁷;
  - `fold1d.make_data(200, seed)`; budget 32,000 steps.
  - **Crossing**: the first step at which G₊(φ_v) > 0, checked every step with exact extrema. R₂ at the
    crossing is ‖v‖₁Γ̂₂/2.
- **Sample**: 80 seeds per activation (600,000–600,079). At least 60 crossings are required per
  activation; otherwise 20 seeds are added at a time, up to 160.
- **Registered predictions**, each scored per activation:
  - **W0 (thresholds)**: R₂,glob and R₂,solve per activation, frozen with the validation above, before
    training.
  - **W1 (primary)**: unconstrained training first becomes correct above the conditional threshold.
    - ≥ 90% of crossing runs have crossing R₂ ≥ R₂,glob;
    - and the bootstrap 95% interval of the median crossing R₂/R₂,glob − 1 lies above 0.
  - **W2 (own-seed thresholds)**, stated in advance on their own terms, not carried over from width 1.
    Each run's own R₂,glob on its 400 points is computed before training, by the same search with
    reduced restarts (validated on a subset).
    - W2a: the own thresholds vary across training sets, with IQR of own/pop ≥ 2%.
    - W2b: crossing R₂ increases with the own threshold (Spearman ρ > 0, one-sided p < 0.05).
    - W2c: **no sign is registered** for median(own/pop) − 1. At width 2 the finite-sample shift has no
      derived direction; it is reported with its interval.
  - **W3 (branch structure)**, stated in advance:
    - W3a: on the population objective, the retained minimiser near R₂,glob belongs to a branch whose
      group images all have equal loss (to 1e−9);
    - W3b: on own training sets those images split, with median relative threshold gap across the images
      > 0 (interval above 0);
    - W3c: runs commit before crossing: the branch at 90% of the time to crossing equals the branch at
      the crossing in ≥ 95% of runs;
    - W3d: crossing R₂ correlates more strongly with the occupied branch's own threshold than with the
      global own threshold (the ρ difference, with a bootstrap interval above 0).
- **The tanh arm.** Tanh can solve the task at width 2 (Theorem 2 gives Γ₂ = 1 > 0).
  - **Two hypotheses**, stated in advance:
    - **H-general**: the conditional threshold predicts tanh too, so the mechanism concerns output scale
      generally.
    - **H-nonmonotone**: it predicts f_a but not tanh, so it is specific to non-monotone activations.
  - **Expected: H-general.** The conditional argument only uses the fact that, at a fixed output scale,
    the logistic loss trades gap against fit. Nothing in it uses non-monotonicity.
  - **Criterion that decides**:
    - H-general iff W1 passes for tanh and for both f_a arms;
    - H-nonmonotone iff W1 passes for both f_a arms and fails for tanh;
    - if W1 fails for an f_a arm, neither hypothesis is supported, and that is reported;
    - if tanh's threshold is undefined (no sign change, §3), the arm is "not applicable" and neither
      hypothesis is decided by it.

## 5. Validity checks and stop conditions

Each is exercised on constructed pass and fail cases before registration (`tests/`), including every file
write-and-read round trip it depends on (exact round-trip parsing).

| check | stop condition | constructed fail case |
|---|---|---|
| Theorem 1 on calibration networks | any m > ½‖w₂‖₁Γ̂₂·(1 + 1e−9) | a network with its margin inflated past the bound |
| identity (§2) on random networks and calibration checkpoints | any mismatch | a bias placed just outside the feasible interval, a wrong orientation |
| Γ̂₂ search against its certificate or the independent search | disagreement beyond 1e−6 relative, or search value > certified upper end | a perturbed G implementation |
| Theorem 2 numerics (tanh) | any search value ≥ 1 | a bounded activation with range > 2 |
| conditional search audit | a discarded candidate below the retained one | a planted lower candidate marked discarded |
| restart convergence and stricter/independent search | lower loss found beyond 1e−9 | a search with its restarts truncated |
| symmetry invariance and canonicalisation | any generator changes loss or G₊ beyond 1e−12 | a deliberately asymmetric window |
| training determinism | the same seed twice not bit-identical | a nondeterministic perturbation |
| crossing detection | disagreement with an independent every-step dense-grid check | an off-by-one step |

## 6. Compute and hardware (this laptop)

- **Priority**: runs only after the registered items in flight (the c₁ test, the certificates, the lag
  test, the prospective test), within the reduced worker cap.
- **Worst-case memory**: the Γ̂₂ certificate and the conditional searches are the risks. Every vectorised
  batch is chunked to ≤ 13 MB per array. Each job's peak memory is measured on a small pilot, and a job
  whose pilot extrapolates past 1 GB per worker is redesigned before launch.
- **Estimates**, to be refined by the pilots:
  - Γ̂₂: hours per activation (multistart, differential evolution, branch and bound);
  - conditional scans: about 40 scales × 2,000 restarts × 3 activations, plus the own-seed thresholds
    for 240 training sets at reduced restarts;
  - training: 240 + 120 calibration runs.

## 7. Order of work after approval

1. Proofs: complete Lemma 4. Verify Theorem 2 numerically.
2. Implement. Construct every check's pass and fail cases; tests pass.
3. Γ̂₂; the identity and Theorem 1 checks on calibration runs.
4. Conditional scans and thresholds (W0), with the full validation.
5. **Register** W1–W3 and the tanh criterion, with the frozen thresholds and hashes.
6. Train, then score.
