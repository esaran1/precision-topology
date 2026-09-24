# Design (approved with amendments): the width-2 extension (Route A)

**Design only, approved with the amendments marked below.** Nothing here has been computed or trained.
It supersedes `block6_width2_design.md`.

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
  - A second sign change, or none at all, is reported.
  - **Scanned range (amendment)**, fixed for every activation before any result exists: R₂ = s·Γ̂₂/2 ∈
    [0.02, 1.00] on a grid of 0.01 in R₂, then bisection. In |w₂|₁ that is s ∈ [0.04, 2.0]/Γ̂₂; for tanh
    (Γ₂ = 1), s ∈ [0.04, 2.0].
  - **"Not applicable", precisely (amendment)**: for an activation, the directional gap G₊ of the
    retained global conditional minimiser has **no sign change over the scanned range**. That is, it is
    ≤ 0 at every grid point, or > 0 at every grid point, both judged after the stricter search of step 3.
    The activation's threshold is then undefined, and its registered predictions (W1, W4) are recorded as
    **not applicable**, neither passed nor failed.

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
  - **W1 (co-primary with W4)**: unconstrained training first becomes correct above the conditional
    threshold, **and not far above it**.
    - ≥ 90% of crossing runs have crossing R₂ ≥ R₂,glob;
    - the bootstrap 95% interval of the median crossing R₂/R₂,glob − 1 lies above 0;
    - **(amendment) an upper bound**: the median crossing R₂/R₂,glob ≤ **1.25**, so that W1 fails if the
      threshold sits far below the crossings.
      - The value is disclosed as informed by the width-1 free-training offsets (1.10–1.16). It is a
        tolerance, not a predicted magnitude.
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
- **W4 (co-primary with W1; amendment): fixed-scale test**, the width-2 analogue of Blocks 4 and 5. W1 alone
  shows only that the threshold is a lower bound; W4 tests whether it is **where** placement switches.
  - **Checkpoints**:
    - pre-correct states (G₊(φ_v) ≤ 0) saved at 60 log-spaced steps of the W1 training runs, with the
      full Adam state;
    - one per run per stratum of current R₂/R₂,glob: [0.2, 0.5), [0.5, 0.8), [0.8, 1.0), [1.0, 1.3).
      Within a stratum, the latest qualifying checkpoint.
  - **Intervention**: (w₂, b) rescaled **jointly** by k > 0 so that R₂/R₂,glob lands on the grid
    {0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0}.
  - **Replay**: train (θ, v, b) with **‖v‖₁ held fixed**. After every Adam step, v is projected onto the ℓ₁
    sphere of the held radius, so the output direction still trains. Horizon **64,000 steps**, recorded
    at 4,000 / 16,000 / 64,000.
    - Optimiser state: "preserved" (primary) and "reset" (secondary).
  - **Outcome**: correct (sign-correct, with exact extrema) at 64,000 steps.
  - **Reference: each training set's own threshold.** Own R₂,glob, and the own threshold of each branch
    (canonical class, §3), are computed on each run's 400 points before any replay. **No expected sign or
    magnitude is carried over from width 1.** Only the method is carried over.
  - **W4a**: the correct fraction at 64,000 steps rises with held R₂ (no significant decrease between
    levels, paired exact McNemar, one-sided p < 0.05). Its **50% point** (linear interpolation, in
    R₂/R₂,glob) lies within **±0.05 of the median own/pop ratio** over the replays' seeds.
  - **W4b**: per replay, "held R₂ above the replay's own threshold **on the branch it occupies at the
    replay's start**" predicts correctness at 64,000 steps, with **agreement ≥ 0.90**.
    - The starting branch is known when the replay starts. The end branch is reported beside it, labelled
      circular.
  - **Validity checks** (the width-1 set, each with constructed pass and fail cases):
    - **k = 1 reproduction**: at k = 1, the replay reproduces an **independent** reference continuation
      that truly freezes ‖v‖₁ (it resets v onto the sphere after every step, as in width-1 amendment 2)
      to 1e−10 over its first 25 steps, on 20 checkpoints;
    - **decisions preserved** at the moment of rescaling: the sign of N at every training point is
      unchanged by (w₂, b) → k(w₂, b);
    - **a true freeze of ‖v‖₁**: |‖v‖₁ − held| ≤ 1e−12 at every step of every replay.
    - Any failure is a stop condition.
- **The tanh arm.** Tanh can solve the task at width 2 (Theorem 2 gives Γ₂ = 1 > 0).
  - **Two hypotheses**, stated in advance:
    - **H-general**: the conditional threshold predicts tanh too, so the mechanism concerns output scale
      generally.
    - **H-nonmonotone**: it predicts f_a but not tanh, so it is specific to non-monotone activations.
  - **Expected: H-general.** The conditional argument only uses the fact that, at a fixed output scale,
    the logistic loss trades gap against fit. Nothing in it uses non-monotonicity.
  - **Criterion that decides**:
    - H-general iff **W1 and W4** pass for tanh and for both f_a arms;
    - H-nonmonotone iff W1 and W4 pass for both f_a arms and either fails for tanh;
    - if W1 or W4 fails for an f_a arm, neither hypothesis is supported, and that is reported;
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
| W4 k = 1 reproduction (true-freeze reference) | difference > 1e−10 | a reference that only zeroes the gradient (the width-1 amendment-2 error) |
| W4 decisions preserved at rescaling | any sign change | k < 0 |
| W4 true freeze of ‖v‖₁ | any drift > 1e−12 | a replay without the projection |

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
6. Train, then score W1–W3. Then run the W4 replays (after W1's training, which provides the
   checkpoints) and score W4.

## Revision 2 — 2026-09-24 (for review; nothing run beyond the exploratory pilot below)

### What the pilot found (EXPLORATORY; `src/width2_pilot.py`, 100 restarts per scale, unvalidated)

- **Cosine identity (checked to 1e−14)**: f_a(αx + π/2) − f_a(αx + 3π/2) = −π + 2a·cos(αx). With opposite output
  weights, the linear parts cancel and the network represents a cosine.
- **The single-cosine problem**: sup over α of G(cos(αx)) = **1/√2 exactly**, at α = 5π/8.
  - *Proof*: G₊ ≤ 0 because the inner maximum is 1 at x = 0. For G₋ = min_I cos(αx) − max_O cos(αx), the
    inner minimum is cos(0.8α) while 0.8α ≤ π. The outer maximum is max(cos 1.2α, cos 2α) while
    [1.2α, 2α] ⊂ [π/2, 3π/2]; cos is symmetric about π there.
  - For 1.6α < π the outer maximum is cos 1.2α, and dG₋/dα = −0.8 sin 0.8α + 1.2 sin 1.2α > 0 near the switch.
    For 1.6α > π it is cos 2α, and dG₋/dα = −0.8 sin 0.8α + 2 sin 2α < 0 there.
  - So the maximum is where cos 1.2α = cos 2α, i.e. 1.2α + 2α = 2π,
    α = 5π/8. There G₋ = cos(π/2) − cos(3π/4) = 1/√2.
  - Outside that interval G₋ is smaller, which the grid check over α ∈ [0.01, 12] confirms
    (`width2_pilot_single_cosine.csv`). ∎
- **The width-2 maximiser is not the single cosine.** The pilot's best point has two different frequencies
  (α₁ ≈ 7.7, α₂ ≈ 1.7) with linear parts that cancel (c = Σṽᵢαᵢ ≈ 0). Its value is Γ̂₂/a ≈ 0.911–0.913, above
  1/√2.
  - When c = 0 exactly, G(φ)/a is the a-free problem κ = sup G(Σṽᵢ sin(αᵢx + βᵢ)) subject to Σṽᵢαᵢ = 0.
  - So **Γ₂ = κ·a holds if the maximiser always has c = 0**. That is not proved; a width-2 function cannot be
    symmetrised within width 2.
  - The full searches at a = 1.30 and 1.50 and the κ search (`width2_pilot gamma`, `kappa`) test it
    numerically. It is reported as validated, not proved, unless a proof closes.
- **Conditional minimisers are placed at very small output scale.**
  - For f_a (a = 1.30, 1.50), G₊ of the retained minimiser changes sign between R₂ = 0.05 and 0.10, i.e.
    ‖w₂‖₁ ≈ 0.08–0.17. That is about 30–50× below the width-1 switch scale.
  - Above it the minimiser is a symmetric pair (α₁ = α₂ ≈ 1.79, β = π/2 and 3π/2, equal weights).
  - For tanh the minimiser is placed at every scanned scale (G₊ = 1), with α diverging: the conditional infimum
    is not attained.
- The scans at a = 0.5 and 1.0 (monotone f_a) test whether width 2 places even without non-monotonicity; the
  cosine identity predicts it does. They are exploratory.

### Decisions recorded (author, 2026-09-24)

1. **Γ̂₂**: reported as validated by the two independent searches (4,000-start Nelder–Mead; differential
   evolution with a different parametrisation), agreeing to 1e−6 relative, and **labelled not certified**. The
   5D branch and bound was measured not to fit: about 10²⁰ cells at 1e−6. Lemma 4 is not closed.
2. **tanh: "not applicable"**, because the conditional infimum is not attained (α diverges; G₊ = 1 at every
   scale). **The audit's non-convergence stop is waived for tanh only.** Both facts are disclosed. The tanh
   criterion (H-general vs H-nonmonotone) is therefore not decided.

### Revised design (replaces §4's initialisation; W1 and W4 criteria unchanged)

**Primary arm: matched initialisation.** Only the output weights v are scaled at initialisation. The hidden layer
(α, β) and the output bias b are unchanged, since b does not enter ‖w₂‖₁. The scale makes the median initial
‖w₂‖₁ relative to the population threshold equal to the width-1 ratio.
- **Width-1 ratio**:
  - r₁(a) = median initial |w₂| / |w₂|_glob(a).
  - Median initial |w₂| = **0.45899**, the width-1 protocol (`blockG_windows.train`: torch.manual_seed(seed),
    U(−1, 1)⁴ in float32, then double) over the W1 seeds 600,000–600,079.
  - |w₂|_glob(a) = the midpoint of the certified population bracket: **4.95625** at a = 1.30
    ((4.9500, 4.9625]) and **2.53125** at a = 1.50 ((2.5250, 2.5375]).
  - So **r₁(1.30) = 0.09261** and **r₁(1.50) = 0.18133**.
- **Width-2 standard initialisation**: median ‖v‖₁ = **0.97946** over the same seeds (torch.manual_seed(seed),
  U(−1, 1)⁷ in float64).
- **Rule**: v₀ ← k(a)·v₀, with **k(a) = r₁(a)·s₂,glob(a)/0.97946**.
  - s₂,glob(a) is the midpoint of the validated W0 bracket in ‖w₂‖₁, converted from R₂ with Γ̂₂.
  - k(a) is computed from W0 and frozen, with a hash, before any W1 run.
  - The median matched initial ‖w₂‖₁/s₂,glob then equals r₁(a) exactly on these seeds.
- tanh has no threshold, so it has no matched arm.

**Secondary arm: standard initialisation (descriptive only).** The same seeds with U(−1, 1)⁷ unscaled. The stated
expectation is that the runs begin above the threshold: median ‖w₂‖₁ = 0.979 against s₂,glob ≈ 0.1–0.2 in the
pilot. So crossing is not preceded by growth through the threshold. No criterion.

**W1 and W4**: as approved (W1 with its upper bound 1.25; W4 co-primary), **on the primary arm**, per f_a arm.

**W4 horizon, chosen from pilot convergence times that do not test the outcome.**
- **Pilot**: calibration seeds 500,000–500,019, matched initialisation. Replays held at **2.0×** the threshold
  (a level where placement is expected under every hypothesis), both optimiser variants. The measure is the step
  after which sign-correctness no longer changes.
- **Rule**: H = the smallest of {16k, 32k, 64k, 128k} steps that is at least 4× the 95th percentile of that time.
  Outcomes are recorded at H/16, H/4 and H. H is stated in the registration.
- No W1 seed and no other level is used in the pilot.

**Iteration caps.**
- The cap-hit rate is reported per activation at every scan scale.
- For f_a, if any restart hits the cap within ±0.1 in R₂ of a threshold, the cap is raised (2,000 → 5,000 →
  20,000 → 100,000) until none do there. The final cap is stated.
- For tanh the rate is reported; the stop is waived (decision 2).

**Per-unit realisation breakdown (descriptive; at every W1 crossing and every W4 endpoint)**:
- the weight share maxᵢ |vᵢ|/‖v‖₁;
- the linear-cancellation index |Σvᵢαᵢ|/Σ|vᵢαᵢ|;
- the knockout class, using G₊ with unit i removed (vᵢ = 0; G₊ is invariant to b):
  - **single-unit**: exactly one unit alone keeps G₊ > 0;
  - **shared**: neither unit alone has G₊ > 0;
  - **redundant**: each alone has G₊ > 0.
- Fractions are reported per activation and arm, with Clopper–Pearson 95% intervals.

**Unchanged**: W0 (the validated conditional scan; the constant predictor included; the audit; the restart
ladder; the 10× stricter search; the CMA-ES search), W2, W3, every validity check and stop condition of §5, and
compute priority.

### Pilot numbers, recorded 2026-09-24 (EXPLORATORY; `width2_pilot.py` → `width2_pilot_*.csv`)

- **Cosine identity**: holds to ≤ 1e−14 at a ∈ {0.5, 1.0, 1.3, 1.5} (`width2_pilot_identity.csv`).
- **Single cosine**: sup_α G(cos αx) = 0.70710678… = 1/√2 at α = 5π/8 = 1.9635 (`width2_pilot_single_cosine.csv`).
- **The a-free (c = 0) two-sinusoid problem**: κ ∈ [0.91297872170245, 0.91297872170254] (exact enclosure at the
  point found by a 3,000-start search; `width2_pilot_kappa.csv`). The optimiser is two cosines:
  - frequencies α₁ ≈ 7.642 and α₂ ≈ 1.750;
  - weights 0.1863 and 0.8137, with |ṽ₁|α₁ = |ṽ₂|α₂;
  - phases β ∈ {±π/2, 3π/2}.
- **Γ̂₂ (4,000-start Nelder–Mead)**:

  | a | Γ̂₂ | Γ̂₂/a |
  |---|---|---|
  | 1.30 | 1.1868729 | 0.9129791 |
  | 1.50 | 1.3694652 | 0.9129768 |

  - Both have the same two-cosine maximiser with c ≈ 0.
  - Coarse 400-start values: Γ̂₂/a = 0.9128 at a = 0.5 and 0.91276 at a = 1.0.
  - So **Γ₂ = κ·a to about 5e−7 relative at the registered a**. This is validated, not proved; see "Γ₂ = κa?"
    above.
- **Differential evolution (60 × 500) converged to the single-cosine local optimum 1/√2·a at both a.** This
  **disagrees** with Nelder–Mead, which is a §5 stop condition for the registered run.
  - **Change, fixed before any registered Γ̂₂ run**: the independent search becomes **20 independent DE runs
    (seeds 1–20), each population 200 × 2,000 generations**, keeping the best. Agreement with Nelder–Mead is
    still required to 1e−6 relative, and a disagreement still stops.
  - The DE budget is set so the search can escape the single-cosine basin. It is not tuned to any outcome.
- **Coarse conditional scans** (100 restarts per scale, population objective; `width2_pilot_scan.csv`):
  - **f_a at a = 1.30**: the retained minimiser's G₊ changes sign between R₂ = 0.02 and 0.05.
  - **f_a at a = 1.50**: it changes sign between R₂ = 0.05 and 0.075.
  - Above the sign change the minimiser is the equal-weight cosine pair.
  - **Monotone f_a**: a = 0.5 is placed at every scanned R₂ (0.02–1.0); a = 1.0 from R₂ = 0.05.
- **Iteration caps**: restarts hit the 2,000-iteration cap near the thresholds. The counts per 100 restarts are:

  | scale | cap hits |
  |---|---|
  | a = 1.30, R₂ = 0.02 | 20 |
  | a = 1.30, R₂ = 0.05 | 15 |
  | a = 1.50, R₂ = 0.02 | 21 |
  | a = 1.50, R₂ = 0.05 | 21 |
  | a = 1.50, R₂ = 0.075 | 11 |

  - Elsewhere the counts are 0–3.
  - This confirms that the cap-raising rule is needed for f_a near the thresholds.
