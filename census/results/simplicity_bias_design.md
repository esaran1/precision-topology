# Track T design: transfer to simplicity bias (linear + 3-slab, width-4 tanh)

**Design commit, 2026-09-26 ~01:35 EDT. Nothing on the landscape has been computed.** The only numbers here come from
the parameter rule (`python -m src.simplicity_bias rule` → `results/simplicity_bias/rule.json`), which is a
deterministic computation on the data and involves no network and no training. Step 1 (the pilot) is EXPLORATORY.
Step 2 is **not registered here**. It is registered only after a gate pass and the author's go-ahead.

## Citation (verified 2026-09-26)

- Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain, Praneeth Netrapalli, "The Pitfalls of Simplicity
  Bias in Neural Networks", *Advances in Neural Information Processing Systems 33* (NeurIPS 2020).
- Proceedings page: https://proceedings.neurips.cc/paper/2020/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html
- PDF: https://proceedings.neurips.cc/paper/2020/file/6cfe0e6127fa25df2a0ef2ae1067d915-Paper.pdf
- Author list and title were checked against the PDF.
- **Construction used (their §3.1, "One-dimensional Building Blocks").**
  - Linear block: positives uniform on [0.1, 1], negatives uniform on [−1, −0.1].
  - Noisy linear block with noise parameter p: a fraction 1 − p is distributed as the linear block, and a fraction p is
    uniform on [−0.1, 0.1]. The Bayes accuracy of a linear classifier is 1 − p/2.
  - k-slab block: "k well-separated, alternating regions".
  - The paper's datasets LMS-k and L̂MS-k put a (noisy) linear block in coordinate 1 and k-slab blocks in the
    others; the default noise is p = 0.1. LSN and Figure 1 pair a linear coordinate with a single 3-slab coordinate.
- **Where this design deviates:**
  1. The linear coordinate is the **noisy** linear block (the L̂MS-k variant). The classes overlap on x₁, so x₁
     separates the class means but misclassifies p/2 of each class. The paper's clean-linear LMS-k is not used.
  2. d = 2, with one **3-slab** coordinate (as in LSN and Figure 1), rather than d − 1 copies of 5- or 7-slabs.
  3. The slab layout is fixed by the rule below. The paper gives no slab widths in the main text.
  4. The data is a deterministic quantile grid, not an i.i.d. sample.
  5. p = 0.2 in the pilot (10% misclassified), chosen by the rule below. The paper's default is p = 0.1.

## Data (`simplicity_bias.make_data`)

- **Fixed n per class: 400.** Each class is a product grid: 20 x₁ levels × 20 x₂ levels, the class-conditional
  quantiles.
  - Hence x₁ ⟂ x₂ | y exactly, as in the paper, where the blocks are independent given y.
  - Classes are balanced.
- **x₁ (noisy linear block, noise p).**
  - Class 1: the quantiles at (k + ¼)/20 of (1 − p)·U[0.1, 1] + p·U[−0.1, 0.1].
  - Class 0: the mirror image.
  - The ¼ offset makes the two classes' levels interleave in the noise band with no cross-class tie. With a ½
    offset the noise levels coincide across classes. An oblique unit could then use x₂ to break those ties, which
    is a discretisation artefact that the rule computation exposed (Δ_all = 1.65 > Δ_lin = 1.6 at the ½ offset).
    The offset was fixed at this point, before any landscape computation.
- **x₂ (3-slab).**
  - Three equal-width slabs separated by gaps of 0.2, the same gap as the linear block's [−0.1, 0.1]: 3w + 0.4 = 2.
  - Class 0 is in the middle slab, [−0.2667, 0.2667].
  - Class 1 is in the two outer slabs, ±[0.4667, 1], half of its mass in each.
  - Levels are at (k + ½)/n within each slab.
  - x₂ separates every point.
- **Population and training-set convention.**
  - The same fixed 800-point set is the landscape objective (Step 1) and the training set (Step 2).
  - In Step 2, seeds vary the initialisation only.
  - There is no sampling gap between the objective whose switch is predicted and the one that is trained.

## Parameter rule (computed from the objective; `gap_rule`, `rule_ok`)

- With balanced classes and b profiled, the fixed-scale objective at small s is
  L*(s) = log 2 − (s/4)·Δ* + O(s²).
  - Δ* = sup over the unit-ℓ₁ hull of tanh units of E[φ | y = 1] − E[φ | y = 0].
  - So **Δ*/4 is the class-mean gap per unit output scale**, the exact first-order slope. This is tested in
    `test_first_order_slope_is_quarter_gap`.
- **Reduction to halfspaces (exact).**
  - Δ is linear in φ, so the sup over the ℓ₁ hull is the sup over single units.
  - tanh(z) = ∫ sign(z − τ) dF(τ), with F = (1 + tanh)/2 a CDF. So a unit's gap is an average of halfspace gaps.
  - Hence Δ* = sup_θ Δ(θ), with Δ(θ) = 2·max_τ |P₁(u > τ) − P₀(u > τ)| and u = x·(cos θ, sin θ).
  - Δ(θ) is computed exactly on the point set for 7,200 directions.
- Δ_lin = Δ(0) (x₁ only). Δ_slab = Δ(π/2) bounds **every** x₂-only configuration, including the full two-unit slab,
  by linearity. The oblique units are covered by sup_θ.
- **The rule:**
  1. Δ_lin / Δ_slab ≥ 1.2, so the linear feature has a larger mean gap per unit scale than any slab configuration,
     with a stated margin.
  2. No direction beats the pure-x₁ direction: Δ(0) = sup_θ Δ(θ). The small-scale minimiser is then predicted to
     use x₁.
- **In closed form:**
  - Δ_lin = 2(1 − p): the best threshold at 0 has Youden index 1 − p.
  - Δ_slab = 1: a halfspace in x₂ captures one outer slab, and the two-unit slab function has range 1 at unit ℓ₁ norm.
  - Condition 1 therefore holds for p ≤ 0.4, that is, at most 20% misclassified.
- **Pilot geometry: p = 0.2** (10% misclassified, the author's example).
  - Result (`rule.json`): Δ_lin = 1.6, Δ_slab = 1.0, ratio 1.6, Δ_all = 1.6 (attained at x₁ and at tilts
    ≤ 1.2°), best x₁ error rate 0.100. **Rule passes.**
  - The first-order slopes are 0.40 (linear) and 0.25 (slab).

## Network and scale convention

- N(x) = s·Σₖ₌₁⁴ ṽₖ tanh(wₖ·x + cₖ) + b, with wₖ ∈ ℝ², width 4 and ‖ṽ‖₁ = 1.
  - **s = ‖w₂‖₁.** tanh is odd, so each output sign can be absorbed into its unit, and ṽ is taken on the simplex:
    ṽₖ = ηₖ²/‖η‖². This is smooth, with no ℓ₁ kink, and is tested to cover signed outputs.
  - The search is over 16 coordinates (W, c, η), with one flat direction (the norm of η).
  - The output bias b is profiled exactly (safeguarded Newton, `width2_conditional.profile_b_batch`).
  - The gradient comes from the envelope theorem and is tested against finite differences.
- **Separation quantity:** G₊(φ) = ½(min_{y=1} φ − max_{y=0} φ).
  - G₊ > 0 if and only if some bias classifies every point correctly with positive margin.
  - The pilot's switch is the scale where G₊ of the retained conditional minimiser turns positive. This is the
    analogue of the paper's placed/unplaced switch.
  - Sign-correctness with the profiled b is also logged.

## Feature event and feature-usage measure

- **Feature event (Step 2 crossing):** the first Adam step at which every training point is classified correctly
  with positive margin. x₁ alone cannot do this, because 10% of each class is on the wrong side.
- **Feature usage (primary): the paper's S-randomisation made deterministic.**
  - V_j = mean over all pairs (i, k) of |φ(x_i) − φ(x_i with coordinate j replaced by x_{k,j})|.
  - ρ₂ = V₂/(V₁ + V₂).
  - Classification: **linear** if ρ₂ < ½, **slab** if ρ₂ > ½.
- **Why ½.** For hard units, write the network as α·(linear unit) + (1 − α)·(two-unit slab). Then V₁ = α,
  V₂ = (1 − α)/2 and G₊ = (1 − 3α)/2.
  - So *ρ₂ > ½ exactly when every point is separated*: the threshold is the event's own boundary, not a tuned number.
  - This is tested for α from 0.1 to 0.9 in `test_feature_share_and_separation_agree_on_hard_mixtures`.
- **Also logged:** the gradient sensitivities S_j = mean_i |∂φ/∂x_j(x_i)|, as the author suggested. They are not the
  classifier, because saturated tanh units have near-zero derivatives at the data.

## Validated search (per scale; `evaluate_scale`)

- **Restart ladder.**
  - 800 BFGS restarts (gtol 1e−8, max 3,000 iterations), with nested sub-ladders 200 → 400 → 800.
  - The constant predictor is an explicit candidate.
  - Every restart is retained with its loss, gradient norm and flags.
  - Excluding flags: non-finite, or not converged.
  - A run is not converged if its gradient is above 1e−6 **and** its loss is still improving by more than 1e−10 over
    the last 10% of iterations. A saturating tail that has stopped improving is flagged informational,
    `info:stalled_tail`.
  - The ladder passes if the retained loss at 400 and 800 agrees to 1e−7, and G₊'s sign and the feature class agree
    at 200, 400 and 800.
  - The 1e−7 tolerance, rather than width 2's 1e−9, is because tanh units may saturate: the loss approaches its infimum
    along a divergent direction, with loss − inf ≈ gradient/rate.
- **Two independent restart sets.**
  - Set A: W, c ~ U[−6, 6], η ~ U[0.2, 1], seed family 11.
  - Set B: W ~ N(0, 3²), c ~ N(0, 2²), η ~ |N(0, 1)| + 0.05, seed family 23.
  - Different seeds and different start distributions. Each set runs on its own worker.
- **Audit:** no excluded candidate below the retained loss by more than 1e−7.
- **Independent search:** CMA-ES (`src/cmaes.py`) in a different parametrisation.
  - Signed output weights u ∈ ℝ⁴, normalised to the ℓ₁ sphere, with no simplex or odd-symmetry reduction.
  - 16 starts × 300 generations.
  - Run at the two final bracket ends of each set.
  - Stop if it finds a loss lower than the retained one by more than 1e−7.
- **Scale grid:** s = 0.5·1.2ᵏ, k = 0…23 (0.5 to 33.1).
  - Each set then bisects its first G₊ sign change at geometric midpoints until hi/lo − 1 ≤ 0.5%.
  - The switch estimate is the geometric mean of the final bracket.
- **Compute:** two workers (A and B), nice 15, one thread each, well under 3 GB, and resumable (one JSON per scale in
  `results/simplicity_bias/pilot_parts/`). The cap is 90 minutes. If it is reached, the pilot stops and reports what it
  has.

## Gate (committed before the pilot; `gate`, tested on constructed pass and fail cases)

All four conditions must hold:

- **G1, one clean switch.** In each set, G₊ of the retained minimiser changes sign **exactly once** over all computed
  scales (grid and bisection): ≤ 0 below and > 0 above, inside the scanned range. There is no alternation.
- **G2, feature usage confirms.** In each set, every computed scale below the switch is **linear** (ρ₂ < ½), and every
  computed scale above it is **slab** (ρ₂ > ½).
- **G3, agreement.** The two sets' refined switch scales agree within 2%: |s_A − s_B| / min(s_A, s_B) ≤ 0.02.
- **G4, validation.** At each set's final bracket ends, the ladder passes, the audit passes, and CMA-ES finds nothing
  lower.

**If the gate fails:** stop, report what the landscape does instead (from `pilot_scan.csv`), and register nothing.

## Step 2 (outline only; NOT registered by this commit)

- Four geometries (four values of p), chosen by the rule above before any training.
- For each geometry: the frozen, hashed switch s* from this validated search; κ from its landscape at the switch
  (`lag_law` construction: joint Hessian, tangent, ∇G₊, and Adam's P at crossing); and the prediction
  crossing = s*·(1 + κχ), with χ from pre-crossing information only.
- 40 fresh seeds per geometry, Adam with standard settings, detection at every step.
- The author's T1–T3 and the validity conditions.
- The details (κ's P, χ's estimator, the baseline for T3, the seed range) are fixed in the Step 2 registration.
