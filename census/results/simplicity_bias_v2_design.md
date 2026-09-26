# Track T redesign (v2): weight decay and the slab-usage path

**Design and gate commit, 2026-09-26 ~02:40 EDT, before any v2 computation.**
- One gated redesign attempt, approved by the author.
- The hard stop is 04:15 EDT.
- The author's text is binding; the interpretations the author approved are marked [AI-1] to [AI-5].
- v1, whose gate failed, is `simplicity_bias_design.md`. Its pilot is the reason for this redesign: the hidden weights
  diverged, the minimiser was not attained, and the "switch" was an addition of the slab feature, not a replacement.
- Producers: `src/simplicity_bias_v2.py` → `results/simplicity_bias_v2/`. Tests: `tests/test_simplicity_bias_v2.py`.

## Unchanged from v1

- **Data:** noisy-linear x₁ with p = 0.2 (10% misclassified), a 3-slab x₂, 400 points per class on a deterministic
  grid (`sb.make_data`). The parameter rule is unchanged (`simplicity_bias/rule.json`).
- **Network:** width-4 tanh, s = ‖w₂‖₁, output bias profiled in the landscape and trained in training.
- **Feature usage ρ₂:** the deterministic S-randomisation share. The same function (`sb.feature_usage`) is used for
  the landscape and for training.

## (1) Weight decay on the hidden weights [AI-2]

- **Fixed-scale objective:** L_λ(θ; s) = min_b mean ℓ(s·Σ ṽₖ tanh(wₖ·x + cₖ) + b, y) + (λ/2)(‖W‖² + ‖c‖²).
  - The decay applies to all hidden parameters, weights and biases, and not to the output layer.
- **Training:** the same λ, added to the loss as (λ/2)(‖W‖² + ‖c‖²) and minimised by plain Adam (not AdamW).
- **A-priori bound.** θ = 0 gives L_λ = log 2, so every minimiser satisfies ‖(W, c)‖ ≤ B(λ) = √(2 log 2 / λ).
  - L_λ is continuous and coercive in (W, c), and ṽ lies on a compact simplex, so the minimiser is attained for every
    λ > 0.
  - Tested: every point outside B(λ) is worse than the origin.
- **Rule for λ (`simplicity_bias_v2 lambda`).**
  - λ is the smallest value on the grid {1e−4, 3e−4, 1e−3, 3e−3, 1e−2} that passes the **attainment check** at
    s ∈ {1, 4, 16}, using set A's start distribution with 200 restarts. This is the check computation named in the
    rule, and nothing else informs λ.
  - **Attainment check**, at one scale:
    - The retained minimiser carries no flag: gradient max-norm ≤ 1e−6 and not a stalled saturating tail.
    - Its norm satisfies ‖(W, c)‖ ≤ B(λ).
    - At least 2 restarts reach the retained loss within 1e−7.
  - The computation is serial over λ and stops at the first value that passes.
  - If no grid value passes, the gate fails.

## (2) The path and q [AI-2]

- **Path.** The retained minimiser of L_λ at each scale of the grid s = 0.5·1.25ᵏ, k = 0…20 (0.5 to 43.4). Each of the
  two restart sets A and B (v1's start distributions and seeds) uses 800 restarts, with the nested ladder
  200/400/800 reported.
- **q rule.** q = ½(ρ̄₂(s_min) + ρ̄₂(s_max)), where ρ̄₂ is the mean of the two sets' retained minimisers at the first
  and last grid scales. This is midway between the small-scale and large-scale ends of the path, and a deterministic
  function of the gated landscape.
- **s_q, per set.**
  1. Take the first upward passage of q by ρ₂ along the grid.
  2. Bisect at geometric midpoints until hi/lo − 1 ≤ 0.5%.
  3. s_q is the geometric mean of the final bracket.
  - The number of passages is reported.
- **Registered s_q:** the geometric mean of the two sets' s_q.

## Gate (committed before the pilot; `gate`, tested on constructed pass and fail cases)

All three conditions must hold:

- **A, attained.** At each set's final bracket ends:
  - the attainment check passes;
  - the Hessian of the profiled objective is positive definite (the projected smallest eigenvalue is > 0) once the
    exactly flat η-radial direction is removed.
- **B, agreement.** Both sets reach q inside the grid, and their s_q agree within 2%:
  |s_A − s_B| / min(s_A, s_B) ≤ 0.02.
- **C, independent search.** CMA-ES finds no loss lower than the retained one by more than 1e−7 at any bracket end.
  - It runs in v1's signed-ℓ₁ parametrisation, with the same decay, 16 starts and 300 generations.

**If the gate fails:** stop, report, and register nothing.

## Registration content (frozen and hashed only after a gate pass; the rules are fixed now)

**κ_q [AI-3]: the 1A derivation applied to ρ₂.**
- r = (s_c − s_q)/s_q = κ_q·χ, with κ_q = λ_min·[∇ρ₂·(PH)⁻¹θ*′]/[∇ρ₂·θ*′] and λ_min = λ_min(P^{1/2}HP^{1/2}).
  This reuses `lag_law.kappa`.
- **Coordinates:** θ = (W, c, b), the hidden coordinates plus the output bias, as in 1A. The output vector is held at
  v = s_q·ṽ*(s_q).
- **Stated approximation:** the output direction is treated as following its path, not lagging. Its motion enters
  only through θ*′.
- **H:** the Hessian of the weight-decayed loss, unprofiled b, at the path minimiser at s_q. Computed by autograd in
  float64 after refining the minimiser at s_q with BFGS to a gradient of 1e−11.
- **θ*′:** the path derivative, (θ*(s_q(1 + h)) − θ*(s_q(1 − h)))/(2h·s_q) with h = 1e−3. Each end is a warm-started
  re-minimisation from θ*(s_q), and b is the profiled optimum.
- **∇ρ₂:** the gradient in (W, c) by autograd, with b-component 0. It is checked against central and one-sided finite
  differences (step 1e−6), which must agree to 1e−3 relative, as in 1A.
- **P, per run and pre-crossing only:** Adam's P = diag(1/(√v̂ + 1e−8)) at the run's step t₀.
  - It is made permutation- and sign-invariant by using the within-block median: one value for all W entries, one for
    all c entries, and b's own value.
  - Reason: training units are not matched to landscape units.

**χ [AI-1]: pre-crossing information only.**
- χ = (ṡ/s_q)/(lr·λ_min), with ṡ = (s(t₀) − s(t₀ − 100))/100.
- t₀ is the last upward passage of 0.5·s_q in the recorded trajectory.
- A run with no such passage has no prediction and counts as invalid.
- Prediction: s_q·(1 + κ_q·χ). Baseline: s_q.

**Training [AI-5].**
- Seeds **2,710,000–2,710,039** (40 seeds). Integer search of src/, results/ and tests/ found no use of them.
- torch float64, `torch.manual_seed(seed)`, then the PyTorch default initialisation of Linear(2, 4) and Linear(4, 1).
- Full batch on the same 800 points. Adam with lr 0.01, betas (0.9, 0.999) and eps 1e−8.
- Loss: BCE-with-logits (mean) + (λ/2)(‖W‖² + ‖c‖²).
- **Stop rule, on s only:** stop at the first step with s = ‖v‖₁ ≥ 3·s_q, or at the budget of 40,000 steps.
- Every step records s, all parameters and Adam's v̂. **ρ₂ is not evaluated** until the predictions are committed and
  hashed, the same discipline as Track A.

**Crossing [AI-1].**
- The first **upward passage** of q by ρ₂(t): the first step t with ρ₂(t) ≥ q after some earlier step with ρ₂ < q.
- Reason: a random initialisation can already have ρ₂ ≥ q, and "reaches q" means from below.
- Evaluated at every step.
- Crossing scale: s at that step.
- A crossing at or before t₀ makes the prediction use post-crossing information. Such a run is reported and counted as
  not valid.

## Registered tests (fixed now; `score_rule`, tested)

- **T1:** the median |log(s_cross / (s_q(1 + κ_q·χ)))| over the valid crossings is at most 0.10.
- **T3 [AI-4]:** the prediction beats the baseline s_q. The paired run-level bootstrap 95% interval
  (10,000 resamples, seed 12345) of mean(|log err_pred| − |log err_base|) lies entirely below 0.
- **Validity:** at least 30 valid crossings of 40.
- **PASS** means valid, T1 and T3.
- χ at crossing is reported beside the width-1 range.

## Amendment 1 (02:41 EDT, before the freeze, before any registered training, and with no ρ₂ from any training)

**Change:** the ṡ window is **min(100, t₀)** steps, not 100. So ṡ = (s(t₀) − s(t₀ − w))/w with w = min(100, t₀), and a
run is invalid only if t₀ is undefined.

**Reason:**
- The gated pilot's grid path puts s_q between 2.98 and 3.73, so 0.5·s_q is about 1.6.
- Adam grows s = ‖v‖₁ by about 0.03 per step from about 1 at the default initialisation.
- The last upward passage of 0.5·s_q therefore falls within the first ~30 steps, and the fixed 100-step window would
  have made every run invalid.

**Disclosure:** the growth rate was seen in a machinery test of `train_run`. It used seed 2,000,002, outside the
registered range, with s_q set to 3.0 by hand. That test recorded s only; no ρ₂ and no crossing were computed.

## Gate verdict (02:46 EDT): **PASS** (`simplicity_bias_v2/pilot_summary.json`, `pilot_scan.csv`)

- **λ:** 1e−4 (`lambda.log`, `lambda_rule.json`). This is the smallest value on the grid, and it passed the attainment
  check at s = 1, 4 and 16, with 184, 13 and 199 hits.
- **Path:** 54 computed scales, all passing the attainment check.
  - ‖(W, c)‖ ≤ 16.5, against B = 117.7.
  - Every ladder passes and every audit passes.
  - Sets A and B agree in retained loss to about 1e−15 at every grid scale.
- **q:** ½(3.2e−9 + 0.7828) = **0.39141**.
- **s_q:** both sets give the same bracket, [3.58512, 3.597642], so **s_q = 3.59138** and the agreement gap is 0.0%.
  - The projected Hessian's smallest eigenvalue is 1.0e−4 at both ends. This equals λ, the direction of an idle hidden
    unit, which is held only by the decay.
  - CMA-ES found 0.149896 and 0.149875 at the lower end, against a retained 0.149834, and 0.149183 and 0.149224 at the
    upper end, against a retained 0.149104. It finds nothing lower.

**Caveat, recorded before the freeze and before any training.**
- **ρ₂ jumps at s_q.** It goes from 0.182 to 0.448 across a 0.35% bracket, and G₊ goes from −0.153 to +0.071.
  - The retained minimiser switches between two branches there. This is a first-order switch, like the paper's
    width-1 switch.
  - Along the path, ρ₂ is 0 up to s = 1.22, rises continuously on the lower branch from 0.085 to 0.182, jumps to 0.448,
    and then rises continuously to 0.783.
- **Consequence for κ_q.** The κ_q formula assumes ρ₂ reaches q continuously along one branch. Here q is crossed by the
  jump, so κ_q, computed as registered on the branch that is lower in loss at s_q, is a formal application.
- The registered test is run as designed. This caveat goes into the writer inputs whatever the outcome.
