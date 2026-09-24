**Superseded by `width2_design.md` (Route A, full design for review). Kept as the record of the first draft.**

# Block 6 design (Route A): does conditional minimisation at fixed output scale predict width 2?

**Design for review. Nothing in this block is run until this is approved.**
Date: 2026-09-23. Producers named below exist only as planned code until then.

## Question

The paper's explanation is that conditional loss minimisation at fixed output scale predicts where
correct configurations become preferred, and that training becomes correct where that switch occurs,
up to a lag.
- At width 2, two monotone units can form a bump, so Proposition 1's monotone obstruction no longer
  applies.
- The test is therefore not "does width 2 solve", but whether **the same conditional-minimisation
  account locates the width-2 onset**.
- tanh is included because it is monotone yet solves at width 2. It separates "the account is about
  output scale in general" from "the account is about non-monotone activations".

## Model and task

- Same 1D task: I = [−0.8, 0.8] (class 0) and O = ±[1.2, 2.0] (class 1).
- Width-2 network: N(x) = v₁u(α₁x + β₁) + v₂u(α₂x + β₂) + b.
- u ∈ {f_a with a = 1.30, f_a with a = 1.50, tanh}.
- Logistic loss, labels {0, 1}, mean reduction, as at width 1.

## The geometric quantity Γ*₂ and the margin bound

**Definition.** For hidden parameters θ = (α₁, β₁, α₂, β₂) and an output direction ṽ with ‖ṽ‖₁ = 1,
let φ_{ṽ,θ}(x) = ṽ₁u(α₁x + β₁) + ṽ₂u(α₂x + β₂). With G(φ) = max(min_O φ − max_I φ, min_I φ − max_O φ),
define

  Γ*₂ = sup over ‖ṽ‖₁ = 1 and all θ of G(φ_{ṽ,θ}).

**Bound, derived.**
- Let N be correct with margin m: N ≤ −m on I and N ≥ m on O (or the mirror).
- Write φ = N − b = ‖v‖₁ φ_{ṽ,θ}, with ṽ = v/‖v‖₁.
- Then min_O φ − max_I φ ≥ 2m.
- Each of G's two terms is positively homogeneous of degree 1 in φ, so G(φ) = ‖v‖₁ G(φ_{ṽ,θ}) ≤ ‖v‖₁Γ*₂.
- Hence **m ≤ ½‖v‖₁Γ*₂**. Define **R₂ = ‖v‖₁Γ̂₂/2**, where Γ̂₂ is the value computed below.

**Relation to width 1, derived.**
- G is concave in φ: each orientation is a minimum of linear functionals minus a maximum of linear
  functionals, and the maximum of two concave functions is not concave. So state it per orientation:
  G₊(φ) = min_O φ − max_I φ is concave.
- For ṽ ≥ 0 with ṽ₁ + ṽ₂ = 1, concavity gives G₊(ṽ₁φ₁ + ṽ₂φ₂) ≥ ṽ₁G₊(φ₁) + ṽ₂G₊(φ₂).
- Taking both units equal to the width-1 optimiser gives **Γ*₂ ≥ G* (the width-1 Ĝ)**.
- Γ*₂ may exceed G*.
- **Per-unit width-1 gaps must not be summed.** For tanh, every single unit has G ≤ 0: a monotone
  function cannot exceed its inner values on both sides. Yet two tanh units with opposite-signed
  output weights form a bump with G > 0. Concavity bounds the combination from below, not above.

**Numerical verification** (`src/width2_bound_check.py`, run first):
1. **Homogeneity**: G(cφ) = cG(φ) for c > 0, on 1,000 random (ṽ, θ), to 1e-12.
2. **Concavity** of G₊ on 1,000 random pairs.
3. **The bound**: m ≤ ½‖v‖₁Γ̂₂ on every correct network found by unconstrained training in the
   calibration runs (item 5). Report the slack distribution.
4. **Counterexample to summing**: exhibit tanh units with per-unit G ≤ 0 whose combination has G > 0.

**Computing Γ̂₂.** The sup is 5-dimensional: θ ∈ ℝ⁴ plus one parameter on the ℓ₁ circle, over the four
sign patterns of ṽ.
- **Evaluation**: G is evaluated on dense grids of the windows. For f_a the extrema of a two-unit sum
  are not in closed form. A Lipschitz correction bounds the grid error:
  |φ′| ≤ |ṽ₁||α₁| max|u′| + |ṽ₂||α₂| max|u′|.
- **Search**: multi-start (2,000 starts, Nelder–Mead then L-BFGS-B), with a bounded compact domain
  justified per activation by a growth argument like W(s, a).
- **Upper bound**: a coarse 5D Lipschitz branch-and-bound over that domain, reporting the enclosure
  [Γ̂₂, Γ̂₂ + slack]. If the 5D certificate is too expensive to close, Γ̂₂ is reported as a certified
  lower bound (an attained value) with an uncertified gap to the multi-start optimum, and labelled so.

## Conditional minimisation at fixed ‖v‖₁

- **Problem**: at fixed s = ‖v‖₁, minimise the logistic loss over θ ∈ ℝ⁴ and the direction ṽ on the
  ℓ₁ circle. The bias b is profiled exactly (the same safeguarded Newton as Block 1b). That is
  5 dimensions.
- **Direction parameterisation**: ṽ = (t, σ(1 − |t|)), t ∈ [−1, 1], σ ∈ {±1}. Both branches are
  searched.
- **Threshold**: R₂^glob = the smallest s at which the global conditional minimiser is correct in
  placement, G(φ_{ṽ,θ}) > 0, expressed as R₂ = sΓ̂₂/2. Similarly R₂^solve (sign-correct everywhere).
- **Search**: 1,000 restarts per s (random θ in a box, t uniform, both σ), Adam then L-BFGS to a
  gradient norm ≤ 1e-8. Scales on a grid in s, refined by bisection to one grid step. The grid step
  is set so that R has the same 0.05·Ĝ/2 resolution as at width 1.

**Validation, since 5D can't be certified exhaustively** (reported as validation, not certification):
1. **Restart convergence**: the minimum at each s is unchanged (to 1e-8) when restarts go
   250 → 500 → 1,000 → 2,000, near each threshold.
2. **Independent optimiser**: a second search (CMA-ES, a different box) reproduces the retained
   minimum near each threshold.
3. **Retain everything**, as in 1a: every restart's final point, loss, G and gradient norm, and a
   check that no discarded candidate beats the retained one.
4. **Local certificate**: at the retained minimiser near each threshold, a certified Hessian
   eigenvalue bound (interval arithmetic on the profiled 5D Hessian over a small box). This
   establishes a strict local minimum, not global optimality.
5. **Subproblem certificate**: with the direction ṽ and one unit fixed at the retained values, the
   remaining 2D problem in (αᵢ, βᵢ) is certified by the Block 1c method, which checks the retained
   point is globally optimal in that slice. Stated as a slice certificate only.

## Unconstrained training (after registration only)

- **Protocol**: width-2 networks, Adam lr 1e-2, 200 points per class (`fold1d.make_data`), float64,
  PyTorch default initialisation, 12,000 steps.
- **Crossing**: the first step at which the output function's dense-grid G > 0 (placement), checked
  every step, as in phase 2b.
- **Sample**: 80 seeds per activation. Registered minimum: 60 crossings; add 30 seeds at a time
  otherwise.
- **Calibration runs** (bound check only): a separate 40 seeds per activation, used for item 3 of
  the numerical verification and never for the predictions.

## Predictions, registered before any unconstrained width-2 training

For each activation, u ∈ {f_{1.30}, f_{1.50}, tanh}:
- **W1 (location)**: the median crossing R₂ lies in [R₂^glob, 1.25·R₂^glob].
  - The lower edge says training does not become correct before the conditional switch.
  - The upper edge is the largest lag seen at width 1 (log-ratio ≤ 0.18 across 18 settings,
    ×1.20), rounded up to 1.25.
  - **No width-2 calibration is used.** The width-2 lag is not known in advance, and λ is not
    transferred from width 1.
- **W2 (ordering)**: the median crossing R₂ is ordered across the three activations as R₂^glob is.
- **W3 (tanh, the discriminating case)**: tanh obeys W1.
  - If W1 holds for f_a but fails for tanh, the account is specific to non-monotone activations, and
    the paper says so.
  - If it holds for all three, the account is about output scale in general.
- **Falsifiers**:
  - W1 fails if the median crossing R₂ lies outside the band for that activation.
  - W2 fails if the orderings disagree.
  - The paper reports each failure as such.

## Stop and report if

- any validation check (1–3) finds a lower-loss point than the retained minimiser near a threshold;
- Γ̂₂'s multi-start optimum and the branch-and-bound enclosure disagree beyond the enclosure;
- the numerical bound check (item 3) finds a correct network violating m ≤ ½‖v‖₁Γ̂₂.

## Cost and timing (estimate)

- **Γ̂₂**: minutes to hours per activation, depending on how far the 5D branch-and-bound closes.
- **Conditional scans**: about 1,000 restarts × about 40 scales × 3 activations, around 2–4 hours at
  6 workers.
- **Training**: 360 runs, about 30 minutes.

Not for the 25 September submission. **This is a rebuttal-revision block**, run after approval, under
its own tag.
