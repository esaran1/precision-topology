# Theorem 1: hypotheses for the sine family, proved or checked (Track D.1)

**Producer:** `src/theorem1_checks.py` → `results/theorem1_checks.json`. It runs in one process at nice 15, takes
25 s and needs well under 1 GB. **Tests:** `tests/test_theorem1_checks.py` exercises each check function on
constructed pass and fail cases, and checks the committed results.

## Which statement

- **No `.tex` source of the paper is in the repository**, so "Theorem 1" was identified from the math note and
  the writer inputs (WP-9, WP-12). **The coordinator should confirm this mapping.**
- **The statement.** "Theorem 1" is taken to be the small-scale criterion of math note §10.1. Suppose the data gap
  Γ_n > 0 is attained. Then output scale gates placement (the conditional minimiser is unplaced below some scale
  and placed above it) iff the small-scale conditional minimiser is unplaced.
- **What it rests on:**
  - Lemma 1: L* = log 2 − (s/4)Δμ + (s²/8)Var + R, with |R| ≤ K s⁴ m₄;
  - Corollaries 1–2 (s → 0): Corollary 1 assumes the minimisers stay in a compact set, and its part (ii) assumes
    quadratic growth of Δμ away from its maximiser set M;
  - Lemma 2 (s → ∞): Γ_n attained, and compactness again;
  - §11 Steps 1–2: Δμ = a·sin b₁·D(w₁), and G, G_n > 0 ⇒ |w₁| < a/1.4.
- **The four hypotheses** checked below are the author's: attainment, compactness, uniform remainder, and quadratic
  growth near α*.
- **Setting:** width 1, f_a(t) = t + a sin t. The data are the 800-point population, and the continuous windows are
  I = [−0.8, 0.8] and O = ±[1.2, 2].
- **a values:** 1.30, 1.35, 1.40, 1.45, 1.50, 1.60, plus 1.02, 1.05, 1.10 for attainment.
  - Theorem 1 is asymptotic in the scale s, not in ε. The small-ε values are included only where they are cheap.

## Summary table

| hypothesis | status | where | result |
|---|---|---|---|
| **A1. Attainment of the data gap Γ_n = sup G_n > 0** (Lemma 2) | **PROVED** for every a with Ĝ(a) > 0 | §11 Step 2 (G_n > 0 ⇒ \|w₁\| < a/1.4); continuity and 2π-periodicity in b₁ (§12.2 Lemma 0, same argument for G_n); Γ_n ≥ G ≥ Ĝ_cert > 0 (the data lie in the windows) | **Checked** at all 9 a: the maximiser found has \|w₁\| ≤ a/1.4 − 0.098 (strictly interior), and Γ_n found ≥ certified Ĝ_lo at every a (for example 0.086103 ≥ 0.086102 at a = 1.30) |
| **A2. Attainment of sup Δμ (the class-mean maximiser α\*)**, continuous windows | **CHECKED** (computer, float Lipschitz bounds); the decay bound is analytic | \|D_c(α)\| ≤ 3/(0.8α) (elementary), so only α ≤ 2.384 matters; scan with \|D_c′\| ≤ 2.0 | α\*_c = 1.7922917 is the **unique global maximiser** on (0, ∞): \|D_c\*\| = 1.573242, separation certified outside α\* ± 0.05 |
| **A2, data points (the 800-point population)** | **PROVED** (exact arithmetic): the sup is attained, **but not at α\*** | Every x is an integer multiple of q = 0.4/79401, with inner integers even and outer integers odd. So D(α) is 2π/q-periodic, and at α = π/q = 623,614 every inner cos is +1 and every outer cos is −1 | **\|D(π/q)\| = 2 > \|D\*\| = 1.571067.** α\* = 1.7913244 is the global maximiser only on **(0, 622,047)** (adaptive Lipschitz scan; the first α with \|D\| ≥ \|D\*\| is 622,046.94) |
| **C. Compactness of the conditional minimisers**, s → ∞ | **PROVED** | §11 Proposition (ii) + Step 2: for s > s₁(a) every minimiser has G_n > 0, so \|w₁\| < a/1.4 (compact mod 2π) | s₁(data) = 163.8 (a = 1.30) … 62.9 (a = 1.60) |
| **C, s → 0** | **NOT PROVED** as a fixed compact set; proved at each s > 0 | Localisation lemma §5.0: every minimiser has \|w₁\| ≤ W(s), but W(s)·s → 4.51 (a = 1.30), so W(s) → ∞ | W(0.1) = 47.3, W(0.01) = 453.5, W(1e−3) = 4,516. W(s) < π/q for s > 7.24e−6 (a = 1.30). Below that scale the alias is not excluded |
| **R. Uniform remainder**, \|R(θ; s)\| ≤ K s⁴ m₄(θ) | **Proof sketch in the note; K not explicit** | §10 Lemma 1 step 5; §10.1 | **Checked**: max \|R\|/(s⁴m₄) = 0.0052084 (= 1/192 to 1e−5) over about 1,570 (θ, s) pairs per a, with \|w₁\| ≤ 10, s = 0.3, 0.1, 0.03, 0.01, at a = 1.30, 1.50, 1.60. It is the same at every s and a, so uniform; empirically K = 1/192 |
| **Q. Quadratic growth of Δμ near α\*** | **CHECKED** (computer, float Lipschitz bound on D″) together with an elementary bound in b₁ | Δμ = a·sin b₁·D(w₁) (§11 Step 1, proved). D″ keeps one sign on α\* ± 0.05, with \|D″\| ≥ 2.454 (data) and 2.455 (continuous). 1 − sin b ≥ (2/π²)(b − π/2)² (elementary) | \|D\*\| − \|D(α)\| ≥ 1.227(α − α\*)² on α\* ± 0.05. Away from α\*: separated (continuous: on all α > 0; data: on (0, 622,047)). Near the full data maximiser set (the aliases) it was not examined |

## What the data-point finding means

- **What was assumed.** The paper, the math note §10–§11 and WP-9/WP-12 treat α* = 1.7913244 as the maximiser of
  |D| on the data points. The §11 search covered α ∈ (0, 50) (`scale_limits.alpha_star`).
- **What is true.** On the 800-point population the supremum of Δμ over all θ is 2a. It is attained at the alias
  |w₁| = π/q ≈ 623,614 (and its periodic copies), not at α*. On the continuous windows α* (1.7922917) is the
  global maximiser.
- **Where each one wins.** By the second-order expansion, checked directly at 40 digits at 0.1× and 10× the
  crossover:
  - the alias beats θ* only for s below 2.0e−12 (a = 1.30) to 2.5e−12 (a = 1.60);
  - the first competitor (α ≈ 622,047) only below 2.2e−15 to 2.8e−15.
  - For s > 7.24e−6 the localisation lemma excludes every |w₁| ≥ π/q rigorously.
- **Width 1: the conclusion is unchanged.** Both α* and the alias have |w₁| ≥ a/1.4, so both are unplaced (§11
  Step 2). The width-1 criterion verdict (unplaced at small scale, hence gating) holds whichever maximiser the
  limit selects. What changes is only which unplaced solution is the literal s → 0 limit on the finite
  population.
- **Width 2: not checked here; the coordinator should check it.**
  - The same aliasing applies to the width-2 selection (§10.1, WP-9). A cancelling pair of alias units gives
    φ = const + a·cos(πx/q) up to sign. That is +a on every inner point and −a on every outer point, so the data
    gap G_n = 2a (placed on the data), while the window gap G is strongly negative.
  - The registered width-2 verdict was computed with α* and does not consider this. It concerns scales many
    orders of magnitude above the crossover. Its literal s → 0 statement on the data population would need
    restating ("on the continuous windows", or "for s above ~1e−11").

## Say / Do not say

**Say:**
- "The attainment hypothesis of Lemma 2 holds for the sine family: a positive data gap forces |w₁| < a/1.4, so Γ_n is a
  supremum over a compact set. It is attained, and it is at least the certified Ĝ(a) > 0."
- "Compactness holds as s → ∞ (proved). As s → 0 it is not proved as a fixed compact set. At each scale the
  localisation lemma confines the minimisers to |w₁| ≤ W(s) ≈ 4.5/s."
- "The remainder bound of Lemma 1 is given with a proof sketch. Numerically |R| ≤ s⁴m₄/192 over every sampled
  parameter and scale."
- "Near α*, the class-mean gap has quadratic growth (checked with a Lipschitz bound on D″: |D*| − |D| ≥ 1.23(α − α*)²).
  On the continuous windows, α* is the unique global maximiser."
- "On the finite 800-point population, the class-mean gap is maximised by an aliased weight, |w₁| = π/q ≈ 6.2e5,
  not by α*. This matters only at scales below about 1e−11. It is unplaced too, so the width-1 conclusion is
  unchanged."

**Do not say:**
- "All hypotheses of Theorem 1 are proved": compactness as s → 0 is not proved, and quadratic growth and the
  continuous-window maximiser are computer-checked in floating point, not proved.
- "α* is the global class-mean maximiser on the data": it is only on (0, 622,047), and the data maximiser is the
  alias.
- "The small-scale limit on the data population is the cosine at α*" without the qualifier "on the continuous
  windows" or "for s above ~1e−11".
- "The remainder constant K is 1/192": that is empirical, from sampled parameters.
- Anything about width 2 and aliasing, until it is checked.

## Method notes

- **"Computer-checked"** means a Lipschitz grid argument. A cell is cleared if f(centre) + L·h/2 < level. The
  argument is evaluated in float64 (the lattice closed form agrees with the direct sum to 3e−11), not in interval
  arithmetic, so it is rigorous up to floating-point rounding only.
- **The remainder check** excludes (θ, s) with s⁴m₄ ≤ 1e−9, where float64 cannot resolve R.
- **The crossover L\* comparisons** use mpmath at 40 digits.
- **The Γ_n search** uses a grid plus the rescaled small-ε window, with multi-start pattern refinement. It is not a
  certificate. Attainment itself is proved, and Γ_n ≥ Ĝ_cert holds by containment.
