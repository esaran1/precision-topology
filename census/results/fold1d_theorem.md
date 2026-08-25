# Part 5: a lower bound on the amplification a solving network needs

The one piece of theory in an otherwise empirical project. Verification
code: `src/fold1d_theorem.py`. The bound was derived first and checked
afterwards against solver weights recorded weeks earlier for a different
purpose; that check is in §3, and the derivation's first consequence was
to overturn a committed empirical claim (`box_emptiness_correction.md`).

## 1. Setting and statement

Task: realize `sign(|x| − 1)` on the sampled regions
`I = [−0.8, 0.8]` (class 0) and `O = ±[1.2, 2.0]` (class 1) with

  N(x) = w₂·f(w₁x + b₁) + b₂,  f = f_a(t) = t + a·sin t.

Write, for a fixed (w₁, b₁),

  G(w₁, b₁) = min_{x∈O} f(w₁x + b₁) − max_{x∈I} f(w₁x + b₁)

for the orientation w₂ > 0 (mirror the roles for w₂ < 0), and let a
solution have **logit margin** m > 0, meaning N ≤ −m on I and N ≥ +m
on O.

**Theorem.** Any solving network with margin m satisfies

  |w₂| ≥ 2m / G(w₁, b₁) ≥ 2m / G\*(a),  where G\*(a) = max_{w₁,b₁} G.

Moreover G\*(a) = κ(a)·D(a) with D(a) = f(t_max) − f(t_min) the dip
depth of f_a and **κ(a) ∈ [0.305, 0.328]** for a ∈ [1.02, 3.0] —
so with c := 2m/κ,

  **|w₂| ≥ c / D(a).**

*Proof.* Pick x⁺ ∈ O attaining the min and x⁻ ∈ I attaining the max of
f(w₁·+b₁). Then N(x⁺) − N(x⁻) = w₂·G ≥ m − (−m) = 2m, and since
G > 0 is required for any separation at this (w₁,b₁), dividing gives
|w₂| ≥ 2m/G. Maximizing the denominator over (w₁,b₁) gives the
uniform form. ∎

Three notes the brief asked for explicitly:

- **b₂ is fully handled and cannot relax the bound.** It cancels in the
  difference N(x⁺) − N(x⁻); the theorem is therefore about the
  *optimal* b₂ automatically, and no choice of b₂ evades it. The
  optimal b₂ is the interval midpoint, b₂ = −w₂(min_O f + max_I f)/2.
- **w₁ cannot substitute for w₂.** w₁ enters only through which part of
  f is traversed, i.e. only through G, and G is bounded above by
  G\*(a) = κ·D(a) no matter how w₁ is chosen. Amplification and
  traversal are not interchangeable: the genuinely constrained
  combination is **|w₂|·G(w₁,b₁) ≥ 2m**, and the activation-only
  form follows by bounding G.
- **The constant is task geometry, not activation.** κ measures what
  fraction of the dip the windows I and O can straddle, fixed by the
  0.8/1.2/2.0 window edges. Its mild drift (0.305 → 0.328) reflects the
  dip changing shape as a grows, not the task changing.

## 2. Why D(a) → 0 is the whole story near threshold

D(a) = 2(√(a²−1) − arccos(1/a)) → 0 as a → 1⁺, like (8/3)·(a−1)^{3/2}.
So the required amplification diverges as

  |w₂| ≳ c / D(a) ~ (3c/8)·(a−1)^{−3/2},

which is the shape the empirical `A_req` measurements followed in the
link setting (T27). The bound is what makes "possible for every a > 1"
compatible with "requires unbounded scale as a → 1⁺".

## 3. Verification against a measurement taken before the theorem existed

66 solving networks were re-derived deterministically from
`fold1d_sweep.csv` / `fold1d_refine.csv` (seeds recorded in 2026-08;
theorem written 2026-08-25) and their (w₁, b₁, w₂, b₂, margin)
extracted.

- **Violations: 0 of 66.** Every solver satisfies |w₂| ≥ 2m/G\*(a).
- **Tightness: median slack |w₂|/bound = 1.27×, minimum 1.04×.** The
  bound is not loose; the closest solver sits 4% above it.

**On the pre-registered c ≈ 1.2.** The brief anticipated that the
derived c would equal the measured "minimal |w₂|·D(a) ≈ 1.2, constant
across a". It does not, and per instruction the derivation is not
adjusted to fit. Diagnosis:

- The measured |w₂|·D(a) at solvers is **not** constant: 2.30, 2.32,
  3.80, 6.54 at a = 1.45, 1.5, 2.0, 3.0 (minimum over solvers).
- The reason is that **c = 2m/κ depends on the margin m the optimizer
  happens to reach**, and m grows with a (median margin 0.07, 0.11,
  0.68, 1.54 across those values) because larger dips permit larger
  margins at no cost. The theorem's constant is fixed only once a
  margin is fixed; the earlier "1.2" summarized solvers at small a,
  where margins are small, and was read as a universal constant.
- Fixing the margin restores constancy: at the smallest-margin quintile
  of solvers, |w₂|·D(a) is 2.30–2.64 at a = 1.45–1.5 — and the
  theorem's own prediction there, 2m/κ, is 0.93–1.29 at the median
  margin, bracketing the earlier figure once the factor κ ≈ 0.31 is
  included.

So the earlier measurement was a **margin-dependent statistic reported
as a constant**, and the theorem explains both its approximate value at
small a and its drift at large a. This is recorded as a corrected
reading of an old number, not a confirmation.

## 4. Tightness: is the infimum attained?

**No, and the gap is characterized.** As m → 0⁺ the bound gives
|w₂| ≥ 2m/G\* → 0, and indeed solutions exist with |w₂| = 1 at a = 1.02
(`box_counterexample.py`) — the infimum over solutions of |w₂| is not
positive, because the margin requirement can be made arbitrarily small.
The correct statement is that **|w₂|·G(w₁,b₁) ≥ 2m is attained exactly**
by the centred-b₂ construction: choosing b₂ at the interval midpoint
makes both inequalities equalities simultaneously, giving margin
m = |w₂|·G/2 with no slack. Constructed exhibits in
`box_counterexample.csv` achieve this: e.g. a = 1.02, w₂ = 1,
G = 5.48e−4, margin 2.74e−4 = |w₂|·G/2 exactly.

The practical form of the bound is therefore: **to solve with margin m
you need |w₂| ≥ 2m/(κ·D(a)); to solve at all you need only |w₂| > 0,
but with margin shrinking like |w₂|·D(a).** Vanishing-margin solutions
are exactly the ones with vanishing basins (§5).

## 5. What the theorem does not cover

- **It bounds what a solution must look like, not what SGD finds.** The
  findability claim needs, additionally, that training does not reach
  the required scale *with a usable margin* — which is measured
  (T27, T29, T30), never proven. The theorem is silent on optimization.
- **It says nothing about basins.** The sharpest empirical finding —
  solutions at ordinary norm (|w₂| = 1, a = 1.02) with *zero* basin —
  is consistent with the theorem and unexplained by it. The theorem
  says such solutions must have tiny margin (m = |w₂|·G/2 ≈ 2.7e−4);
  connecting small margin to zero basin volume is exactly the gap
  between this theorem and a theory of findability.
- **Width > 1 breaks the argument.** With several units the network is
  Σᵢ w₂ᵢ f(w₁ᵢx + b₁ᵢ) + b₂ and the two sign changes can be produced by
  *different* units, so no single |w₂ᵢ| need be large: the argument
  bounds only the aggregate Σᵢ|w₂ᵢ|·Gᵢ ≥ 2m, which is far weaker and
  admits cancellation between units. Nothing here extends to the
  width-3 link setting, where the corresponding statement would need
  the topological obstruction rather than a sign-change count.
- **It is specific to a single fold-carrying unit and a 1-D task.** The
  extension to width d in R^d is open and is not a corollary.
