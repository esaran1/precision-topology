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

## 3. Verification, and a registered expectation that failed

Three things happened, in this order. The order matters, so it is
stated explicitly rather than left to be reconstructed.

### 3.1 The registered expectation failed

The brief registered, before the derivation: *"You measured solvers'
minimal |w₂|·D(a) ≈ 1.2, constant across a. The derived c should equal
that."* It does not, and the derivation was **not** adjusted to fit.

### 3.2 The theorem was verified against independent prior data

66 solving networks were re-derived deterministically from
`fold1d_sweep.csv` / `fold1d_refine.csv` — seeds recorded in early
2026-08, weeks before this theorem was written (2026-08-25), for the
entirely different purpose of locating the findability onset. Their
(w₁, b₁, w₂, b₂, margin) were extracted and checked against the bound:

- **Violations: 0 of 66.** Every solver satisfies |w₂| ≥ 2m/G\*(a).
- **Tightness: median slack |w₂|/bound = 1.27×, minimum 1.04×.** The
  closest solver sits 4% above the bound.

This is the real verification: a bound derived from geometry, tested on
weights nobody chose with the bound in mind, holding without exception
and without slack.

### 3.3 The failure was diagnosed, not explained away

Why the registered c ≈ 1.2 did not appear:

- The measured |w₂|·D(a) at solvers is **not constant across a**:
  2.30, 2.32, 4.02, 6.54 (minimum over solvers at a = 1.45, 1.5, 2.0,
  3.0). The "1.2, constant" reading came from a narrower slice.
- The reason it drifts: **c = 2m/κ depends on the margin m the
  optimizer happens to reach**, and m grows with a (median margin 0.07,
  0.11, 0.68, 1.54 across those values) because a deeper dip permits a
  larger margin at no extra cost in |w₂|. The theorem's constant is
  fixed only once a margin is fixed.
- Fixing the margin restores constancy: at the smallest-margin quintile
  of solvers, |w₂|·D(a) is 2.30–2.64 at a = 1.45–1.5, and the theorem's
  own prediction there (2m/κ at the median margin) is 0.93–1.29,
  bracketing the earlier figure once κ ≈ 0.31 is included.

**The failure is a better outcome than a match would have been.** The
quantity the registration expected the theorem to reproduce was a
margin-dependent statistic reported as a constant. Had the derived c
come out at 1.2, it would have agreed with an artifact of how that
statistic was summarized, and the drift (2.30 → 6.54) would have gone
unnoticed. Instead the mismatch exposed the artifact, and the theorem
now accounts for both the old number's approximate value at small a and
its growth at large a. Recorded as a corrected reading of an old
number, not as a confirmation of the theory.

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
but with margin shrinking like |w₂|·D(a).**

## 5. The seam: where a bound on solutions stops and findability begins

This is the precise boundary of what the theorem covers, and it is
where a reviewer should look first.

**The infimum of |w₂| over solutions is 0, not positive.** The bound
|w₂| ≥ 2m/(κ·D(a)) is vacuous as m → 0⁺, and that is not a weakness of
the derivation — it is the true state of affairs, exhibited: at
a = 1.02 a network with |w₂| = 1 solves the task exactly
(`box_counterexample.csv`), with margin m = |w₂|·G/2 = 2.74e−4. Nothing
forbids solutions at ordinary norm arbitrarily close to the threshold.
**So no lower bound of this form can explain unfindability.** A theorem
about what solutions must look like cannot, even in principle, rule out
the solutions that exist; it can only say what they must cost in
margin.

**Vanishing-margin solutions are exactly the zero-basin ones.** At
a ≤ 1.25 the exact basin fraction is 0 (`grid_measure_audit.md`) while
solutions exist with |w₂| = 1 — measure ~1e−6, margin ~1e−4, no
attracting volume. This is the empirical content the theorem is silent
about.

**And margin alone does not draw the line.** The obvious conjecture —
"SGD finds solutions above some margin threshold" — is false as stated
in our data. At a = 1.5 the smallest margin SGD actually reached among
found solvers is **0.0168**, *below* the |w₂| = 1 construction's margin
at the same a (**0.0384**). The found solution has the smaller margin
and a basin; the constructed one has the larger margin and none. What
differs is not margin but where the solution sits: found solvers carry
|w₂| ≈ 4–5 (median 5.25 at a = 1.5), the construction |w₂| = 1.

**Statement of the gap.** The theorem gives a *necessary condition on
the parameters of any solution with a given margin*. Findability
requires a *sufficient condition for SGD to converge to one*, which
depends on basin geometry — a property of the loss landscape around a
solution, not of the solution's coordinates. Our evidence for the
second is entirely measured: solve rates (T30), basin volumes
(exact grids), and the scale-manipulation interventions (T27, T29).
**Connecting small margin and low |w₂| to zero basin volume is exactly
the missing theory**, and this project does not supply it. What it
supplies is the two endpoints: a proven necessary condition on
solutions, and measured basin geometry — with the seam between them
named rather than papered over.

## 6. What the theorem does not cover

- **It bounds what a solution must look like, not what SGD finds.** The
  findability claim needs, additionally, that training does not reach
  the required scale *with a usable margin* — which is measured
  (T27, T29, T30), never proven. The theorem is silent on optimization.
- **It says nothing about basins** — see §5, which states that seam in
  full rather than as an aside.
- **Width > 1 breaks the argument.** With several units the network is
  Σᵢ w₂ᵢ f(w₁ᵢx + b₁ᵢ) + b₂ and the two sign changes can be produced by
  *different* units, so no single |w₂ᵢ| need be large: the argument
  bounds only the aggregate Σᵢ|w₂ᵢ|·Gᵢ ≥ 2m, which is far weaker and
  admits cancellation between units. Nothing here extends to the
  width-3 link setting, where the corresponding statement would need
  the topological obstruction rather than a sign-change count.
- **It is specific to a single fold-carrying unit and a 1-D task.** The
  extension to width d in R^d is open and is not a corollary.
