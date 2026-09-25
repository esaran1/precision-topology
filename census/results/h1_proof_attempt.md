# H1 (localisation of the gap maximiser): a proof

Status: **PROVED** for 0 < ε ≤ ε₀ = 0.029, both in the "fixed compact" form and in the exact form stated in
`scaling_proposition.md` (limit coordinates in C = {1 ≤ |u| ≤ 2.2, |v| ≤ 2}). The proof is analytic. It uses one
exact rational evaluation, G₀(1.6, 1.2) = 27088/46875, and no numerical certificate. §8 lists what is *not* covered.
The numerical sanity checks (`src/h1_checks.py`) are in §9, kept apart from the proof.

The route differs from the one suggested. The key tool is an exact trigonometric identity for f_a on a pair of
points, the finite-ε analogue of the cubic identity in `math_note_v2.md` §8. With it the regime w₁ = O(1)
(many periods) needs no separate argument: |sin y|/y ≤ 1/π < 1/a for y ≥ π disposes of it in one line (Lemma 2b).

---

## 0. The statement being proved

`scaling_proposition.md`, Hypotheses:

> **H1 (localisation).** For ε small, Ĝ(a) is attained at a placement whose limit coordinates lie in C, up to the
> symmetries t ↦ t + 2π (f_a(t + 2π) = f_a(t) + 2π) and σ ↦ −σ, x ↦ −x.
> C = {1 ≤ |u| ≤ 2.2, |v| ≤ 2}. Limit coordinates: w₁x + b₁ = π + √ε σ(x), σ(x) = ux + v.

Here (u, v) = (p, q) in the notation of `math_note_v2.md` §1: w₁ = √ε p, b₁ = π + √ε q. H1 is used in step 4 of
that note ("Taking sups over C (H1 for the left side)"). In `math_note_v2.md` it enters only through
K(ε) := Ĝ(a)/ε^{3/2} → K (§1 "Units", §8 bookkeeping).

**Setting.**
- f_a(t) = t + a sin t, a = 1 + ε with ε > 0. The windows are I = [−0.8, 0.8] and O = [−2, −1.2] ∪ [1.2, 2].
- φ(x) = f_a(w₁x + b₁).
- G(w₁, b₁) = max(G₁, G₂), with G₁ = min_O φ − max_I φ and G₂ = min_I φ − max_O φ.
- Ĝ(a) = sup G.
- Rescaled gap: G̃_ε(p, q) = G(√ε p, π + √ε q)/ε^{3/2}. This is the gap of φ_ε∘σ, with
  φ_ε(σ) = [f_a(π + √εσ) − π]/ε^{3/2}.
- G₀ is the same functional with h(σ) = −σ + σ³/6 in place of φ_ε, and K = sup G₀.

## 1. Theorem

**Theorem.** Let 0 < ε ≤ 0.029. Then:
1. Ĝ(a) > 0, and Ĝ(a) is attained.
2. Every maximiser (w₁, b₁) of G has w₁ ≠ 0. Using (w₁, b₁) ↦ (−w₁, −b₁) and b₁ ↦ b₁ + 2πk, it has a representative
   whose rescaled coordinates satisfy

   **1.0155 ≤ K(ε)/0.4 ≤ p < √6/1.4 = 1.74964,   |q| < 1.7956.**

In particular (p, q) ∈ C, so H1 holds as stated in `scaling_proposition.md`, with the explicit ε₀ = 0.029.

**Corollary (H1 in the "fixed compact" form, larger ε).**
- For every ε ∈ (0, 0.6878] with Ĝ(a) > 0, every maximiser has a representative with 0 < p < 1.74964 and
  |q| < √(2 + ε) + 0.6p < 2.690.
- Ĝ(a) > 0 is proved here for ε < 0.0859 (Lemma 5).

**Corollary (convergence, replacing H1 in step 4).** For 0 < ε ≤ 0.029,
  |K(ε) − K| ≤ 2ε[sup_{|σ|≤S}|r| + ε(S⁵/120 + (1+ε)S⁷/5040)] ≤ 23.3 ε,   with S = 5.295, r(σ) = σ³/6 − σ⁵/120.

**Limit version (ε = 0; sharpens the domain lemma of `math_note_v2.md` §8).**
- G₀(p, q) ≤ 0.4|p|.
- If G₀(p, q) > 0 (with p > 0), then (q − 0.6p)² + (2.8p)²/12 < 2 or (q + 0.6p)² + (2.8p)²/12 < 2.
- Hence G₀ > 0 only for |p| < √24/2.8 = 1.74964 and |q| < 1.76126. The note's lemma gives 4.0825 and 3.8637.
- The argmax of G₀ has p ≥ K/0.4 ≥ 1.4447.

## 2. Lemma 0: existence and symmetries

- **Continuity.** φ is jointly continuous in (x, w₁, b₁), and the windows are compact. So min_O φ, max_I φ, etc. are
  continuous, and G is continuous on ℝ².
- **Symmetries.** f_a(t + 2π) = f_a(t) + 2π, so b₁ ↦ b₁ + 2π shifts φ by a constant and leaves G unchanged.
  f_a is odd, so (w₁, b₁) ↦ (−w₁, −b₁) maps φ to −φ. This swaps G₁ and G₂ and leaves G unchanged.
- **w₁ = 0.** φ is constant and G = 0.
- **Attainment.** By Lemma 2(b), G ≤ 0 whenever 1.4|w₁| ≥ x_a. So when Ĝ(a) > 0 the supremum is a supremum over the
  compact set |w₁| ≤ x_a/1.4, b₁ ∈ [0, 2π], and it is attained. Every maximiser has G = Ĝ > 0, so w₁ ≠ 0. Take
  w₁ > 0 from now on.

## 3. Lemma 1: the pair identity (exact)

For t_o < t_i, with d = t_i − t_o and m = (t_o + t_i)/2 − π:

  **f_a(t_o) − f_a(t_i) = 2a cos(m) sin(d/2) − d.**

*Proof.*
- f_a(t_o) − f_a(t_i) = −d + a(sin t_o − sin t_i) = −d + 2a cos((t_o + t_i)/2)·sin(−d/2).
- cos((t_o + t_i)/2) = cos(π + m) = −cos m.
- So the difference is −d + 2a cos(m) sin(d/2). ∎

In rescaled units (t = π + √εσ, d = √εδ, m = √εμ) this is the exact analogue of the cubic identity
h(α) − h(α + δ) = δ[1 − ((α + δ/2)² + δ²/12)/2] of `math_note_v2.md` §8. Here α + δ/2 = μ, and the cubic identity
is its ε → 0 limit.

## 4. Lemma 2: the finite-ε domain lemma (every ε with 1 < a < π, w₁ > 0)

Let x_a be the unique root in (0, π) of sin x = x/a, and let β = arccos(1/a) ∈ (0, π/2). Write y = 1.4w₁.

**(a) G ≤ 0.4 ε w₁.** Equivalently, G̃_ε(p, q) ≤ 0.4p for every ε.
- G₁ ≤ φ(−1.2) − φ(−0.8), because a minimum is at most any member and a maximum at least any member.
- By Lemma 1 with d = 0.4w₁, this is 2a cos(m) sin(0.2w₁) − 0.4w₁ ≤ 2a|sin(0.2w₁)| − 0.4w₁ ≤ 0.4(a − 1)w₁.
- G₂ ≤ φ(0.8) − φ(1.2) gives the same bound.
- If G ≤ 0 the claim is trivial.

**(b) G > 0 ⇒ 0 < y < x_a (< π).**
- G₁ ≤ φ(−2) − φ(0.8). With d = 2.8w₁ = 2y, this is 2a cos(m₁) sin(y) − 2y ≤ 2a|sin y| − 2y, where
  m₁ = b₁ − 0.6w₁ − π.
- G₂ ≤ φ(−0.8) − φ(2) = 2a cos(m₂) sin(y) − 2y, where m₂ = b₁ + 0.6w₁ − π.
- So G > 0 ⇒ |sin y|/y > 1/a.
- For y ≥ π, |sin y|/y ≤ 1/π < 1/a. On (0, π), sin y/y is strictly decreasing. Hence y < x_a.

**(c) If G_j > 0, then cos m_j > 1/a.** So after b₁ ↦ b₁ − 2πk we may take |m_j| < β.
- By (b), 0 < sin y < y.
- G_j > 0 gives 2a cos(m_j) sin y > 2y > 0. So cos m_j > y/(a sin y) > 1/a.
- cos m > 1/a exactly when m ∈ (2πk − β, 2πk + β) for some k.

**(d) The ellipse.** Assume in addition ε ≤ 0.6878 (so that x_a² ≤ 6ε, Lemma 3). If G_j > 0 and |m_j| < β, then in
rescaled units μ_j = m_j/√ε:

  μ_j²/2 + c′p² < Λ(ε),   with c′ = 1.96/6,  Λ(ε) = 1/[(1 + ε)·min(κ₁, κ₃)],
  κ₁ = 1 − ε(2 + ε)/12,   κ₃ = (1 − 0.3ε)(1 − ε(2 + ε)/2).

*Proof.*
1. Use cos m ≤ 1 − (m²/2)(1 − m²/12) and sin y/y ≤ 1 − (y²/6)(1 − y²/20). Both are Taylor bounds with alternating
   remainder and hold for all real m and all y > 0.
2. By (b), (c) and Lemma 3, m² < β² ≤ ε(2 + ε) and y² < x_a² ≤ 6ε. So 1 − m²/12 ≥ κ₁ and 1 − y²/20 ≥ 1 − 0.3ε.
3. Each right side is at least the positive quantity it bounds. So
   1/a < cos m · sin y/y ≤ (1 − A)(1 − B), with A = κ₁m²/2 and B = (1 − 0.3ε)y²/6.
4. (1 − A)(1 − B) = 1 − A − B + AB, and AB ≤ B·m²/2 ≤ B·ε(2 + ε)/2.
5. So A + B(1 − ε(2 + ε)/2) < 1 − 1/a = ε/(1 + ε).
6. Divide by ε and use m² = εμ² and y² = 1.96εp². This gives
   κ₁μ²/2 + κ₃c′p² < 1/(1 + ε), which implies the stated inequality. ∎

**(e) q from μ.**
- q = μ₁ + 0.6p (from m₁) or q = μ₂ − 0.6p (from m₂).
- By Cauchy–Schwarz, |μ| + 0.6p ≤ √(2 + 0.36/c′)·√(μ²/2 + c′p²).
- Hence |q| < √(3.10204·Λ(ε)). Also, directly from (c) and Lemma 3, |q| < √(2 + ε) + 0.6p.

**Scope.** Only the points x = −2, −1.2, −0.8, 0.8, 1.2, 2 enter (a)–(e). These are window endpoints and also points
of the 800-point population data (`linspace` includes its ends). So Lemma 2 holds verbatim for the data gap G_n.
It does not hold for a random 400-point training set, whose points need not include the window endpoints.

## 5. Lemma 3: explicit bounds on x_a and β

**(i) x_a² ≤ 6ε for 0 < ε ≤ 0.6878.**
1. At x = x_a, 1/a = sin x/x ≤ 1 − (x²/6)(1 − x²/20).
2. So (x_a²/6)(1 − x_a²/20) ≤ ε/(1 + ε).
3. First pass: x_a < π gives 1 − x_a²/20 > 1 − π²/20 = 0.50652. So x_a² < 11.846ε, and hence
   1 − x_a²/20 > 1 − 0.5923ε.
4. Second pass: x_a² ≤ 6ε/[(1 + ε)(1 − 0.5923ε)].
5. The denominator is ≥ 1 if and only if ε(0.4077 − 0.5923ε) ≥ 0, that is, ε ≤ 0.6883. ∎

Hence G > 0 ⇒ p = y/(1.4√ε) < √6/1.4 = √24/2.8 = 1.74964. This is the ε-uniform bound, and it equals the limit
threshold exactly.

**(ii) β² ≤ ε(2 + ε) for all ε > 0.** Since tan β = √(a² − 1), β = arctan √(a² − 1) ≤ √(a² − 1) = √(ε(2 + ε)).

**Numerical values for ε ≤ 0.029.**
- κ₁ ≥ 0.99510 and κ₃ ≥ (1 − 0.0087)(1 − 0.029·2.029/2) = 0.96214.
- Since (1 + ε) ≥ 1, Λ ≤ 1/0.96214 = 1.03935.
- So |q| < √(3.10204 × 1.03935) = 1.7956. This bound is deliberately crude and monotone-safe.
- The value of Λ at ε = 0.029 itself gives 1.7701.

## 6. Lemmas 4–5: sup-norm remainder, and a lower bound on K(ε)

**Lemma 4 (remainder).**
- With s = √εσ and the Lagrange form sin s = s − s³/6 + s⁵/120 + R₇, |R₇| ≤ |s|⁷/5040 (all real s):

  φ_ε(σ) − h(σ) = ε[σ³/6 − (1 + ε)σ⁵/120] − aR₇/ε^{3/2},   |φ_ε − h| ≤ ε|σ³/6 − (1+ε)σ⁵/120| + (1+ε)ε²|σ|⁷/5040.

  This is the same remainder as step 2 of `scaling_proposition.md`.
- G is **2-Lipschitz** in the sup-norm of the profile on the windows. Each of G₁ and G₂ is a min minus a max, so each
  moves by at most 2δ, and so does their maximum.
- So |G̃_ε(p, q) − G₀(p, q)| ≤ 2·sup{|φ_ε − h|(σ) : σ ∈ σ(I ∪ O)}.

**Lemma 5 (exact value at (1.6, 1.2), and K(ε) ≥ K_low(ε)).**
1. At (p, q) = (8/5, 6/5), in exact rational arithmetic:
   - **I**: σ ∈ [−0.08, 2.48]. The only critical point of h inside is +√2, a local minimum, so the maximum is at an
     endpoint: max(3746/46875, 2914/46875) = 3746/46875.
   - **O₋**: σ ∈ [−2, −0.72]. The only critical point inside is −√2, a local maximum, so the minimum is at an
     endpoint: min(2/3, 10278/15625) = 10278/15625.
   - **O₊**: σ ∈ [3.12, 4.4], where h is increasing; the minimum is 30342/15625.
   - Membership of ±√2 is decided exactly, by comparing squares of rationals with 2.
   - Hence G₀(1.6, 1.2) ≥ G₁ = 10278/15625 − 3746/46875 = **27088/46875 = 0.5778773…**
2. Over all windows, σ ∈ [−2, 4.4]. The function g(σ) = σ³/6 − cσ⁵/120 (c = 1 + ε) is odd.
   - On [0, √(12/c)], 0 ≤ g ≤ (12/c)^{3/2}/15 ≤ 12^{3/2}/15 = 2.7713.
   - Beyond √(12/c), g decreases. For c ≤ 1.2, g(4.4) = 14.197 − 13.743c ≥ −2.296.
   - So sup_{|σ|≤4.4}|g| ≤ 2.7713 for ε ≤ 0.2.
3. By Lemma 4,
   K(ε) ≥ G̃_ε(1.6, 1.2) ≥ K_low(ε) := 27088/46875 − 2ε[2.7713 + (1 + ε)ε·4.4⁷/5040],  with 4.4⁷/5040 = 6.3349.
4. K_low > 0 for ε < 0.0859, and K_low > 0.4 for ε < 0.02998. K_low(0.029) = 0.4062.

## 7. Proof of the Theorem and corollaries

**Theorem.** Let 0 < ε ≤ 0.029.
1. By Lemma 5, Ĝ(a) ≥ ε^{3/2}K_low(ε) > 0. By Lemma 0 the supremum is attained, and every maximiser has w₁ ≠ 0.
   Reduce to w₁ > 0.
2. Let j be an orientation with G_j = Ĝ > 0. By Lemma 2(c), reduce b₁ so that |m_j| < β.
3. **Upper bound on p.** Lemma 2(b) and Lemma 3(i) give p < √6/1.4 = 1.74964 ≤ 2.2.
4. **Bound on q.** Lemma 2(d), (e) and the numerical values in Lemma 3 give |q| < 1.7956 ≤ 2.
5. **Lower bound on p.** Lemma 2(a) at the maximiser gives 0.4p ≥ G̃_ε = K(ε) ≥ K_low(ε) ≥ K_low(0.029) = 0.4062.
   K_low is decreasing in ε, so p ≥ 1.0155 ≥ 1. ∎

**Fixed-compact corollary.** For ε ≤ 0.6878 with Ĝ(a) > 0, steps 1–3 and 4 (with the direct bound of Lemma 2(e)) go
through unchanged, and give 0 < p < 1.74964 and |q| < √(2 + ε) + 0.6·1.74964. ∎

**Convergence corollary.**
- Let C′ = [1, 1.7497] × [−1.7956, 1.7956] ⊂ C. On C′, |σ| ≤ 2(1.7497) + 1.7956 = 5.295.
- For ε ≤ 0.029 every maximiser of G̃_ε lies in C′ (the Theorem), so sup_{C′} G̃_ε = K(ε).
- In the limit, Lemma 2(a) and the limit ellipse (§1, proved below) put every maximiser of G₀ in C′, since its p is at
  least 0.57788/0.4 = 1.4447. So sup_{C′} G₀ = K.
- Hence |K(ε) − K| ≤ sup_{C′}|G̃_ε − G₀| ≤ 2ε M(S, ε), by Lemma 4.
- With S = 5.295: sup_{|σ|≤S}|r| = 9.943, and the bound is ≤ 23.3ε for ε ≤ 0.029. The measured deviation is
  about 5ε (`scaling_proposition.md` step 4).
- This is exactly step 4 of `scaling_proposition.md` with C′ in place of C. Its own constant (S = 6.4, ≈ 94ε) is also
  valid, since C′ ⊂ C.

**Limit version.**
- Apply the cubic identity to the pairs (−1.2, −0.8) and (0.8, 1.2) (δ = 0.4p): G₀ ≤ δ = 0.4p.
- Apply it to the pairs (−2, 0.8) and (−0.8, 2) (δ = 2.8p, α + δ/2 = q ∓ 0.6p). A positive G_j needs
  (q ∓ 0.6p)² + (2.8p)²/12 < 2.
- That inequality is μ²/2 + c′p² < 1 with μ = q ∓ 0.6p. Cauchy–Schwarz as in Lemma 2(e) gives |q| < √3.10204 = 1.76126. ∎

(The §8 lemma used δ = 1.2p for the p-threshold. The pair (−2, 0.8) with δ = 2.8p is sharper and gives
1.74964 instead of 4.0825. The §8 lemma remains correct.)

## 8. What is NOT proved here (honest limits)

1. **The range of ε is small, and it is an artefact of the proof.**
   - ε₀ = 0.029 comes only from the crude lower bound K_low(ε), which uses one test point and a sup-norm remainder.
   - The localisation lemmas themselves hold up to ε ≈ 0.69 (the fixed compact form) and up to ε ≈ 0.29 (the |q| < 2
     bound, where √(3.102Λ(ε)) = 2).
   - For the C-form at a *given* larger ε, the only missing input is K(ε) ≥ 0.4 (for p ≥ 1).
   - At a = 1.02, 1.05, 1.10, 1.15 and 1.25, the rigorous Ĝ lower ends in `ghat_rigorous.csv` give K(ε) = 0.575,
     0.569, 0.559, 0.549 and 0.532. With Lemma 2 this gives the C-form at those a. That step is **certified, not
     proved**: it rests on the Ĝ branch-and-bound values.
   - At a = 1.30 (ε = 0.3) the q-bound of Lemma 2(d) gives 2.0008 > 2. It just fails, so the C-form at a ≥ 1.30 is not
     covered by this argument. The grid check in §9 still places the argmax well inside C.
   - Ĝ(a) > 0 for every a > 1 is proved only for ε < 0.0859. Beyond that it rests on the certified values.
2. **H1 localises; it does not identify the maximiser.**
   - The stronger claim in `math_note_v2.md` §8 is that K(ε) = K_loc(ε), that is, the global maximiser is the tied
     corner continued from ε = 0. §8 uses it to compute k₁ as a two-sided derivative of the global K(ε).
   - That claim needs, in addition, uniqueness (up to symmetry) of G₀'s global maximiser. That uniqueness is
     **certified** by the K branch and bound; it is not proved here.
   - Given it, the Theorem plus the convergence corollary put the finite-ε maximiser within o(1) of the corner. §8's
     parametric Krawczyk certificate (unique sharp corner for |ε| ≤ 0.05) then finishes the job.
   - So that chain is "proved + certified", not "proved".
3. **Only the continuous-window gap and the population data gap are covered.**
   - For a random training set, the endpoint pair bounds of Lemma 2 need the extremal data points in place of the
     endpoints. The analogous argument works with slightly different constants, but it is not written out here.
4. **H1 concerns object 1 only** (the gap maximiser). It says nothing about object 3 (the global conditional-loss
   minimiser at finite a). The note correctly leaves that to Block 1c.

## 9. Numerical sanity checks (not part of the proof)

`nice -n 15 python -m src.h1_checks`: one process, about 1 minute. The venv has no scipy, so root-finding is by
bisection.

- **(A) The exact value** 27088/46875 = 0.5778773 at (1.6, 1.2). The code asserts the critical-point membership
  pattern (I: +√2 only; O₋: −√2 only; O₊: none). The other orientation is ≤ −9.72 < 0.
- **(B) Lemma 3 bounds.** max x_a²/(6ε) = 1.000000 (→ 1 as ε → 0, never above) and max β²/(ε(2+ε)) = 1.000000, over
  800 values of ε in [1e−8, 0.6878].
- **(C) Constants.**
  - sup_{[0,4.4]}|g| = 2.7713 at ε = 0 and 2.2943 at ε = 0.2, against the bound 2.7713.
  - K_low > 0 for ε < 0.08593, and K_low > 0.4 for ε < 0.02998.
  - Λ(0.029) = 1.01006.
- **(D) 800,000 placements at each ε ∈ {0.5, 0.2, 0.05, 0.029, 0.01, 0.001}.**
  - Half are uniform over w₁ ∈ (0, 3], b₁ ∈ [0, 2π). Half are in the rescaled window p ∈ [0, 3], |q| ≤ 4.
  - Gap by exact extrema. The vectorised gap agrees with `src.exact_extrema.exact_gap` to 0.0 on 200 points.
  - 18,000–22,000 placements per ε have G > 0. All satisfy 1.4w₁ < x_a, |m_j| < β for the positive orientation, and
    the ellipse inequality of Lemma 2(d).
  - max G/(0.4εw₁) = 0.990–0.991, so Lemma 2(a) is nearly tight.
  - max G/(maximal drop 2(√(a²−1) − β)) ≈ 0.31.
- **(E) Grid argmax** of G̃_ε on p ∈ (0, 1.8] × q ∈ [−2.6, 2.6] (901 × 1301):

  | ε | K_grid | argmax (p, \|q\|) | max over p < 1 | max over \|q\| ≥ 2 | proved K_low |
  |---|---|---|---|---|---|
  | 0.2 | 0.54039 | (1.496, 1.132) | 0.392 | −0.0035 | (not valid) |
  | 0.05 | 0.56867 | (1.576, 1.184) | 0.387 | −0.0030 | 0.2675 |
  | 0.029 | 0.57302 | (1.588, 1.192) | 0.387 | −0.0029 | 0.4062 |
  | 0.01 | 0.57710 | (1.598, 1.200) | 0.386 | −0.0028 | 0.5212 |
  | 0.001 | 0.57922 | (1.604, 1.204) | 0.385 | −0.0028 | 0.5723 |
  | 0.0001 | 0.57927 | (1.604, 1.204) | 0.386 | −0.0028 | 0.5773 |

  The argmax is inside C at every ε checked, and it approaches K's argmax (1.6056, 1.2042). The maximum over p < 1
  stays below 0.4, consistent with Lemma 2(a).
  K_low is conservative by 0.17 at ε = 0.029. That slack is the only thing that limits ε₀.

## 10. Remark on the suggested route

The suggested route bounds G by the drop over the decreasing region plus the rise elsewhere, with a separate
many-period argument for w₁ = O(1). It is unnecessary.
- Lemma 1 turns every outer–inner pair into the closed form 2a cos(m) sin(d/2) − d. Choosing the pairs
  (−1.2, −0.8) [or (0.8, 1.2)] and (−2, 0.8) [or (−0.8, 2)] gives all three bounds: on G, on p and on q.
- The many-period regime is the single inequality |sin y|/y ≤ 1/π < 1/a for y = 1.4w₁ ≥ π.
- As a by-product, the finite-a branch-and-bound domain w₁ ∈ (0, a/1.4] (`ghat_bnb.py`) could be shrunk to
  w₁ < x_a/1.4. At a = 1.30, for example, that is 0.873 against 0.929.
