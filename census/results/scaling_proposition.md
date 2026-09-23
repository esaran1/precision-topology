# Proposition: the scaling reduction of the 1D fold task

Checks: `src/scaling_proposition.py` → `results/scaling_proposition_checks.csv`
(39 checks, all pass). Check IDs in brackets.

## Setting

`f_a(t) = t + a sin t`, `a = 1 + ε`, `ε > 0`. Width-1 network
`N(x) = w₂ f_a(w₁x + b₁) + b₂` on windows `I = [−0.8, 0.8]` (label 0),
`O = ±[1.2, 2.0]` (label 1); `|x| ≤ 2` on `I ∪ O`.
Limit coordinates: `w₁x + b₁ = π + √ε σ(x)`, `σ(x) = ux + v`.
Compact placement set `C = {1 ≤ |u| ≤ 2.2, |v| ≤ 2}`, so `|σ| ≤ S = 6.4` on `I ∪ O`.
`h(σ) = −σ + σ³/6`, `r(σ) = σ³/6 − σ⁵/120`.
Class gap `G(φ) = max(min_O φ − max_I φ, min_I φ − max_O φ)` for a function `φ`
of `x`; `Ĝ(a)` its supremum over placements; `K = sup_{(u,v)} G(h∘σ) = 0.579454926`
(certified, T64). `R = |w₂| Ĝ(a)/2`.

## Hypotheses

- **H1 (localisation).** For `ε` small, `Ĝ(a)` is attained at a placement whose
  limit coordinates lie in `C`, up to the symmetries `t ↦ t + 2π`
  (`f_a(t + 2π) = f_a(t) + 2π`) and `σ ↦ −σ`, `x ↦ −x`.
  *Evidence:* every certified argmax for `a ∈ [1.02, 1.6]` lies in `C`, at
  `|u| = 1.341–1.594`, `|v| = 1.021–1.196` [H1(·)].
- **H2 (nondegeneracy).** At `R_glob^∞` the minimiser of `L_∞` below is unique on its
  branch, with positive-definite Hessian, and the class gap of the minimiser
  crosses zero with nonzero derivative in `R`.
  *Evidence:* at `W = 2R/K = 0.690`: min Hessian eigenvalue `0.1470` [6b];
  `d(gap)/dW = 1.656` [6a]; minimiser moves ≤ `0.0036` per 0.005 step in `W`
  across the crossing [6c].

## Statement

Under H1 and H2, as `ε → 0`:

1. `f_a(π + s) = π − εs + (1+ε)s³/6 − (1+ε)s⁵/120 + O(s⁷)`.
2. `f_a(π + √ε σ) = π + ε^{3/2} h(σ) + ε^{5/2} r(σ) + ρ_ε(σ)`, with
   `|ρ_ε(σ)| ≤ ε^{7/2}(|σ|⁵/120 + (1+ε)|σ|⁷/5040)`.
3. With `b₂' = b₂ + w₂π`, the logit is `b₂' + w₂ε^{3/2}h(σ) + w₂ε^{5/2}(r(σ) + ε^{-5/2}ρ_ε)`.
4. `Ĝ(a) = K ε^{3/2}(1 + δ_ε)`, `|δ_ε| = O(ε)`; hence `w₂ε^{3/2} = (2R/K)/(1 + δ_ε)`.
5. Let `L_a(u, v, b₂'; R)` be the population logistic loss at output weight `w₂`
   and `L_∞(u, v, b₂'; R) = E ℓ(b₂' + (2R/K) h(σ(x)), y)`. Then, uniformly on
   `C × [−B, B]`,
   `|L_a − L_∞| ≤ (2R/K)·(ε M' + |δ_ε| H)/(1 + δ_ε) = O(Rε)`,
   with `M' = sup_{|σ|≤S} |r| + ε(S⁵/120 + S⁷/5040)` and `H = S + S³/6`.
   `L_∞` depends on the output weight only through `R`, and on the task only
   through the windows.
6. Each switch point of the conditional landscape (`R_glob`, `R_solve`), expressed
   in `R`, converges to the corresponding switch point of `L_∞` at rate `O(ε)`.

## Proof

**1.** `f_a(π + s) = π + s + (1+ε) sin(π + s) = π + s − (1+ε) sin s` [1b]; insert
the sine series [1].

**2.** Substitute `s = √ε σ`. The `ε^{3/2}` coefficient is `h` [2a], the `ε^{5/2}`
coefficient is `r` [2b], and no lower power of `√ε` occurs [2c]. The remainder is
`−ε^{7/2}σ⁵/120 + (1+ε)R₇(s)` with `|R₇(s)| ≤ |s|⁷/5040` (alternating series),
which gives the bound. It holds with the true remainder at 3.9–6.6% of the bound
for `ε ∈ {0.3, 0.1, 0.02, 0.001}` in 60-digit arithmetic [R(·)].

**3.** Algebra from 2.

**4.** For fixed `(u, v)`, `G` is 1-Lipschitz in the sup-norm of `φ` on `I ∪ O`
(each of its terms is a min or max of `φ`), so by 2,
`|G(f_a∘σ) − ε^{3/2}G(h∘σ)| ≤ 2ε^{5/2}M_ε`, `M_ε = sup_{|σ|≤S}|r| + ε(S⁵/120 + (1+ε)S⁷/5040)`,
uniformly on `C`. Taking sups over `C` (H1 for the left side) gives
`|Ĝ(a)/ε^{3/2} − K| ≤ 2εM_ε`. Measured sup deviation: `1.236, 0.4765, 0.1008,
0.02548` at `ε = 0.3, 0.1, 0.02, 0.005`, ≈ `5ε`, against the bound ≈ `94ε` [4(·)].
The rate is `O(ε)`; the constant in the bound is conservative by about 20×.

**5.** `ℓ(z, y) = log(1 + eᶻ) − yz` has `|∂_z ℓ| = |sigmoid(z) − y| ≤ 1`, so
`|L_a − L_∞| ≤ E|z_a − z_∞|`. By 3–4,
`z_a − z_∞ = (2R/K)[(1+δ_ε)^{-1} ε(r + ε^{-5/2}ρ_ε) − δ_ε(1+δ_ε)^{-1} h]`, bounded by
the stated expression. Verified on 1,500 random points of `C × [−3, 3]` at
`ε ∈ {0.1, 0.02, 0.005}`, `R ∈ {0.2, 0.3}`; measured sup `|L_a − L_∞|` scales as
`ε` [5(·)]. `L_∞` involves `w₂` only through `2R/K`.

**6.** Steps 2–5 hold also for derivatives in `(u, v, b₂')` (the remainder is
smooth in `σ`, and `σ` is affine in `(u, v)`), so `∇L_a → ∇L_∞` at rate `O(Rε)`
on `C`. By H2 and the implicit function theorem, the minimiser of `L_a` on its
branch is within `O(ε)` of that of `L_∞`, uniformly for `R` near `R_glob^∞`. The
class gap of the minimiser is then within `O(ε)` of its limit, and because the
limit gap crosses zero transversally (H2), its zero `R_glob(a)` is within `O(ε)`
of `R_glob^∞`. The same argument applies to `R_solve` wherever its defining
crossing is transversal. ∎

## Connection to measured rates

- `(κ(a) − κ₀)/κ₀` against `ε` for `ε ≤ 0.1`: log-log slope **0.9926** [M1].
- `R_glob(a)/R_glob^∞ − 1` over the six measured `a`: log-log slope **0.9472** [M2] (frozen grid; superseded by the
  certified-interval refit in `math_note_v2.md` §5: 0.829, range [0.714, 0.954]).

Both are the `O(ε)` rate of 4 and 6.

## The q-families

For `|x| ≤ 1` the constructed families are `f(x) = (1−a)x + a sgn(x)|x|^{q+1}/(q+1)`.
With `a = 1 + ε`, `x = ε^{1/q}σ`:

    f = ε^{1+1/q} [ h_q(σ) + ε g_q(σ) ],   h_q = −σ + sgn(σ)|σ|^{q+1}/(q+1),   g_q = sgn(σ)|σ|^{q+1}/(q+1)

**exactly**, with no higher terms, while `ε^{1/q}|σ| ≤ 1` [q=·]. So steps 2–6 hold
with `β = 1 + 1/q` in place of `3/2` and remainder exactly `ε g_q`. For `q = 2`,
`√2·h₂(τ/√2) = h(τ)` [q=2 vs sin]: the `q = 2` limit is the sin-family limit up to
scale, which is why their onsets differ by a constant factor (T44).

## Limits of the statement

- **Cross-family test (post-registration finding, `crossfamily_results.md`).**
  For the q-families the reduction holds only while `ε^{1/q}·max|σ| ≤ 1`, because
  they are polynomial only for `|t| ≤ 1`. The registered cross-family test ran
  at `ε ≥ 0.3`, where 25–34% of window points fall on the linear continuation.
  There, q2 matched family A's `R_glob` at all six `a` but **not** its `R_solve`
  (below A by 0.012–0.051). So at finite `ε` the reduction does **not** fix the
  solve threshold across families whose folds agree only near the fold.
  Statement 6 is claimed only inside the reduction's domain.

- **Cross-family follow-up, inside the domain (`crossfamily_followup_results.md`;
  registered after X1 failed).** At `a` = 1.01–1.04, with 0% of window points on
  q2's linear continuation, q2 matches family A on **both** `R_glob` and
  `R_solve` at all four `a` (Y1 passes). Inside its domain, the reduction therefore
  predicts both thresholds across equivalent families. The agreement is at grid
  resolution (`ΔR = 0.00215`), with both families within one step of the limit.
  A registered negative control (Z1, made after Y1) shows that this resolution is
  enough to discriminate: q1, whose limit is not a rescaling of `h`, sits 35–36 grid
  steps from A on `R_glob` and 5 on `R_solve` at the same `a`. So Y1's agreement
  reflects the reduction, not the grid. X1 stays failed as registered.

- H1 and H2 are verified numerically, not proved.
- Step 6 is conditional on H2 at each switch point used; H2 was checked at
  `R_glob^∞` only.
- The remainder constants are explicit but conservative; the rate, not the
  constant, is what the measurements confirm.
