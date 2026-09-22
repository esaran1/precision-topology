# Registration: a training-free cross-family test of the scaling reduction

**Written before any certified Ĝ or landscape threshold has been computed for
the q-families at these `a` values.** Date: 2026-09-22.

## Why the reduction makes a prediction here

For `|x| ≤ 1` the q = 2 family is `f(x) = −εx + (1+ε)x³/3`. With `x = s/√2`,

    f(s/√2) = (1/√2) [ −εs + (1+ε) s³/6 ]

which is exactly the **cubic truncation of the sin family** at the same `ε`,
`f_a(π+s) − π = −εs + (1+ε)s³/6 − (1+ε)s⁵/120 + …`, scaled by `1/√2`. The
input rescaling `s ↦ s/√2` and the shift `π ↦ 0` are absorbed by `(w₁, b₁)`,
and the output factor `1/√2` is absorbed by `w₂`. Because `R = |w₂|Ĝ/2`
normalises by the family's own `Ĝ`, which carries the same `1/√2`, **the two
families' conditional landscapes in `R` differ only through the sin family's
quintic term, a relative `O(ε)` correction** (T70). In the limit they are
identical: `√2·h₂(τ/√2) = h(τ)`.

The q = 1 family reduces to `h₁(σ) = −σ + sgn(σ)σ²/2`, which is **not** a
rescaling of `h`. Its landscape in `R` has no reason to coincide with family A's.

## What is computed

For each family ∈ {A (sin), q2, q1} and each `a` ∈ {1.30, 1.35, 1.40, 1.45, 1.50,
1.60} — family A's Block B grid:

- **`Ĝ`**: certified, exact `x`-extrema plus Lipschitz slack (`kappa_certify` for
  A, `family_certify` for the q-families).
- **`R_glob(a)`**: the smallest `R` at which the conditional minimiser has class gap
  > 0. **`R_solve(a)`**: the smallest `R` at which the conditional minimiser passes
  the dense regional check.
- **Procedure: frozen Block B** (sha256 `9f1b10741d8bf48c`): `best_conditional` with
  50 restarts (24 for `R_solve`), 600-step screen, 8 kept, 3,000 inner steps, Adam
  3e-2 → 5e-3, the log-2 degeneracy exclusion, the 800-point population loss,
  coarse-then-fine bracketing. **Only the activation is substituted.**
- **Grid in `R`, not in `w₂`.** The frozen grid is 0.5 then 0.05 in `|w₂|`. For a
  q-family both steps are multiplied by `Ĝ_A(a)/Ĝ_q(a)`, so the `R`-resolution at
  each `a` equals family A's frozen resolution `Δ_A(a) = 0.05·Ĝ_A(a)/2`. The `w₂`
  range covers `R ∈ [0.05, 0.60]`.
- **Family A** is not recomputed: its values are the committed frozen-grid
  `w2_glob`, `w2_solve` (`blockB_switches.csv`) times the certified `Ĝ_A`.

## Registered predictions

Tolerance at each `a`, for `x ∈ {glob, solve}`:

    tol_x(a) = max( |R_x^A(a) − R_x^∞| , 2·Δ_A(a) )

— the size of family A's own finite-ε correction at that `a`, floored at two grid
steps. `R_glob^∞ = 0.19991`, `R_solve^∞ = 0.30711` (T58).

- **X1 (q2 matches A on the landscape).** `|R_x^{q2}(a) − R_x^A(a)| ≤ tol_x(a)` at
  **all six** `a`, for **both** `R_glob` and `R_solve`.
- **X2 (q1 does not).** `|R_x^{q1}(a) − R_x^A(a)| > tol_x(a)` at **four or more** of
  the six `a`, for at least one of `R_glob`, `R_solve`.

## How the outcome will be read, fixed now

The pooled solve thresholds (T52 restated, same estimator) are family A **0.3723**,
q2 **0.3032**, q1 **0.3585**.

- **If X1 holds**, q2 and family A share a landscape while their pooled `R₅₀`
  differ by 0.069. The difference is then **dynamical** (where trained runs sit
  relative to the landscape's thresholds), not geometric. The paper states that.
- **If X1 fails**, the landscape thresholds differ too. The reduction misses
  something at these `ε`, and **that goes in the math note** (`scaling_proposition.md`)
  as a limitation of the proposition.
- **X2** is the control. If q1 matched A as well, the test would not be
  discriminating: agreement would come from `R`'s normalisation, not from the
  shared polynomial.
