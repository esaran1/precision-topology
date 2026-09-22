# Registration: the scaling limit as an explanation of O1

**Written before Block B has been run on `h(sigma)`, and before any finite-`eps`
correction has been fitted or regressed.** Date: 2026-09-21.

Inputs already in hand and stated here so the registration is auditable: the
symbolic expansion (verified), the numeric dip-depth convergence (verified),
the constant `K = 0.577731` computed on `h` alone, and the Step-5c table showing
`w1*/sqrt(eps)` in [1.31, 1.56] and `Ghat/eps^1.5` in [0.49, 0.56].

---

## What the scaling limit says

With `t = pi + s`, `a = 1 + eps`, `s = sqrt(eps) * sigma`:

    f_a(pi + s) - pi  =  eps^{3/2} * h(sigma) + O(eps^{5/2}),    h(sigma) = -sigma + sigma^3/6

So near the fold the activation *is* `h` up to an amplitude `eps^{3/2}` and a
horizontal rescaling `sqrt(eps)`. Two consequences, both already checked:

- dip depth `D(a) -> (4 sqrt2 / 3) * eps^{3/2}`
- optimal placement `w1* ~ sqrt(eps)`, and `Ghat(a) -> K * eps^{3/2}`

Since `R = |w2| * Ghat / 2`, holding `R` fixed while `eps` varies means holding
`|w2| * eps^{3/2}` fixed up to the constant `K`. **If the conditional training
loss at fixed `|w2|` depends on `|w2|` only through `|w2| * eps^{3/2}`, then the
switch points are functions of `R` alone and O1 follows** — `R` is the right
variable because it is the only scale-invariant combination.

## Already reported as FAILING (recorded before the registration below)

**Step 5, exact loss invariance, does not hold at the `eps` used.** Holding
`|w2| * eps^{3/2}` fixed, the conditional gap drifts `+0.001388` at `a = 1.30`
to `-0.055321` at `a = 1.60`. The invariance is asymptotic, not exact, because
the outer windows `I = [-0.8, 0.8]` and `O = +-[1.2, 2.0]` are **fixed in `x`**
while the fold's width shrinks as `sqrt(eps)`; the placement `w1` absorbs this
only to leading order. At `eps = 0.30 .. 0.60` the neglected `O(eps^{5/2})` term
is 15-25% of the leading one.

This is reported as a failed step, not repaired. The registration below tests
what survives.

## Registered predictions

**S-1 (the constant).** `kappa_0 = K / (4 sqrt2 / 3) = 0.3064` should match the
measured `kappa(a) = Ghat(a) / D(a)` as `a -> 1+`.

> Registered tolerance: **within 3%** at `a = 1.02`.

**S-2 (scaling-limit switch points exist and are a-independent).** Running the
frozen Block B procedure with `f_a` replaced by `h(sigma)` on the rescaled
windows must return finite `R_glob^inf` and `R_solve^inf`. These are
**training-free and carry no `a`**.

> Falsifier: the scan is degenerate on `h` (no non-degenerate minimiser, or no
> sign change in the gap), in which case the scaling-limit account gives no
> switch points and this route is dead.

**S-3 (convergence, with the DIRECTION REGISTERED NOW).** The finite-`a` values
`R_glob(a)` and `R_solve(a)` should approach `R_glob^inf`, `R_solve^inf` as
`a -> 1+`, with the deviation growing in `eps`.

> **Registered direction: the finite-`eps` correction is POSITIVE and INCREASING
> in `eps`** — that is, `R_glob(a) > R_glob^inf` and the excess grows with `eps`.

Grounds, stated before the fit: `Ghat / eps^{3/2}` **decreases** with `eps`
(0.56 at `eps = 0.02` down to 0.49 at `eps = 0.50`), so a finite-`eps` fold is
*less* efficient per unit amplitude than the limit; a given placement quality
therefore needs *more* `|w2|`, pushing `R` up. Independently, measured crossing
`R` rises `+10.0%` and `R_glob` rises `+6.5%` over `a = 1.30 -> 1.60`, both
upward in `eps`. Both lines of reasoning give the same sign.

> **Falsified if** the deviation is negative (finite-`a` switch points sit
> *below* the scaling-limit values), or if it does not increase monotonically in
> `eps` across the six `a`. Either outcome means the `eps^{3/2}` scaling is not
> what sets the drift, and the +10% drift in O1 needs a different account.

**S-4 (leading order of the correction).** Only if S-3 holds: the correction
should be `O(eps)` relative, since the next term in the expansion of `f_a` is
`O(eps^{5/2})` against a leading `eps^{3/2}`.

> Registered: fitting `R_glob(a) = R_glob^inf * (1 + c1 * eps)` should give
> `c1 > 0` and residuals below the spread already present in `R_glob`
> (grid step 0.05 in `|w2|`, about 0.002 in `R`). **A required exponent far from
> 1 — outside [0.5, 2] — falsifies S-4** while leaving S-3 intact. S-4 is the
> weaker, more easily broken claim and is labelled as such.

## What is at stake

If S-1, S-2 and S-3 hold, the paper can say `R` is the right variable **because
it is the scaling-invariant combination of the fold's amplitude and the output
weight**, derived rather than observed. If S-3 fails, O1 stands as a measured
regularity with no derivation, which is what the draft currently claims.
