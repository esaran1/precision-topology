# Items #13 and #2: exact critical points replace grid sampling

Junyu's theory review items **#13** (exact extrema for every headline separation,
cross-checked against the Lipschitz certificate) and **#2** (the same exact
extrema inside the certified `Ĝ` search). Artifacts: `exact_separation.csv`,
`exact_vs_grid.csv`, `kappa_certified_exact.csv`, `src/exact_extrema.py`,
`src/kappa_certify.py`.

---

## The exact computation

`f_a(t) = t + a sin t`, so

    d/dx f_a(w₁x + b₁) = w₁·(1 + a·cos(w₁x + b₁))

which vanishes iff `cos(w₁x + b₁) = −1/a`. For `a > 1` that has solutions

    w₁x + b₁ = ±arccos(−1/a) + 2πk        k ∈ ℤ

so `x = (±arccos(−1/a) + 2πk − b₁)/w₁`. The extrema of `f_a(w₁x+b₁)` on an
interval are the **endpoints plus whichever of these finitely many points lie
inside** — no grid, exact up to float64 evaluation of `arccos` and `sin`.

For `a ≤ 1` there are **no** interior critical points (`|a cos| ≤ 1` with equality
only at isolated points), so the activation is monotone and the extrema are the
endpoints. The implementation returns the empty list in that case, which is a
check on the monotone branch rather than an approximation of it.

## #13: exact vs grid on every recorded run

| check | result |
|---|---|
| Phase 1 runs tested | **4,800** |
| exact separation agrees with grid `solves()` | **4,800 / 4,800** |
| disagreements | **0** |
| of the 866 Lipschitz-**certified** runs, exact confirms | **866 / 866** |

The two figure-1 networks, checked individually:

| network | `a` | exact separates | grid `solves()` | inner max logit | outer min logit | interior critical points (I / O+ / O−) |
|---|---:|---|---|---:|---:|---|
| solver, seed 0 | 1.5 | **True** | True | −5.068e−01 | +4.905e−02 | 1 / 1 / 0 |
| monotone, seed 38 | 1.0 | **False** | False | +1.838e−01 | −1.242e+00 | **0 / 0 / 0** |

The monotone network has **zero** interior critical points at every window, as
`a = 1` requires — the exact computation reproduces the monotonicity argument that
makes family A's `a ≤ 1` case provably unsolvable, rather than assuming it.

### The invariant that is forced, and a correction

Exact extrema search endpoints **plus** interior critical points, while a grid
samples. So on each window **separately**:

    exact max_I  ≥  grid max_I        exact min_O  ≤  grid min_O

**Verified: 0 violations in 400 runs**, worst discrepancy 0.00e+00. A grid can
only under-state how bad a window is, which is the direction that matters for a
soundness claim.

**Correction to an earlier statement of mine.** I first wrote that the *oriented*
gap must satisfy `exact ≤ grid`. That is **wrong**: the oriented gap is
`max(min_O − max_I, min_I − max_O)`, a maximum over two orientations whose terms
both shift, so its direction is **not** forced. Measured, exact exceeds grid in
4,798 of 4,800 runs and falls below in 2 — and those 2 differ only at float64
noise (identical to 6 decimals, `a = 1.25` seed 191 and `a = 1.35` seed 56). The
per-window inequalities above are the correct invariant and they hold exactly.

Max `|exact − grid|` on the oriented gap across 4,800 runs: **1.47e−02**, on gaps
of order 0.01–3, i.e. the grid's sampling error — which is precisely why the
exact version was worth computing.

## #2: the certified `Ĝ` search with exact x-extrema

The outer certificate is unchanged: a grid in `(w₁, b₁)` with Lipschitz slack
`L·h/2`, `L = (1+a)·max(1, max|x|)`. What changed is the **inner** evaluation of
`G` at each grid point — previously a 4,001/2,000-point sample over the windows,
now exact.

| `a` | `h` | `Ĝ` lower | `Ĝ` upper | rel. width | `κ` lower | `κ` upper |
|---|---:|---:|---:|---:|---:|---:|
| 1.02 | 6.4e−07 | 0.001626745 | 0.001628038 | 0.080% | 0.307747 | 0.307992 |
| 1.05 | 6.4e−07 | 0.006360086 | 0.006361398 | 0.021% | 0.308396 | 0.308459 |
| 1.10 | 3.2e−06 | 0.017671951 | 0.017678671 | 0.038% | 0.309406 | 0.309523 |
| 1.25 | 1.6e−05 | 0.066505622 | 0.066541622 | 0.054% | 0.312236 | 0.312405 |
| 1.30 | 1.6e−05 | 0.086101794 | 0.086138594 | 0.043% | 0.313092 | 0.313226 |
| 1.35 | 1.6e−05 | 0.106913551 | 0.106951151 | 0.035% | 0.313909 | 0.314019 |
| 1.40 | 1.6e−05 | 0.128771896 | 0.128810296 | 0.030% | 0.314688 | 0.314782 |
| 1.45 | 1.6e−05 | 0.151545319 | 0.151584519 | 0.026% | 0.315435 | 0.315517 |
| 1.50 | 1.6e−05 | 0.175126200 | 0.175166200 | 0.023% | 0.316152 | 0.316224 |
| 1.60 | 8.0e−05 | 0.224360202 | 0.224568202 | 0.093% | 0.317490 | 0.317784 |

**Certified `κ` over `a ≤ 1.60`, exact x-extrema: [0.307747, 0.317784]**, width
**3.261%**.

Against the sampled-x certificate `[0.307747, 0.317616]`, width 3.207%: the
**lower endpoint is identical to 6 digits** and the upper moves by 0.05%. Per-`a`
differences in `Ĝ_lower` are at most **6.4e−06** — inside the certificate slack at
every `a`.

**So the certificate did not materially rest on the x-grid.** That is the
reassuring outcome rather than a null one: the concern was legitimate (Block G's
G4 episode showed a fixed grid mis-measuring `Ĝ` by 18.8% in the `(w₁,b₁)`
directions), and it is now excluded for the `x` direction by exact computation.
The `(w₁,b₁)` grid remains a grid, but that is what the Lipschitz slack bounds.

**Interval arithmetic** was not used. It would tighten the float64 evaluation of
`arccos`/`sin` to a rigorous enclosure, but the certificate slack
(2.3e−04 to 9.3e−02 relative) is **four to six orders of magnitude** above float64
rounding on these quantities, so it would not move any reported digit. Recorded as
deliberately omitted, with the reason, rather than overlooked.

## What this buys

`Ĝ`, `κ`, and every headline separation now rest on:

- **exact** evaluation in `x` (endpoints plus closed-form interior roots),
- a **certified** bound in `(w₁, b₁)` (Lipschitz slack, widths 0.02–0.09%),
- agreement with the previous grid criterion on **4,800 / 4,800** runs and with
  the Lipschitz certificate on **866 / 866**.

The sign-correctness criterion is a statement about the continuum at every point
where the paper makes it.
