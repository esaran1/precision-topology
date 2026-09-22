# Registration: certifying the κ interval by a compact-domain Lipschitz argument

**Written before any certified bound has been computed.** Date: 2026-09-22.

## Why this is needed, and a naming correction

T50 reports `κ = Ĝ(a)/D(a) ∈ [0.30544, 0.31543]` over `a ≤ 1.60`, from a **grid
search at resolution 600** over placements `(w₁, b₁)`. That is a **sampled**
maximum: `Ĝ` is a supremum over a continuum, and a finite grid can only
**under-estimate** it, so the reported κ is a lower bound on the truth with no
error bar. Every downstream use — the theorem chain `|w₂| ≥ 2m/(κ·D(a))`, the
scaling-limit constant `κ₀`, and T58's 0.061% agreement — inherits that gap.

Block G already showed this is not hypothetical: a fixed placement grid
**mis-measured** `Ĝ` badly at small ε (G4_shifted off by −18.8% at `a = 1.002`)
because the optimum scales as `w₁ ~ √ε`. The same failure mode threatens κ.

**Naming.** The priority list calls this "Block H", but Block H is already the
growth-rate dose-response (registered `3c1c0f4`, run, reported in
`blockH_rate_results.md`). To keep the record unambiguous this is registered as
**the κ certification**, not Block H.

## The argument to be verified

`Ĝ(a) = sup over (w₁,b₁) of G(w₁,b₁)` where
`G = max(min_O f_a(w₁x+b₁) − max_I f_a(w₁x+b₁), min_I − max_O)`.

Two reductions make the domain compact and the function Lipschitz:

1. **Compactness.** `f_a` is `2π`-periodic up to the linear term `t`, and only the
   fold near `t = π` can separate the classes, so the optimum satisfies
   `|w₁| ≤ W` and `|b₁ − π| ≤ B` for explicit `W, B`. Any placement outside cannot
   beat the best interior one. **To be verified numerically**, by showing `G` on the
   boundary shell is strictly below the interior maximum.
2. **Lipschitz constant.** `|∂G/∂w₁| ≤ (1+a)·max(|x|)` over the windows and
   `|∂G/∂b₁| ≤ (1+a)`, since `|f_a'| = |1 + a cos t| ≤ 1+a` and `G` is a max/min of
   compositions. So `G` is Lipschitz with `L = (1+a)·max(1, max|x|)` in the
   sup-norm on `(w₁,b₁)`.

Then a grid of spacing `h` certifies

    Ĝ_grid  ≤  Ĝ  ≤  Ĝ_grid + L·h/2

which is the same certification device already used for `solves()` in T57 (grid
margin against `L·h/2`), applied to the supremum instead of the sign.

## Registered predictions

**C-1 (the certificate closes).** With `h` refined until `L·h/2 < 0.001·Ĝ`, the
certified interval for each `a` has **relative width below 0.2%**.

> **Falsified** if the certificate cannot be driven below 0.2% at any `a ≤ 1.60`
> with `h ≥ 1e-6` (below that, float64 accumulation in the window sums becomes
> the limiting error and the argument needs a different treatment).

**C-2 (T50's interval survives).** The certified κ interval over `a ≤ 1.60`
**contains** T50's reported `[0.30544, 0.31543]`, and its endpoints move by
**less than 1%** relative.

> **Falsified** if the certified interval excludes either endpoint, or if any
> endpoint moves by more than 1%. Since a grid can only under-estimate `Ĝ`, the
> certified κ can only move **up**; a downward move would indicate a bug.

**C-3 (the compactness reduction is sound).** On the boundary shell
`|w₁| = W` or `|b₁ − π| = B`, `G` is **strictly below** the interior maximum at
every `a` tested, by a margin exceeding the certificate width.

> **Falsified** if any boundary point matches or exceeds the interior maximum,
> which would mean the domain was cut too small and `Ĝ` is not what was computed.

**C-4 (κ₀ inherits the certificate).** The scaling-limit constant
`K = 0.579454926` and `κ₀ = 0.307302` are certified by the same device applied to
`h(σ)`, whose Lipschitz constant on the relevant σ-range is explicit
(`|h'| = |−1 + σ²/2|`).

> Registered: `K`'s certified interval has relative width **below 0.1%** and
> contains the reported value.

## What this does and does not buy

It converts κ from a sampled estimate into an interval with a proof obligation
discharged numerically. It does **not** make κ constant — T50 already reports the
3.2% drift and that stands. The point is that the drift is now bounded away from
being an artifact of grid resolution, which after Block G's G4 episode is a real
concern rather than a formality.
