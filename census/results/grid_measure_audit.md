# Audit of every claim resting on grid-derived measures (2026-08-25)

Follow-up to `box_emptiness_correction.md`. Since
`fold1d_geometry.csv`'s `solution_fraction` is a lower bound at grid
resolution rather than a measure, every claim built on it needed
re-deriving. Generator: `src/solution_measure.py`.

## The replacement instrument

The solution set has analytic structure that makes gridding
unnecessary. For fixed (w₁, b₁, w₂), the set of admissible b₂ is an
**interval** — for w₂ > 0 it is (−w₂·min f(outer), −w₂·max f(inner)),
non-empty exactly when gap(w₁,b₁) = min f(outer) − max f(inner) > 0 —
of width |w₂|·gap. So the exact measure is

  μ(a) = (2B)⁻⁴ ∫∫∫ |w₂| · [gap(w₁,b₁)]⁺ dw₁ db₁ dw₂
       = (2B)⁻² · (B²/2) · E_{w₁,b₁}[gap⁺] ,

with the b₂ and w₂ integrals done analytically and only the (w₁, b₁)
expectation sampled (20,000 draws, dense inner/outer grids). This is a
Monte-Carlo estimate of an exact expression, not a scan: it has no
resolution floor.

## Result: the grid undercounts ~10–19× everywhere, and the shape holds

| a | true measure (B = 5) | grid-41 value | ratio |
|---|---|---|---|
| 1.05 | 7.59e−7 | **0** | ∞ |
| 1.10 | 4.07e−6 | **0** | ∞ |
| 1.25 | 3.74e−5 | 3e−6 | 12.5× |
| 1.35 | 7.50e−5 | 4e−6 | 18.7× |
| 1.50 | 1.38e−4 | 8e−6 | 17.2× |
| 2.00 | 6.27e−4 | 4.8e−5 | 13.1× |
| 3.00 | 1.85e−3 | 1.87e−4 | 9.9× |

The undercount factor is not constant — it shrinks from ~19× to ~10× as
`a` grows, exactly as the resolution account predicts (detectability
improves as gap widens). **So the concern was well founded: part of the
grid's apparent growth was its own resolution improving.**

## The 2a claim, re-derived: it survives, on better evidence

"Solution measure grows smoothly from the threshold while basin volume
jumps at the onset" is **borne out by the exact measure**, and more
cleanly than by the grid:

- The true measure is **strictly monotone in a** (Spearman 1.0) and
  **positive at every a > 1**, including 1.05 and 1.10 where the grid
  read zero. Smooth growth from the threshold is now a statement about
  the geometry, not about detectability.
- Consecutive growth factors are modest and even: ×5.4, ×9.2, ×2.0,
  ×1.8, ×4.6, ×3.0 — no discontinuity anywhere.
- Across the findability onset (1.25 → 1.35 → 1.50) the true measure
  grows ×2.0 then ×1.8, while **basin fraction goes 0 → 0.4% → 5.2%**
  (×14 then ×10 where defined, from an exact zero). The separation
  between the two quantities is *larger* under the corrected
  instrument, because the corrected measure is smoother.

The registered prediction is therefore unaffected in direction and
strengthened in support. What changes is the wording: the measure never
"turns on" at some a — it is positive immediately above the analytic
threshold and grows steadily, which is what "possible everywhere past
threshold, findable only later" requires.

## Other claims audited

| claim using grid measures | verdict |
|---|---|
| "Solution sets are thin sheets (inradius < 1 cell)" | **stands, strengthened** — the sheets are thinner than the grid could resolve; the b₂ interval width |w₂|·gap is now known analytically (5.5e−4 at a = 1.02, w₂ = 1) |
| "components 8 → 12 → 24 → 64 → 80" | **withdrawn as a count of components** — these are counts of *grid-detected* clusters, which change with resolution; the true set's component structure was never measured |
| "basin-to-solution ratios 400–10,000×" | **withdrawn** — numerator (SGD-measured) and denominator (grid-measured) are not commensurable; recomputed against the true measure the ratios are ~10× smaller and still large, but the quantity is not a clean one and is not needed by any claim |
| "basin volume jumps at the onset" | **unaffected** — basins were measured by SGD from 6,561 initializations, never from the solution grid |
| "solutions with zero basins at a = 1.25–1.35" | **unaffected and strengthened** — see `box_emptiness_correction.md` and the restated version in T30 |
| pwl / GELU rows of the geometry table | **downgraded to lower bounds**; family B's larger measure at reachable scales is preserved in ordering (its gap function is far wider), but the numbers are grid values |

## What this does not touch

Nothing in the findability strand: solve rates, onset location, basin
measurements, and the provable monotonic zero are all training or
SGD measurements with no grid dependence.
