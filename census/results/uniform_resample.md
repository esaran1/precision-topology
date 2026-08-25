# 1f: the sampler's non-uniformity, quantified, and the volume-uniform re-check

Generator: `src/uniform_resample.py`. Data: `uniform_resample.csv`.

## The density, exactly

`linked_tori` draws the tube angle θ uniformly and cross-sections
area-uniformly. The solid-torus volume element is
dV = ρ·(R + ρ·cosφ)·dρ·dφ·dθ, so the sampler's density per unit volume
is

  p(x) / p_uniform = R / (R + ρ·cosφ),

a smooth function of position only through ρ·cosφ (signed distance of
the point's tube offset toward the outside of the torus). At
r/R = 0.2: **+25% at the inner equator, −17% at the outer equator,
max/min ratio exactly 1.5**, bounded below at 0.833× uniform
everywhere. Verified empirically: mean ρ·cosφ = −0.0003 ± 0.0003
(theory 0) for the sampler, +0.0106 (theory r²/4R = 0.01) for the
volume-uniform version.

## What regional separation means under this sample

A zero-error result on n area-uniform points implies, for any
misclassified region of volume fraction v, that (1 − 0.833·v)ⁿ ≥
(observed zero's likelihood) — i.e. the usual coverage argument holds
with the constant 0.833 in place of 1. **Zeros lose at most a factor
1.2 in effective n.** Error *rates* (nonzero counts) are measured under
the ±25% tilt and are the quantity that could shift under a uniform
sampler.

## Volume-uniform re-verification (rejection sampling, exact)

- **84 baseline-geometry width-3 dense survivors** (width, threshold,
  protocol sweeps): **79 at exactly 0 errors** on fresh 100k
  volume-uniform points; 5 show 1–2 errors. The paired control column
  resolves those five: sin(1.5) d5 s19 and pwl(−0.25) d8 s4 show the
  *same* 1–2 errors on a fresh **area-uniform** control sample — the
  flip-prone band near 1-in-100k margins already documented in
  `dense_check.md` — and the other three (2 GELU, 1 sin) are 1–2-in-
  100k events with clean controls. No run shows an error excess
  attributable to sampler shape.
- **The four interval-defining bisection runs** (sin 1.09 d5 s4,
  sin 1.10 d5 s4 and d8 s8, pwl −0.25 d8 s5): **0 errors each** —
  T5/T22's transition intervals do not move under volume-uniform
  sampling.
- **Both exhibits at 1,000,000 volume-uniform points each**: Part 2a
  witness **0 errors**; a = 1.02 offset witness **0 errors**.

## Scope

Parametrization- and corrugation-sweep survivors are **excluded by
design** here: their links are deformed geometries and this module's
sampler generates baseline tori (a first draft evaluated them anyway
and produced garbage counts — discarded, and the lesson is recorded in
the module docstring). A volume-uniform check for the deformed families
needs per-family Jacobians; until then their dense results carry the
quantified area-uniform caveat above, which bounds the exposure at a
1.2× effective-n factor on zeros.

## Which claims would change under a uniform sampler

None found. Zeros (T1 strata, T5/T22 intervals, T7 exhibits) are
either re-verified directly above or protected by the 0.833 coverage
bound; rate-valued claims (T8's 56/163, dense-verified rates) could in
principle shift within ±25% per-point weighting, but the five marginal
runs found here are consistent with the already-documented flip-prone
band rather than with any directional sampler effect.
