# Registered predictions: the monotonicity threshold sweep

Written **before** any implementation or training. Predictions are never
revised; outcomes are appended in marked sections or reported in
`threshold_results.md`.

## Why this experiment exists

Every monotonic-versus-non-monotonic comparison in this project rests on four
fixed activations: three monotonic (tanh, ReLU, leaky-ReLU) and **one**
non-monotonic (GELU). The monotonic zero — 0 separations in 3,330 width-3 runs
— is a claim about a category supported on the non-monotonic side by a single
member. Attributing the difference to monotonicity rather than to something
else GELU happens to have is an interpretation with n = 1 support.

This sweep replaces the categorical comparison with a **critical-point
prediction**: two continuously parametrized activation families, each with an
analytically known monotonicity threshold, asked whether separation ability
changes at exactly that threshold.

## The two families

**Family A (smooth):** `f_a(x) = x + a·sin(x)`. Derivative `1 + a·cos(x)`,
non-negative everywhere iff `a ≤ 1`. **Threshold: a = 1**, analytic, no
fitting. At `a = 1` the derivative touches zero only at isolated points
(`x = π + 2πk`), so the function is still monotonic (non-decreasing, in fact
strictly increasing). At `a = 0` the function is the identity, so the network
is affine end-to-end.

**Family B (piecewise linear):** `g_α(x) = x` for `x ≥ 0`, `αx` for `x < 0`.
Strictly monotonic for `α > 0`, ReLU at `α = 0` (monotonic, non-decreasing),
non-monotonic (V-shaped) for `α < 0`. **Threshold: α = 0.** At `α = 1` the
function is the identity — the same affine-network anchor as Family A's
`a = 0`, reached from the other end.

The families share nothing but the property under test: A is smooth, bounded
derivative oscillation, infinitely many turning points once `a > 1`; B is
piecewise linear with a single kink. If separation turns on at the analytic
threshold in **both**, monotonicity is the operative property. If it turns on
elsewhere, or at inconsistent relative positions, it is not.

## Grid

- Family A: `a ∈ {0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0}` (13 values, dense near 1)
- Family B: `α ∈ {−1.0, −0.5, −0.25, −0.1, −0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0}` (11 values, dense near 0)
- Baselines in the same sweep, same seeds: tanh, ReLU, leaky-ReLU, GELU
- Widths 3 and 4, depths 3/5/8/12, seeds 0–19, baseline linked tori,
  our protocol (full-batch Adam, 2,000 steps, lr 1e-2, 1,000 points per class)

Total: 28 conditions × 2 widths × 4 depths × 20 seeds = **4,480 runs**.

## Predictions

**P1 (the flagship).** At width 3, the separation count is **exactly zero for
every monotonic parameter value** — Family A `a ≤ 1.0` inclusive, Family B
`α ≥ 0.0` inclusive — and **positive for at least some non-monotonic values**
in each family. The transition therefore lies in `(1.0, 1.05]` for Family A
and `[−0.05, 0.0)` for Family B, at the resolution of this grid.

**P2 (asymmetric sharpness).** The sharp edge is on the **monotonic side**.
Theory (Theorem 4.7) forbids separation for monotonic activations at width 3;
it does not promise that SGD exploits an arbitrarily small non-monotonicity.
So we predict an exact zero everywhere on the monotonic side, but on the
non-monotonic side the rate may rise **gradually** with distance from the
threshold — `a = 1.05` may separate rarely or never in 160 runs while
`a = 3.0` separates often. A zero at `a = 1.05` with positives at larger `a`
does **not** count against the barrier reading; a positive at `a ≤ 1.0` or
`α ≥ 0.0` **overturns it**, and takes the 3,330-run zero down with it.

**P3 (floor structure).** The error-floor structure seen throughout this
project — monotonic runs never entering the low-error band, GELU populating
0 through 15 — should appear as a function of the parameter: monotonic-side
values show a hard floor (no runs below ~9–16 errors on this link), and
non-monotonic-side values populate the band below it.

**P4 (width 4).** At width 4 every parameter value with meaningful
nonlinearity separates at some rate, and the threshold structure visible at
width 3 is absent or strongly attenuated. The affine anchors (`a = 0`,
`α = 1`) never separate at any width, since the end-to-end map is affine and
the classes are not linearly separable.

**P5 (the two families agree).** The transition sits at the analytic
threshold in both families — not at a common value of some other quantity
(amplitude of nonlinearity, derivative range) that happens to differ between
them.

## What would count against the monotonicity reading

Stated in advance, so it cannot be rationalised afterwards:

- **Any separation at a monotonic parameter value** (Family A `a ≤ 1`,
  Family B `α ≥ 0`, width 3). This is the overturn condition. It must be
  reported immediately, prominently, and unsoftened, and it invalidates the
  central claim of the project, not just this sweep.
- **A transition located away from the threshold in either family** — for
  instance, separations beginning only at `a ≥ 2`, while Family B separates
  immediately at `α = −0.05` — would suggest the operative variable is the
  *magnitude* of non-monotonicity (depth of the dip), not its presence,
  which is a different and weaker claim than the theorem's.
- **Inconsistent relative positions between families** — e.g. Family A
  turning on exactly at its threshold while Family B stays at zero deep into
  `α < 0` — would suggest smoothness, curvature, or gradient scale is doing
  the work.
- **A transition coinciding with an optimization discontinuity** — if final
  loss, gradient norms, or dead-unit rates jump at the same grid point where
  separation turns on, the transition may be about trainability rather than
  expressivity, and must be reported as confounded.

## Known confounds, stated before running

**Effective nonlinearity scales with the parameter.** In Family A, `a` past 1
increases amplitude and curvature as well as introducing non-monotonicity. If
separation turns on at `a = 1` in Family A alone, monotonicity and
nonlinearity-strength are confounded. Family B is the control: crossing
`α = 0` changes monotonicity while changing the function's scale hardly at
all (`|g_α|` on the negative side goes from `0.05|x|` to `−0.05|x|`). The
joint pattern is the evidence; neither family alone suffices.

**Gradient scale changes with the parameter.** Family A's derivative range is
`[1−a, 1+a]`; Family B's negative-side derivative is `α`. We record final
train loss, final-step gradient norm, and inactive-unit fraction per run, and
will report whether any of these is discontinuous at the threshold.

**Degenerate anchors.** `a = 0` and `α = 1` are affine networks and serve as
in-sweep correctness checks: if either ever separates, the harness is broken
and every other number in the sweep is suspect.

## What is measured

Primary: **separation count (eval errors = 0) per parameter value** at width
3, with exact binomial confidence intervals. Then: full error distributions
per value (median, quartiles, band counts, per the standing rules); the
transition interval, stated at grid resolution with no interpolation; width-4
rates; final loss / gradient norm / inactive-unit fraction across the
parameter, checked for discontinuities at the threshold.

Per `notes/reporting_rules.md`: counts of exact zeros are the primary
statistic; any minimum appears with its distribution; no correlation between
quantities that share a definition.
