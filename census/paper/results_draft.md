# §3 The setting and the theorem

We study a one-dimensional classification task with a provable impossibility
side. The target is `sign(|x| - 1)`, sampled on two regions: the inner region
`|x| <= 0.8` (class 0) and the outer region `1.2 <= |x| <= 2.0` (class 1). The
network is a single hidden unit,

    N(x) = w2 * f(w1 x + b1) + b2,

four parameters in total. If `f` is monotone then `N` is monotone, so `N` has
at most one sign change; the task requires two. **No monotone activation can
solve this task at any parameter setting**, and the impossibility is one line.

We use two activation families indexed by a parameter that crosses an analytic
monotonicity threshold. Family A is `f_a(t) = t + a sin t`, monotone for
`a <= 1` and non-monotone for `a > 1`. Family B is `f_alpha(t) = max(t, alpha t)`,
monotone for `alpha >= 0`. Above threshold both develop a fold, and the fold's
depth `D(a) = f_a(t_max) - f_a(t_min)` vanishes as the threshold is approached.

**Theorem.** Let `N` solve the task with logit margin `m > 0`, meaning
`N <= -m` on the inner region and `N >= +m` on the outer region. Then

    |w2| >= 2m / G(w1, b1) >= 2m / (kappa * D(a)),

where `G(w1, b1)` is the class gap realized by that traversal, and
`kappa = G*/D` with `G*` the maximum gap over all `(w1, b1)`.

The bound is stated in terms of `D(a)`. In the limit `a -> 1+`,
`D(a) ~ (8/3)(a-1)^{3/2}`; **we state this asymptotic separately because it is
inaccurate over the range we measure**, overstating `D` by 42.7% at `a = 1.02`
and by 136.1% at `a = 3.0`. All computations use `D(a)` exactly. The local
logarithmic slope of `D` over the range our onsets span (`eps = 0.03` to
`0.60`) is **1.4364**, so the asymptotic exponent 3/2 describes the measured
range to 4.2% even though its prefactor does not.

Two features of the bound are worth stating precisely. `b2` cancels in the
derivation, so the bound holds for the optimal `b2` and no choice of `b2`
evades it. And `w1` cannot substitute for `w2`: `w1` enters only through `G`,
which is bounded above by `kappa * D(a)` however `w1` is chosen.

**`kappa` is a numerically obtained constant, not a derived one.** We prove
that it depends only on the window *ratios* and not their absolute scale --
rescaling all three window edges by 2x changes it by 0.2% (0.3152 to 0.3147) --
so `kappa = kappa(r1, r2)` is a property of the task geometry rather than of
the activation. We did not obtain a closed form; the natural quadratic-dip
model gives 0.278 against a measured 0.315. Over `a` in `[1.02, 3.0]`,
`kappa` in `[0.305, 0.328]`, computed on a 33x400 grid in `(b1, w1)`.
**No exponent in this paper depends on `kappa`**: it scales the required-`|w2|`
prefactor and cancels from `beta`, from `alpha`, and from the onset exponent.
Every downstream conclusion holds unchanged for `kappa` anywhere in
`[0.1, 1.0]`, a tenfold range.

**Verification.** We re-derived 66 solving parameter vectors from seeds
recorded weeks before the theorem was written, for the unrelated purpose of
locating the findability onset, and checked each against the bound using its
own measured margin and the measured `G*(a)`. **Zero violations in 66, with
minimum slack 1.0394.** What this verifies requires care. The bound uses
`G*(a)`, the maximum over all traversals, while a solver achieves
`G(w1,b1) <= G*`; a solver placing its fold poorly would have `G << G*` and
would violate the bound at small `|w2|`. Minimum slack of 1.04 shows solvers
sit close to the constraint rather than trivially far from it, so **the
content of the verification is that trained networks place their folds
near-optimally.** It does not connect the theorem to our empirical onsets: our
empirical criterion is sign-correctness, the `m -> 0` limit, where the bound
degenerates to `|w2| >= 0`. Redefining separation at fixed `m > 0` leaves the
budget law unchanged (§5), which is how the two are reconciled.

# §4 Correctly-classifying regions that gradient descent does not reach

Above the analytic threshold, solutions exist at every `a > 1`. We exhibit
them explicitly: at `a = 1.02` a network with `|w2| = 1` classifies both
regions correctly, verified on 500,000 points, with every coordinate inside
`[-5, 5]`. Training at that value succeeds in **0 of 200 runs** at the
standard budget.

These points are **not minima of the loss**. Their gradient norms are 0.019 to
0.272, which is **20 to 227 times** the gradient norm training itself has when
it terminates at the same `a` (0.0006 to 0.021), and their loss is 0.22 to
0.69 against 0.02 to 0.24 at found solutions. `lambda_min` is negative at six
of the eight values tested, so they are saddles. Throughout this paper,
**"solution" means a parameter vector that classifies both regions correctly**,
never a minimum of the training loss; the two coincide for found solutions and
do not coincide for these.

Five standard explanations for gradient descent failing to reach a point are
excluded by direct measurement on the same objects.

**Energy barrier.** Minimum-energy-path barriers, computed by a string method
from 5 initializations per value, are **exactly 0.000** at every `a`. This
establishes that **the landscape admits a barrier-free path** from typical
initializations to these solutions, which excludes an energetic explanation.
It does not establish that the trajectory gradient descent actually follows is
barrier-free; the string method finds *a* low-barrier path, not *the* path
training takes.

**Sharpness.** Exact 4x4 Hessians give `lambda_max * eta_eff` of 0.025 to
0.030 at these points and 0.0005 to 0.0036 at found solutions, against the
edge-of-stability threshold of 2 -- between 70 and 4000 times inside the
stable regime. **We had registered the prediction that this quantity would
separate the two populations at 2; it does not**, and neither population is
near the threshold. The Ahn-Zhang-Sra result does not apply here for a more
basic reason than either of its eigenvalue conditions: it concerns stationary
points, and these points are not stationary.

**Distance.** These solutions sit 4.7 to 5.5 from typical initialization in
parameter norm, while found solutions sit 10.2 to 17.7 away -- **the unreached
solutions are the nearer ones**. Because Euclidean distance in raw coordinates
is not the metric Adam moves in, we recomputed under Adam's own preconditioner,
the bias-corrected square root of the accumulated second moment measured from
real trajectories: found-to-unreached distance ratios are **1.72 to 3.79**
against Euclidean 1.85 to 3.50. The reversal survives in the optimizer's own
metric. **We had registered the opposite prediction**, that unreached
solutions would be the more distant ones.

**Margin.** At `a = 1.5` the unreached solution has logit margin 0.0384, and
**5 of 20 found solutions have a smaller margin than that** (minimum 0.0038).
Destination margin does not separate the populations. Margin along the
approach is a different quantity and is untested.

**Criticality.** The gradient-norm comparison above is between two solution
populations rather than against an arbitrary baseline: found solutions sit at
0.006 to 0.014, at the same scale as training's terminal gradient norm, while
the unreached ones sit at 0.019 to 0.272. An independent finite-difference
implementation reproduces the autograd gradient norms to **ten significant
figures** and agrees on the sign of `lambda_min` at all six negative values.

# §5 The budget law

**Onsets are population thresholds.** For each budget we locate the smallest
grid value of `a` at which at least 50% of 40 independent runs produce a
correctly-classifying network, requiring a strictly smaller grid value below
50%; unbracketed cells are reported as bounds and excluded from fits. Claims
in this section are about **where the population threshold sits**, not about
how far any individual run travels.

**The effective boundary moves with budget.** Over budgets 2,000 to 128,000 --
a 64x range, with **6 of 6 cells bracketed** -- the onset moves from
`a = 1.60` to `a = 1.03`. The fitted exponent is **-0.7340** against a
prediction of **-0.7448 registered before any onset was measured**, with an
acceptance band of `[-0.895, -0.595]` derived from the measured `alpha` and
the grid resolution rather than chosen. Sampling uncertainty was bootstrapped
by resampling seeds within each cell and redetermining every onset through the
bracketing rule: **95% interval `[-0.761, -0.707]`, half-width 0.027**, which
is 0.24x the grid-resolution term of 0.114 that dominates it. The onset
estimator is stable because bracket sharpness is high in most cells; the
exception is the 2,000-step cell, where the rates straddling the threshold are
0.550 and 0.450 and an independently written reimplementation located the
onset one grid step away.

**The clearest single statement requires no fit.** At `a = 1.25`, the solve
rate is **0 of 200 at the standard 2,000-step budget and 40 of 40 at 80,000
and at 160,000 steps** -- the same task, architecture and analytic threshold,
with one hyperparameter changed.

**Compute cost.** Inverting the onset exponent, the budget at which a majority
of runs succeed within `eps` of the analytic threshold grows as

    B(eps) ~ eps^{-1.36},   interval [1.151, 1.670],

so halving `eps` costs **2.57x** compute (2.22x to 3.18x). **This is a
median**: at the onset budget, half of runs still fail, and we have not
measured other quantiles. Measuring along the other axis -- bisecting on
budget at fixed `a`, 5 of 5 cells bracketed -- gives **-1.0242**, consistent in
sign and order of magnitude; that check has resolution `+-0.431`, wider than
the band it tests, so it cannot adjudicate and we report it as a consistency
check only.

**Asymptotics.** As `B -> infinity`, `eps_onset -> 0`: the gap closes in the
infinite-compute limit and the analytic boundary is recovered. **The
displacement is a finite-budget claim, not a permanent gap.** Further,
`alpha` is not range-stable: fitted over every contiguous sub-range it falls
monotonically from **1.5123** (1k-4k) to **0.7193** (40k-160k) with `R^2 >=
0.983` throughout, which is local power-law behaviour with a drifting exponent
-- the signature of a sub-power-law process. **Extrapolation of `B(eps)`
beyond the measured range of 160,000 steps is unsupported.**

# §6 The geometric relationship, and its scope

**Construction.** We build activation families with `f'(x) = (1-a) + a|x|^q`
near the origin, integrated in closed form, giving fold-depth exponent
`beta = 1 + 1/q` analytically. Each `beta` is verified numerically against its
derivation to within 0.7%. As a control, `q = 2` reproduces family A's
`beta = 1.4959` against the sin family's 1.4963, with a constant prefactor
ratio of `sqrt(2)` across three decades -- same geometry, different function.

**The relationship.** Predicted onset exponents `-alpha/beta` were registered
before any of these families was trained, using `alpha = 1.1173` measured on a
separate experiment and never fitted to them. With 3 to 4 bracketed cells per
family:

| family | beta | predicted | measured |
|---|---|---|---|
| q4 | 1.25 | -0.8938 | **-0.8305** |
| q2 | 1.50 | -0.7449 | **-0.6749** |
| family A (independent) | 1.50 | -0.7449 | **-0.7340** |
| q1 | 2.00 | -0.5586 | **-0.6521** |
| q0.667 | 2.50 | -0.4470 | **-0.5000** |

Every measured exponent lies within its propagated interval. Plotted against
`1/beta`, the through-origin slope is **1.1240 with 95% interval
`[1.008, 1.240]`**, against the independently measured `alpha = 1.1173` with
interval `[0.999, 1.236]`: **the intervals overlap**. The point estimates
differ by 0.6%, but both carry intervals of roughly `+-10%`, so this is
agreement within uncertainty and not a sub-percent measurement.

**Sensitivity to the choice of `beta`.** Each `beta` above is the analytic
`a -> 1+` exponent. Over the `eps` range each family's onsets actually span,
the effective local exponent is smaller -- by 1.7% (q4) to 10.4% (q0.667) --
and substituting it moves the through-origin slope to **1.0547**, which is
5.6% from `alpha` rather than 1.7%. **Both values lie inside `alpha`'s
confidence interval and the conclusion is unchanged, but the headline
agreement is 1.7% under asymptotic `beta` and 5.6% under effective `beta`.**
We use asymptotic `beta` because it is analytically defined, so `1/beta`
carries no measurement error; using the effective value would make the slope a
regression of two measured quantities. Per-family, the effective value
improves agreement for q1 and q0.667 and worsens it for q4, q2 and family A.

**A counterexample.** Family B has a diverging requirement -- its activation
is positively homogeneous, so the amplification measure is `|w1 w2|` rather
than `|w2|`, and the maximum class gap is `0.4|alpha|` exactly, giving
`beta = 1`, the steepest of any family here. The law therefore predicts the
largest onset exponent in magnitude, `-1.1173`. **Measured over a matched 64x
budget range, family B's onset exponent is `+0.25`** -- the opposite sign.
Training exceeds its required amplification by 900 to 1,200 times and the
solve rate *falls* with budget. Family B is an unexplained counterexample, not
a limiting case.

**Scope: this is a single-optimizer result.** We tested transfer to plain SGD
three ways and all three failed or proved unmeasurable. The dynamical form
fails quantitatively: the exponent ratio between optimizers is **0.447**
against the `alpha` ratio **0.643**, a 30% gap on the quantity where grid
resolution partially cancels. The ratio test, in which `alpha` cancels
exactly, **could not execute**: SGD never solves the `beta = 2.5` family at
any `eps` or budget tested, and we do not pursue it to larger `eps` where the
near-threshold expansion defining `beta` may not hold. The same-family
comparison fails decisively: at `beta = 1.25`, with 4 of 4 cells bracketed on
a refined grid, SGD's onset exponent is **`+0.0056`, bounded `|exponent| <
0.0185`, against Adam's `-0.8305`** -- at least 45 times smaller in magnitude,
where the law predicts this family should have the *steepest* exponent under
either optimizer. Family A does move under SGD (`-0.3255`), so SGD is not
globally static.

Taken together these are one finding rather than three caveats: **the
relationship is real under Adam, and every attempt to extend it beyond Adam
has either failed or proved unmeasurable.** Whatever governs the
geometry-to-onset relationship is not optimizer-independent.
