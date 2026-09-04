# A thin-target failure mode: what is ours, and what is measured versus inferred

Positioning for the paper. Written to be honest to the point of understatement.

## The object

A four-parameter network `w2*f_a(w1*x + b1) + b2` on `sign(|x| - 1)`, with
`f_a(t) = t + a*sin(t)`. Monotonic `f` (a <= 1) provably cannot solve it: a
monotone logit has at most one sign change, the task needs two. For every
a > 1 solutions exist. Between a = 1 and a ~ 1.35 they exist and are never
found: 0/200 SGD runs, exact basin fraction 0 through a = 1.25.

## The claim

A solution set can be **unreachable by gradient descent while being stable,
downhill-connected, nearby, and adequately-margined**. The obstruction is the
solution set's *thinness as a target*, which is a different quantity from the
loss surface's sharpness -- quantities that coincide in most settings and come
apart here.

## What excludes the standard accounts (all measured, `exclusion_table.md`)

| account | what it requires | measured value | status |
|---|---|---|---|
| Edge of stability (Ahn-Zhang-Sra Thm 1) | lambda_max * eta > 2 | **0.025-0.030** | 70x inside stable regime |
| Energy barrier | positive barrier to escape | **MEP exactly 0.000** | downhill-connected |
| Distance | solution far from init | **4.7-5.5 vs 10.2-17.7 for found** | nearer than the found ones |
| Margin | too small to be found | **5/20 found solutions have smaller margin** | populations overlap |

Each is a measurement on the same objects, and each could have come out the
other way. Two did come out the other way from our own registered
predictions: sharpness was predicted to separate the populations at 2 and did
not, and distance was predicted to be larger for zero-basin solutions and is
smaller.

## What is new here, stated strictly

1. **The exclusion table itself.** Four standard explanations ruled out by
   direct measurement on one set of objects. The EoS literature studies
   sharpness along trajectories; it does not exhibit a minimum that is stable
   by its own criterion and still unreachable.
2. **A zero-barrier demonstration.** MEP barriers exactly 0 at every a, for
   solutions never reached in 200+ runs. Energetic accounts are excluded by
   construction of the example, not by argument.
3. **A controlled family across an analytic expressivity boundary.** `a`
   interpolates from provably-impossible (a <= 1) through
   possible-but-unfound to routinely-found, with the fold depth known in
   closed form, D(a) = 2(sqrt(a^2-1) - arccos(1/a)) ~ (8/3)(a-1)^{3/2}.
4. **A proven necessary condition on solutions** (`fold1d_theorem.md`):
   |w2| >= 2m/(kappa*D(a)), kappa in [0.305, 0.328] set by task geometry, not
   the activation. Verified on 66 solvers recorded weeks before the theorem:
   0 violations, minimum slack 1.04x.
5. **Exhibits.** Explicit solutions at |w2| = 1 in the ordinary parameter
   region, verified on 500k+ points, with zero basin.
6. **Three-way manipulation of the gap**: initialization scale up and down
   (17/40 vs 0/400 at a = 1.02; GELU 3%/8%/17% dose-response), and step size
   (0/200 -> 83/200 at a = 1.25 with Adam lr 3e-2).

## What is not ours

- That GD cannot converge to minima sharper than 2/eta: Ahn-Zhang-Sra,
  Lee et al. via the Stable Manifold Theorem. **We do not use it; we measure
  that it does not apply.**
- The 2/eta threshold, progressive sharpening, the edge-of-stability
  phenomenon: Cohen et al.
- The idea that flat minima are preferentially found: long-standing.
- The topological expressivity results the project began from: Ren and Lim.

## What is measurement and what is inference

**Measurements** (each could have falsified the claim, several did falsify
our predictions):
- the exact-zero solve counts, at stated n, with exact-region checks;
- MEP and linear barriers;
- exact 4x4 Hessian eigenvalues;
- distances and margins in the table above;
- basin fractions from grid initializations and perturb-retrain;
- solve rates under initialization and step-size intervention;
- reached-|w2| distributions.

**Inferences, labelled as such**:
- that *no* standard account applies -- we checked four, which is not
  exhaustiveness;
- that the mechanism is thin-target capture -- supported by `reach` (T40) but
  **not derived**: the closed-form capture cross-section |w2|*G(a)/step was
  tested and **fails under step intervention** (ratio flat at 36-40 while the
  rate moves 0 -> 0.5, `capture_results.md`);
- that the 1D account transfers to the link setting -- argued from the shared
  fold structure, measured only in 1D.

**Known open**: we can predict findability from reached-|w2| plus the
theorem's requirement, but we cannot derive reached-|w2| from the optimizer's
hyperparameters. That is the missing law, and it is stated as missing.

## The honest one-sentence version

We exhibit minima that gradient descent does not reach although they are
stable by the standard criterion, separated from initialization by no barrier
and by less distance than the minima it does reach; we show by measurement
that the usual explanations do not apply, and we do not yet have a
quantitative law that replaces them.

> **Terminology correction (2026-08-28).** The objects called "zero-basin
> solutions" here are **not critical points of the loss** (gradient norm
> 0.02–0.27 versus training's terminal 0.0006–0.021; loss 0.22–0.69;
> λ_min < 0 at six of eight values). Read "solution" throughout as
> "correctly-classifying parameter vector", never as "minimum of the loss".
> The measurements are unaffected; what changes is what they are about. See
> `criticality_results.md` and `terminology_correction.md`. In particular the
> statement that Ahn–Zhang–Sra does not apply was tested on one of its two
> conditions only, and is corrected there.
