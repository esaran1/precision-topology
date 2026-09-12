# Registration: does the budget law hold in a standard setting?

**Written before any measurement in this task.** Nothing below was adjusted
after seeing data. Amendments, if any, are appended as dated blocks with the
original left intact.

Date: 2026-09-11.

---

## The question

Sections 3–6 establish, in a four-parameter task, that the boundary of what
gradient descent finds moves with training budget as `eps_onset ~ B^(-alpha/beta)`.
The claim is currently scoped to that task. This experiment asks whether the
same displacement occurs in a standard setting — MNIST, an MLP, Adam — using
our own activation family so that the fold parameter is continuous and its
threshold is known analytically.

## Architecture and family

`784 -> 256 -> [w] -> 128 -> 10`, ReLU everywhere except the **bottleneck
layer**, whose activation is `f_a(x) = x + a*sin(x)`.

- Monotone iff `a <= 1` (since `f_a'(x) = 1 + a*cos(x) >= 0` for all `x` iff
  `a <= 1`). Threshold **exactly `a = 1`**, analytically.
- Fold depth `D(a) ~ (a-1)^{3/2}` as `a -> 1+`, derived in
  `results/fold1d_theorem.md`.
- Therefore **`beta = 3/2`** for this family, the same value family A carries
  in the 1D task.

The bottleneck is the only place width is a single unambiguous number, which
is the reasoning that closed the earlier width-axis attempt (T39).

## Capability criterion — a choice, and registered as one

Real tasks have no exact-separation criterion, so capability is defined as
**reaching a fixed test-accuracy threshold within the budget**. The threshold
is chosen from a pilot (reported in full) under one rule fixed now:

> Pick the threshold so that at the narrow bottleneck, monotonic `a` fails and
> clearly non-monotonic `a` succeeds, with both margins visible in the pilot.

This is a choice and it is load-bearing. **Sensitivity is mandatory**: every
onset is reported at the registered threshold and at two others (one higher,
one lower), and if the onset ordering changes across them, that is reported as
the headline rather than the registered number.

## Part 2 controls — both must hold, failure conditions fixed now

**2a. Narrow bottleneck (`w` near ID): monotonic fails.**
Predicted: at `w ~ ID`, activations with `a <= 1` fail the capability
criterion at a rate consistent with 0, and some `a > 1` succeeds at >= 50%.

*Failure condition*: if monotonic `a` reaches the criterion at >= 50% at the
narrow bottleneck, **the setting does not exhibit the phenomenon and the
experiment stops.** That result would say real-data structure supplies what
the fold supplies in the toy task, and it would be reported as such.

**2b. Wide bottleneck (`w >> 2 x ID`): no `a`-dependence.**
Predicted: at wide `w`, every `a` including monotonic ones reaches the
criterion, so no onset exists.

*Failure condition*: if `a` still matters at a wide bottleneck, the effect is
about the activation in general rather than about the fold at a capability
boundary, **the mechanism connection is lost, and the main sweep is not run.**

This two-sided structure is what makes the experiment a test. 2a alone could
be produced by any activation-dependent difficulty; 2b is the prediction that
would be surprising if the effect were generic.

## Part 3 predictions

**3a. `alpha` is measured here, not assumed.** `alpha = 1.1172` was measured on
the 1D task and `alpha` is demonstrably setting-dependent — it is not even
range-stable within the 1D task (1.5109 over 1k-4k falling to 0.7193 over
40k-160k). Terminal bottleneck-layer weight scale is fit against budget over
>= 5 budgets spanning >= 2 orders of magnitude, with sub-range stability
reported as in `results/alpha_stability_results.md`.

**3b. Onset exponent.** With `alpha` measured here and `beta = 3/2`, the
predicted onset exponent is `-alpha/beta`. The band is derived from alpha's
fit uncertainty and the `a`-grid resolution **combined in quadrature**, and is
computed from those two numbers rather than rounded — the specific value is
registered in `results/mnist_alpha_registration.md` after 3a completes and
before any onset is measured.

**3c. Bracketing.** Identical to the 1D protocol: an onset is located only
when some `a` succeeds at >= 50% **and a strictly smaller `a` fails at < 50%**.
Unbracketed cells are reported as bounds and excluded from fits. **The
bracketed count is stated before any exponent.** >= 4 budgets spanning >= 64x.
Seeds >= 15 per cell, raised near the onset.

## Part 4 — what each outcome means, fixed in advance

1. **Onset moves at the predicted exponent** (within the registered band):
   the law holds in a standard setting on real data. The paper's scope claim
   changes from one toy task to a toy task and a standard one.
2. **Onset moves, exponent outside the band**: the displacement phenomenon
   transfers, the quantitative law does not. Reported as such — still
   substantial, since it places the effect outside the toy setting.
3. **Onset does not move** (flat within resolution across >= 64x budget): the
   budget-dependence is specific to the toy task. This bounds the paper's
   claim and is reported without softening.
4. **Either control fails**: the setting does not exhibit the phenomenon.
   Reported plainly; the paper stays scoped to the toy setting.

Outcomes 3 and 4 are reported with the same prominence as 1 and 2. The
abstract depends on which of these obtains, and a negative resolves it as
usefully as a positive.

## Scope conditions known in advance

- One dataset, one architecture family, one optimizer (Adam). The 1D law is
  already known to be Adam-specific; nothing here tests other optimizers.
- The capability criterion is a threshold on test accuracy, not an exact
  separation. It has no analogue of the 1D task's exact-zero criterion.
- ID estimators are expected to disagree; we use them to *choose a width*, not
  to test a prediction about ID, so a rough estimate suffices. The spread is
  reported either way.
