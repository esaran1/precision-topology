# Width sweep at d = 3: design proposal

Status: **approved with three changes**, recorded below. The accuracy and
pass-rate half is approved to run; the linking-number half remains blocked on
estimator validation.

Approved changes to the original proposal:

1. **Activations.** Keep tanh, ReLU, and leaky-ReLU, but label them a
   *robustness check within one expressivity class*, not a comparison. Add
   **GELU** as a fourth.
2. **Split the work.** Run the accuracy and pass-rate half now; it needs no
   estimator. Estimator validation blocks only the linking-number half.
3. **Trimodality is a primary outcome**, not a caveat. Report distributions,
   never means alone.

## Framing

The paper's `R^3` evidence varies **depth** at fixed width 3 (Tables 2, 3, 8;
Appendix G.2: "All architectures use width 3"). This sweep varies **width** at
`d = 3`. The two are complementary axes, and the paper holds constant exactly
the variable we would move. This sweep would therefore be the **first width
evidence at `d = 3` in either project**, not a replication. Nothing in the paper
predicts its outcome at widths 4-8, because the paper never ran it. Any writeup
must say so rather than presenting agreement or disagreement as reproduction.

## 1. Width range

Two sources disagree about how the obstruction is relieved as width grows.

| Source | Scaling | Evidence | Applies to |
|---|---|---|---|
| Junyu (author, beyond paper) | additive, `~<3+5` | not published | 3D links specifically |
| Table 7 | multiplicative, saturating `~5d` | `R^7`, `S^3 ⊔ S^3`, `k=10`, 15 seeds | `d=7`, different link type |

**Primary: Junyu's `~<3+5`.** It is specifically about `d = 3` and 3D links,
which is our exact setting, and it comes from an author of the theory. Table 7
is a different ambient dimension, a different link type (`S^3 ⊔ S^3` rather than
two circles), and a different copy count (`k = 10` rather than 1). Transporting
its scaling to `d = 3` is the error corrected in
`icml_paper_reconciliation.md`; I will not repeat it by designing around it.

**But the range should extend past 8, because that is precisely where the two
sources diverge.** At `d = 3`, additive `3+5 = 8` and multiplicative `5d = 15`
make different predictions, and the gap between them is the open question. A
sweep stopping at 8 cannot distinguish them: it would show behaviour consistent
with the additive account without ever testing the alternative. Extending to 15
costs little (see runtime below) and converts an assumption into a measurement.

**Proposed widths: 3, 4, 5, 6, 7, 8, 10, 12, 15.**

Rationale: 3 is the theorem-obstructed width; 4 is theoretically sufficient
(Theorem D.1); 5-8 is the additive transition region; 10, 12, 15 test whether
anything is still changing out to `5d`. We already have width 5, 15, 30, and 50
data, so 15 also ties the new sweep to the existing census at a shared point,
which is a useful consistency check.

Note the existing data already bears on this. At width 5, the linked-tori pass
rate is **50%** (30 of 60 runs), and ReLU eval accuracy at width 5 is trimodal:
four runs at exactly 0.5000 (chance), fourteen clustered at 0.959-0.971, two at
1.0000. That is not a smooth ceiling of the kind Table 2 reports for width-3
ReLU (~90-92%); it looks like optimization failure coexisting with success. Any
width sweep must report the full distribution per width, not a mean, or this
structure will be averaged away.

## 2. Seed count

The paper uses 30 seeds per cell for the `R^3` tables and 15 for Table 7. We
used 5.

**Runtime is not the binding constraint.** Measured on this machine: 1.50 s per
run at width 5, 4.73 s at width 50, so the narrow widths in this sweep are all
under ~2 s. For 9 widths x 10 seeds x 1 depth x 3 activations that is roughly
9 x 10 x 3 x 2 s ≈ 9 minutes. Even 30 seeds crossed with 4 depths is well under
two hours. The full original 480-run census took ~15 minutes.

**Proposed: 20 seeds**, with 30 if depth is crossed only lightly.

What that buys, using the observed seed-level SD of final-layer margin
(2.4202 steps, averaged over widths):

| Seeds | SE | ~2 SE resolvable difference |
|---:|---:|---:|
| 5 | 1.0823 | 2.1647 steps |
| 10 | 0.7653 | 1.5307 steps |
| 15 | 0.6249 | 1.2498 steps |
| 20 | 0.5412 | 1.0823 steps |
| 30 | 0.4419 | 0.8837 steps |

The cost of 5 seeds is that only differences above ~2.2 steps are resolvable,
which is large relative to the width-invariance effect we observed (12.30 to
13.52 steps across a tenfold width change — a difference of 1.22 steps, *below*
what 5 seeds can resolve, and one reason that observation was flagged as open
rather than claimed). 20 seeds brings the resolvable difference to ~1.08 steps,
just under that effect. This matters more than usual here because a plausible
outcome of the sweep is a *small* width effect, and 5 seeds cannot tell a small
effect from none.

Pass/fail rate needs seeds even more than margin does: at width 5 we observed
50% pass, so estimating a pass rate to within ~10 percentage points needs on the
order of 20-30 seeds per cell.

## 3. Depth: crossed, not fixed

**Argue for crossed.** The paper's strongest `R^3` evidence is the depth axis
(Table 2 spans depths 3-20 at 30 seeds; Table 3 spans 3-8), and its most
striking reported result is that plain ReLU mean accuracy *degrades* with depth
(84.3% at depth 3 falling to 50.3% at depth 20) while ResNet stays at 96.6-98.5%
throughout. Fixing depth would discard the axis on which their evidence is
strongest, and would make our result non-comparable to Tables 2 and 3 at the one
place where comparison is available.

There is also a specific confound to avoid. If we fix depth and observe a width
effect, we cannot tell whether the effect is width or an interaction between
width and the fixed depth. The existing census has this problem in mild form:
its width effect is pooled over depths 4, 6, 8, 10 without a crossed analysis.

**Proposed depths: 3, 5, 8, 12** — a subset of Table 2's grid, chosen to overlap
their reported values so the depth trend can be placed alongside theirs, while
keeping the cell count manageable.

Full grid: 9 widths x 4 depths x 20 seeds = 720 runs per activation. At ~2 s per
run that is ~24 minutes per activation, ~1.2 hours for three. Acceptable.

## 3b. Activations: three monotonic plus GELU

**tanh, ReLU, and leaky-ReLU are a robustness check, not a comparison.** All
three are continuous and coordinate-wise monotonic, so all three satisfy the
same hypothesis of Theorem 4.7 and occupy one expressivity class. Differences
among them cannot bear on the paper's ordering. Retaining all three tests
whether a width effect is robust across activation choice *within* that class,
which is worth knowing but must not be described as an expressivity comparison.

**GELU is added as the fourth**, for two reasons. It is non-monotonic, so it is
the only activation in this sweep that can speak to the ordering at all. And it
is the paper's own Table 2 comparison (ReLU vs GELU on the Hopf link, 30 seeds
per cell, depths 3-20), so our depths 3, 5, 8, and 12 overlap their grid and the
result is checkable against published numbers. Their reported GELU means are
89.3, 90.0, 91.1, and 91.2 at depths 3, 5, 8, and 12 with maxima of 92.9, 100.0,
100.0, and 100.0; their ReLU means are 84.3, 77.1, 63.1, and 57.7 with maxima
92.8, 92.5, 92.6, and 91.6. Those are at width 3, which is the one width where
our sweep and their table meet.

**Confound, recorded explicitly.** GELU is non-monotonic *and* unbounded. tanh
is monotonic *and* bounded. A GELU-versus-tanh difference therefore conflates
monotonicity with boundedness and cannot separate them. Isolating boundedness
would require a bounded non-monotonic activation, which does not exist as a
standard choice; constructing one (for instance a clipped GELU or a bounded
sinusoidal fold) would be a deliberate departure from standard practice and is
not proposed here. The cleanest available contrast is GELU versus ReLU, since
both are unbounded and they differ only in monotonicity — and that is exactly
the contrast the paper's Table 2 reports, which is a further reason to include
GELU rather than a bounded alternative.

Activation summary for this sweep:

| Activation | Monotonic | Bounded | Role |
|---|---|---|---|
| tanh | yes | yes | robustness within class; existing census comparator |
| ReLU | yes | no | robustness within class; paper's Table 2 baseline |
| leaky-ReLU | yes | no | robustness within class |
| GELU | **no** | no | tests the ordering; paper's Table 2 contrast |

## 4. Primary measured quantity

**Honest answer: we do not currently have one, and the sweep should not run
until we do.**

The two candidates both fail right now:

- **Collision counting cannot detect topology change.** Established and recorded
  in FINDINGS.md: `collisions(F)` is empty in every accepted run because exact
  coincidence of real-valued outputs is measure-zero, so any smooth map is
  injective on a finite sample almost surely. Every collision we have measured
  is quantization-induced. Running a width sweep with collision rate as the
  primary outcome would measure quantization behaviour as a function of width,
  which is a real question but not the topological one.
- **Linking-number estimation is unvalidated on sampled representations.** We
  have never run it. The paper's own estimator (Appendix H: PCA to `R^3`,
  ε-filtered k-NN graphs, fundamental cycle basis, Gauss integrals over basis
  pairs) is described but its Table 8 application is a single best seed with no
  uncertainty, and the paper labels its CIFAR-10 results projection-dependent
  and correlational. We additionally have the problem recorded in the prior
  notes: our class supports are sampled *solid tori*, not the 1-D closed curves
  the linking number is defined on, so ordered core-circle samples must be
  propagated separately.

**The work is therefore split.**

**Half A, approved to run now: accuracy and pass rate versus width.** These need
no estimator, are what the paper's own `R^3` tables report, and directly address
the width question. This half stands alone as the first width evidence at
`d = 3` in either project, and remains a complete result even if the estimator
work later fails.

**Half B, blocked: estimated linking number of the two class cores per layer.**
Gated on estimator validation, below. Validation failure is a legitimate
stopping point and would leave Half A intact and publishable as an
accuracy-versus-width study.

### Trimodality is a primary outcome of Half A

Not a caveat. The existing width-5 ReLU data is trimodal (four runs at exactly
0.5000, fourteen at 0.959-0.971, two at 1.0000), which is a mixture of
optimization failures and successes rather than the smooth ~90-92% ceiling
Table 2 reports for width-3 ReLU. That distinction matters: a ceiling implies
expressivity is binding, whereas a mixture implies optimization may be binding,
and the two have different consequences for the paper's claims.

Half A therefore reports, for every width x depth x activation cell:

- the full per-width accuracy **histogram**, not a summary;
- the **fraction at exactly chance** (0.5000);
- the **fraction at exactly 1.0000**;
- the **pass rate** under the existing gate;
- mean and seed-level SD, **never reported alone**.

It also asks explicitly whether the trimodal structure is specific to width 5
and ReLU or appears across widths and activations. That is a question about
whether width or optimization is binding, and it is answerable from Half A
alone.

**Gate for Half B: validate the estimator before it runs.** Specifically:

1. Implement the Gauss double integral on ordered polygonal cycles, the direct
   Appendix G.8 approach rather than the k-NN cycle detector, since we know our
   core parametrizations exactly.
2. Validate on synthetic configurations with **known** linking number: an
   unlink (0), a Hopf link (±1), a `(2,4)` torus link (±2), and a chain of
   three (pairwise 0, 1, 0). Confirm the estimator returns the integer within
   tolerance and that the sign flips with orientation reversal.
3. Calibrate a **noise floor**: sample density, quadrature subdivision, and
   distance to intersection all affect the estimate. Report the estimator's
   dispersion across seeds and discretizations, which is exactly what the paper
   does not provide for Table 8. Without this we cannot distinguish "link
   changed" from "estimator noise".
4. Establish the **failure mode**: linking number is undefined once the two
   images intersect, and the paper's own Table 8 reports fractional artifacts
   (0.50, 0.18) at exactly that point, starred as meaningless. Our estimator
   must detect and report the intersection condition rather than emitting a
   fractional number.
5. Only then propagate ordered core-circle samples through trained networks.

This is a prerequisite, not a parallel task. A width sweep whose primary
quantity is an uncalibrated estimator produces numbers nobody can interpret.

## 5. What would count as informative

Stated in advance, including negative outcomes.

**Informative positive results**

- Pass rate or accuracy rises sharply between width 3 and some width `w*`, then
  flattens. `w*` near 8 supports the additive account; `w*` near 15 or continued
  improvement past 8 supports the multiplicative account. Either is a genuine
  result because it is the first width evidence at `d = 3`.
- Estimated linking number goes from ±1 at the input to 0 at some layer, with
  minimum inter-class distance staying positive, and the layer at which this
  happens depends on width. That would be the direct analogue of Table 8 with
  seed-level uncertainty the paper does not report.
- The width at which linking is resolved differs between monotonic and
  non-monotonic activations. This requires adding a non-monotonic activation
  (GELU, Swish/SiLU, or Mish), without which the ordering cannot be tested at
  all, as recorded in FINDINGS.md.

**Informative negative results**

- **No width effect between 3 and 15.** Given that width 4 is already
  theoretically sufficient (Theorem D.1), a flat curve from 4 upward is fully
  consistent with the theory and would indicate the obstruction is relieved
  immediately rather than gradually. This would sit against *both* the additive
  and multiplicative accounts, and would be worth reporting.
- **Accuracy at width 3 is not obviously capped.** Theorem 3.7 forbids perfect
  separation of the continuous curves, not high accuracy on a finite sample of
  thickened tubes. If width-3 runs reach high accuracy, that does not refute the
  theorem, and saying so clearly is itself a useful contribution given how
  easily it would be misread.
- **The estimator fails validation.** If step 2 above does not recover known
  linking numbers, or the noise floor in step 3 swamps the effect, the correct
  outcome is to report that the measurement cannot currently be made and to stop
  before running 720 networks. This is a real possible outcome and should not be
  treated as a failure of the project.
- **Trimodal accuracy persists.** If narrow widths keep producing the observed
  0.50 / ~0.96 / 1.00 structure, the mean is meaningless and the honest report
  is a distribution plus a pass rate, with the note that optimization failure
  and expressivity limits are not separable by this design.

**What would not be informative**

- Any collision-rate-versus-width curve presented as evidence about topology.
- Agreement with Junyu's `~<3+5` treated as confirmation of the paper, given
  that the paper does not contain the claim.
- A single best seed at any width, for any quantity.

## Resolved: final grid for Half A

| Axis | Values | Count |
|---|---|---:|
| Width | 3, 4, 5, 6, 7, 8, 10, 12, 15 | 9 |
| Depth | 3, 5, 8, 12 | 4 |
| Activation | tanh, ReLU, leaky-ReLU, GELU | 4 |
| Seeds | 0-19 | 20 |

2,880 runs. At the measured ~1.5-2 s per run at these widths, roughly 1.5 hours
on one CPU thread. All seeds recorded per run; no run excluded from the
pass/fail record, since pass rate is itself a primary outcome.

Half B remains blocked on estimator validation and is not scheduled here.
