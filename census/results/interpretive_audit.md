# Part 2: interpretive audit

Every substantive error in this project was interpretive, not arithmetic: the
number was right and the claim about it was wrong (T30 grid resolution, T28
wrong variable, "unreachable minima", the probe-dominated barrier, P7's
inverted ledger row). Recomputation cannot catch that class, because it
reproduces the same interpretation.

For each load-bearing claim, three questions. **Q3 is answered adversarially
throughout**: the strongest instrument failure that would make the claim wrong
while the number stays right, named even where the claim survives. A claim
that survives a named attack is checkable; one that survives an unasked
question is not.

---

## 1. The monotonic zero: 0 in 5,580

**Q1. What does the number measure?** The count of runs, among 5,580 distinct
training runs with analytically-monotonic activations at width 3, whose final
network scored 0 errors on a 2,000-point held-out evaluation sample. The
operation is `perfect_eval`: argmax over 2 logits equals the label, at every
one of 2,000 sampled points. Monotonicity is analytic (`parametric_monotonic`:
sin family a <= 1, pwl alpha >= 0), not fitted.

**Q2. Does the claim assert exactly that?** The ledger says "no
monotonic-activation width-3 network has reached 0 eval errors under SGD".
That is exactly the measurement. **The adjacent claim to avoid is "monotonic
networks cannot separate linked tori"**, which is Theorem 3.7's statement
about disjoint 1-D curves, not about our 3-D solid samples (recorded in
`notes/icml_paper_notes.md` item 2). We measure sample separation on
thickened tubes; the theorem concerns the underlying cores. **T1 is a
measurement, not a test of the theorem, and the documents say so.**

**Q3 (adversarial).** *Strongest failure:* the 5,580 runs are not independent
draws from "monotonic architectures" but from one narrow optimizer basin --
all share Adam/Kaiming ancestry except the 360-run protocol stratum. If
monotonic solutions exist but require an initialization scale or optimizer
outside that family, the zero measures our sampling of parameter space, not
representability. This is not hypothetical: **the same failure mode produced
the T30 error**, where a grid's reach was read as the object's extent.
*Why it survives:* four non-SGD search families were run precisely to break
this -- swap-descent, annealing (12 traces), CMA-ES (0/120 restarts with a
positive control at 12/20), and 400 restarts -- and the protocol stratum uses
the authors' own settings (minibatch, lr/10, early stopping). All returned
zero. *What would still overturn it:* a monotonic width-3 separation found by
any method. The claim is falsifiable and has survived deliberate attempts.
**Residual exposure: the bound is over a heterogeneous mixture, so it does not
license a per-condition rate.** Already stated in T2's threats column.

---

## 2. The four-family law

**Q1. What is on each axis, what was fitted, what was independent?**
x-axis: **1/beta**, beta = 1 + 1/q **analytic**, verified numerically to
<=0.7% -- no training, no fitting. y-axis: the onset exponent, a log-log
regression of onset epsilon against budget over bracketed cells only. The
predicted line's slope is **alpha = 1.1173, measured on a separate experiment
(terminal weight growth) and never fitted to these families**. Family A is an
out-of-sample point measured before the constructed families existed.

**Q2. Does the claim assert exactly that?** The claim is "the onset exponent
scales as 1/beta with slope alpha". Measured through-origin slope **1.0984
[1.008, 1.240]** against alpha **1.1173 [0.999, 1.236]** -- overlapping
intervals, **not** a 0.6% agreement (corrected in 3c). **Adjacent claim to
avoid: that this is a derivation.** alpha is measured, not derived (P-1a and
P-1c falsified). It is a law with one empirical input.

**Q3 (adversarial).** *Strongest failure:* the families were **constructed by
us** to have specified beta, so the relationship could be an artifact of the
construction rather than a fact about folds. Specifically, if the constructed
f's differ from each other in some property that co-varies with beta and
independently drives the onset, the fit would hold for the wrong reason.
*Why it partly survives:* family A (`x + a sin x`) is **not** one of our
constructions, was measured first, and lands on the line -- and q2 reproduces
family A's beta via a different function with a constant sqrt(2) prefactor
ratio, so the construction is faithful at least there. *Why it is not fully
answered:* four of five points are our constructions, all from the same
one-parameter template f'(x) = (1-a) + a|x|^q. **A reader is entitled to ask
whether a second, structurally different construction route gives the same
line, and we have not run one.** Recorded as an open exposure.

---

## 3. The budget law: onsets are 50% population thresholds

**Q1. What does the number measure?** For each budget, the smallest grid value
of a at which **>= 50% of 40 independent runs** produce a network passing the
exact-region check, with a strictly smaller grid value below 50%. It is a
**population quantile**, over a population that includes runs stalled at the
log-2 saddle.

**Q2. Does the claim assert exactly that?** "The effective boundary moves with
budget" is a claim about the population, and the measurement supports it.
**The adjacent claim to avoid is "individual runs reach further with more
budget"** -- which the population threshold does not establish. It could in
principle move because a fixed subpopulation always succeeds and simply grows,
rather than because any individual trajectory travels further. *Checked*: the
terminal |w2| distribution's **median** grows 0.87 -> 264 across budgets, and
that is a per-run quantity, so individual runs do travel further. But the
onset is set by the **lower part** of the distribution, and the stall fraction
falls 0.267 -> 0.000 over the same range, so **both mechanisms are present and
the measurement does not separate them.** Stated in T42.

**Q3 (adversarial).** *Strongest failure:* the 50% threshold is a quantile of
a **mixture** (escapers and stallers). If budget acts almost entirely by
converting stallers into escapers, the onset would move without the
*geometry* of reachability changing at all -- the law would be measuring
escape from a saddle, not fold amplification. *Why it survives, partly:*
stalling reaches exactly 0.000 by 80k while the onset keeps moving (1.06 at
32k to 1.03 at 128k), so onset movement continues after the staller
population is exhausted. *Residual:* over the 1k-16k range where most of the
exponent's leverage sits, stalling falls 0.267 -> 0.100, so the two effects
are entangled exactly where the fit is most sensitive. **This is the weakest
link in the budget law and should be stated as such.**

---

## 4. The exclusion table: five exclusions

For each: what would make the measurement right and the exclusion wrong?

**(a) Barrier (MEP exactly 0).** *Measures:* the maximum loss along a string-
method path minus max(endpoint losses), converged to 41 images x 400 steps.
*Adversarial failure:* the string method finds **a** path, not **the** minimum-
energy path; if it converges to a low-barrier path that SGD would never
follow, "no barrier" is true of the landscape and irrelevant to the dynamics.
*Status:* **not fully excluded.** The claim should be "no barrier exists on
the found path", which is what rules out an energetic account, not "SGD's
trajectory is barrier-free".

**(b) Sharpness (lambda_max * eta ~ 0.03 vs threshold 2).** *Measures:* exact
4x4 Hessian eigenvalue at the point, times realized step. *Adversarial
failure:* Ahn-Zhang-Sra concerns sharpness **at the stationary point being
approached**; these points are not stationary (grad norm 0.02-0.27), so
comparing their curvature to 2/eta may be answering a question the theorem
does not ask. *Status:* **this is exactly right and is now the stated
position** -- the points fall outside the theorem's scope for a more basic
reason than either eigenvalue condition. The original framing ("EoS does not
apply because lambda_max is small") tested one of two conditions and was
corrected.

**(c) Distance (4.7-5.5 vs found 10.2-17.7).** *Measures:* Euclidean norm in
raw parameter space between the constructed point and 20 typical
initializations. *Adversarial failure:* Euclidean distance in unnormalized
parameter coordinates is not the metric SGD moves in -- Adam's per-coordinate
normalization means the effective metric is closer to L-infinity in
gradient-scaled units. Under a different metric the ordering could reverse.
*Status:* **RESOLVED 2026-09-08 -- attack answered.** Recomputed under Adam's
own preconditioner (bias-corrected sqrt of accumulated second moment, measured
from trajectories): found/constructed ratios **3.79 / 3.04 / 2.30 / 1.72** at
a = 1.45 / 1.50 / 2.00 / 3.00, against Euclidean 3.50 / 2.80 / 2.56 / 1.85.
**The reversal survives in the metric the optimizer actually uses**, so it is
not a coordinate artifact. The claim strengthens.

**(d) Margin (5/20 found solutions have smaller margin).** *Measures:* minimum
logit margin over dense sweeps of both class regions. *Adversarial failure:*
margin at the *found* solution is an endpoint property; SGD's ability to reach
a solution may depend on margin **along the approach**, not at the
destination. *Status:* the exclusion is of "destination margin explains
findability", which is what was claimed. Approach-margin is untested and is a
different quantity.

**(e) Criticality (grad norm 20-227x training's terminal norm).** *Measures:*
gradient norm at the constructed point vs the median gradient norm at step
2,000 of ordinary runs. *Adversarial failure:* the comparison denominator is
training's *terminal* gradient norm, which is small because training has
converged **in loss**, not because it has found a critical point -- so the
ratio could be large for both. *Checked:* the found solutions' own gradient
norms are 0.006-0.014, i.e. at the same scale as the terminal norm, while the
constructed points are 0.02-0.27. The comparison is between two solution
populations, not against an arbitrary baseline. **Survives.** Independently
confirmed to 10 significant figures by finite differences (1e).

---

## 5. The theorem's verification: 66 solvers, 0 violations

**Q1. What does it verify?** That every one of 66 recorded solving parameter
vectors satisfies |w2| >= 2m/G*(a), where m is **that solver's own measured
margin** and G* is the measured maximum class gap. Minimum slack 1.04x.

**Q2. Does the claim assert exactly that?** Yes, and the wording matters.
**The adjacent claim to avoid: that the theorem constrains the empirical
onsets.** It does not, because the empirical criterion is sign-correctness
(m -> 0), where the bound degenerates to |w2| >= 0. The verification confirms
the **inequality holds at the margins solutions actually have**; it does not
connect the theorem to the onset measurement.

**Q3 (adversarial).** *Strongest failure:* the verification is **near-
tautological**. Given m is computed from the same theta, |w2| >= 2m/G* may
follow from the definitions with little content -- if m is defined as the
achieved margin and G* bounds the achievable gap, the inequality could hold
for any theta whatsoever. *Checked:* it does not. The bound uses **G\*(a), the
maximum over all (w1,b1)**, while the solver achieves gap G(w1,b1) <= G*.
A solver with a poorly-placed fold has G << G* and would violate the bound if
its |w2| were small; minimum slack 1.04x shows solvers sit close to the
constraint rather than trivially far from it. **The content is that solvers
place their folds near-optimally.** *Residual:* with 0 violations and minimum
slack 1.04x, the test cannot distinguish "the bound is tight" from "SGD only
finds near-optimal fold placements", and those are different statements.

---

## 6. Compute-cost divergence B(eps) ~ eps^-1.36

**Q1. What does it measure?** An algebraic inversion of the onset exponent,
which is itself a population-threshold measurement (see 3). So B(eps) is
**the budget at which 50% of runs succeed**, not the budget any individual run
needs.

**Q2. Does the claim assert exactly that?** "The compute required to realize a
capability diverges" invites reading B as a per-run requirement. **It is a
median.** Half of runs at the onset budget still fail. The honest phrasing is
"the budget at which a majority of runs succeed diverges".

**Q3 (adversarial).** *Strongest failure:* if the run-to-run distribution
widens with eps -> 0 -- which the SGD shape check suggests it can -- the median
budget and the *mean* budget diverge at different rates, and quoting one
exponent for "the compute required" is then ill-defined. *Status:* **not
excluded.** We measured only the 50% quantile. The exponent for the 90%
quantile could differ, and we did not measure it. This should be stated.

---

## 7. CIFAR margin shift: 5.65x across budgets

**Q1. What is held fixed, what varies?** Fixed: architecture (depth-8 CNN),
dataset, optimizer (Adam lr 1e-3), batch size, seeds (100-107), and the
activation pair. Varies: **epochs only** (2, 5, 12, and the converged run).
The number is the ratio of GELU-over-ReLU mean test-error advantage at 2
epochs (750.4) to that at convergence (132.7).

**Q2. Does the claim assert exactly that?** "The magnitude of a reported
advantage varies 5.65x with budget" -- yes. **Adjacent claim to avoid: that
architecture comparisons are generally budget-dependent.** One architecture
family, one dataset, one optimizer, no augmentation, single learning rate.
Stated in the not-licensed list.

**Q3 (adversarial).** *Strongest failure:* the converged endpoint (132.7)
comes from a **different run** (n = 10, `cifar_convergence.csv`) than the
budget sweep (n = 8, seeds 100-107), so the 5.65x ratio spans two experiments
with different seed sets. If the two populations differ systematically, the
ratio conflates a budget effect with a between-experiment difference.
*Checked:* the sweep's own 12-epoch cell gives 150.6, within 13% of the
converged 132.7 and on the same trajectory, so the two experiments agree where
they overlap. **Using only the sweep's own cells (2 -> 12 epochs) gives
750.4/150.6 = 4.98x**, still a large effect. *Action for the paper: quote
4.98x from a single experiment, or state that 5.65x spans two.*

---

# Summary: claims needing rewording before the paper

Q2 answered "something adjacent" for **five** of seven. Each needs the stated
rewording, not a caveat elsewhere.

| # | claim | current wording drifts to | required wording |
|---|---|---|---|
| 3 | budget law | "individual runs reach further" | "the population threshold moves"; both mechanisms present and not separated over 1k-16k where the fit's leverage sits |
| 4a | barrier | "no barrier exists" | "no barrier on the found path" -- excludes an energetic account, not a claim about SGD's trajectory |
| 4c | distance | "the nearer solutions are the unreached ones" | "nearer **in parameter norm**"; the reversal is metric-dependent and Adam does not move in that metric |
| 6 | compute cost | "the compute required diverges" | "the budget at which a **majority** of runs succeed diverges"; other quantiles unmeasured |
| 7 | CIFAR 5.65x | one number, one experiment | **4.98x** within a single experiment (2 -> 12 epochs, n=8, seeds 100-107), or state that 5.65x spans two experiments |

Verified while writing this audit: single-experiment advantages are 750.4 /
418.4 / 150.6 at 2 / 5 / 12 epochs (n = 8 each), giving **4.98x**; the
converged 132.7 comes from n = 10 with different seeds, and the 12-epoch cell
agrees with it to 13%, so the experiments are consistent where they overlap.
Theorem verification re-checked: n = 66, 0 violations, **minimum slack
1.0394**.

# Exposures that survive the audit and cannot be reworded away

1. **Four of five families in the law are our own constructions** from one
   template. A structurally different construction route has not been tried.
2. **The budget law's staller/escaper entanglement** over 1k-16k, where the
   exponent has most leverage.
3. **The string method finds a path, not the path** -- the barrier exclusion
   is about the landscape, not the trajectory. **The exclusion table now says
   this explicitly** rather than only the audit.
5. **The theorem's verification cannot separate "tight bound" from "SGD finds
   near-optimal fold placements"** (0 violations, min slack 1.04x).
6. **B(eps) is a median**; the distribution's other quantiles are unmeasured.


> **Correction (2026-09-11).** The through-origin slope quoted here was a **four-point** fit that merged q2 and family A into a single point at beta = 1.5 and used family A's value (−0.7340), discarding q2's measured −0.6749. The correct **five-point** fit, using every measured family, is **1.0984 [0.958, 1.239]**, which is **1.7%** from alpha = 1.1173 rather than 0.6%. Intervals still overlap; the agreement is weaker than stated. See `paper/rounded_source_audit.md`.
