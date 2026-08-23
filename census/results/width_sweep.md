# Width sweep at d = 3, Half A: accuracy and pass rate

**This is the first width evidence at `d = 3` in this project or in Ren and
Lim.** Their `R^3` experiments fix width at 3 and vary depth (Tables 2, 3, 8;
Appendix G.2: "All architectures use width 3"). This sweep varies width at
`d = 3`. It is not a replication, and nothing in that paper predicts its outcome
at widths 4 through 15.

Grid: widths 3, 4, 5, 6, 7, 8, 10, 12, 15 × depths 3, 5, 8, 12 × activations
tanh, ReLU, leaky-ReLU, GELU × seeds 0-19. **2,880 runs, no run excluded**,
since pass rate is a primary outcome rather than a filter. 64.5 minutes on one
CPU thread. Linked tori, tube radius 0.2, 1,000 points per class per split,
float32, full-batch Adam, 2,000 steps. All seeds recorded in
`width_sweep.csv`.

Activation roles, per the approved design: tanh, ReLU, and leaky-ReLU are all
continuous and coordinate-wise monotonic and are a **robustness check within one
expressivity class**, not a comparison. **GELU is non-monotonic** and is the
only activation here that can bear on the paper's ordering.

## Which accuracy is reported

**Every accuracy in this document is held-out evaluation accuracy, never
training accuracy.** "Perfect" means `final_eval_accuracy >= 1.0`.

The evaluation split is generated from a different data seed than the training
split (`20000 + seed` against `10000 + seed`), giving 2,000 evaluation points
and 2,000 training points, balanced 1,000 per class in each. The two share no
points: zero exact duplicates, and the nearest training point to any evaluation
point is 5.78e−03 away (mean 0.0538). The training loss is computed only on the
training split; the evaluation split is never used for the gradient, for early
stopping, or for any model selection — the reported checkpoint is
unconditionally the final step.

The acceptance gate required **both**: exactly 100% training accuracy *and* at
least 99% evaluation accuracy. The gate is separate from the "perfect" column,
which is evaluation-only.

The two quantities do differ in this sweep — 2,125 runs reach 1.0 on training
against 1,958 on evaluation, with 167 runs perfect on training but not on
evaluation — so the distinction is not academic. At width 3 the headline result
is unchanged either way: on evaluation accuracy GELU has 6 perfect runs of 80
and the monotonic activations 0 of 240; on training accuracy GELU has 7 and the
monotonic activations still 0. The result rests on held-out accuracy.

## Accuracy distributions, n = 80 per width

Counts, not means. Bands separate the two exact endpoints from the interior.

**tanh**

| Width | chance (=0.50) | (0.50,0.90) | [0.90,0.99) | [0.99,1.00) | perfect (=1.00) |
|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 4 | 76 | 0 | **0** |
| 4 | 0 | 1 | 35 | 3 | 41 |
| 5 | 0 | 0 | 11 | 4 | 65 |
| 6 | 0 | 0 | 2 | 2 | 76 |
| 8 | 0 | 0 | 0 | 2 | 78 |
| 15 | 0 | 0 | 0 | 2 | 78 |

**ReLU**

| Width | chance (=0.50) | (0.50,0.90) | [0.90,0.99) | [0.99,1.00) | perfect (=1.00) |
|---:|---:|---:|---:|---:|---:|
| 3 | 49 | 1 | 30 | 0 | **0** |
| 4 | 36 | 2 | 40 | 0 | 2 |
| 5 | 22 | 0 | 38 | 1 | 19 |
| 6 | 16 | 1 | 30 | 8 | 25 |
| 7 | 6 | 0 | 19 | 8 | 47 |
| 8 | 4 | 0 | 10 | 5 | 61 |
| 10 | 1 | 0 | 1 | 6 | 72 |
| 15 | 0 | 0 | 0 | 4 | 76 |

**leaky-ReLU**

| Width | chance (=0.50) | (0.50,0.90) | [0.90,0.99) | [0.99,1.00) | perfect (=1.00) |
|---:|---:|---:|---:|---:|---:|
| 3 | 8 | 1 | 71 | 0 | **0** |
| 4 | 1 | 0 | 63 | 11 | 5 |
| 5 | 1 | 0 | 35 | 13 | 31 |
| 6 | 0 | 0 | 12 | 17 | 51 |
| 8 | 0 | 0 | 1 | 7 | 72 |
| 15 | 0 | 0 | 0 | 4 | 76 |

**GELU (non-monotonic)**

| Width | chance (=0.50) | (0.50,0.90) | [0.90,0.99) | [0.99,1.00) | perfect (=1.00) |
|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 0 | 73 | 1 | **6** |
| 4 | 0 | 0 | 20 | 6 | 54 |
| 5 | 0 | 0 | 15 | 4 | 61 |
| 6 | 0 | 0 | 0 | 3 | 77 |
| 8 | 0 | 0 | 0 | 5 | 75 |
| 15 | 0 | 0 | 0 | 7 | 73 |

## Headline: GELU versus ReLU at width 3

**The cleanest contrast in this sweep is GELU against ReLU**, because both are
unbounded and they differ only in monotonicity, which is the property the
paper's ordering turns on. It is also the larger effect.

At width 3, over 80 runs each:

| | GELU | ReLU |
|---|---:|---:|
| Monotonic | **no** | yes |
| Bounded | no | no |
| Perfect runs (=1.0000) | **6 / 80** | **0 / 80** |
| Max accuracy | **1.0000** | 0.9735 |
| Runs at exactly chance | **0 / 80** | **49 / 80** |
| Mean accuracy | 0.9634 | 0.6735 |

Widening to all three monotonic activations, none reaches 1.0000 at width 3 in
any run:

| Activation | Monotonic | Max accuracy at width 3 | Perfect runs |
|---|---|---:|---:|
| tanh | yes | 0.9870 | 0 / 80 |
| ReLU | yes | 0.9735 | 0 / 80 |
| leaky-ReLU | yes | 0.9700 | 0 / 80 |
| **All monotonic** | yes | **0.9870** | **0 / 240** |
| GELU | **no** | **1.0000** | **6 / 80** |

So the result is **6 / 80 for the one non-monotonic activation against 0 / 240
across all three monotonic ones**, with the monotonic maxima falling short at
0.9870, 0.9735, and 0.9700 respectively. GELU's six perfect runs are spread
across all four depths (2 at depth 3, 2 at depth 5, 1 at depth 8, 1 at depth
12), so they are not a single lucky configuration.

At width 4 every activation reaches 1.0000 **in at least one run**, which is not
the same as every activation succeeding at width 4 — and the distinction
matters. The per-activation rates at width 4 are GELU 67.50%, tanh 51.25%,
leaky-ReLU 6.25%, and **ReLU 2.50% (2 runs of 80)**, with ReLU additionally at
exactly chance in 45.00% of runs. ReLU cannot be reliably trained to perfect
accuracy at width 4 in this setup; it merely does so occasionally. Earlier
wording here said "at width 4 every activation reaches 1.0000" without that
qualification and overstated the result.

This is the direction Theorem 4.7 and the Table 1 ordering describe, at the
width where the obstruction is stated. It is a **finite-sample result on
thickened tubes and is not a test of the theorem**, which concerns perfect
separation of the underlying continuous curves; see the caveats.

The GELU-versus-tanh comparison is reported below for completeness but is the
weaker inference: tanh is monotonic *and* bounded, so a difference against GELU
conflates monotonicity with boundedness. GELU versus ReLU isolates monotonicity
as far as any standard activation allows.

## Pipeline validation: independent agreement with the paper's Table 2

**This is the reason to believe the width-3 result**, and it is reported here
rather than among the caveats. Our width-3 runs and the paper's Table 2 meet at
exactly one place — width 3, depths 3, 5, 8, 12, ReLU and GELU — and they agree.

Mean accuracy (%), side by side:

| Depth | Paper ReLU | **Ours ReLU** | Paper GELU | **Ours GELU** |
|---:|---:|---:|---:|---:|
| 3 | 84.3 | **81.6** | 89.3 | **96.5** |
| 5 | 77.1 | **73.1** | 90.0 | **96.5** |
| 8 | 63.1 | **61.3** | 91.1 | **96.1** |
| 12 | 57.7 | **53.5** | 91.2 | **96.2** |

Maximum accuracy (%), side by side:

| Depth | Paper ReLU | **Ours ReLU** | Paper GELU | **Ours GELU** |
|---:|---:|---:|---:|---:|
| 3 | 92.8 | **97.0** | 92.9 | **100.0** |
| 5 | 92.5 | **97.4** | 100.0 | **100.0** |
| 8 | 92.6 | **95.9** | 100.0 | **100.0** |
| 12 | 91.6 | **96.6** | 100.0 | **100.0** |

Three features reproduce independently:

1. **ReLU means degrade with depth** — 84.3 → 57.7 in theirs, 81.6 → 53.5 in
   ours. The magnitude of the decline matches to within a few points at every
   depth.
2. **GELU means are flat in depth** — 89.3 → 91.2 in theirs, 96.5 → 96.2 in
   ours.
3. **GELU reaches 100.0 max at depths 5, 8, and 12 while ReLU never does** in
   either study.

Our GELU means sit 5-6 points above theirs and our ReLU maxima 3-5 points above,
which is expected: the datasets, tube geometry, optimiser, and step budget all
differ. This is agreement in direction, shape, and ordering, not a reproduction
of values, and it was obtained without tuning toward their numbers. It gives an
independent check that the training pipeline, the linked-tori construction, and
the accuracy measurement behave as the published setting does at the one point
where the two overlap.

## Result 3: the trimodality is dying ReLU, not a ceiling

The design flagged the width-5 ReLU trimodality from the original census as a
primary question: is it specific to ReLU and width 5, or general? It is
**almost entirely ReLU, and it scales with depth, not width**.

Runs at exactly chance (0.5000), pooled over all widths and depths:

| Activation | Chance runs | Fraction |
|---|---:|---:|
| ReLU | 134 / 720 | 18.61% |
| leaky-ReLU | 10 / 720 | 1.39% |
| tanh | 0 / 720 | 0.00% |
| GELU | 0 / 720 | 0.00% |

ReLU chance fraction by depth (all widths): 5.56% at depth 3, 9.44% at depth 5,
25.56% at depth 8, **33.89% at depth 12**. Within ReLU, by width × depth:

| Width | d=3 | d=5 | d=8 | d=12 |
|---:|---:|---:|---:|---:|
| 3 | 30.0 | 50.0 | 75.0 | 90.0 |
| 5 | 5.0 | 10.0 | 40.0 | 55.0 |
| 8 | 0.0 | 0.0 | 5.0 | 15.0 |
| 15 | 0.0 | 0.0 | 0.0 | 0.0 |

This is the signature of dying ReLU: units driven permanently negative, the
network collapsing to a constant predictor. It is an **optimization failure, not
an expressivity ceiling**, and it is why means must not be reported alone —
ReLU's mean at width 3 (0.6735) is an average over a bimodal mixture of dead
runs and runs near 0.95 and describes neither.

Excluding chance runs, ReLU's surviving accuracy still shows a genuine ceiling
at width 3: max 0.9735 over 31 live runs, rising to 1.0000 at width 4. So both
effects are present and separable — optimization failure dominated by depth, and
a width-3 accuracy ceiling that survives removing it.

## Result 4: additive scaling holds at d = 3; 5d scaling does not

**This is the first evidence bearing on which scaling applies at `d = 3`.**
Neither source supplies it: the author's `~<3+5` figure is unpublished, and
Table 7 is a different setting. The sweep was extended to width 15 precisely so
the two accounts could be told apart, since they diverge at `3+5 = 8` versus
`5d = 15`.

**At `d = 3`, nothing improves from width 8 to width 15.** Fraction perfect:

| Activation | w=8 | w=10 | w=12 | w=15 | Change 8→15 |
|---|---:|---:|---:|---:|---:|
| tanh | 0.975 | 0.963 | 0.988 | 0.975 | 0.000 |
| GELU | 0.938 | 0.912 | 0.887 | 0.912 | −0.026 |
| leaky-ReLU | 0.900 | 0.950 | 0.988 | 0.950 | +0.050 |
| ReLU | 0.762 | 0.900 | 0.938 | 0.950 | +0.188 |

Three of four activations are flat within seed noise across that range; two move
slightly downward. Only ReLU climbs materially, and that is the dying-ReLU rate
resolving (its exactly-chance fraction falls from 5.0% at width 8 to 0.0% at
width 12), not an expressivity change.

Width at which each activation first exceeds 50% perfect runs:

| Activation | First width ≥ 50% perfect |
|---|---:|
| GELU | 4 |
| tanh | 4 |
| leaky-ReLU | 6 |
| ReLU | 7 |

**Conclusion: the additive account is supported in this setting and `5d`
scaling is not.** Every activation is saturated by width 8, and three of four by
width 6, which sits inside the additive `~<3+5 = 8` range. If accuracy at
`d = 3` followed Table 7's multiplicative pattern, substantial gains would
continue out to width 15; they do not.

**This does not contradict Table 7.** That table is `R^7`, with `S^3 ⊔ S^3`
rather than two circles, and `k = 10` copies rather than one — a different link
type, a different ambient dimension, and a different copy count. The two results
together indicate that the scaling is **setting-dependent** rather than that
either is wrong. The open question recorded in
`notes/icml_paper_reconciliation.md` is now partly answered: additive scaling is
what holds at `d = 3` for a single Hopf link, whatever governs `R^7` at
`k = 10`.

## Caveats

- **This is not a test of Theorem 3.7 or 4.7.** Those forbid perfect separation
  of disjoint continuous curves. We sample thickened solid tori and evaluate on
  a finite set. A width-3 run reaching 1.0000 on 2,000 sampled points does not
  refute the theorem, and the monotonic activations failing to reach it does not
  establish it.
- **GELU versus tanh conflates two properties.** GELU is non-monotonic *and*
  unbounded; tanh is monotonic *and* bounded. The cleaner contrast is GELU
  versus ReLU, both unbounded and differing only in monotonicity — and there the
  gap at width 3 is large (6 perfect runs versus 0, and 0% versus 61.25% at
  chance). Isolating boundedness would need a bounded non-monotonic activation,
  which is not a standard choice.
- **Optimization is confounded with expressivity throughout.** The paper's
  theory is about what a network can represent; this sweep measures what Adam
  finds in 2,000 steps. The dying-ReLU result is a direct illustration.
- **Accuracy is not a topological measurement.** Nothing here measures linking.
  That is Half B, which remains blocked on estimator validation.

## Status

Half A complete. Half B (linking-number estimation) not started and blocked on
the validation gate in `notes/width_sweep_design.md`.

---

> **Status note (2026-08-22).** Dense verification (`dense_check.md`): 5 of
> the 6 width-3 GELU separations recorded here survive 100,000 fresh points;
> the sixth (depth 5, seed 1) shows 1 error on the primary dense sample and
> 0 on two further samples — marginal, not decisively failed. Width-4 dense
> sampling found ReLU-family separations the least robust (ReLU 1/5,
> leaky-ReLU 3/10 in the sampled subset). Counts in this document are
> sample-level.
