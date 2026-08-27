# Task F: what the bridge attempt produced, and where it stops

Final state of the bridge from the fold account to the widely reported
GELU-over-ReLU advantage. Part 2 (the width prediction on real data) is
**closed untested**, by decision, and the reasons are a result in their
own right.

## 1. The standalone finding: the advantage is not smoothness

This is what restoring the positive control bought, it stands on its
own, and **it does not depend on the width prediction ever being
tested.**

In the setting where the folk GELU-over-ReLU advantage reproduces —
CIFAR-10, depth-8 small CNN, trained to convergence with 24 of 25 runs
plateaued — the three activations rank:

| activation | monotonic | smooth | mean test errors (n) | mean train errors /10k |
|---|---|---|---:|---:|
| GELU | **no** | yes | **2,047.0** (10) | 173 |
| ReLU | yes | no | 2,179.7 (10) | 184 |
| tanh | yes | **yes** | **2,591.8** (5) | **744** |

- GELU beats ReLU by 133 errors, two-sided permutation **p = 3e−5**.
- **tanh is worst of the three**: 545 behind GELU (**p = 4e−4**) and
  412 behind ReLU (**p = 4e−4**), with **4× the training error** — it
  is underfitting the task, not generalizing differently.

**The standard smoothness explanation of GELU's advantage is
contradicted here.** tanh is smooth and monotonic; if smoothness were
the operative property it should track GELU, and instead it is not
merely no better than ReLU but substantially worse. The property GELU
has that tanh lacks, and that the fold account names, is
non-monotonicity.

What this does **not** establish: that the fold mechanism *causes* the
CIFAR advantage. Establishing that is what the width prediction was
for, and it is untested (§2). The finding above is a constraint on
explanations — it rules smoothness out in this setting — not a
demonstration of the mechanism.

## 2. The endpoint: no available setting supplies both requirements

Testing the width prediction on real data needs two things at once:

- **(a) a restored positive control** — the GELU advantage must
  actually exist in the setting, or a null width result is
  uninterpretable; and
- **(b) a defensible width axis** — a flat representation whose width
  is a single unambiguous number, with an intrinsic-dimension estimate
  stable enough that "near ID" and "≥ 2×ID" name disjoint width
  regions at the sweep's resolution.

Three experiments establish that neither setting we could construct has
both:

| setting | (a) control | (b) width axis | evidence |
|---|---|---|---|
| MNIST bottleneck MLP | **absent** — GELU never beat ReLU at any width; the folk phenomenon simply is not present | present — bottleneck width is unambiguous, ID spread [7, 13] | T34, `bottleneck_results.md` (300 runs) |
| CIFAR-10 deep MLP | **absent** — depth-8 pilot gap failed n = 10 replication (p = 0.07); depth 12 p = 0.98 | present by construction | T35, `control_restoration.md` (MNIST arm; CIFAR MLP untested but MNIST-like) |
| CIFAR-10 depth-8 CNN | **restored** — all four pre-registered criteria met at convergence | **absent** — conv width is ambiguous (channels 128 / spatial 4 / product 512); at the one unambiguous location, ID estimators disagree 1.99× systematically, implying overlapping test regions | T35, `cifar_width_axis_verdict.md` (`cifar_intrinsic.csv`) |

The pattern is not accidental. The folk advantage reproduces in
convolutional settings, and convolutional settings are exactly where
"width" stops being a single number. The one flat location inside the
working CNN (the 512-d flatten point) has an intrinsic dimension whose
estimate is method-dependent by a factor equal to the sweep's own
resolution — TwoNN 45–51 against MLE 26–30, a systematic 1.7× gap
present in every representation, not sampling noise.

**Why we stopped rather than running the CIFAR MLP arm.** Its control
would have to be established first, and the MNIST MLP evidence makes
that roughly a coin flip. The bad outcome is not failure — it is a
*narrow* pass, which would rest a width prediction on a weak control
and a marginal effect. That is where results go soft, and the cost of
finding out is about a day of compute for the control alone.

## 3. Specification: what a setting satisfying both would need

Stated so the test is well posed for anyone who finds such a setting.

1. **A reproducible GELU-over-tanh advantage** (tanh, not ReLU: our own
   width-8 result isolates a ReLU-family optimization deficit unrelated
   to monotonicity, T32), meeting the four criteria in
   `control_criterion.md`: effect, p < 0.05 at n ≥ 10 seeds, replication
   in ≥ 2 nearby configurations, and training-loss plateau.
2. **A flat bottleneck** whose width is one number, insertable without
   destroying the advantage — verified by two control arms, `w` = full
   width and an identity-insert arm at full width, so layer-insertion
   and narrowing are separable (`cifar_width_axis.md`).
3. **A stable intrinsic dimension for the bottleneck's input**: ≥ 3
   nonlinear estimators agreeing within the sweep's per-step
   resolution, so that "at/below ID" and "at/above 2×ID" are disjoint
   under every estimator.
4. **Bottleneck input dimension ≫ 2×ID**, so the wide end of the sweep
   is wide relative to the intrinsic dimension rather than merely
   approaching the layer's own size (at the CIFAR flatten point,
   2×ID ≈ 95 against a 512-d input — only 5×, uncomfortably tight).
5. **The registered prediction**, unchanged: the GELU-over-tanh
   advantage is present at bottleneck widths near the estimated
   intrinsic dimension and gone by roughly twice it (the boundary
   derived from our own setting, where the monotonicity-specific
   advantage vanishes by width 6 = 2d, T32).

Plausible candidates we did not try: a transformer MLP block (flat by
construction, but its width is typically ≫ ID, which is the account's
own prediction of no effect); a tabular or sequence task with a genuine
representational bottleneck; or a conv net with a deliberate flat
low-dimensional neck trained from scratch.

## 4. Where Task F leaves the bridge

- **Established**: the folk advantage reproduces at convergence in one
  concrete setting, and **it does not track smoothness** — the standard
  explanation is contradicted there (§1).
- **Mapped**: where the advantage does and does not reproduce, which is
  narrower than the folk claim implies (absent in shallow MLPs, absent
  on MNIST at every depth tried, present in CIFAR CNNs; erased by a
  500-step warmup in the pilot, an initialization-trajectory
  interaction no smoothness account predicts — n = 3, a lead, not a
  claim).
- **Untested**: whether the advantage tracks bottleneck width relative
  to intrinsic dimension — the fold account's distinguishing
  prediction. Not refuted; untested, for the reason in §2.
- **Honest scope sentence**: *"The mechanism is established in the toy
  setting. It predicts that activation choice matters near width ≈
  intrinsic dimension and not above it; on real data we confirmed the
  advantage exists and is not a smoothness effect, but could not test
  the width prediction, because no setting we could construct supplied
  both a reproducible advantage and an unambiguous width axis."*
