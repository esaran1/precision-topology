# §7 On real data: a budget-dependent comparison, and a failed bridge

*Draft in publication prose. Follows §3–§6 (`paper/results_draft.md`).
Verification against raw artifacts in `paper/link_cifar_verification.md`.*

---

The preceding sections establish a budget-dependent findability boundary in a
four-parameter task. The obvious question is whether anything of that survives
contact with a setting people actually train in. This section reports the one
place we could test it, what it shows, and — equally — the prediction we could
not test and why that failure is itself informative.

## 7.1 The setting, and the control that licenses it

All results here use a depth-8 convolutional network on CIFAR-10, trained with
Adam on fresh seeds. This configuration was not chosen for convenience: it is
the only setting among six we examined in which the folk GELU-over-ReLU
advantage reproduces under a pre-registered criterion.

That criterion, fixed in `control_criterion.md` before the confirming runs,
required four things: an effect in the expected direction; `p < 0.05` at
`n >= 10` fresh seeds; replication in at least two nearby configurations; and
a **training-loss plateau**, defined as under 1% relative improvement in the
final epoch. The fourth condition exists because this project has already been
misled once by a mid-descent snapshot, and we did not want to rank activations
from a point where all arms were still descending.

At 12 epochs the first three criteria passed and **the plateau criterion
failed everywhere** — median final-epoch improvement was 9–20% by cell, with
every cell still mid-descent. We therefore reran the depth-8 cells to
convergence, stopping at under 1% relative improvement on two consecutive
epochs, up to a 40-epoch cap:

| arm | n | mean test errors | SD | median epochs | plateaued |
|---|---:|---:|---:|---:|---:|
| GELU | 10 | 2,047.0 | 47 | 22 | 9/10 |
| ReLU | 10 | 2,179.7 | 45 | 31.5 | 9/10 |
| tanh | 5 | 2,591.8 | 36 | 21 | 5/5 |

**23 of 25 runs plateaued.** The two censored runs (GELU seed 10, ReLU seed 4)
hit the 40-epoch cap and are reported as censored rather than counted as
converged; dropping them entirely moves nothing (2,049 vs 2,183, `p = 5e-5`).
GELU leads ReLU by 132.7 errors at `p = 4e-5` (two-sided permutation, 100,000
resamples). The control is restored in full, at convergence, on fresh seeds.

Worth recording: the mid-descent snapshot was not, in this instance,
misleading. The GELU–ReLU gap was 132 errors at 12 epochs and 133 at
convergence. The criterion that forced the rerun was still correct to impose —
§7.3 shows a comparison in this very setting where the short-budget ranking
*is* an artifact — which is why we did not weaken it retrospectively once it
had cost us a day of compute.

## 7.2 tanh is worst, and this contradicts the smoothness account

The standard explanation for GELU beating ReLU is smoothness. That account
makes a prediction about tanh, which is smooth and monotonic, and the
prediction fails.

**tanh is the worst of the three at convergence**: 545 errors behind GELU
(`p = 4e-4`) and 412 behind ReLU (`p = 4e-4`). Its training error is roughly
four times the others' — 744 per 10,000 against about 180 — so it is
**underfitting the task rather than generalizing differently**. A smooth
monotonic activation here is not merely no better than ReLU; it is
substantially worse, on training error first.

This is the strongest content in the section. In the one setting where the
folk advantage reproduces under a pre-registered criterion, the advantage
tracks **non-monotonicity rather than smoothness** — the same axis that
separates the activations in §2 and §3, arrived at independently on real data.
We state the limits plainly: this is one architecture on one dataset, tanh's
arm has `n = 5` against 10, and the result rules out smoothness as *the*
explanation in this setting rather than establishing folding as the mechanism.

## 7.3 A registered budget-dependent flip

If a standard activation comparison is budget-dependent, the ranking should be
reversible by training budget alone. It is, for one pair, and this was
registered in advance.

| budget | tanh | ReLU | ordering | p |
|---|---|---|---|---|
| 2 epochs | 3,425 ± 150 | 3,924 ± 180 | **tanh ahead by 499** | 0.00044 |
| converged (30–40 ep) | 2,592 ± 36 | 2,180 ± 45 | **ReLU ahead by 412** | 0.00039 |

Same task, architecture, optimizer and data; only the budget differs. Both
ends are individually significant, as the pre-registered criteria require.
This is the pair **P-flip** named in advance, for the stated reason — that
tanh underfits at convergence. Its companion **P-no-flip-GELU** was also borne
out: GELU leads ReLU at 2 epochs (3,173 vs 3,924, `p = 0.00023`) and at
convergence, so the flip is specific to one pair rather than general
instability of the setup.

**Where the crossing sits, reported against the rule registered before the
tanh cells landed.** At 2 epochs mean test error is 31.7–39.2% — a third of
examples wrong. The crossing is **bounded between 2 and 30 epochs** by present
data, and we have not narrowed it further.

The registered interpretation rule, restated so it cannot be chosen after the
fact: a flip whose crossing exists only in the ~35%-error regime supports the
narrow claim that early-training rankings are unstable, which is closer to
known than to new, and **must not be led with**. It carries weight only if the
crossing sits at 5 epochs or later. **That question is undetermined.** The
flip exists and is significant at both ends; whether it clears the regime bar
depends on cells that are not in. We therefore report it as the extreme case
rather than the headline.

## 7.4 The magnitude of the advantage varies about fivefold with budget

Within a single experiment — `n = 8` per cell, seeds 100–107, one architecture
and one data pipeline — the **GELU-over-ReLU** advantage measured in test
errors is 750.4 at 2 epochs, 418.4 at 5, and 150.6 at 12: a **4.98x** change
across the budget range. (The contrast here is GELU against ReLU, not against
tanh; the GELU-over-tanh advantage in the same experiment runs the other way
with budget, 251.4 to 478.4, which is the §7.3 flip seen from the other side.)

We quote 4.98x rather than the larger figure that appears in our own earlier
notes. **5.65x is not reported here**, because it pairs the 2-epoch cell from
this sweep with a converged endpoint (132.7) measured in a *different
experiment* with `n = 10` and a different seed set. The two experiments agree
where they overlap — this sweep's 12-epoch cell gives 150.6 against the other
run's converged 132.7, within 13%, on the same trajectory — so there is no
inconsistency between them. But a ratio spanning two seed populations
conflates a budget effect with a possible between-experiment difference, and
the within-experiment number loses nothing worth having. This was registered
as P-margin (a change exceeding 1.5x) before measurement.

**What this licenses, and what it does not.** It licenses: *at least one
standard activation comparison on real data has a magnitude, and in the
extreme case a direction, that depends on training budget.* It does not
license: that activation comparisons in general are budget-dependent, that
published comparisons are wrong, or that the 1D budget law of §5 governs this
setting. We measured one architecture on one dataset; the law of §5 is Adam-
specific even in the toy setting, and nothing here tests its exponent.

## 7.5 The bridge we could not build

The §5 account makes a prediction for real data: the monotonicity-specific
advantage should be present when a network's bottleneck is near the intrinsic
dimension of its input and gone by roughly twice it, mirroring the width-3 to
width-6 boundary of §2.5. **We could not test it**, and the reason is a
finding rather than a shortfall of effort.

The prediction needs a setting with two properties at once: a reproducible
GELU-over-tanh advantage, and a flat bottleneck whose width is a single
unambiguous number with a stable intrinsic-dimension estimate. Three
experiments show that no setting available to us has both:

- **MNIST bottleneck.** The width axis is defensible (ID spread [7, 13]), but
  the control is **absent**: GELU never beat ReLU at any width.
- **MNIST and CIFAR deep MLPs.** The axis is defensible; the control is
  **absent** (depth-8 replication `p = 0.07`, depth-12 `p = 0.98`).
- **CIFAR depth-8 CNN.** The control is **restored** — all four criteria, §7.1
  — but the axis is **not defensible**. Convolutional width is not one number
  (channels 128, spatial 4, product 512), and at the one flat location the
  intrinsic-dimension estimators disagree by **1.99x systematically** (TwoNN
  45–51 against MLE 26–30, a 1.7x gap in every representation, not noise),
  which would make "at ID" and "at 2x ID" overlapping rather than disjoint
  test regions.

The pattern is structural, and it is the finding: **the folk advantage
reproduces in convolutional settings, and convolutional settings are exactly
where "width" stops being a single number.** The two requirements are in
tension for a reason that is not accidental to our choices.

We closed this by decision rather than by running more cells, because the
remaining candidate — a CIFAR MLP control — was about a coin flip on prior
evidence, and a narrow pass would have rested the prediction on a weak
control. The prediction is therefore **untested, not refuted**. So that it is
well posed for anyone with a better setting, a qualifying one must supply:

1. **A reproducible GELU-over-tanh advantage** — tanh, not ReLU, since our own
   width-8 result isolates a ReLU-family optimization deficit unrelated to
   monotonicity — meeting the four criteria of §7.1.
2. **A flat bottleneck** whose width is one number, insertable without
   destroying the advantage, verified by two control arms (full width, and an
   identity insert at full width) so that layer insertion and narrowing are
   separable.
3. **A stable intrinsic dimension** for the bottleneck's input: at least three
   nonlinear estimators agreeing within the sweep's per-step resolution, so
   that "at or below ID" and "at or above 2x ID" are disjoint under every
   estimator.
4. **Bottleneck input dimension much greater than 2x ID**, so the wide end of
   the sweep is wide relative to the intrinsic dimension rather than merely
   approaching the layer's own size. At the CIFAR flatten point `2x ID ~ 95`
   against a 512-d input — only 5x, uncomfortably tight.
5. **The registered prediction, unchanged**: the advantage is present at
   bottleneck widths near the estimated intrinsic dimension and gone by
   roughly twice it.

Candidates we did not try, and which may satisfy all five: a transformer MLP
block, flat by construction though typically much wider than the intrinsic
dimension, which is the account's own prediction of no effect; a tabular or
sequence task with a genuine representational bottleneck; or a convolutional
network with a deliberate flat low-dimensional neck trained from scratch.
