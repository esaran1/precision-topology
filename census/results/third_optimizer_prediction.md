# Registration: does R govern under a third optimiser?

**Written before any scored measurement.** The only numbers in hand are an
8-seed pilot used solely to locate budgets whose runs land inside the
transition band; those pilot cells are reported below and are **not** part of
the scored result. Nothing else was computed before this file was written.

Date: 2026-09-14.

---

## 1a. The choice: AdamW with decoupled weight decay, and why

**AdamW, `weight_decay = 0.01`.** Chosen because it is the **most adversarial**
of the three candidates, not the most convenient.

Our account says solvability is governed by `R = |w2| G*(a) / 2` — margin
capacity, which is weight scale converted into achievable margin. Decoupled
weight decay applies a multiplicative shrink to `|w2|` at every step, **acting
directly against the quantity the account says drives the effect**. If R still
governs under an optimiser that actively suppresses weight growth, that is a
materially stronger statement than adding a third adaptive method whose
dynamics resemble Adam's.

Rejected alternatives, with reasons:

- **RMSprop** — isolates preconditioning from momentum, which is a question
  about Adam's internals rather than about R. It would add a third adaptive
  method that behaves like the first.
- **SGD with momentum** — sits *between* the two optimisers we already have, so
  agreement would be the least surprising outcome available and would add the
  least.

AdamW also differs from both existing arms in the way that matters for the
secondary prediction: it should reach **different terminal weight scales**,
which is what makes the test non-trivial.

## 1b. What will be run

Matched to the existing two-optimiser comparison wherever matching is possible:

| | existing arms | AdamW arm |
|---|---|---|
| task | `sign(\|x\|-1)`, `a = 1.25` | same |
| seeds per cell | 30 | 30 |
| criterion | `solves()`, exact-region dense check | same |
| learning rate | 1e-2 | 1e-2 |
| budgets | 1k, 2k, 4k, 8k, 16k, 40k | **1k, 2k, 4k, 5k, 6k, 8k, 16k, 40k** |

**The budget grid is deliberately different, and this is the point.** The Adam
comparison was weak because Adam's runs jump from median `R = 0.234` at 4k to
`0.578` at 8k, straddling the transition band with only **10 runs inside it**.
An 8-seed pilot locates AdamW's in-band window at **B = 5,000-8,000** (median
`R` = 0.282, 0.341, 0.447). We therefore **add 5k and 6k specifically to
populate the band**, keeping the six original budgets so the growth exponent is
measured over the same range as the other two arms.

Stating this plainly: budgets were chosen *after* a pilot, *to land in the
transition band*. That is a design choice about coverage, not about outcome —
the pilot measured where runs land, not whether they solve at a given R.

Terminal `|w2|` is recorded for every run so `R` is computable.

## 1c. Predictions

**Primary.** AdamW runs fall on the **same `P(solve | R)` curve** as Adam and
SGD. Tested by the same three-regime comparison (`R < 0.3`, `0.3 <= R <= 0.5`,
`R > 0.5`), Fisher exact two-sided, against each existing arm; and by `R50`
with a bootstrap interval, reported alongside the existing two.

**Secondary.** AdamW reaches **different terminal weight scales** and therefore
a different growth exponent `alpha`, while `R` still governs solving. `alpha`
is fitted over the six shared budgets (1k-40k) so it is comparable with
`alpha_adam = 1.2627` and `alpha_sgd = 0.7188` on the same window.

**Prediction on direction, registered**: weight decay opposes growth, so
`alpha_adamw < alpha_adam`. This is a risky sub-prediction and is scored.

## Failure condition, fixed now

> If AdamW's `P(solve | R)` curve shifts materially — `R50` differing from the
> pooled value by more than **0.5 W = 0.0241**, the threshold registered in
> `r_collapse_prediction.md`, **with non-overlapping intervals and a
> significant Fisher comparison in a well-populated regime** — then **R is not
> optimiser-independent**, and §5's claim must be **narrowed rather than
> strengthened**.

In that case the paper text changes before the deadline, not after. This
outcome is reported immediately and with the same prominence as a success.

**A point-estimate difference alone does not trigger this.** The existing Adam
arm already exceeds 0.0241 on point estimates while its interval contains SGD's
value and every regime-wise Fisher test is null — which is exactly why §5
currently hedges. The failure condition therefore requires interval separation
*and* a significant regime-wise difference, not a point-estimate gap on sparse
coverage.

## What each outcome licenses

1. **Three arms agree** (no material shift): the hedge in §5 is replaced. The
   claim becomes that the optimiser enters **only through where on the R axis
   its runs land**, supported by three optimisers with different dynamics —
   including one whose weight decay opposes the mechanism. A spare degree of
   freedom, not a two-point line.
2. **AdamW shifts materially**: R is optimiser-dependent. §5 narrows to the
   arms where it holds, and the shift is characterised rather than explained
   away.
3. **Ambiguous** (point estimates differ, intervals overlap, Fisher null —
   i.e. what the Adam arm currently shows): the hedge **stays**, now supported
   by better in-band coverage, and we say the third arm did not resolve it.

Outcome 3 is a real possibility given how narrow the band is, and it is written
down here so it cannot be reported as a success.
