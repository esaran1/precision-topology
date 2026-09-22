# Registration: is the jump/cold stall SATURATION or TRAPPING?

**Written before any gradient norm, final loss or extended-budget run has been
inspected for the jump and cold_high arms.** Date: 2026-09-21.

Block E found that pre-supplying `|w2|` makes placement *worse*: `jump` places
9/40 against control's 37/40 (Fisher p = 1.3e-10) and `cold_high` places 2/40.
Two mechanisms explain a stall, and they differ in signature.

## The two accounts and their distinct signatures

**Saturation.** With `|w2|` large at initialisation the logit `w2 f(w1 x + b1) + b2`
is far from 0 for most inputs, so the sigmoid saturates and
`dL/dtheta ~ sigma'(logit)` collapses. The run is not in a basin; it is on a
plateau with almost no gradient signal.

> Signature: **gradient norms collapse** after the jump (orders of magnitude below
> control at a matched step), and **a much longer budget eventually lets the run
> place**, because the tiny gradient still points the right way.

**Trapping.** The jumped run falls into a genuine basin whose placement is wrong —
a stationary point with `G <= 0` — and its loss is **above** the conditional
minimum at the same `R`, because a better configuration exists at that `|w2|` and
the run is not at it.

> Signature: **gradient norms are NOT small** (comparable to control), the run
> **converges** to a stationary point with `G <= 0`, its **final loss exceeds the
> conditional minimum at the same `R`**, and **a longer budget does not help**.

## Measurements, fixed now

1. **Gradient norm** `||dL/d(w1,b1,b2)||` (the trained coordinates; `w2` is
   released in `jump` so it is included there) at steps 100, 500, 2,000, 8,000 and
   12,000, for `jump`, `cold_high` and `control`, 40 seeds, reported as medians.
2. **Final loss against the conditional minimum at the same `R`.** For each stalled
   run, take its final `|w2|`, compute `best_conditional` at that `|w2|` with the
   frozen Block B procedure, and report `loss_run - loss_min`.
3. **Extended budget**: 5x the extra budget. `jump` and `cold_high` rerun at
   **60,000 steps** on a subset of **15 seeds** each, recording whether placement
   is ever achieved.

## Registered predictions

**S-A (discriminating).** If **saturation**: median gradient norm after the jump is
**below 10% of control's** at the matched step, and the 60,000-step runs place in
**more than 25%** of seeds (against 9/40 = 22.5% at 12,000 — so the criterion is a
real increase, not noise).

**S-B (discriminating).** If **trapping**: median gradient norm after the jump is
**above 50% of control's**, the 60,000-step placement rate is **within 10
percentage points** of the 12,000-step rate, and the median
`loss_run - loss_min` is **positive and above 0.005** (the loss scale of the
Block B gap differences, ~0.01-0.05).

**Registered expectation, stated before measuring: TRAPPING.** Grounds: the pilot
showed the jumped run's gap *improving* slowly and monotonically (-0.104 to
-0.029 over 19,500 steps) rather than frozen, which is movement, not a dead
plateau; and `R` crept up 0.225 -> 0.236, so `w2` was receiving gradient. A
saturated run would show neither.

> **If neither S-A nor S-B is satisfied** — for instance small gradients *and* no
> benefit from a longer budget — the result is reported as **mixed** and the
> annealing framing in item 3 is **not** adopted, per the instruction that it be
> reported only if trapping is shown.

## Consequence for the annealing account

The annealing framing (output scale as a temperature-like parameter that must be
raised slowly) requires **trapping**: a fast-growing or pre-supplied `|w2|` commits
the run to a wrong basin. Saturation would instead mean the stall is an
optimisation artifact of large initial logits and says nothing about ordering.

**Per the standing instruction, the annealing framing enters
`paper_claims_delta.md` only if this registration returns TRAPPING and item 3's
dose-response shows a monotone decline.**
