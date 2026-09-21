# Registration: does output scale enable placement, or does placement enable output growth?

**Written before any checkpoint was logged.** No trajectory quantity existed
when this was written; Phase 1 measured only terminal parameters.

Date: 2026-09-21.

---

## The problem with Phase 1's result

Phase 1 found that within each cell, `P(G > 0)` rises monotonically with
**terminal** `|w2|` (correlation +0.78 to +0.95, lowest quartile 0.000 in every
cell). That is correlational and **terminal**, and there is a plausible reverse
mechanism that produces the identical pattern.

Under cross-entropy on a separable problem, once placement and bias are good
the loss keeps decreasing as the logits grow, so `|w2|` should **accelerate
after** the run becomes sign-correct. Runs that placed early would then have
had longer to grow and would end with large `|w2|` — reproducing the within-cell
correlation with the causal arrow reversed.

Both mechanisms predict the Phase 1 result. They differ in the trajectory.

## The two hypotheses, with distinct signatures

**H-forward — output scale enables placement.** Growing `|w2|` steepens the
loss surface seen by `(w1, b1)` and drives the unit into a traversing position.

*Signatures:*
1. Among runs **unsolved at checkpoint `t`**, higher `|w2|_t` (equivalently
   `R_t`) predicts a **later placement crossing**.
2. At the step placement first becomes good, `R` is **already near or inside
   the transition band** (0.30–0.50).
3. `|w2|` growth rate is **not** systematically higher after the crossing than
   before.

**H-reverse — placement enables output growth.** Placement is achieved by
`(w1, b1)` dynamics largely independent of `|w2|`; once the classes are
separable in the unit's image, cross-entropy drives `|w2|` up to sharpen the
logits.

*Signatures:*
1. `|w2|` **growth rate jumps after** the crossing step.
2. `R` at the crossing step is **typically well below 0.30**, climbing past the
   band only afterwards.
3. `R_t` among unsolved runs predicts final outcome **poorly**.

## The decisive statistic

> **The distribution of `R` at the first good-placement step, reported per
> cell.**

If its median sits **inside or above** the transition band (≥ 0.30), scale
precedes placement — H-forward. If it sits **well below** (say ≤ 0.15 median),
placement precedes scale — H-reverse.

Secondary, per run: the `|w2|` growth rate in a window **before** the crossing
against the same-length window **after**, reported as a ratio. H-reverse
predicts a ratio materially above 1; H-forward predicts near 1.

## Registered prediction

**H-reverse**, on two grounds stated now so the reasoning is on record:

- Phase 1 found the lowest `|w2|` quartile has `P(G > 0) = 0.000` in every
  cell, but those runs have `|w2| ≈ 0.4–0.5`, i.e. **they barely moved at all**.
  That is consistent with "runs that did not train" rather than with a scale
  threshold acting on placement.
- The cross-entropy argument above is a known property of the loss, not a
  conjecture, and it predicts post-crossing acceleration directly.

**Registered numeric form**: median `R` at the crossing step **below 0.20**, and
median post/pre growth-rate ratio **above 1.5**.

I expect to be reporting against my own Phase 1 framing. If H-reverse holds,
`R` is a *consequence* marker rather than an enabling condition, and the paper
cannot claim output scale drives placement on this evidence.

## Instrumentation required

- **Log-spaced checkpoints** across the budget, plus a **fine record** once `G`
  comes within a small margin of zero, so the crossing step is located
  precisely rather than bracketed by a coarse grid.
- At each checkpoint: full `(w1, b1, w2, b2)`, `G`, placement and bias status,
  `R`, `rho`, loss, and error count.
- Crossing step defined as the **first** step at which `G > 0`, with
  re-crossings recorded (placement may be lost again).

## Consequences fixed in advance

- **If H-reverse holds**, Phase 3's **scale-up arm becomes the only remaining
  test** of whether output scale can causally help placement, and its result is
  the paper's mechanism claim in either direction. Phase 3's primary outcome
  stays the placement rate and time to `G > 0`.
- **If H-forward holds**, the Phase 1 correlation is supported by ordering and
  Phase 3 tests it by intervention.
- **Either way the paper reports this ordering test**, because the Phase 1
  correlation alone cannot distinguish the two.

## Phase 2b risk, registered separately

Independent of the ordering result: `R_t` may predict final outcome poorly
among unsolved runs for reasons unrelated to mechanism. Headline metrics are
**log loss and calibration at the cell level**, not run-level AUC alone, and
the held-out comparison is against **M0 (cell difficulty: `a`, `log B`,
optimiser)**, which may match or beat `R_t`.
