# Registration: growth-rate dose-response — is output scale an annealing parameter?

**Written before any imposed-rate run has been executed.** Date: 2026-09-21.

Block E established that `R > R_glob` is necessary for sign-correct placement to
be **stable** but not sufficient for it to be **found**: cold starts above the
threshold place 2/40, and jump-started runs place 9/40 against control's 37/40.
The natural reading is that `|w2|` must grow *while* `(w1, b1)` are being shaped —
an ordering constraint. If that is right, then **how fast** `|w2|` grows should
matter, and growing it faster should hurt.

## The experiment

From a **matched early checkpoint** (step 400, before any run has placed —
Block A's earliest crossing is step 1,860), impose a schedule on `|w2|` while
`(w1, b1, b2)` train normally under Adam:

    |w2|(t) = |w2|(400) + m * r_nat * (t - 400)

with `r_nat` the natural local Adam growth rate measured on control runs, and
multiplier `m` in **{0.5, 1, 2, 4}**. `|w2|` is projected onto the schedule after
every step; its sign is never changed; `b2` is **not** rescaled (the `jump` arm
showed rescaling `b2` is separately harmful, and holding it fixed isolates the
rate).

`m = 1` is the control-matched arm and its placement rate must reproduce the
unconstrained control's to within sampling error; that is the validity gate.

`a = 1.30`, 40 seeds per arm, 12,000 steps, float64, Adam lr 1e-2, the
shared-dtype initialisation used throughout.

## Registered predictions

**H-1 (monotone decline).** Placement rate falls monotonically in `m`:

> `P(placed | m=0.5) >= P(placed | m=1) >= P(placed | m=2) >= P(placed | m=4)`,
> with the endpoints separated: **`P(m=0.5) - P(m=4) >= 0.25`** (25 percentage
> points), and Fisher exact `m=0.5` vs `m=4` **p < 0.01**.

**H-2 (the solve rate follows).** The same ordering holds for the solve rate.

**H-3 (validity gate).** `m = 1` places within **10 percentage points** of the
unconstrained control (37/40 = 0.925). If it does not, the projection itself is
perturbing the run and H-1/H-2 are uninterpretable; the block is reported as
inconclusive and the annealing framing is not adopted.

## Falsifiers

- **Non-monotone or flat** in `m`: growth *rate* does not govern success, only the
  attained level does, and the annealing reading is wrong. Report as such.
- **`m = 0.5` worse than `m = 1`**: slower is not better, so there is no
  "anneal slowly" principle; at most there is a penalty for being too fast.
  This would weaken H-1 to a one-sided claim and must be reported that way.
- **H-3 fails**: inconclusive, as above.

## Connection to Block F

SGD grows `|w2|` about **2.8x slower** than Adam near the crossing (0.00075 vs
0.00207 per step, `blockF_lag_prediction.md`). The lag account (F-1) predicts SGD
crosses at a **lower** `R` — closer to `R_glob` — because it carries less residual
deficit. The annealing account predicts SGD should **succeed at least as often**,
since slower growth is better. These are compatible and both are already
registered; this note records that the annealing reading makes the *success-rate*
prediction while F-1 makes the *threshold-location* prediction, so Block F tests
them jointly rather than either alone.

## Reporting condition, per the standing instruction

**The annealing framing enters `paper_claims_delta.md` only if both** (i) the
saturation-vs-trapping test (`blockE_stall_prediction.md`) returns **trapping**,
and (ii) **H-1 shows the decline**. If either fails, the ordering observation from
Block E is reported as-is, without the annealing interpretation.
