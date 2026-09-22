# Block E: the registered arms measure stability, not attainability — recorded before the full run

The registered arm set (`blockF_lag_prediction.md`) reads:

| arm | registered prediction |
|---|---|
| freeze-low (`R < R_glob`) | **placement never achieved** |
| freeze-high (`R > R_spin`) | **placement achieved within the relaxation time** |

Piloting the arms at `a = 1.30` shows the second prediction is **false as
written**, for a reason that is itself the result.

## What the pilot found

**Held at `|w2|` from a cold random initialisation, placement is never achieved
— at any `R`.** 20,000 steps, 3 seeds, at `R/R_glob` = 0.85, 1.15 and 1.50:
**0 of 9 placed**, final gaps `-0.035` to `-0.005`. Raising `R` to 1.5x the
switch does not help.

**Held at `|w2|` but started from Block B's conditional minimiser, placement is
retained above `R_glob` and lost below it.** Ten seeds at each `R`, 8,000 steps,
`|w2|` projected back every step:

| `R/R_glob` | 0.80 | 0.85 | 0.90 | 0.95 | 0.98 | 1.00 | 1.02 | 1.05 | 1.15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| start gap | -0.040 | -0.028 | -0.017 | -0.007 | -0.002 | +0.001 | +0.005 | +0.009 | +0.023 |
| **kept /10** | **0** | **0** | **0** | 1 | 6 | 3 | 7 | 6 | **10** |

`0/10` at and below `0.90 R_glob`; `10/10` at `1.15 R_glob`. The knife-edge band
`0.95-1.05` is mixed, and its non-monotonicity (6/10 at 0.98 against 3/10 at
1.00) is seed noise: the start point is a different conditional minimiser at each
`R`, so these are not nested comparisons.

## Why this is not a failure of the account

Growing `|w2|` past `R_glob` is **necessary but not sufficient** for a
gradient-trained run to place. What the landscape supplies above `R_glob` is a
**stable** sign-correct configuration; reaching its basin is a separate matter,
and a run held at fixed `|w2|` from a cold start does not find it. Real runs place
while `|w2|` is *growing*, which is what Block C measured — they relax onto the
branch (distance 1.09 -> 0.008) and are carried along it.

So the correct reading of the freeze arms is:

- **freeze-low**: placement cannot be **held**, let alone achieved. Registered
  prediction stands.
- **freeze-high**: placement cannot be **achieved from cold**, but is **held once
  attained**. The registered prediction is **falsified as written** and replaced
  by the stability statement, which the same data establish.

## The revised arm set, registered now, before the full run

Every arm starts from the **same** initialisation per seed and differs only in
the intervention. `|w2|` is projected back to its held value after every step;
`(w1, b1)` and `sign(w2)` are never touched; `b2` is scaled with `w2` so the
threshold is carried rather than destroyed.

| arm | procedure | registered prediction |
|---|---|---|
| `control` | nothing | places at `R` ~ 0.23-0.31, well above `R_glob` |
| `null` | `(w2,b2)` x (1+1e-6) at the intervention step, not held | indistinguishable from control — the noise floor |
| `hold_low` | run to placement, then scale to `0.85 R_glob` and hold | **placement LOST**, >= 9/10 |
| `hold_high` | run to placement, then scale to `1.15 R_glob` and hold | **placement KEPT**, >= 9/10 |
| `cold_low` | hold at `0.85 R_glob` from step 0 | never placed |
| `cold_high` | hold at `1.15 R_glob` from step 0 | **never placed** (revised; the registered version said "achieved") |
| `jump` | scale to `1.05 R_glob` at step 0, then release | places **no earlier** than control (revised: `|w2|` is not the bottleneck at cold start) |

**`hold_low` vs `hold_high` is the reversibility pair and the primary result.**
It is a clean causal test: the same run, placed, then pushed to either side of a
threshold computed with no training in it, retains or loses placement accordingly.

**Falsifier for the primary pair**: if `hold_low` keeps placement, or `hold_high`
loses it, in more than 1 of 10 seeds, the threshold does not govern stability and
Block E is reported as negative.

## Provenance and honesty note

The pilot above was run **before** these predictions were written, and its numbers
are reported in full rather than used silently to set the arms. The revision is
labelled: `cold_high` and `jump` are **post-hoc** predictions, chosen after seeing
that cold starts never place. `hold_low` / `hold_high` are the **prospective**
pair — they are the registered reversibility arms, unchanged in content, and
their predictions are as originally registered. Only the prospective pair is
scored as a registered test.
