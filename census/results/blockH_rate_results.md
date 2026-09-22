# Block H: the dose-response is there, but the validity gate FAILED

Scored against `results/blockH_rate_prediction.md` (registered `3c1c0f4`, before
any imposed-rate run). Artifacts: `blockH_rate.csv`, `src/blockH_rate.py`.
`a = 1.30`, 40 seeds per arm, 12,000 steps, float64, schedule imposed from step 400.

---

## H-3, the validity gate: FAILED

| | placed |
|---|---:|
| unconstrained control | **37/40 = 0.925** |
| `m = 1` (control-matched rate) | **20/40 = 0.500** |

Gap **42.5 percentage points** against a registered tolerance of 10, Fisher exact
**p = 4.3e-05**. The gate fails decisively, not marginally.

**Per the registration, H-1 and H-2 are therefore uninterpretable as registered
tests.** They are reported below as observations, explicitly not as passes.

## Why the gate failed — and it is informative

The schedule is **linear** from step 400. The natural `|w2|` trajectory is **not**:

| seed | natural `\|w2\|` at steps 400 / 1000 / 2000 / 3000 | `m=1` schedule at the same steps |
|---|---|---|
| 0 | 0.371 / 0.371 / **0.162** / 2.758 | 0.371 / 1.704 / 3.927 / 6.150 |
| 1 | 0.897 / 1.985 / 3.938 / 6.113 | 0.897 / 2.231 / 4.454 / 6.677 |
| 3 | 0.824 / 1.748 / 3.463 / 5.468 | 0.824 / 2.158 / 4.381 / 6.604 |

**Some runs shrink `|w2|` before growing it** — seed 0 falls from 0.371 to 0.162
between steps 400 and 2,000, then rises to 9.45 by step 6,000. A linear schedule
forces growth through exactly the phase in which the run naturally *reduces*
output scale while re-orienting `(w1, b1)`.

So "the same average rate as control" is not "the same trajectory as control", and
the `m = 1` arm is a genuine intervention rather than a null. **That is itself
support for the ordering claim** — the natural non-monotonicity appears to matter —
but it is not the experiment that was registered, and the registered comparison
cannot be rescued by reinterpreting its control.

## H-1 and H-2, reported as observations only

| `m` | placed | solved | median placed step | median final `R` |
|---|---:|---:|---:|---:|
| 0.5 | 22/40 (0.550) | 22/40 | 4,275 | 0.5842 |
| 1.0 | 20/40 (0.500) | 20/40 | 2,375 | 1.1371 |
| 2.0 | 15/40 (0.375) | **0/40** | 1,700 | 2.2431 |
| 4.0 | **0/40 (0.000)** | 0/40 | — | 4.4550 |

Placement is **strictly monotone decreasing** in `m`, endpoint separation
**0.550** (registered `>= 0.25`), Fisher `m=0.5` vs `m=4` **p = 8.4e-09**
(registered `< 0.01`). The solve rate is monotone too, and collapses earlier:
**0/40 already at `m = 2`**, where placement is still 15/40 — the same
placement-without-solving dissociation Block E produced, here because a fast
schedule overshoots `R_solve` before `(w1, b1)` have been shaped.

`m = 0.5` vs `m = 1`: 22/40 against 20/40, **Fisher p = 0.823 — not separated.**
There is no evidence that slower-than-natural is *better*, only that faster is
worse. The registered falsifier for a two-sided "anneal slowly" reading is
therefore triggered: at most there is a penalty for excessive speed.

## What is and is not claimed

**Not claimed**: that growth rate governs success. H-3's failure means every arm,
including the nominal control, is perturbed relative to free training, so the
decline in `m` is confounded with the degree of departure from the natural
trajectory. A faster schedule departs further, and departure alone could produce
the ordering.

**Claimed, and supported**: **forcing `|w2|` to grow on a fixed linear schedule
hurts, and hurts more the faster the schedule** — 22/40 at half the natural
average rate down to **0/40 at four times it**, against 37/40 when `|w2|` is left
free. Combined with Block E's `jump` result (pre-supplying `|w2|` gives 9/40), the
consistent finding is that **`|w2|` must be allowed to follow its own trajectory**,
which is non-monotone, and that constraining it — in level or in rate — degrades
placement.

## The annealing framing is not adopted

Two independent reasons, both fixed in advance:

1. `blockE_stall_results.md` returned **saturation**, not trapping, and the
   registration made the annealing framing conditional on trapping.
2. H-3 failed, so the dose-response is not a valid test of rate.

Per the standing instruction, **the annealing interpretation does not enter
`paper_claims_delta.md`**. What enters instead is the weaker, better-supported
statement above.

## What a valid version would require

A control-matched arm must reproduce the *shape* of the natural trajectory, not
its average slope: for instance, replay each seed's own recorded `|w2|(t)`
time-warped by a factor `m`, so `m = 1` is the natural trajectory exactly and the
gate passes by construction. That is the right experiment and it is **not run
here**; it is recorded as the way to do it properly rather than attempted in the
remaining time.
