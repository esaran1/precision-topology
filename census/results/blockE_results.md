# Block E: intervening on `|w2|` — the causal test

Registered in `blockF_lag_prediction.md` (reversibility pair) and revised in
`blockE_redesign.md` after a pilot reported there in full. `a = 1.30`, 40 seeds
per arm, 12,000 steps, Adam lr 1e-2, float64. Artifact:
`results/blockE_intervene.csv`.

`R_glob = 0.21446`, so `hold_low` holds `R = 0.18229` and `hold_high`
`R = 0.24663`. `|w2|` is projected back to its held value after every step;
`(w1, b1)` and `sign(w2)` are never touched; `b2` is scaled with `w2` so the
decision threshold is carried rather than destroyed.

---

## The primary result: the reversibility pair

Both arms run the **same seed** to placement, wait 200 steps, then scale `|w2|`
to one side of a threshold computed with **no training in it**, and hold.

| arm | held at | kept placement | Clopper–Pearson 95% |
|---|---:|---:|---|
| `hold_low` | `0.85 R_glob` | **0 / 37** | [0.0000, 0.0949] |
| `hold_high` | `1.15 R_glob` | **33 / 37** | [0.7458, 0.9697] |
| `noise_floor` (x 1+1e-6) | — | **37 / 37** | [0.9051, 1.0000] |

**Fisher exact `hold_high` vs `hold_low`: p = 1.2e-16.**
`noise_floor` vs `hold_high`: p = 0.115 (null, as it should be).
`noise_floor` vs `hold_low`: p = 1.1e-21.

Three of forty seeds never placed within 12,000 steps and so were never
intervened on; they are excluded from the pair and counted in the `40` column
below. The `noise_floor` arm confirms the intervention machinery itself does
nothing: a 1-part-in-10^6 rescale changes no outcome.

**Placement is lost in 50 steps or fewer — the first checkpoint after the
intervention — in all 37 `hold_low` runs** (median 50, min 50, max 50). It is
not a slow drift; the configuration is simply not stable below the threshold.

This is the causal statement the paper needed. The same network, at the same
point in its own training, keeps or loses sign-correct placement according to
which side of a training-free threshold its output weight is pinned to.

## All arms

| arm | placed | solved | median final `R` |
|---|---:|---:|---:|
| `control` | 37/40 | 37/40 | 1.1687 |
| `noise_floor` | 37/40 | 37/40 | 1.1687 |
| `hold_low` | 0/40 | 0/40 | 0.1823 |
| `hold_high` | **33/40** | **0/40** | 0.2466 |
| `cold_low` | 0/40 | 0/40 | 0.1823 |
| `cold_high` | 2/40 | 0/40 | 0.2466 |
| `jump` | 9/40 | 6/40 | 0.2205 |

## `hold_high` separates the two thresholds, unprompted

`hold_high` **places in 33 of 40 runs and solves in 0**. This was not what the
arm was designed to test, and it is a direct confirmation of the placement/bias
decomposition (T57):

- `R_glob = 0.2145` governs **placement** — `hold_high` sits 15% above it and
  places.
- `R_solve = 0.3067` governs **solving** — `hold_high` sits **19.6% below** it
  and never solves.

Holding `R` between the two thresholds produces placed-but-unsolved networks in
33 of 40 runs. The decomposition predicted exactly this regime; Block E exhibits
it under intervention.

## Two registered predictions FAILED, both informative

**`cold_high` — registered "placement achieved within the relaxation time".
FALSIFIED: 2 of 40.** Held at `1.15 R_glob` from a cold start, placement is
essentially never achieved. The pilot extended this to `1.50 R_glob` and 20,000
steps: **0 of 9**. Above the threshold the landscape offers a *stable*
sign-correct configuration, but gradient descent starting cold does not find its
basin.

**`jump` — registered "placement well before control". FALSIFIED IN THE OPPOSITE
DIRECTION: 9 of 40 against control's 37 of 40, Fisher p = 1.3e-10.** Pre-supplying
`|w2|` at initialisation and then releasing it makes placement *less* likely, not
more. Dose-response over 5 seeds at 20,000 steps: control 4/5, `w2` scaled alone
2/5 (and ~5x later), `w2` and `b2` scaled together 0/5. A jumped run stalls — `R`
moves 0.225 -> 0.236 in 19,500 steps while control passes through the same `R`
around step 4,000 and grows to 1.80.

## What Block E establishes, and what it does not

**Establishes**: above `R_glob` sign-correct placement is **stable**; below it,
**unstable**, lost within 50 steps. The threshold is computed from the loss
landscape with no training run in it, and the intervention is causal — same run,
same step, one parameter changed.

**Does not establish**: that crossing `R_glob` *causes* placement to be found.
It does not: cold starts above the threshold do not place (2/40), and jump-started
runs place *worse* than controls (9/40 vs 37/40). `R > R_glob` is **necessary for
placement to persist and not sufficient for it to be discovered**.

The mechanism is therefore **coupled and ordered**: `|w2|` must grow *while*
`(w1, b1)` are being shaped. This is consistent with Block C (runs relax onto the
conditional branch, distance 1.09 -> 0.008, then track it) and with the rejection
of the reverse mechanism (no post-crossing acceleration in `|w2|`). Neither
"growth causes placement" nor "placement causes growth" is right on its own.

**The paper should say**: `R` predicts *when sign-correct placement is stable*,
verified causally; it does not by itself predict *when gradient descent finds it*,
and the intervention data say plainly that it cannot.
