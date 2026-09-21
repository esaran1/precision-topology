# Blocks B and C: the conditional landscape, and whether runs track it

Registered in `results/blockB_prediction.md` (commit before prospective cells).
float64 throughout. Population loss = dense uniform sample of the windows,
800 points.

---

## B: the switch points at `a = 1.30`

| quantity | `|w2|*` | `R` |
|---|---:|---:|
| `R_glob` — global minimiser first has `G > 0` | 5.00 | **0.2145** |
| `R_spin` — upper spinodal, branch followed upward | 5.00 | **0.2145** |
| hysteresis window `R_spin - R_glob` | — | **0.0000** |

**REGISTERED FALSIFIER TRIGGERED: there is no hysteresis window.**

Following the `G <= 0` branch upward by warm continuation from `|w2| = 2.0`,
the gap rises **smoothly through zero**:

| `|w2|` | 4.85 | 4.90 | 4.95 | **5.00** | 5.05 | 5.10 |
|---|---:|---:|---:|---:|---:|---:|
| gap | −0.00369 | −0.00196 | −0.00027 | **+0.00139** | +0.00301 | +0.00460 |
| d(gap) | +0.00177 | +0.00173 | +0.00169 | +0.00166 | +0.00162 | +0.00159 |

The increments are monotone and continuous, with no jump and no vanishing
branch. **The crossing is transcritical, not a saddle-node**: there is one
branch, not two, so there is nothing to be bistable and nothing to be
hysteretic.

**Consequences, per the registration**: the "bifurcation with hysteresis"
framing is **dropped, not repaired**. Block E runs the **reduced arm set**
(control, null, freeze-low, freeze-high, jump) at the single switch point;
freeze-mid and down-hold-mid are undefined here and are not run.

**What survives**: a single, sharp, training-free switch point at
`R = 0.2145`, against a measured crossing of **0.2332** — an offset of
**+8.7%** that Block C explains.

## C: runs relax onto the branch, then track it

Distance between the run's `(w1, b1, b2)` and the conditional minimiser
**warm-started from the run itself** — i.e. the minimiser on the branch the run
occupies — 408 checkpoints over 30 runs at `a` = 1.30 and 1.40:

| phase | median relative distance |
|---|---:|
| steps 0–100 | 1.0883 |
| steps 100–300 | 0.8341 |
| steps 300–1,000 | 0.3908 |
| steps 1,000–3,000 | 0.1490 |
| **within 400 steps of crossing** | **0.0080** |
| after crossing | **0.0017** |

**Registered criterion (median < 0.2 = tracking): met**, overall median 0.0065,
and met from about step 1,000 onward.

**Read this correctly.** The large early distances are **not** a failure of the
adiabatic picture: runs start at random initialisation, far from any minimiser,
and *relax onto* the branch. By the time the switch matters they are within
**0.8%** of it, and after crossing within **0.17%**. The picture holds where it
is being used.

**The offset, quantified.** A run tracking at relative distance 0.008 sits
slightly short of the exact minimiser. The branch has slope
`d(gap)/d|w2| ≈ 0.033` per unit, and the measured offset
`0.2332 − 0.2145 = 0.0187` in `R` is `0.44` in `|w2|`, i.e. a residual gap
deficit of **0.0144** carried at the switch. That is the lag, and it is small,
one-signed and consistent — a run crosses slightly *after* the landscape says
it could, because it is still catching up.

**An artifact of the earlier "before crossing" pooling, recorded so it is not
repeated**: pooling all pre-crossing checkpoints gives median distance 0.834
and makes the runs look untracked. That figure mixes step-50 checkpoints
(distance ~1.09, barely moved from init) with near-crossing ones (0.008). The
pre-crossing median is dominated by early steps and says nothing about the
switch. Split by step, as above.

## What this means for the paper

- The threshold is **derived, not fitted**: `R = 0.2145` at `a = 1.30` comes
  from the loss landscape with **no training run in it**.
- The mechanism is **a continuous switch in the conditional minimiser**, not a
  bifurcation. The paper should say so plainly; "bifurcation" and "hysteresis"
  are not supported.
- Runs track the conditional branch near the switch to under 1%, which is what
  makes the derived threshold predictive rather than coincidental.
