# Registration: the relaxation-lag account, made predictive

**Written before any crossing `R` has been measured for AdamW or SGD, and
before any per-`a` offset has been regressed on growth rate.** The only inputs
are (i) the Block B switch points, frozen and committed, and (ii) `|w2|` growth
rates at `a = 1.25` already in `alpha_composition.csv` and `r_adamw.csv`.

Date: 2026-09-21.

---

## The account to be tested

Block C showed runs relax onto the conditional branch and then track it, and
attributed the **+8.7%** offset between measured crossing (0.2332) and predicted
switch (0.2145) at `a = 1.30` to **relaxation lag**: a run carries a residual
gap deficit because `(w1, b1, b2)` chase a target that `|w2|` keeps moving.

That is currently a **reconciliation**. It becomes a **result** only if the lag
predicts something not used to construct it. A lag of this kind should scale
with the **ratio of the driving rate to the relaxation rate**: the faster `|w2|`
grows relative to how fast `(w1,b1,b2)` can re-minimise, the further behind the
run sits when the switch passes, and the larger the overshoot in `R`.

## Measured growth rates at `a = 1.25`, near the crossing region

Local `d|w2|/dstep` where `|w2|` passes through the crossing region:

| optimiser | local rate | relative to Adam |
|---|---:|---:|
| Adam | **0.00207** | 1.00 |
| AdamW | **0.00184** | 0.89 |
| SGD | **0.00075** | 0.36 |

## Registered predictions

**F-1 (ordering).** Measured crossing `R` at `a = 1.25`, by optimiser:

> **`R_cross(Adam) > R_cross(AdamW) > R_cross(SGD)`**, and **AdamW and SGD both
> sit closer to `R_glob` than Adam does.**

Slower growth means less lag means a crossing nearer the landscape's own switch.

**F-2 (magnitude).** Taking Adam's offset at `a = 1.25` as the reference, the
offsets scale with the rate ratio:

> predicted offset(optimiser) ≈ offset(Adam) × (rate_optimiser / rate_Adam)

With Adam's offset assumed comparable to the `a = 1.30` value (+8.7%), this
gives predicted offsets of roughly **+8% (Adam)**, **+7% (AdamW)** and
**+3% (SGD)**. Registered tolerance: **each within ±4 percentage points**, and
the **ordering** in F-1 must hold, which is the sharper requirement.

**F-3 (across `a`, same account).** Within Adam, the per-`a` offset
`R_measured(a)/R_spin(a) - 1` should **correlate positively** with the local
`|w2|` growth rate at that `a`. Registered: **Spearman correlation > 0** across
the six `a` values.

## What falsifies the account

- **Ordering reversed or flat** in F-1: the offset is not driven by growth rate,
  and "relaxation lag" is a label rather than a mechanism. Report as such and
  drop it.
- **F-3 correlation <= 0**: same conclusion within a single optimiser, which is
  the cleaner test since nothing else changes.

If F-1 and F-3 both hold, the lag is a **result**: the same mechanism explains
the offset's size across three optimisers and six activation values, and Block
F's "three optimisers agree on `R`" becomes explained rather than observed.

## D-1. The DISCRIMINATING test: `R_spin + lag` against `R_glob + lag`

**Registered before any lag is fitted.** Added 2026-09-21 after the timestamp
audit; this is a *new* registration, not an amendment to an existing one.

**Why it is needed.** The registered "crossings track `R_spin`" test is **not
discriminating**. `R_spin >= R_glob` by construction and every measured crossing
exceeds `R_spin`, so "closer to `R_spin`" follows from the ordering alone and
would hold equally if the true switch were `R_glob` and the run simply
overshot it. Both accounts predict the same sign.

They differ in **how much** of the offset the lag must carry:

- **Spinodal account**: the run leaves when its branch vanishes, at `R_spin`.
  Residual offset above `R_spin` is lag, and should be **small**.
- **Global-switch-plus-lag account**: the landscape switches at `R_glob`; the
  run overshoots by lag alone, which must therefore be **larger**, covering
  `R_measured - R_glob`.

**The test.** Predict lag from growth rate alone, with a single proportionality
constant `c` fitted **once** across all `a`:

    lag(a) = c * rate(a)

where `rate(a)` is the local median `d|w2|/dstep` near the crossing, measured
and fixed now:

| `a` | 1.30 | 1.35 | 1.40 | 1.45 | 1.50 | 1.60 |
|---|---:|---:|---:|---:|---:|---:|
| rate (per step) | 0.00239 | 0.00241 | 0.00238 | 0.00241 | 0.00250 | 0.00260 |

Then compare two one-parameter models against the six measured crossings:

- **M-spin**: `R_pred(a) = R_spin(a) + c * rate(a)`
- **M-glob**: `R_pred(a) = R_glob(a) + c * rate(a)`

**Scored by RMS relative error across the six `a`, each fitting its own `c`.**

**Registered tolerance and verdict rule**:

> **M-spin wins** if its RMS relative error is **below M-glob's by at least a
> factor of 1.5**. **M-glob wins** on the mirror. Otherwise **neither is
> distinguished**, and the paper says the data do not separate the spinodal
> from a global switch with lag.

**Registered prediction: neither will be distinguished.** Grounds, stated now:
the measured rates vary only **9%** across `a` (0.00239 to 0.00260) while the
offsets vary more, so `c * rate(a)` is nearly a *constant* offset in this data
and both models reduce to "predicted switch plus a constant". The data lack the
leverage to separate them. **This is registered as the expected outcome so that
a null result is not reported as a failure of the landscape account** — the
landscape account survives either way; what is undecided is *which* switch
point the run leaves at.

**If this is right, the discriminating evidence must come from Block E**, where
`freeze-mid` (hold `R_glob < R < R_spin`) directly separates them: placement
should be **achieved** under the global-switch account and **never achieved**
under the spinodal account. That is why the `a = 1.60` grid refinement below
matters.

## Block B procedure, frozen before this comparison

Committed and unchanged from here on:

| setting | value |
|---|---|
| `RESTARTS` | 50 |
| `STEPS` | 3,000 (convergence-justified at `a = 1.30`) |
| `SCREEN` / `KEEP` | 600 / 8 |
| degeneracy test | `\|loss − log 2\| < 1e-4` or `\|w1\| < 1e-3` |
| grid | coarse 0.5 then fine 0.05, range [1.5, 11.0] |
| loss data | `population_data()`, dense uniform, 800 points |
| inner optimiser | Adam, lr 3e-2 → 5e-3 at half |
| source sha256 (first 16) | `9f1b10741d8bf48c` |

**Any further change must be justified by convergence diagnostics alone, never
by agreement with measured crossings, and must be logged in `SESSION_LOG.md`.**

## Block E, reduced arm set — reversibility registered

The branch is continuous at `a = 1.30` (`R_fold ≈ R_glob ≈ R_spin ≈ 0.21`
within a grid step), so the test is **reversibility**, which is still a clean
causal test:

| arm | registered prediction |
|---|---|
| control | reference |
| null (`(w2,b2)` × (1 + 1e-6)) | rate equal to control; the noise floor |
| freeze-low (`R < R_glob`) | **placement never achieved** |
| freeze-high (`R > R_spin`) | placement achieved within the relaxation time |
| jump (scale to just above `R_spin`) | placement well before control |
| **down-hold-low** (placed, then scaled and frozen below `R_glob`) | **placement lost** |
| **down-hold-high** (placed, then scaled down but held above `R_glob`) | **placement kept** |

The last two are the reversibility pair. At `a >= 1.35` Block B reports small
windows (`R_spin - R_fold` up to 0.028), so those `a` additionally permit a
narrow freeze-mid arm; it is **not** registered as a primary test because the
window is only a few grid steps wide.
