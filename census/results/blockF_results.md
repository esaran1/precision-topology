# Block F: the lag account's ordering survives as a point estimate; its magnitude does not

Scored against `results/blockF_lag_prediction.md`. Artifacts:
`blockF_optimisers.csv`, `src/blockF_optimisers.py`. `a = 1.25`, 60 seeds per
optimiser, 20,000 steps, float64. SGD lr 0.3 (the only value that solves at all);
Adam/AdamW lr 1e-2.

**No existing artifact had per-optimiser crossing `R`** — `r_adamw.csv`,
`alpha_composition.csv` and `r_pooled.csv` all record *terminal* `|w2|`, not the
placement-crossing step — so the crossing was instrumented fresh, the way Block A
did it.

---

## Measured

| optimiser | crossed | median cross `R` | median cross step | median local rate | rate / Adam | registered rate |
|---|---:|---:|---:|---:|---:|---:|
| Adam | 57/60 | **0.23094** | 3,975 | 0.002471 | 1.000 | 0.00207 |
| AdamW | 60/60 | **0.22980** | 4,362 | 0.001908 | 0.772 | 0.00184 |
| SGD | 43/60 | **0.22888** | 7,950 | 0.000631 | 0.255 | 0.00075 |

The measured rates reproduce the registered ones closely (within 6–19%), and the
**3.9× spread** in growth rate across optimisers is the lever the whole account
rests on.

`R_glob(1.25) = 0.21066` (`|w2| = 6.36`), computed here with the frozen Block B
procedure at grid 0.01 — Block B itself ran `a ≥ 1.30`, so this value is new.

## F-1: the ordering holds as a point estimate, but is NOT statistically supported

`R(Adam) = 0.23094 > R(AdamW) = 0.22980 > R(SGD) = 0.22888` — **the registered
ordering holds exactly.**

It is not meaningful. Bootstrap over seeds (4,000 resamples of the median):

| optimiser | 95% CI of the median crossing `R` |
|---|---|
| Adam | [0.22542, 0.23829] |
| AdamW | [0.22398, 0.23627] |
| SGD | [0.22332, 0.23823] |

- `P(median Adam > median AdamW)` = **0.589**
- `P(median AdamW > median SGD)` = **0.613**
- **`P(full ordering)` = 0.290**, against **1/6 = 0.167** by chance

The intervals overlap almost completely. A 3.9× change in growth rate moves the
median crossing `R` by **0.9%**, which is inside the within-cell seed scatter.
**Reported as: the ordering is in the predicted direction and the data cannot
distinguish it from noise.**

## F-2: FAILS

Predicted offset = Adam's offset × (rate ratio), each within ±4 pp:

| optimiser | measured offset | rate ratio | predicted offset | error |
|---|---:|---:|---:|---:|
| Adam | 9.626% | 1.000 | 9.626% | 0.000 pp |
| AdamW | 9.084% | 0.772 | 7.432% | 1.652 pp |
| SGD | **8.645%** | 0.255 | **2.458%** | **6.188 pp** |

**SGD misses by 6.19 pp, outside the registered ±4 pp.** The account predicts
SGD's offset should be about a quarter of Adam's; measured, it is **90%** of it.

The offsets span **8.645% to 9.626% — a spread of 0.98 pp — while the rate spans
3.9×.** The offset is essentially **constant** across optimisers. Two consequences:

1. **F-2 is falsified.** The lag's *size* does not scale with the driving rate.
2. **F-1's "pass" is vacuous for the same reason.** With all three offsets inside
   1 pp, the ±4 pp tolerance could not have failed for AdamW either; only SGD's
   3.9× rate ratio made the test bite at all, and it bit.

## F-3: passes, but the test is confounded

Restated against `R_glob` (`R_spin` no longer exists, T60 — the substitution is
forced, not chosen):

| `a` | 1.30 | 1.35 | 1.40 | 1.45 | 1.50 | 1.60 |
|---|---:|---:|---:|---:|---:|---:|
| offset | 0.08646 | 0.10119 | 0.11255 | 0.11961 | 0.12770 | 0.14409 |
| rate | 0.00239 | 0.00241 | 0.00238 | 0.00241 | 0.00250 | 0.00260 |

**Spearman = +0.7827** (registered `> 0`), Pearson +0.8386.

**The caveat is fatal to its evidential value**: rate and offset both rise
monotonically with `a` (`corr(a, rate) = +0.894`, `corr(a, offset) = +0.991`), and
the rates vary only **9%** across the whole range. Any quantity increasing in `a`
would correlate. F-3 cannot separate "rate drives the offset" from "both increase
with `a`", and F-1/F-2 — which vary rate 3.9× at fixed `a` — say the offset does
**not** follow rate.

## Verdict: the relaxation-lag account is a label, not a mechanism

The registration set this out explicitly: *"the offset is not driven by growth
rate, and 'relaxation lag' is a label rather than a mechanism. Report as such and
drop it."*

That is the outcome. The falsifier as written was "ordering reversed or flat in
F-1"; the ordering is neither, but **F-2 — the quantitative form of the same
claim — fails, and F-1 has no power to detect anything**. A 3.9× change in the
driving rate produces a 0.9% change in the offset.

**What the paper should say.** Measured crossings sit **above** `R_glob` by a
**consistent ~9–14%** across six activation values and three optimisers
(8.6–9.6% at `a = 1.25`; 8.6–14.4% over `a = 1.30–1.60`). That offset is **robust
and one-signed** — which is itself worth reporting, and is what makes the
training-free prediction useful. But its **size is not explained** by growth-rate
lag, and the paper must not claim it is.

**What survives from Block C**: runs do relax onto the conditional branch and
track it (distance 1.09 → 0.008, then under 1%). That observation is unaffected —
it says runs follow the branch, not that the residual offset scales with how fast
they are driven along it.

**Cross-reference.** This is the second independent failure of a rate-based
account: Block H's dose-response also could not establish one, its validity gate
failing because the natural `|w2|` trajectory is non-monotone
(`blockH_rate_results.md`). Two different experiments, neither supporting rate as
the governing variable.
