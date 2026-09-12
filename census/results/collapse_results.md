# Items 1-3: data collapse, threshold independence, held-out prediction

Registered in `results/collapse_prediction.md` before any of this was
computed. Pure reanalysis of committed artifacts; no training.

Date: 2026-09-12.

---

## Headline

**Registered outcome 2 for Item 1, and a registered FAILURE for Item 3.**

The rate curves' **location** scales as the law predicts — the collapse
midpoints agree to 1.39x across a 32x budget range, and the minimizing
exponent `theta = 0.7250` sits inside the onset-derived `0.7340`'s interval.
But the curves' **shape** does not transfer: the logistic slope rises
**7.6-fold**, `k ~ B^{0.612}` with `r = 0.974`, so the transition sharpens
with budget rather than translating. The registered factor-2 selectivity
condition is **not met** (1.62x and 1.48x).

**The law describes the median, not the distribution.** That is a real
narrowing and it belongs in the abstract.

Separately, **the held-out prediction fails**: fitting the four smallest
budgets predicts `eps_onset = 0.01872` at 128k with 95% prediction interval
`[0.01403, 0.02497]`; the measured value is **0.030**, outside the interval,
60.3% above the point estimate.

---

## Item 1: data collapse

### Family A, 36 cells, 6 budgets spanning 64x

Scaling variable `u = (a - 1) * B^{theta}`, single logistic `F(log u)` fit by
least squares on the rate.

| `theta` | RMS residual |
|---|---:|
| 0.0 | 0.2885 |
| 0.5 | 0.2187 |
| **0.7250 (minimum)** | **0.1780** |
| 0.7340 (onset fit) | 0.1780 |
| 1.0 | 0.2416 |
| 1.5 | 0.2634 |

**The minimizing exponent is `theta = 0.7250`**, bootstrap 95% CI over cells
(300 resamples) **[0.5000, 0.8881]**, which contains the onset-derived
**0.7340**. Two estimators using different information — all 36 rate cells
versus 6 crossings — agree.

**But the registered pass condition is not met.** It required the minimum
residual to sit a factor of 2 below the residuals at deliberately wrong
exponents. Observed: **1.62x** against `theta = 0` and **1.48x** against
`theta = 1.5`. The residual curve has a real minimum in the right place, but
it is shallow: wrong exponents are not strongly rejected.

### Why: the curves sharpen rather than translate

Fitting a logistic to each budget separately, in the collapse variable `u`
(rescaling `u` shifts midpoints but cannot change the slope, since
`log u = log eps + theta log B` is a pure translation):

| budget | cells | slope `k` | midpoint (log `u`) | residual |
|---:|---:|---:|---:|---:|
| 4,000 | 5 | 3.94 | +4.846 | 0.0254 |
| 8,000 | 6 | 7.41 | +4.660 | 0.0786 |
| 32,000 | 8 | 21.02 | +4.679 | 0.0636 |
| 64,000 | 7 | 28.76 | +4.777 | 0.0367 |
| 128,000 | 8 | 29.82 | +4.990 | 0.0040 |

- **Midpoints collapse**: spread 0.330 in `log u`, a factor of **1.39** in `u`
  across 32x of budget. This is what the law predicts and it holds well.
- **Slopes do not**: **7.6x** from 3.94 to 29.82, monotone in budget,
  `k ~ B^{0.612}` with `r = 0.974`.

Per-budget residuals are small (0.004-0.079), so each individual curve *is*
well described by a logistic. What fails is that they are not the **same**
logistic. The pooled residual of 0.178 is **3.55x** the RMS binomial sampling
noise (0.050 at 40 seeds/cell), so the misfit is real and not noise.

The transition width in `log eps` therefore narrows as `B^{-0.612}`: at large
budget the boundary between "usually solves" and "usually fails" becomes
sharper, approaching a step. The *location* of that boundary obeys the law;
its *width* is an additional budget-dependent quantity the law does not
describe.

### Other families: the sharpening is not universal

| family | slope scaling | `r` | verdict |
|---|---|---:|---|
| A | `k ~ B^{+0.612}` | 0.974 | sharpens |
| q2 (`beta = 1.5`) | `k ~ B^{+0.428}` | 0.993 | sharpens |
| q1 (`beta = 2.0`) | `k ~ B^{-0.718}` | −0.906 | **broadens** |

q1 moves the opposite way. Whatever governs the shape is not a single
universal function of budget across families, which rules out the simplest
repair (adding one shape exponent to the law).

q4 and q0.667 have too few cells per budget (fewer than 3 at most budgets) to
fit a shape, and are not reported.

### Pooled all-family collapse

84 cells across A, q1, q2, with each family scaled by its own
`theta = alpha/beta_family` (`alpha = 1.1172`):

- per-family `theta = alpha/beta`: residual **0.2340**
- no budget scaling (`theta = 0`): residual **0.2824**, only **1.21x** worse
- best *single* common `theta = 0.700`: residual **0.2269**

The pooled collapse is weak, and a single common exponent fits marginally
**better** than per-family `alpha/beta`. This does not refute the four-family
law — that law is about where onsets sit, and it was verified on onsets — but
it means **the pooled rate curves do not provide independent support for it**.
We report this rather than omitting the negative pooled result.

---

## Item 2: threshold independence

Onsets extracted at 25%, 50%, 75% by linear interpolation between bracketing
cells, bracketing counts stated before the exponents.

| level | bracketed | exponent | SE | 95% CI |
|---|---:|---:|---:|---|
| 25% | **3 of 6** | −0.5037 | 0.0251 | [−0.8223, −0.1851] |
| 50% | **6 of 6** | −0.7325 | 0.0394 | [−0.8420, −0.6230] |
| 75% | **5 of 6** | −0.7325 | 0.0628 | [−0.9323, −0.5326] |

All three intervals overlap on [−0.8223, −0.6230], and **50% and 75% agree to
four decimals** (−0.7325 both). The 50% figure also reproduces the committed
−0.7340 to within 0.0015.

**The 25% level differs by 0.23**, and its interval is wide because only 3 of 6
budgets bracket it — at low budgets no measured cell falls below 25%.

**This is the linkage registered in advance.** The registration predicted that
if Item 2 showed systematic drift, Item 1's collapse would fail, and vice
versa. Both happened, and they are quantitatively consistent: propagating the
fitted per-budget shapes forward predicts a 25% exponent of **−0.6177** and a
75% exponent of **−0.7518**, against measured −0.5037 and −0.7325. The
sharpening explains the direction and rough size of the 25% deviation.

**Verdict**: the 50% choice is not arbitrary in the range where it is well
bracketed — 50% and 75% give the same exponent. But threshold independence is
**not** established across the full range, because the sharpening makes low
thresholds behave differently, and the 25% level is under-bracketed.

---

## Item 3: held-out prediction — FAILED

Registered procedure: fit on the four smallest budgets, record the prediction
and its interval, then reveal the 128k cell.

**Fit on `B` in {2k, 4k, 8k, 32k}**: exponent **−0.8261**.

**Prediction recorded before revealing the held-out cell:**

> `eps_onset(128k) = 0.01872`, 95% prediction interval **[0.01403, 0.02497]**
> (equivalently onset `a` in [1.01403, 1.02497]).

**Measured: `eps_onset(128k) = 0.030`** (onset `a = 1.03`).

**FAIL.** The measured value lies outside the interval, **60.3% above the
point estimate**. The four-budget fit's exponent (−0.8261) is steeper than the
six-budget fit (−0.7340), so extrapolating from short budgets **overestimates
how far the onset moves**.

Two honest qualifications, neither of which rescues the prediction:

1. The 128k onset sits 6 grid steps above `a = 1` at a grid spacing of 0.005,
   so the measured 0.030 carries roughly +-0.005 of grid resolution. The
   interval's upper end is 0.02497; even at 0.025 the measurement is outside.
2. The prediction interval from a 4-point fit with 2 degrees of freedom is
   wide (`t = 4.303`) and still failed, which makes the failure stronger
   rather than weaker.

This is the same curvature visible in `alpha`'s non-stability: exponents
measured over short budget ranges do not extrapolate to long ones. The
consistent reading across Items 1-3 is that **the power law is a good local
description whose exponent drifts with the range it is fitted over.**

### Constructed families

q2 is the only constructed family with >= 4 bracketed cells. Fitting its three
smallest budgets and predicting 128k is a 3-point fit with 1 degree of
freedom, giving an interval too wide to test anything (`t = 12.71`). Not
reported as a test.

---

## What Items 1-3 change

**Retained**: onsets move with budget; the 50% exponent is reproducible
(−0.7325 from interpolation, −0.7340 committed); the collapse minimizer agrees
with it independently (0.7250, CI [0.500, 0.888]); midpoints collapse to 1.39x
over 32x of budget.

**Narrowed**:
1. The law describes the **median crossing**, not the full distribution. Rate
   curves sharpen with budget (`k ~ B^{0.612}` for family A), so the
   distributional form is not budget-invariant. Registered outcome 2.
2. **Threshold independence holds only for 50% and 75%.** The 25% exponent
   differs by 0.23, consistently with the sharpening.
3. **The held-out prediction fails.** Short-budget fits over-extrapolate.
4. **The pooled all-family collapse is weak** (1.21x over the null), so rate
   curves do not independently corroborate the four-family law.

These belong together in the paper: the law's *location* claim survives four
tests and its *distributional* claim does not survive any of them.
