# Registration: ε-dependence of weight growth, and the corrected onset exponent

Junyu's theory review item **#10**. **Written before any two-dimensional
`|w₂|(B, ε)` fit has been run.** Date: 2026-09-22.

## The gap being closed

The onset law is `ε_onset ~ B^{−α/β}`, derived by combining

- `|w₂|` growth with budget: `|w₂| ~ B^α`, committed **α = 1.1172** (eight
  budgets 1k–160k, stallers included, `alpha_derivation_results.md`), and
- the requirement's divergence with ε: `|w₂|_req ~ ε^{−β}`, **β = 1.4963**
  analytically for family A (`β = 1 + 1/q` with q = 2).

Setting `B^α ≈ ε^{−β}` gives the onset. **The derivation assumes `α` is
independent of ε** — that `|w₂|(B, ε)` factorises as `B^α` times a function of ε
with no cross-term. If instead

    |w₂|(B, ε)  ~  B^α · ε^{−ν}

with `ν ≠ 0`, then the growth *itself* is ε-dependent and the balance
`B^α ε^{−ν} ≈ ε^{−β}` gives

    **ε_onset ~ B^{−α/(β − ν)}**

so the committed exponent −α/β = −0.7465 is an approximation valid only for
`ν ≈ 0`. This has never been measured: `budget_alpha.csv` contains a **single**
`a` value (1.25), so no ε-dependence of `α` could have been detected.

## Measurement

Two-dimensional grid, **8 budgets × 6 ε**, 40 seeds per cell, float64, Adam
lr 1e-2, family A. Budgets 1k/2k/4k/8k/16k/32k/64k/128k; `a` = 1.05, 1.10, 1.20,
1.30, 1.45, 1.60 so `ε = a − 1` spans 0.05–0.60, a 12× range bracketing the
onset region.

Per cell record the **median terminal `|w₂|`**, matching the estimator
`budget_alpha.csv` used for the committed α. Fit
`log|w₂| = c + α·log B − ν·log ε` by least squares; uncertainty on `ν` by
**cluster bootstrap over cells** (2,000 resamples), which is the convention used
for every other interval in this paper.

**Near the moving onset**: the brief asks for the fit *near the onset*, so the
primary fit is restricted to cells within a factor of 3 in ε of that budget's own
onset, read from `onset_law.csv` (`ε_onset ~ B^{−0.7275}`). The unrestricted fit
over all 48 cells is reported alongside as a robustness check, not as the primary.

## Registered predictions

**N-1 (the value of ν).** No mechanism predicts a nonzero ν: `|w₂|` growth under
Adam is driven by the logit-scaling pressure of cross-entropy, which does not
reference the activation's fold depth. Registered:

> **`ν` is consistent with zero**, meaning its 95% cluster-bootstrap interval
> contains 0.

**N-2 (the consequence, either way).** Stated in advance so neither outcome can
be reported selectively:

> - If `ν`'s interval **contains 0**: the committed `−α/β` stands **as an
>   approximation**, and the paper says so explicitly, quoting `ν`'s interval as
>   the bound on the correction.
> - If `ν`'s interval **excludes 0**: the corrected exponent `−α/(β − ν)` replaces
>   `−α/β` throughout, and the change in the predicted onset exponent is reported
>   against the **grid resolution 0.114** already used as the materiality scale in
>   T50 and T44.

**N-3 (materiality threshold).** Registered before fitting: the correction matters
only if it moves the predicted onset exponent by more than **0.114**, the onset
grid resolution. Solving, `−α/(β−ν)` differs from `−α/β` by more than 0.114 when

    |ν| > β²·0.114/(α + β·0.114) ≈ 1.4963² · 0.114 / (1.1172 + 0.1706) ≈ **0.198**

> So: **`|ν| < 0.198` ⟹ immaterial even if statistically nonzero**, and that is
> the number to compare against, not zero. Registered now so the comparison is
> not chosen after seeing the interval.

## Falsifiers

- The 2D fit has `R² < 0.90`: the power-law form is wrong and neither `α` nor `ν`
  is well defined. Report and stop.
- `α` from the 2D fit differs from the committed **1.1172** by more than its own
  bootstrap interval: the two estimators disagree and the discrepancy must be
  resolved before `ν` means anything.
