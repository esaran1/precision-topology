# Part 1: the cost inversion. Appendix consistency check (DEMOTED before it ran).

**Status fixed before the numbers existed** (`arrhenius_prediction.md`,
committed 2026-09-07): this check's resolution (+-0.4307) **exceeds the
half-width of the band it tests** (0.26), so it **cannot falsify the
prediction**. It can confirm sign and order of magnitude and catch a gross
disagreement between the two measurement axes or an arithmetic error in the
inversion. It is reported as an appendix consistency check, **not** as a
headline and **not** as a figure.

## The inversion

From eps_onset ~ B^{-0.734}, the compute needed to realize a capability within
eps of the analytic threshold is

    **B(eps) ~ eps^{-1.36}**   interval [1.151, 1.670]

**Halving eps costs 2.57x compute** (interval 2.22x-3.18x). The uncertainty
propagates from the onset exponent's interval [-0.8691, -0.5989].

## Direct verification along the other axis

Bisecting on **budget** at fixed a (not on a at fixed budget), 30 seeds per
cell, 2x geometric ladder:

| a | eps | budget for 50% | bracketed |
|---|---|---|---|
| 1.50 | 0.50 | 4,000 | yes |
| 1.35 | 0.35 | 4,000 | yes |
| 1.25 | 0.25 | 8,000 | yes |
| 1.15 | 0.15 | 16,000 | yes |
| 1.10 | 0.10 | 16,000 | yes |

**BRACKETED CELLS: 5 of 5.**

**Measured cost exponent: -1.0242**, against the registered -1.3624 and band
[-1.6698, -1.1506]: **outside the band by 0.1264**.

## Interpretation, constrained by the pre-registered limitation

- The miss (0.1264) is **0.29 of one resolution unit** (+-0.4307), and the
  distance to the prediction (0.3382) is **within** one resolution unit.
- **This check cannot adjudicate**, as stated before it ran. It neither
  confirms nor falsifies -1.3624.
- What it **does** establish: the cost exponent measured along the budget axis
  is **negative, of order 1, and within one resolution unit of the value
  inverted from the onset axis**. No gross disagreement between the two axes,
  and no arithmetic error in the inversion.

**The headline cost statement therefore rests on the onset-axis measurement
(6 bracketed cells, resolution +-0.114) and its algebraic inversion**, with
this budget-axis check recorded as consistent-but-not-decisive.

## Asymptotics, stated so the claim is not misread

As B -> infinity, eps_onset -> 0: **the gap closes in the infinite-compute
limit and the analytic boundary is recovered.** The claim is not a permanent
gap. At every *finite* budget the effective boundary is displaced, by an
amount that grows as the threshold is approached.

**Extrapolation caveat.** alpha is not range-stable -- it falls monotonically
from ~1.51 (1k-4k) to ~0.72 (40k-160k) -- so growth is sub-power-law and
**B(eps) must not be extrapolated beyond the measured range (<=160k steps)**.
The cost exponent quoted here is the value over that range, not an asymptotic
constant.
