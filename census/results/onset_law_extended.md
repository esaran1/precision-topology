# 2a: six bracketed onsets. The exponent test is no longer underpowered.

Same bracketing standard as before: an onset is *located* only if some `a`
gives rate >= 50% and a strictly smaller one gives < 50%. Data:
`onset_law_extended.csv`, `onset_curves_extended.csv`.

| budget | onset | bracketing evidence |
|---|---|---|
| 2,000 | 1.60 | 1.60: 0.550 -> 1.55: 0.450 |
| **4,000** | **1.32** | 1.32: 0.525 -> 1.30: 0.450 |
| 8,000 | 1.18 | 1.18: 0.850 -> 1.16: 0.475 |
| 32,000 | 1.06 | 1.06: 0.650 -> 1.05: 0.000 |
| **64,000** | **1.04** | 1.04: 0.675 -> 1.035: 0.000 |
| 128,000 | 1.03 | 1.03: 0.700 -> 1.025: 0.000 |

**BRACKETED CELLS: 6 of 6.**

**Six-point exponent = -0.7340, R^2 = 0.9900**, against the registered
prediction **-0.7448** (band [-0.895, -0.595], fixed before any onset was
measured). **Within band**, 1.5% from the point prediction.

The four-point fit gave -0.7275 (R^2 = 0.9865). Adding two independent
budgets moved the exponent by 0.0065 and raised R^2 -- the law is stable
under extension rather than an artifact of four points.

Status change: the exponent test was previously reported as "underpowered,
consistent-with". At n = 6 spanning a 64x budget range with R^2 = 0.99 it
carries an R^2 legitimately. It remains a single-family, single-optimizer
measurement (2b tests the second family).
