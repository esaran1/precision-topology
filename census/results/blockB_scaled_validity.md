# Deviation from the frozen Block B procedure, for the cross-family follow-up only

**Justified by a validity-gate failure alone, not by any threshold value.**

## The failure

At the follow-up's `a` (1.01–1.04) the frozen procedure's random starts
(`w₁ ∈ [−2, 2]`, `b₁ ∈ [−4, 4]`, Adam lr 3e-2) are set in raw units, while the
fold is `~√ε ≈ 0.14` wide and `|w₂|` is in the hundreds. At `a = 1.02`:

- **48 of 50** screening starts collapse to the degenerate constant predictor, at
  every `R` tried (0.25, 0.35, 0.5);
- **a start placed at the certified optimum also collapses** at `R ≥ 0.35`, because
  one Adam step moves the placement out of the dip.

It returned no threshold at all for family A at `a = 1.01` or `1.02`.

## The change

`src/blockB_scaled.py`: the same minimisation in limit coordinates,
`w₁ = s·u`, `b₁ = c + s·v`, logit `= W·(f(c + sσ) − f(c))/ε^{3/2} + b₂′`, with
`W = w₂ε^{3/2}`. Restarts, screen/keep, inner steps, Adam and its schedule, the
log-2 exclusion and the population data are the frozen values. The random box is
the frozen box read in `(u, v, b₂′)`, with `W` in place of `|w₂|`. Gap and dense
check are evaluated on raw parameters.

## Gate, passed before use

At `a = 1.30`, where the frozen procedure works, the scaled procedure must
reproduce the frozen thresholds within one frozen grid step (`ΔR = 0.00215`):

| family | threshold | frozen | scaled | difference | pass |
|---|---|---:|---:|---:|---|
| A | `R_glob` | 0.215254 | 0.213593 | −0.00166 | yes |
| A | `R_solve` | 0.307814 | 0.306153 | −0.00166 | yes |
| q2 | `R_glob` | 0.204983 | 0.204983 | +2.4e-07 | yes |
| q2 | `R_solve` | 0.295390 | 0.295390 | +1.3e-07 | yes |

All four pass. The scaled procedure is used for **both** families at every
follow-up `a`, so the comparison is like for like. The frozen procedure is
unchanged everywhere else.
