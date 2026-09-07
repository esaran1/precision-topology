# Part 2b: does the onset law survive a different optimizer?
# DIRECTIONAL-ONLY. Registered prediction missed; alpha-dependence holds in
# direction and order of magnitude, not quantitatively.

Registered 2026-09-07 **before any SGD onset run**, including the
partial-outcome scoring and the band decomposition. Data: `sgd_onsets.csv`,
`sgd_onset_curves.csv`.

## BRACKETED CELLS: 4 of 4

| budget | SGD onset | Adam onset (same budgets) |
|---|---|---|
| 2,000 | 1.60 | 1.60 |
| 8,000 | 1.38 | 1.18 |
| 32,000 | 1.22 | 1.06 |
| 128,000 | 1.16 | 1.03 |

Every cell located under the standing criteria (some a at >=50%, a strictly
smaller a below).

## Verdict: OUTSIDE the registered band

| quantity | value |
|---|---|
| measured SGD onset exponent | **-0.3255** |
| registered prediction (-alpha_SGD/beta) | -0.4792 |
| registered band | [-0.5973, -0.3611] |
| Adam's measured exponent | -0.7340 |

**Missed the band by 0.0356.** Scored by the rule registered in advance:

- distance to the SGD prediction: **0.1537**
- distance to Adam's value: **0.4085** -- **2.7x further**
- therefore "clearly nearer -0.4792 than -0.7340" => **DIRECTIONAL-ONLY**

**This is reported as directional-only, not as a qualified success**, per the
rule fixed before the number was visible: *the alpha-dependence holds
directionally while the quantitative prediction misses.* Weaker than a hit,
considerably stronger than a null.

## Band decomposition, as registered

The +-0.1181 band is **93% grid resolution** (+-0.1140) and 7% alpha
uncertainty (+-0.0311). **The miss (0.0356) is smaller than one grid
resolution unit (0.1140)** -- about a third of it. So the miss is consistent
with the onset grid being unable to resolve the prediction, rather than with
alpha being wrong. Recorded because "the prediction is wrong" and "our
instrument cannot resolve the prediction" are different findings, and the
decomposition was fixed in advance so this diagnosis could not be chosen
afterwards.

## What is robust: the optimizers separate, and largely as predicted

The finding that discriminates the hypotheses is the **separation**, which is
large and unambiguous:

- SGD's onset moves from 1.60 to 1.16 across 64x budget; Adam's moves 1.60 to
  1.03 across the same range.
- SGD's exponent is **-0.3255** against Adam's **-0.7275** on matched cells.
- Had SGD reproduced Adam's exponent, the law's alpha-dependence would be
  **falsified**. It did not.

## The diagnostic that cuts against us, reported plainly

The law predicts the exponent ratio should equal the alpha ratio:

    exponent ratio (SGD/Adam) = -0.3255 / -0.7275 = **0.447**
    alpha ratio     (SGD/Adam) =  0.7188 / 1.1172 = **0.643**

**These do not match** (0.447 vs 0.643, a 30% discrepancy). SGD's onset moves
*more slowly* than its alpha alone predicts. So the optimizer does not enter
**only** through alpha, as the strong form of the law claims -- something else
about SGD's trajectory also slows the onset. This is stated as a limitation of
the law, not smoothed over: the ratio test is a cleaner comparison than either
exponent alone, because grid resolution partially cancels, and it fails.

## Consequence for the rest of the paper

With the strong form unsupported here, **the four-family relationship (T44) is
now the only relationship in this project that holds quantitatively rather
than directionally** -- through-origin slope 1.1240 [1.008, 1.240] against an
independently measured alpha of 1.1173 [0.999, 1.236], with all four families
overlapping their propagated intervals. It is correspondingly more
load-bearing, and its own limitations (n = 4 families, per-family grid
resolution +-0.114 to +-0.171, alpha not range-stable) matter more than they
did when the SGD result was expected to corroborate it.

## Status of the claim

- **Not an Adam artifact.** The law's qualitative content -- onsets move with
  budget, and slower-growing optimizers move them more slowly -- holds under
  two optimizers with different growth exponents.
- **Not quantitatively optimizer-independent.** The strong form ("the
  optimizer enters only through alpha") is **not supported**: the exponent
  ratio is 0.447 against the predicted 0.643.
- 2c (a second family under SGD) is **not run**: its premise was that 2b
  landed, and it did not.
