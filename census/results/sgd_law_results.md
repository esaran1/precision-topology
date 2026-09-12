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
than directionally** -- through-origin slope 1.0984 [0.958, 1.239] (five-point fit, q2 and family A both included at matched beta = 1.5) against an
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

## Where the missing dependence may live: one look at distribution SHAPE

The discrepancy has a sign -- SGD moves **slower** than its alpha predicts,
not faster -- which prompted a single check: do the two optimizers differ in
the *shape* of their terminal weight distribution, or only in its scale?
Scale-free metrics, matched a = 1.25 and budget, n = 60:

| budget | optimizer | median \|w2\| | IQR/median | log-sd | p10 / p90 |
|---|---|---|---|---|---|
| 8,000 | Adam | 17.45 | **0.22** | 0.737 | 9.86 / 20.34 |
| 8,000 | SGD | 7.02 | **0.90** | 1.153 | **0.57** / 7.38 |
| 32,000 | Adam | 78.38 | **0.05** | 0.817 | 71.14 / 81.20 |
| 32,000 | SGD | 17.07 | **0.70** | 1.373 | **0.73** / 17.69 |

**The distributions differ in shape, not merely in scale.** Adam's terminal
weights concentrate tightly and tighten further with budget (IQR/median
0.22 -> 0.05); SGD's stay broadly dispersed (0.90 -> 0.70), with a p10 that
barely moves (0.57 -> 0.73) while its median grows 2.4x.

So under SGD a substantial minority of runs are **left behind** near |w2| ~ 1
regardless of budget, while the median advances. Since the onset is a **50%
solve-rate threshold**, it is set by the *lower* part of the distribution, not
the median -- and SGD's lower tail is nearly budget-independent. That would
depress the onset exponent below what the median-based alpha predicts, which
is the observed direction.

**Stated as a hypothesis consistent with one observation, not a result.** It
was not registered, it rests on two budgets at one a, and it was not pursued
further. What it establishes is only that **the missing dependence plausibly
concerns the spread of where training lands rather than how far it travels on
average** -- which is a different quantity from alpha, and would explain why
alpha alone is insufficient.


> **Correction (2026-09-11).** The through-origin slope quoted here was a **four-point** fit that merged q2 and family A into a single point at beta = 1.5 and used family A's value (−0.7340), discarding q2's measured −0.6749. The correct **five-point** fit, using every measured family, is **1.0984 [0.958, 1.239]**, which is **1.7%** from alpha = 1.1173 rather than 0.6%. Intervals still overlap; the agreement is weaker than stated. See `paper/rounded_source_audit.md`.
