# Registration: Block G — out-of-distribution task windows, predicted training-free

**Written before any run on a modified window.** Date: 2026-09-21.

Every threshold measured so far uses one task: `I = [-0.8, 0.8]` inner,
`O = +-[1.2, 2.0]` outer. `Ghat(a)`, `R_glob(a)`, `R_solve(a)` and the constant
`K = 0.579454926` are all computed **on those windows**. Nothing in the account
is specific to them: `Ghat` is a supremum over placements of a class gap, and
`R = |w2| Ghat/2` is a margin capacity, both defined for any pair of windows.

So the account makes predictions on windows it was never calibrated on, and those
predictions require **no training run at all**.

## The windows

Four variants, each changing one thing about the task and nothing else:

| tag | `I` | `O` | what changes |
|---|---|---|---|
| `base` | `[-0.8, 0.8]` | `+-[1.2, 2.0]` | the calibrated task |
| `G1_wide_gap` | `[-0.6, 0.6]` | `+-[1.4, 2.0]` | a **wider** margin between classes |
| `G2_narrow_gap` | `[-0.95, 0.95]` | `+-[1.05, 2.0]` | a **narrower** margin |
| `G3_far_outer` | `[-0.8, 0.8]` | `+-[1.2, 3.0]` | the outer window **extended outward** |
| `G4_shifted` | `[-0.4, 1.2]` | `+-[1.6, 2.4]` | windows **asymmetric about 0** |

`G4` is the hardest case for the account: the task is no longer symmetric, so the
fold must be placed off-centre and the two outer arms are no longer equivalent.

## What is computed with no training

For each window pair and each `a` in {1.30, 1.50}: recompute `Ghat` as the
supremum of the oriented class gap over placements `(w1, b1)`, then run the
**frozen** Block B procedure (sha256 `9f1b10741d8bf48c`, grid step **0.01** —
the refinement the `a = 1.60` audit showed is necessary) to get `R_glob` and
`R_solve` on those windows.

## Registered predictions

**G-1 (the threshold transfers in `R`, not in `|w2|`).** Measured crossing `R`
on each window will lie within **15%** of that window's own predicted `R_glob`,
the same tolerance the base task met (measured/predicted 1.086 to 1.122).

> **Falsified** if any window's measured crossing misses its predicted `R_glob`
> by more than 15%, or if the miss is not one-signed (the base task's offset is
> always **above** the prediction, attributed to relaxation lag; a below-prediction
> crossing would contradict the lag account as well).

**G-2 (`|w2|` does not transfer).** Across the five windows at fixed `a = 1.30`,
**`CV(R at crossing) < CV(|w2| at crossing) / 2`** — the same criterion as
registered outcome O1, now across *tasks* instead of across *activations*.

> This is the sharper test. O1 varied `a` and held the task fixed; G-2 varies the
> task and holds `a` fixed. If `R` is the right variable, both should collapse.
> **Falsified** if `CV(R) >= CV(|w2|)/2`.

**G-3 (`Ghat` is what moves).** The predicted ordering of `Ghat` at `a = 1.30`,
committed before computing it: `G2_narrow_gap < base < G1_wide_gap`, because a
wider class margin admits a larger achievable gap. `G3_far_outer` is predicted
**>= base** (a superset of the outer window can only help the supremum, since the
inner window is unchanged — this one is a near-certainty and is stated so as to be
checkable, not as a discriminating test). `G4_shifted` is **not predicted**; it is
recorded as exploratory.

**G-4 (the scaling limit still holds).** `kappa = Ghat/D(a)` will differ between
windows — it is window-dependent, unlike `D(a)` — but for each window,
`Ghat/eps^{3/2}` will still converge to that window's own `K` as `a -> 1+`, and
`K` computed on `h(sigma)` for that window will match to within **3%** at
`a = 1.02`, as S-1 held for the base task at 0.38%.

> **Falsified** if any window's `kappa_0` misses its measured `kappa(1.02)` by
> more than 3%. This tests whether the scaling-limit derivation is a property of
> the activation (as claimed) or was a coincidence of the base windows.

## Measurement protocol

Training runs: `a` in {1.30, 1.50}, 40 seeds, budget 12,000 (the base task's
crossings run to 6,839 steps and the Block E pilot shows control runs place around
2,000-4,500, so 12,000 gives margin), Adam lr 1e-2, float64, the shared-dtype
initialisation used throughout. Crossing = first checkpoint with oriented gap > 0,
logged every 50 steps, with `R` recorded at that checkpoint.

Nothing about the windows is tuned. `G1`-`G4` were chosen and written down here
before any of their `Ghat` values were computed.
