# Block G: the machinery transfers to task windows it was never calibrated on

Registered in `results/blockG_prediction.md` (committed `da8621f`), with one
sub-claim corrected before any crossing was measured
(`results/blockG_g3_correction.md`, `e84c833`). Artifacts:
`blockG_switches.csv`, `blockG_crossings.csv`, `blockG_scaling.csv`,
`src/blockG_windows.py`. float64, 40 seeds per cell, 12,000 steps, Adam lr 1e-2.

Every threshold in this paper was calibrated on `I = [-0.8,0.8]`,
`O = +-[1.2,2.0]`. Block G recomputes `Ghat`, `R_glob` and `R_solve` on four
modified window pairs — **with no training** — and then measures crossings there.

---

## Summary

| prediction | verdict |
|---|---|
| **G-1** measured crossing within 15% of that window's own `R_glob` | **6/10 on the letter; 10/10 once the known per-`a` lag is accounted for** |
| **G-2** `CV(R) < CV(\|w2\|)/2` across windows at fixed `a` | **PASS at both `a`**, 3.4x and 3.0x |
| **G-3** `Ghat` ordering `G2 < base < G1` | **PASS**; the `G3_far_outer` sub-claim was **corrected before measuring** |
| **G-4** each window's `kappa_0` from `h(sigma)` matches its measured `kappa(1.02)` within 3% | **PASS 5/5**, all within **0.81%** |

## G-2: the strongest form of the O1 claim — PASS

O1 varied the activation and held the task fixed. G-2 varies the **task** and
holds the activation fixed. Median values at crossing:

| `a` | window | `R` | `\|w2\|` |
|---|---|---:|---:|
| 1.30 | `G2_narrow_gap` | 0.1437 | **13.6542** |
| 1.30 | `G4_shifted` | 0.1679 | 2.3357 |
| 1.30 | `G3_far_outer` | 0.2264 | 6.3294 |
| 1.30 | `base` | 0.2369 | 5.5191 |
| 1.30 | `G1_wide_gap` | 0.2414 | 2.9011 |

**`|w2|` at crossing spans 2.34 to 13.65 — a 5.8x range — while `R` spans 0.144
to 0.241.**

| `a` | `CV(R)` | `CV(\|w2\|)` | ratio | criterion `CV(R) < CV(\|w2\|)/2` |
|---|---:|---:|---:|---|
| 1.30 | 0.2189 | 0.7358 | **3.36x** | **PASS** |
| 1.50 | 0.2304 | 0.6925 | **3.01x** | **PASS** |

Changing the task changes the required output weight by nearly sixfold and leaves
the threshold in `R` recognisably the same quantity.

## G-1: 6/10 on the letter, and the 4 misses are the documented lag — not a window effect

| window | `a` | predicted `R_glob` | measured cross `R` | ratio | within 15% |
|---|---:|---:|---:|---:|---|
| `base` | 1.30 | 0.21290 | 0.23690 | 1.113 | yes |
| `G1_wide_gap` | 1.30 | 0.22300 | 0.24139 | 1.082 | yes |
| `G2_narrow_gap` | 1.30 | 0.12868 | 0.14367 | 1.116 | yes |
| `G3_far_outer` | 1.30 | 0.20386 | 0.22637 | 1.110 | yes |
| `G4_shifted` | 1.30 | 0.15313 | 0.16791 | 1.097 | yes |
| `base` | 1.50 | 0.22107 | 0.25709 | 1.163 | **no** |
| `G1_wide_gap` | 1.50 | 0.23790 | 0.27953 | 1.175 | **no** |
| `G2_narrow_gap` | 1.50 | 0.13192 | 0.15127 | 1.147 | yes |
| `G3_far_outer` | 1.50 | 0.20905 | 0.24527 | 1.173 | **no** |
| `G4_shifted` | 1.50 | 0.16314 | 0.19424 | 1.191 | **no** |

**Every ratio is one-signed and above prediction**, between 1.082 and 1.191, as
the lag account requires (a below-prediction crossing would have falsified it).

**The offset is a property of `a`, not of the window.** At `a = 1.30` the ten
windows' ratios have **sd 0.014**; at `a = 1.50`, **sd 0.016**. The base task's
own offset is 1.113 at `a = 1.30` and 1.163 at `a = 1.50` — so the `a = 1.50`
excess that causes all four misses is **present in the calibrated task too**.

Normalising each window by the base task's offset **at the same `a`**:

| `a` | base | `G1_wide` | `G2_narrow` | `G3_far` | `G4_shifted` |
|---|---:|---:|---:|---:|---:|
| 1.30 | 1.0000 | 0.9728 | 1.0034 | 0.9979 | 0.9855 |
| 1.50 | 1.0000 | 1.0104 | 0.9860 | 1.0089 | 1.0238 |

**All ten within 2.72% of 1.0.** The window contributes essentially nothing to the
offset; `a` contributes all of it, and the `a`-dependence was already measured and
documented as relaxation lag on the base task alone.

Reported honestly: **G-1 fails as registered at 6/10**, because the registered
tolerance was a flat 15% against the raw prediction and did not allow for the
per-`a` lag the base task itself shows. The sharper statement the data support is
that **the window-to-window variation in the offset is under 3%**, which is a
stronger claim than the one registered and is not a post-hoc rescue: the lag's
existence and its growth with `a` were both recorded before Block G ran
(`blockF_lag_prediction.md`, `SESSION_LOG.md` entry on Block C).

## G-3: PASS, with the sub-claim corrected in advance

| window | `Ghat` at `a=1.30` | `Ghat` at `a=1.50` |
|---|---:|---:|
| `G2_narrow_gap` | 0.021043 | 0.042901 |
| `G3_far_outer` | 0.071530 | 0.145175 |
| `base` | 0.085846 | 0.174761 |
| `G4_shifted` | 0.143781 | 0.283729 |
| `G1_wide_gap` | 0.166415 | 0.337440 |

Registered ordering `G2_narrow_gap < base < G1_wide_gap` holds at both `a`.
The registered `G3_far_outer >= base` was **wrong by a provable inequality** and
was corrected before any crossing was measured — enlarging `O` can only lower
`min_O f` and raise `max_O f`, so `Ghat(O', I) <= Ghat(O, I)` for `O' ⊃ O`.
Measured `G3 < base` at both `a`, consistent with the corrected direction.

## G-4: the scaling limit is a property of the activation, not of the base windows — PASS 5/5

For each window, `K` is recomputed as the supremum of `h(sigma) = -sigma + sigma^3/6`'s
class gap over that window's placements, and `kappa_0 = K/(4 sqrt2 /3)` is compared
with the measured `kappa(1.02) = Ghat(1.02)/D(1.02)`:

| window | `K` | `kappa_0` | measured `kappa(1.02)` | error |
|---|---:|---:|---:|---:|
| `base` | 0.579443 | 0.307296 | 0.307483 | **+0.061%** |
| `G1_wide_gap` | 1.132317 | 0.600502 | 0.599298 | -0.201% |
| `G2_narrow_gap` | 0.140190 | 0.074347 | 0.074947 | +0.808% |
| `G3_far_outer` | 0.481585 | 0.255399 | 0.254874 | -0.206% |
| `G4_shifted` | 1.041548 | 0.552364 | 0.548046 | -0.782% |

**5 of 5 within 0.81%, against a registered 3%, across `kappa_0` values spanning
8.1x (0.0743 to 0.6005).** Every one is computed from a cubic, with no reference
to `f_a` and no training run.

**Instrument note (validity gate).** A first pass reported `G4_shifted` at
**-5.75%**, a registered FAIL, and the error *grew* as `a -> 1+` (-4.6% at 1.02,
-18.8% at 1.002) — the wrong direction for a finite-`eps` effect. Cause: `ghat`
searches a **fixed** `(w1, b1)` grid, but the optimum scales as `w1 ~ sqrt(eps)`,
so at `eps = 2e-3` only ~7 grid points lie below `w1*`. Re-measured on a grid
**centred on the predicted scaling optimum**, `G4_shifted` gives -0.36% at
`a = 1.02` and stays within 0.81% down to `a = 1.001`. The fix was applied to all
five windows, which is why `base` tightens from -2.67% to +0.061%.

That the *scaling prediction itself* supplied the grid needed to measure its own
constant is worth noting: the failure was an artifact of not using it.

## What Block G adds to the paper

The account was built on one task. It transfers to four others, unchanged:

- the variable that collapses is `R`, across a **5.8x** spread in `|w2|` (G-2);
- the threshold's location is predicted **training-free** on every window, with the
  only discrepancy being an `a`-dependent lag that varies **under 3%** between
  windows (G-1);
- the geometric constant is derived from the scaling limit on **all five** windows
  to better than **1%**, over an **8.1x** range of its value (G-4).

None of `Ghat`, `R_glob`, `R_solve`, `K` or `kappa_0` was refitted per window. The
only per-window inputs are the four window bounds.
