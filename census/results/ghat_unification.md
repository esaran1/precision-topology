# Item 1: one `Ĝ` everywhere — switch to the certified search

The certified search and the restricted search **target the same supremum**; the
difference is resolution, not definition. All `R`-derived quantities are switched
to the certified `Ĝ`. Artifact: `ghat_certified_all.csv`.

---

## First: are they the same quantity? Yes — verified, not assumed

`maximum_gap()`, which every committed `R` used, is **not** a coarse version of the
certified search. It differs in three ways:

| | restricted (`maximum_gap`) | certified (`certify_exact`) |
|---|---|---|
| `b₁` domain | `π + c·fraction`, `fraction ∈ [0.6, 1.4]` — a **1-D slice** | `\|b₁ − π\| ≤ 6`, 2-D |
| `w₁` domain | `(0.001, 3.0]`, **positive only** | `\|w₁\| ≤ 3`, both signs |
| objective | `class_gap = min_O − max_I`, **one orientation** | oriented gap, both |
| `x` evaluation | 4,001/2,001-point grid | **exact** critical points |

The certified argmax lies **outside** the restricted slice at 9 of 10 `a`, mostly
at **negative `w₁`**. That looked like a definitional difference. It is not:

**The oriented gap is invariant under `(w₁, b₁) → (−w₁, −b₁)`** — verified exactly
at six test placements across three `a`, equality to 1e-12. The sine part of `f_a`
is odd and the task windows are symmetric, so the mirrored orientation is a
relabelling, not new territory. Restricting to `w₁ > 0` loses nothing **provided
the oriented gap is used**, and the restricted search's one-sided `class_gap`
recovers the same maximum over its own slice:

| `a` | restricted, one-sided | same slice, oriented | certified |
|---|---:|---:|---:|
| 1.02 | 0.0016156 | 0.0016156 | 0.0016156 |
| 1.30 | 0.0857831 | 0.0857831 | 0.0857831 |
| 1.60 | 0.2229064 | 0.2229064 | 0.2229064 |

Identical. **So the ~0.4% gap is pure grid resolution** — the restricted search is
33 × 600 over a slice, the certified one refines to `h = 1.6e-05` with exact
`x`-extrema. One quantity, two resolutions, and the certified value is the correct
one (a grid can only under-estimate a supremum).

**Decision: switch everything to the certified `Ĝ`.** No justification for keeping
the restricted one survives once it is established they measure the same thing.

## The certified `Ĝ`, all 13 `a` values in use

| `a` | restricted | certified | change |
|---|---:|---:|---:|
| 1.02 | 0.0016156 | 0.0016267 | +0.690% |
| 1.05 | 0.0063358 | 0.0063601 | +0.384% |
| 1.10 | 0.0175545 | 0.0176720 | +0.669% |
| 1.15 | 0.0316526 | 0.0319184 | **+0.840%** |
| 1.25 | 0.0662464 | 0.0665056 | +0.391% |
| 1.30 | 0.0857831 | 0.0861018 | +0.372% |
| 1.35 | 0.1067706 | 0.1069136 | +0.134% |
| 1.40 | 0.1286030 | 0.1287719 | +0.131% |
| 1.45 | 0.1512840 | 0.1515453 | +0.173% |
| 1.50 | 0.1744777 | 0.1751262 | +0.372% |
| 1.60 | 0.2229064 | 0.2243602 | +0.652% |
| 2.00 | 0.4402489 | 0.4410339 | +0.178% |
| 3.00 | 1.0489010 | 1.0522978 | +0.324% |

Since `R = |w₂|·Ĝ/2`, every `R` rises by the same fraction as its `Ĝ`.

## What moves beyond its stated rounding

### Moves — must be updated in the paper

**Pooled union counts.** The cutoffs are fixed at 0.30 and 0.50, so runs migrate
across them:

| | below 0.30 | above 0.50 |
|---|---|---|
| restricted (committed) | 0 of **2,285** | **460** of **461** |
| **certified** | 0 of **2,281** | **461** of **463** |

The headline "**0 solved below 0.30**" is **unchanged**. Four runs leave the lower
bin; two enter the upper bin, and **both solve**, so the single exception above
0.50 becomes **two of 463**. That exception count is quoted in T55 and in the
abstract's vicinity, so it must be updated: **"460 of 461" → "461 of 463"**.

**Crossing `R` per `a` (Block A)**, moves > 0.0005 at three of six:

| `a` | 1.30 | 1.35 | 1.40 | 1.45 | 1.50 | 1.60 |
|---|---:|---:|---:|---:|---:|---:|
| restricted | 0.2330 | 0.2375 | 0.2418 | 0.2456 | 0.2489 | 0.2563 |
| **certified** | **0.2338** | 0.2378 | 0.2421 | 0.2460 | **0.2498** | **0.2580** |

`CV(R)` 0.0311 → **0.0324**; O1's criteria (`≤ 0.15` and `< CV(|w₂|)/2 = 0.141`)
**still met**. Drift +10.0% → **+10.3%**.

**`R_glob` and `R_solve`** (Block B, fine-window values where available):

| `a` | `R_glob` old → new | `R_solve` old → new |
|---|---|---|
| 1.30 | 0.21446 → **0.21525** | 0.30667 → **0.30781** |
| 1.35 | 0.21568 → 0.21597 | 0.30697 → 0.30738 |
| 1.40 | 0.21734 → 0.21762 | 0.30543 → 0.30583 |
| 1.45 | 0.21936 → 0.21974 | 0.30635 → 0.30688 |
| 1.50 | 0.22071 → **0.22153** | 0.30534 → **0.30647** |
| 1.60 | 0.22402 → **0.22548** | 0.30650 → **0.30850** |

`R_solve`'s across-`a` CV rises **0.00200 → 0.00284** — still remarkably flat, and
the claim "`R_solve` is essentially constant in `a`" is unaffected.

**Three-optimiser crossings** (`a = 1.25`, all shift by the same +0.00090 since
they share one `Ĝ`):

| optimiser | restricted | certified |
|---|---:|---:|
| Adam | 0.23094 | **0.23184** |
| AdamW | 0.22980 | **0.23070** |
| SGD | 0.22888 | **0.22977** |

`R_glob(1.25)`: 0.21066 → **0.21149**. **Block F's conclusions are untouched** —
the ordering, the 0.98 pp offset spread and F-2's 6.19 pp failure all involve
differences at fixed `a`, where a common factor cancels exactly.

### Does not move beyond rounding

**`R50`**: shift **+0.0013**, inside the reported precision.

*Corrected 2026-09-22*: the absolute values first quoted here (0.3715 → 0.3728)
came from a **coarse fit grid**. Converged (900 × 1600) the reimplementation gives
**0.3669 → 0.3682**. The **+0.0013 shift is identical at every grid resolution**
and is the robust quantity; the absolute value is not, and the code producing the
committed 0.3705 is **not in the repository**. See `r50_certified_results.md`.

## Nothing qualitative changes

Every threshold, separation and CV criterion survives: 0 solved below 0.30, O1's
CV test, `R_solve`'s flatness, Block F's null, the Block E reversibility pair
(which is a `|w₂|` intervention at fixed `a`, so `Ĝ` cancels). The switch is a
**uniform ~0.1–0.8% rescaling of the `R` axis**, and the only reported integers
that change are the pooled-union denominators and the exception count.
