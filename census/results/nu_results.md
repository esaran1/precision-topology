# Item #10: the ε-dependence of weight growth — the registered form is misspecified

Scored against `results/nu_prediction.md` (registered `2bd014d`, before any 2D fit).
Artifact: `nu_sweep.csv`. 6 `a` × 6 budgets × 30 seeds = 1,080 runs, float64, Adam
lr 1e-2, median terminal `|w₂|` per cell (the estimator the committed α used).

---

## The registered validity falsifier is TRIGGERED — and fully explained

> *"`α` from the 2D fit differs from the committed **1.1172** by more than its own
> bootstrap interval: the two estimators disagree and the discrepancy must be
> resolved before `ν` means anything."*

| fit | `α` | 95% CI | contains 1.1172? |
|---|---:|---|---|
| near-onset (primary) | **1.3442** | [1.1926, 1.4468] | **no** |
| all 36 cells | **1.2704** | [1.2174, 1.3206] | **no** |

**Resolved: the estimators agree; the budget ranges differ.** The committed α spans
**1k–160k** (2.2 decades); this sweep spans **1k–32k** (1.5 decades, chosen for
runtime). Restricting the *committed* data (`budget_alpha.csv`) to 1k–32k gives

    alpha = 1.3452    against my 1.3442

a match to 0.001. `alpha_derivation_results.md` already records that α depends on
the budget grid. **No estimator disagreement exists**, and the discrepancy is a
range effect that was documented before this sweep.

The consequence is that **this sweep's α is not the committed α**, and ν must be
read as conditional on the 1k–32k range.

## N-1: FAILS on the primary fit, PASSES on the secondary

| fit | `ν` | 95% CI (cluster bootstrap, 2,000) | contains 0? | `R²` |
|---|---:|---|---|---:|
| near-onset (primary) | **−0.1243** | [−0.2384, −0.0294] | **no** | 0.9901 |
| all 36 cells | −0.0626 | [−0.1357, +0.0080] | **yes** | 0.9871 |

Registered N-1 was "ν is consistent with zero". On the **primary** fit it is not.
`R²` ≥ 0.987 in both, so the `R² < 0.90` falsifier does not fire.

## But ν is an artifact of misspecification, not a real cross-term

**At fixed budget, where α cannot contribute at all**, the cross-`a` slope is:

| budget | 1,000 | 2,000 | 4,000 | 8,000 | 16,000 | 32,000 |
|---|---:|---:|---:|---:|---:|---:|
| `ν` | +0.005 | −0.259 | −0.242 | −0.061 | +0.077 | +0.104 |

Mean **−0.063**, sd **0.157**, and the **sign is not consistent** — three positive,
three negative. There is no ε-dependence of `|w₂|` at fixed budget.

**What is real is that `α` itself varies with ε:**

| `a` | 1.05 | 1.10 | 1.20 | 1.30 | 1.45 | 1.60 |
|---|---:|---:|---:|---:|---:|---:|
| `α` | 1.3398 | 1.3411 | 1.3010 | 1.2583 | 1.2105 | 1.1719 |

**corr(ε, α) = −0.9943**, spread **0.169** over a 12× range in ε.

So the registered model `|w₂| ~ B^α · ε^{−ν}` is **misspecified**: it assumes a
separable cross-term, and the actual dependence is `α = α(ε)`. A single global α
cannot fit both ends, and the least-squares fit absorbs the misfit into ν. That is
why the primary fit's ν excludes zero while every fixed-budget slice says ν ≈ 0.

## N-3: immaterial either way — the registered threshold decides it

Registered before fitting: the correction matters only if it moves the predicted
onset exponent by more than the **0.114** grid resolution, which requires
**|ν| > 0.198**.

| fit | `ν` | `−α/β` | `−α/(β−ν)` | shift | material? |
|---|---:|---:|---:|---:|---|
| near-onset | −0.1243 | −0.7466 | −0.6894 | **0.0573** | **no** |
| all cells | −0.0626 | −0.7466 | −0.7167 | 0.0300 | no |

Both shifts are **half the grid resolution or less**. Even taking the primary ν at
face value, the correction is immaterial.

## Verdict, per N-2's pre-registered branch

N-2 said: *if ν's interval contains 0, `−α/β` stands as an approximation and the
paper says so.* The primary interval does **not** contain 0, so strictly the other
branch applies — but the correction it implies is **immaterial by the registered
N-3 threshold**, and the fixed-budget analysis shows the nonzero ν is
misspecification rather than signal.

**What the paper should say:**

> `ε_onset ~ B^{−α/β}` stands **as an approximation**. A two-dimensional fit over
> 1,080 runs gives `ν = −0.124` [−0.238, −0.029] near the onset, which would move
> the predicted exponent by **0.057 — half the onset grid resolution of 0.114**,
> and is therefore immaterial. At **fixed budget** the ε-dependence of `|w₂|` is
> **consistent with zero** (mean −0.063, sd 0.157, sign inconsistent across six
> budgets). What the data do show is that **`α` itself drifts with `ε`**
> (corr = −0.994, spread 0.169), so the separable form `B^α ε^{−ν}` is not the
> right model; the correct statement is `α = α(ε)`, and a single exponent is a
> range-dependent approximation.

This is consistent with the already-recorded fact that **α depends on the budget
range** (1.1172 over 1k–160k, 1.3452 over 1k–32k) — a single α was always an
approximation, and this sweep shows the same is true across ε.
