# Item 1: R governs under a third optimiser — the §5 hedge is replaced

Registered in `results/third_optimizer_prediction.md` before any scored
measurement. **Registered outcome 1.**

Date: 2026-09-14. Data: `results/r_adamw.csv` (240 runs).

---

## The result, aggregate first

> **Across three optimisers at `a = 1.25`: no run below `R = 0.30` solved
> (0 of 361), and every run above `R = 0.50` did (154 of 154). In the
> transition band all nine pairwise Fisher tests are null, minimum
> `p = 0.674`, with all three `R50` intervals overlapping.**

Pooled with the rest of the R dataset (3,150 runs, 12 values of `a`, 8
budgets, three optimisers): **0 of 2,285 below R = 0.30, 460 of 461 above
R = 0.50.**

**Three optimisers, three different terminal weight scales, one
`P(solve | R)` curve.**

| optimiser | n | solved | in-band | `R50` | 95% CI |
|---|---:|---:|---:|---:|---|
| Adam | 180 | 82 | 10 | 0.3350 | [0.3203, 0.3939] |
| **AdamW** (wd 0.01) | **240** | **99** | **57** | **0.3454** | **[0.3342, 0.3597]** |
| SGD | 180 | 35 | 18 | 0.3709 | [0.3587, 0.3753] |

**All three intervals mutually overlap**, on [0.3587, 0.3597].

Regime-wise, Fisher exact two-sided, every pair:

| regime | Adam | AdamW | SGD | pairwise p |
|---|---|---|---|---|
| `R < 0.3` | **0/95** | **0/125** | **0/141** | 1.0000 / 1.0000 / 1.0000 |
| `0.3 <= R <= 0.5` | 7/10 | **41/57** | 14/18 | 1.0000 / 0.6744 / 0.7647 |
| `R > 0.5` | **75/75** | **58/58** | **21/21** | 1.0000 / 1.0000 / 1.0000 |

**All nine tests are null.** Across 361 runs below `R = 0.3` spanning three
optimisers, **not one solved**; across 154 runs above `R = 0.5`, **all solved**.

## Why AdamW makes the agreement mean something

AdamW was chosen because it is **adversarial to the mechanism**, not because it
was convenient. Decoupled weight decay applies a multiplicative shrink to
`|w2|` at every step — it works directly against the quantity the account says
drives solvability. Agreement under an optimiser that suppresses weight growth
is a stronger statement than agreement under a third adaptive method whose
dynamics resemble Adam's.

The suppression is real and measurable. Over the shared 1k–40k window:

| optimiser | `\|w2\|` at 1k | at 40k | growth | `alpha` |
|---|---:|---:|---:|---:|
| Adam | 0.87 | **97.48** | 112x | **1.2627 ± 0.0593** |
| AdamW | 0.81 | **30.55** | 38x | **0.9802 ± 0.1269** |
| SGD | 1.36 | **19.38** | 14x | **0.7188 ± 0.0238** |

Terminal scale at 40k differs by **3.2x** between AdamW and Adam. The three
optimisers arrive at different places on the `|w2|` axis, by different
dynamics, and **solve at the same value of R**.

**The registered directional sub-prediction is confirmed**: `alpha_adamw`
(0.9802) `< alpha_adam` (1.2627), as weight decay opposing growth requires. The
full ordering is Adam > AdamW > SGD.

## Scoring against the registered failure condition

The condition required **both**:

1. `|R50 - pooled|` exceeding **0.0241** with **non-overlapping** intervals —
   **not met**: all three intervals overlap on [0.3587, 0.3597].
2. A **significant regime-wise Fisher test** in a populated regime —
   **not met**: minimum p across all nine tests is **0.6744**.

**Not triggered. Registered outcome 1.**

**The tightening mattered, exactly as anticipated.** On point estimates alone,
AdamW's gap from the pooled `R50` is 0.0251 and Adam's is 0.0355 — both
nominally exceed 0.0241. Had the criterion been point estimates only, this
experiment would have "failed" for the new arm on the same basis the existing
Adam arm already "fails", and the result would have been uninterpretable in
either direction. Requiring interval separation **and** a significant Fisher
test is what makes the outcome mean something. That criterion was fixed before
the data.

## The coverage problem, solved by design

The Adam comparison was weak because Adam's runs **jump across** the transition
band: median `R` goes 0.234 at 4k to 0.578 at 8k, leaving only **10 runs
inside**. An 8-seed pilot located AdamW's in-band window and 5k/6k were added
for that reason.

**The pilot measured where runs land, not whether they solve** — it fixed
coverage, not outcome. This is stated wherever the grid is described.

Result: **57 AdamW runs inside the band**, against Adam's 10 and SGD's 18. The
decisive regime is now well-powered on at least one arm rather than resting on
ten runs.

## What this licenses

The §5 hedge — "undetectable at this coverage, not absent" — is replaced by a
positive claim: **the optimiser enters only through where on the R axis its
runs land.** Three optimisers, `alpha` spanning 0.72 to 1.26, terminal scales
differing 5x, one curve.

**Scope unchanged**: one task, `a = 1.25`, width 1. This is a statement about
optimisers, not about settings, and the family-dependence of the threshold
(T52, T54) is untouched — R remains family-specific.
