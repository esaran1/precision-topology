# Correction to T69: the S²⊔S² sweep used overlapping class regions

Found 2026-09-22 while preparing Block A5d, by reading Ren & Lim's released
multicopy generator (`exp_4_higher_dim_r5_multicopy`) line by line.

## What T69 did

`src/blockS2_budget.py` built one linked pair `tildeA(S²)`, `tildeB(S²)` and
thickened each point by a **uniform draw from the ambient 5-ball** of radius
`ρ = 0.5`, with 1,000 points per class, no held-out set.

## Why that is ill-posed

The cores are `2(√2 − 1) = 0.828427` apart (measured 0.828459). Ambient balls
of radius `ρ` around them are disjoint only when `2ρ < 0.8284`, i.e.
**`ρ < 0.4142`** — the bound Ren & Lim's own generator states in its docstring.
At `ρ = 0.5` the two solids **overlap**. On the overlap both classes have
positive density, so **no classifier can be perfect on the continuum**, whatever
its activation. The 2,000 training points happened to lie 0.1753 apart, which is
why training-set perfection was possible at all.

## Two further departures from their construction

| | their released code | T69 |
|---|---|---|
| thickening | **targeted**: A in its Y-subspace, B in its X-subspace (disjoint for all `ρ < 1`) | ambient 5-ball |
| copies | `k` copies on an L1 grid, spacing 10 | 1 |
| `k = 10` | 10 copies | read as "1,000 points per class", which was wrong |

## What this does to T69's claims

- **The budget-dependence of the ReLU/GELU gap** (+0.0138 → +0.0402) is a
  measurement on an ill-posed problem. It is not withdrawn as a number, but it
  **cannot be attributed to expressivity**.
- **"ReLU is capped at 0/20 perfect, consistent with Theorem 4.7"** is
  **unsupported**. On overlapping regions a capped perfect rate has a second,
  sufficient cause, so the pattern does not bear on the theorem.
- **GELU reaching 8/20 perfect** was perfection on the **training** sample only.

T69 is superseded for every paper-facing purpose by Block A5d
(`blockA5d_prediction.md`), which uses their targeted generator with `k = 10`,
a held-out criterion and `f_a` across the expressivity threshold. The
`paper-submitted` tag keeps T69 as reported. This file is the record that it
was wrong.
