# S²⊔S² budget sweep: the budget law does NOT extend — the gap widens

Scored against `results/blockS2_prediction.md` (registered before any run).
**EXPLORATORY**, as registered — new setting, last experiment day.
Artifacts: `blockS2_budget.csv`, `src/blockS2_budget.py`.

Ren–Lim construction from their own repository: `tildeA`/`tildeB` embeddings
`S² → ℝ⁵`, uniform normal-space thickening `ρ = 0.5`, 1,000 points per class.
Width 5, depth 5, Adam lr 1e-3, 20 seeds per cell. Between-class minimum distance
**0.1753**, so the problem is well-posed.

---

## Result

| budget | ReLU mean acc | GELU mean acc | gap (GELU − ReLU) | ReLU perfect | GELU perfect |
|---|---:|---:|---:|---:|---:|
| 1,000 | 0.9449 | 0.9586 | **+0.0138** | 0/20 | 0/20 |
| 4,000 | 0.9504 | 0.9793 | **+0.0289** | 0/20 | 3/20 |
| 16,000 | 0.9532 | 0.9908 | **+0.0376** | 0/20 | 6/20 |
| 64,000 | 0.9533 | 0.9935 | **+0.0402** | 0/20 | **8/20** |

## S2-1: FAILS, in the registered falsifier direction

Registered: *the gap at 64,000 is **smaller** than at 1,000, by more than 2 pp.*

Measured: the gap **widens** from +0.0138 to +0.0402 — a change of **+2.65 pp**,
the right magnitude and the **wrong sign**. It widens **monotonically across all
four budgets**.

| | 95% CI (4,000 bootstrap) |
|---|---|
| gap at 1,000 | [+0.0035, +0.0233] |
| gap at 64,000 | [+0.0307, +0.0491] |
| **P(gap widens)** | **0.9998** |

The registration named this outcome explicitly: *"Gap widens with budget: the
opposite of the 1D behaviour, which would be a substantive negative result and
must be reported as one."*

## S2-2 (validity gate): PASSES, and explains the failure

Accuracy is non-decreasing in budget for **both** activations, so the runs are
converging and S2-1 is interpretable.

But the two behave completely differently:

- **ReLU is flat**: **0.9449 → 0.9533** over a **64× budget range**, and
  **0/20 perfect at every budget**. More steps buy it almost nothing.
- **GELU improves**: 0.9586 → **0.9935**, perfect **0/20 → 8/20**.

**That asymmetry is the whole result.** In the 1D fold task, budget substitutes
for the activation — the monotonic network is *provably* unable to solve, but the
non-monotonic one merely needs enough steps, so the gap narrows as budget grows.
Here the monotonic network **plateaus below the non-monotonic one and stays
there**, so budget amplifies the difference instead of closing it.

## What this means for the paper

**The budget law is a 1D-fold result, and this is now tested rather than
assumed.** The paper should say:

> The budget law (`ε_onset ~ B^{−α/β}`) is established for the 1D fold task. A
> direct test in the Ren–Lim S²⊔S² setting (width 5, depth 5, ρ = 0.5, four
> budgets over a 64× range, 20 seeds) finds the **opposite** behaviour: the
> ReLU/GELU accuracy gap **widens** monotonically with budget, from 1.4 to 4.0
> percentage points (P = 0.9998), because ReLU plateaus at ~0.953 while GELU
> continues to improve to 0.994. Budget does **not** substitute for the
> activation in this setting.

This is a **negative result about the scope of our own law**, and it strengthens
the paper's honesty rather than weakening its claims: the 1D mechanism is stated
precisely and its boundary is now measured, not left open.

## Caveats — this is exploratory

- **One architecture** (width 5, depth 5) and one `ρ`. Ren–Lim report
  width-dependence; a wider net might behave differently.
- **`k = 10` was interpreted as 1,000 points per class.** Ten points cannot
  populate a thickened `S²`. Stated in the registration as an interpretation.
- **Accuracy, not a dense separation certificate.** The 1D work uses `solves()`
  on a dense grid; here accuracy is on the training sample, so "perfect" means
  perfect on 2,000 points, not region-wide.
- Registered and run on the final experiment day with no pilot beyond a timing
  test. **Present as a first look, not a tested claim.**
