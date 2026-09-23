# Block 1: audit of the conditional threshold — results

Registered in `conditional_audit_prediction.md` (`6b17508`) before any audit computation.
Producers: `src/conditional_audit.py` (1a, 1e), `src/profiled_bnb.py` (1b, 1c bounds),
`src/conditional_certified.py` (1c brackets, 1d). Figure: `results/figures/cond_audit_candidates.pdf`.

## Verdict

**The registered decision rule is met.** The placement transition is identifiable without
excluding any lower-loss candidate. The certified procedure (1c), the stricter Adam search (1e) and
the frozen search agree within their numerical uncertainty at every threshold. **No stop condition
fired.**

| stop condition | result |
|---|---|
| a discarded candidate has lower loss than the retained branch near a threshold | **no**: none at any of 358 scales (6,986 degenerate restarts, 3,396 not carried past the screen, 2,506 full runs not lowest, and the exact constant predictor) |
| certified threshold differs from the reported one by more than one grid step (0.05 in \|w₂\|) | **no**: the certified switch lies inside the frozen grid cell at all 12 thresholds |
| the switch is a basin exchange where the paper describes a continuous crossing | **no**: at the R_glob switch, the G ≤ 0 and G > 0 minimisers are 0.00015–0.0011 apart in (w₁, b₁), one basin crossing G = 0 |

## 1a. Every candidate retained

- **Scope**: the frozen search replayed call by call at every scale it evaluates for R_glob and
  R_solve, at a = 1.30–1.60. That is 358 (a, threshold, scale) searches and 16,468 candidates
  (`cond_audit_candidates.csv`).
- **Reproduction**: the retained candidate equals `best_conditional`'s at all 358 scales.
- **Stationarity**: retained gradient norms have median 7e-16 and maximum 2.3e-6.
- **The constant predictor** (w₁ = 0, loss log 2) is never within 0.207 of the retained loss.
- **Degeneracy filter**: it removes 42% of restarts but never a better solution.
- **Shape** (candidate-loss plots): one continuous branch at every a, with G ≤ 0 below the frozen
  threshold and G > 0 above it.
  - Below the threshold, the best G > 0 candidates lie well above the branch.
  - Above it, the best G ≤ 0 candidates are either higher local minima or the constant predictor.

## 1b–1c. Certified thresholds (profiled bias, exhaustive branch and bound)

- **Profiled bias**: b₂ is the unique root of mean(σ(z) − y) = 0, found by safeguarded Newton to a
  residual ≤ 1e-14 (measured 8e-15).
- **Search domain**: w₂ > 0 (the orientation symmetry covers the other sign), and b₁ ∈ [0, 2π).
  - L* and G are 2π-periodic in b₁, verified to 4e-15.
  - |w₁| ≤ W(s, a). This bound comes from the linear-growth argument: for w₁ > 0, left-outer points
    and inner points with x ≥ c are separated in the wrong direction by Δ = s((1.2 + c)w₁ − 2a), so
    L* ≥ (m/n)·log(1 + e^{Δ/2}), where m/n is the smaller group's share. Beyond W, L* > log 2 ≥ m₋.
    W is minimised over c on each side, the larger side is taken, and it is checked numerically
    (L* > log 2 at |w₁| = W for every b₁).
- **Bounds on a sup-norm cell** with half-widths (h_w, h_b):
  - **L***: L*(c) − |∂_w L*| h_w − |∂_b L*| h_b − ½ s a · mean((|x| h_w + h_b)²). The gradient is exact
    by the envelope theorem, checked against finite differences to 1.4e-9. The curvature term holds
    because the Schur-complement term is dominated by the Gauss–Newton term (Cauchy–Schwarz), and
    |f″| ≤ a.
  - **G**: exact (closed-form interval extrema), with step (1+a)(2.8 h_w + 2 h_b).
- **Tolerance**: m₋ and m₊ are certified to absolute 1e-7, tightened to 1e-9 when their intervals
  overlap; the float64 rounding margin is 1e-10. The frozen quadrature objective has 800 points.
- **Brackets** (reallocated): each frozen grid cell [w₂_frozen − 0.05, w₂_frozen] was certified at
  both ends, then bisected twice to width 0.0125 (`cond_certified_brackets.csv`).

| a | Ĝ (certified) | R_glob, \|w₂\| interval | **R_glob, R interval** | frozen R_glob | R_solve, \|w₂\| interval | **R_solve, R interval** | frozen R_solve |
|---|---:|---|---|---:|---|---|---:|
| 1.30 | 0.086102 | (4.9500, 4.9625] | **(0.21310, 0.21364]** | 0.21525 | (7.1000, 7.1125] | **(0.30566, 0.30620]** | 0.30781 |
| 1.35 | 0.106914 | (4.0250, 4.0375] | **(0.21516, 0.21583]** | 0.21650 | (5.7125, 5.7250] | **(0.30537, 0.30604]** | 0.30738 |
| 1.40 | 0.128772 | (3.3750, 3.3875] | **(0.21730, 0.21811]** | 0.21891 | (4.7375, 4.7500] | **(0.30503, 0.30583]** | 0.30583 |
| 1.45 | 0.151545 | (2.8875, 2.9000] | **(0.21879, 0.21974]** | 0.21974 | (4.0125, 4.0250] | **(0.30404, 0.30499]** | 0.30688 |
| 1.50 | 0.175126 | (2.5250, 2.5375] | **(0.22110, 0.22219]** | 0.22329 | (3.4750, 3.4875] | **(0.30428, 0.30538]** | 0.30647 |
| 1.60 | 0.224360 | (2.0000, 2.0125] | **(0.22436, 0.22576]** | 0.22997 | (2.7000, 2.7125] | **(0.30289, 0.30429]** | 0.30850 |

- **Reading**: the frozen value is always the first grid point at or above the certified switch, so it
  overstates each threshold by less than one grid step. **The paper should quote the certified R
  intervals.**
- **R_solve**: the solve margin of the certified global minimiser changes sign inside each interval,
  for example −0.00051 → +0.00036 at a = 1.30. It is evaluated at the certified argmin, so R_solve is
  numerically checked, not certified. R_glob's sign of m₊ − m₋ is certified.

## 1e. Stricter optimisation

- **Settings**: 10× steps (screen 6,000, full 30,000) and a gradient-norm stop at 1e-8, at every
  frozen grid scale within ±0.1 of each threshold. That is 60 searches (`cond_audit_strict.csv`).
- **Agreement**: all 12 thresholds fall on the same grid point as the frozen search. Losses agree
  with the frozen retained candidates to ≤ 1.7e-16.
- **Convergence**: every search reached the tolerance within its screen stage.

## 1d. Objective used (training sets against quadrature)

Reduced and deprioritised by decision on 2026-09-23 (compute reallocated to Block 3): 20 seeds per a,
run after Block 3's training. Results will be added here when run; nothing is claimed about it yet.
