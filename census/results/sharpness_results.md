# Parts 1a-1d: the edge-of-stability explanation is falsified

Registered in `arrhenius_prediction.md` after reading Ahn-Zhang-Sra and
Cohen et al. Data: `sharpness.csv` (exact 4x4 Hessians, full-batch BCE).

## 1b: no separation at 2. Both populations sit ~2 orders below it.

`lambda_max * eta_eff`, Adam, realized step measured over the final 300 steps:

| a | n found | lambda_max (med) | realized step | **lambda_max * eta_eff** | constructed \|w2\|=1: lambda_max * eta |
|---|---|---|---|---|---|
| 1.45 | 7 | 2.72 | 1.17e-3 | **0.0033** | 0.029 (nominal eta) |
| 1.50 | 15 | 2.61 | 1.14e-3 | **0.0031** | 0.029 |
| 2.00 | 28 | 1.90 | 8.0e-4 | **0.0016** | 0.030 |
| 3.00 | 26 | 1.07 | 4.8e-4 | **0.0006** | 0.025 |

Threshold is **2**. Found solutions: 0.0006-0.0034. Zero-basin constructed
solutions: 0.025-0.030 at nominal lr. **Both populations are far inside the
stable regime, by 60x-3000x.** P-EoS-1b is falsified: there is no separation
at 2, and neither population is anywhere near it.

Sharpness itself barely differs between the populations (found 1.07-2.72,
constructed 2.48-3.02) and is if anything *higher* for the constructed ones
in the same range — but 3 * 0.01 = 0.03, not 2. Nothing here is unstable.

## 1c: sharpness RISES with |w2|, does not fall

P-EoS-1c predicted lambda_max falls with |w2|, crossing 2/eta near 5.
Measured (a = 1.5, along the manifold): **2.91 -> 3.05 -> 3.65 -> 4.08 (|w2|=5)
-> 4.23 -> 5.01 (|w2|=20)**. Monotonically **increasing**, and never within
50x of 2/eta (= 200 nominal, ~1800 at realized step). Falsified in direction
and magnitude.

## 1d: sharpness does not diverge as a -> 1+; it decreases

P-EoS-1d predicted divergence at the expressivity threshold. Measured:

| a | lambda_max at \|w2\|=5 | lambda_max, constructed \|w2\|=1 |
|---|---|---|
| 1.02 | 2.73 | 2.72 |
| 1.10 | 2.91 | 2.73 |
| 1.25 | 4.05 | 2.78 |
| 1.35 | 4.48 | 2.82 |
| 1.50 | 4.08 | 2.91 |
| 3.00 | 1.38 | 2.48 |

The a -> 1+ limit is **2.7, not infinity**. The curve is non-monotonic with a
maximum near a = 1.35, not a divergence. Falsified.

## Why sharpness is bounded here (the mechanism)

The BCE Hessian is `J^T diag(s(1-s)) J / n` plus a curvature term, and
`s(1-s) <= 1/4` always. At a *solution* every point is classified, so `s(1-s)`
is small: measured mean 0.035 at |w2|=20 (loss 0.044). The parameters are
O(1-20) and the data O(1), so `J` entries are O(1). **lambda_max is
structurally O(1) in this task** and cannot diverge: the fold's vanishing
depth as a -> 1+ makes the solution *thin in parameter space* (thickness
~|w2|*gap -> 0) but does not make the loss surface *sharp*, because the
loss at those parameters stays near 0.69 (the constructed |w2|=1 solutions
have loss 0.56-0.69, mean s(1-s) = 0.24-0.25 -- barely separated at all).

**Thinness of the solution set and sharpness of the loss are different
quantities, and only the first is what we have.** A set can be thin in
parameter space while the loss varies gently across it: that is exactly a
region where the logit margin is small, which is what the theorem in
`fold1d_theorem.md` says these solutions must have (m = |w2|*G/2, ~1e-4).

Trajectory sharpness confirms the same picture: along an Adam run at a = 1.5,
lambda_max goes 1.70 -> 1.94 -> 1.78 -> 1.91 -> 2.33. No progressive
sharpening toward 2/eta = 200; the dynamics never approach the edge.

## Verdict

**We are not in the edge-of-stability regime, and the Ahn-Zhang-Sra
mechanism does not apply to our phenomenon.** Their Theorem 1 requires
lambda_max > 2/eta at the stationary point; our stationary points are at
lambda_max ~ 3 with 2/eta ~ 200-1800. Their theorem is not violated and is
not engaged -- it says nothing about these solutions.

This is the outcome the brief asked to be reported as prominently as the
confirmatory one: **the EoS explanation fails, and the zero-basin phenomenon
remains unexplained by it.** What we have is not a sharpness effect.

Consequences:
- Parts 2, 3, 4 as registered are moot: 2a predicts rates from the fraction
  of the manifold with lambda_max*eta < 2, which is 100% everywhere; 2b's SGD
  prediction (findability falls with eta) is not entailed by anything we
  measured; 3 and 4 test an account that does not apply here.
- The 2b SGD direction was independently *already* contradicted by
  `step_size_results.md`: SGD findability is flat-to-non-monotonic in eta,
  not monotonically falling.
- What still explains the data is `reach` (T40): where the optimizer travels,
  not the curvature it encounters.

> **Terminology correction (2026-08-28).** The objects called "zero-basin
> solutions" here are **not critical points of the loss** (gradient norm
> 0.02–0.27 versus training's terminal 0.0006–0.021; loss 0.22–0.69;
> λ_min < 0 at six of eight values). Read "solution" throughout as
> "correctly-classifying parameter vector", never as "minimum of the loss".
> The measurements are unaffected; what changes is what they are about. See
> `criticality_results.md` and `terminology_correction.md`. In particular the
> statement that Ahn–Zhang–Sra does not apply was tested on one of its two
> conditions only, and is corrected there.
