# Registration: the scale-gating criterion for GELU, SiLU and Mish at width 1 (Track 3A, step 2)

**Written and committed before the small-scale conditional minimiser was computed for any of these activations.**
Date: 2026-09-25, about 19:30 EDT. Code: `src/act_general.py`. Tests: `tests/test_act_general.py` (every validity
check and the decision rule below are exercised on constructed pass and fail cases; all pass at this commit).

## Setting

- Width 1: z = w₂·u(w₁x + b₁) + b₂, s = |w₂|, σ = sign w₂. Windows I = [−0.8, 0.8] (class 0), O = ±[1.2, 2.0]
  (class 1); the 800-point population (`width2_conditional.population()`).
- u ∈ {GELU (exact, erf form), SiLU (t·σ(t)), Mish (t·tanh(softplus(t)))}. Each has a single fixed-size dip (a
  minimum at t ≈ −0.75, −1.28, −1.19 of depth −0.170, −0.278, −0.309), no periodicity and hence no 2π winding
  (k = 0 in Track 1's κ).
- Placement of a configuration: G₊(σ·u(w₁x + b₁)) = min_O − max_I > 0, the lower end of the exact-extrema enclosure
  (`width2_geometry.gaps`, with a **validated, not certified** sup|u″| bound: the 9-point sub-grid maximum of |u″|
  plus B3·h/2, B3 a validated global bound on |u‴|).
- The width-1 conditional search: b₂ profiled exactly, BFGS (`width2_conditional.bfgs_batch`) from w₁ ∈ U(0, 10),
  b₁ ∈ U(−10, 10), σ ∈ {±1}; the five lowest distinct losses polished by damped Newton in (w₁, b₁, b₂); the constant
  predictor is a candidate.

## Already known before this registration (steps 0 and 1; not predictions)

- **Step 0.** With f_a (a = 1.30) the generalised search + bisection gives the switch in [4.95796, 4.95835], inside
  the certified bracket [4.95, 4.9625] (`results/act_general/validate_fa.json`); Track 1's s* = 4.95801 is inside it.
- **Step 1 (validated).** A single unit places: Ĝ = sup G₊ = 0.037547 (GELU), 0.050912 (SiLU), 0.054772 (Mish), by
  multistart Nelder–Mead (400 starts) and an independent differential-evolution search agreeing to 1e−6 relative;
  every maximiser has σ = +1 (the dip is placed under I and both outer windows sit higher). With the output bias at
  the feasible midpoint the maximiser is sign-correct on the windows (exact extrema), and a wrong bias is not
  (`results/act_general/ghat.json`). So the "a placed configuration exists" part of the rule below is already
  satisfied for all three.

## Why the class-mean route is not used

On the symmetric windows the linear part of the ramp contributes nothing to Δμ = μ_O − μ_I, but the ramp of these
activations is not odd: u(t) − max(t, 0) → 0 as |t| → ∞, so Δμ(w₁, b₁ = 0) = |w₁|·(mean_O relu(x) − mean_I relu(x))
+ o(|w₁|) = 0.6|w₁| + o(|w₁|) on the continuous windows (0.8 − 0.2). Δμ is unbounded and the class-mean maximiser is
not attained, so math note §10's Corollary does not apply as stated. **The criterion's object is therefore the direct
conditional minimiser at small scale, s = 0.05 and s = 0.1**, by the width-1 search.

## What will be computed (per activation, GELU, SiLU, Mish)

At each of s = 0.05 and s = 0.1 (`act_general.validate_point`, `act_general.criterion`):
1. the retained conditional minimiser with 800 restarts, its exact G₊ enclosure and status (placed iff lower end
   > 0; unplaced iff upper end ≤ 0; else undecided);
2. validation: (i) restart ladder — a separate 200-restart search gives the same retained loss within 1e−9 and the
   same status; (ii) independent CMA-ES search (20 starts, (w₁, b₁, v) with φ = sign(v)·u) not lower by more than
   1e−9; (iii) audit — no eligible candidate within 1e−9 of the retained loss has another status, and the retained
   status is decided.

## Decision rule (registered)

- **"Switch predicted"** iff the validated small-scale minimiser is **unplaced** at both s = 0.05 and s = 0.1, and a
  placed configuration exists (Ĝ > 0, lower end).
- **"No switch predicted"** iff it is **placed** at both scales.
- Otherwise (mixed, undecided, or any validation check failing at either scale): **"undetermined"**, reported as such.

## What follows (not predictions; fixed now)

- Where a switch is predicted: a validated bracket in s — scan s = 10^(k/8), k = −8 … 24 (0.1 to 1,000), 200
  restarts per scale; the first unplaced → placed change; bisection in log s to hi/lo − 1 ≤ 0.01 (200 restarts per
  midpoint); validation (ladder 200 → 800, CMA-ES, audit as above) at both bracket ends; every scan status change is
  reported. Labelled VALIDATED (not certified). If a validation check fails at a bracket end, the bracket is reported
  as not validated and no training test is registered on it.
- Where "no switch predicted": the same scan is run as a check of the prediction (placed at every scan scale is
  consistent with it); any unplaced scan point is reported as evidence against it.
- Training predictions (step 4) are registered separately before any run.

## Amendment before any criterion result (2026-09-25, about 19:50 EDT)

The first `criterion` run stopped at its first exact-extrema evaluation ("extrema: cell cap reached"; nothing was
recorded or printed except the traceback). Cause: the registered d2u_bound's margin B3·h/2 used a *global* |u‴|
bound, so on the flat tails of these activations (|φ′| ≈ 0) cells never became provably monotone. The bound was
replaced by a sharper one that is still validated, not certified: max over the 9-point sub-grid of |u″| plus
(h/2)·(max over the sub-grid of |u‴| + (h/2)·B4), B4 a validated global bound on |u⁗| (grid maximum on [−60, 60]
at step 1e−4 plus 5%); u‴ is in closed form and tested against autograd. The Ĝ enclosures of step 1 are unchanged to
the last digit under the new bound. The decision rule, the scales and every validation check are unchanged.

## Amendment 2 (2026-09-25, about 20:15 EDT): implementation only; what was seen before it

- The second `criterion` run also stopped at the cell cap at GELU, s = 0.05. **Disclosure:** while diagnosing it I
  looked at that search's candidates. The lowest loss (0.4773820939) is a one-sided ramp: w₁ ≈ 775, b₁ ≈ −620.8,
  σ = +1, i.e. the ReLU-like kink just right of x = 0.8 with the inner edge at the dip, the inner window and the left
  outer window deep in the negative tail, and the right outer window on the ramp. With a larger cell cap its
  exact-extrema G₊ enclosure is [−5.1e−14, 0], which the registered rule reads as unplaced. In exact arithmetic its
  G₊ is positive but about 10^(−334,000) (the tail values underflow to −0.0 in double precision). Nothing else about
  any activation's small-scale result was seen.
- Changes (none to the decision rule, the scales or the checks): (i) the extrema cell cap is raised from 400,000 to
  4,000,000 (a resource limit; if still reached, the status is recorded as **undecided**, which the rule already maps
  to "undetermined"); (ii) the logistic σ and GELU's Φ are evaluated with full relative precision in the negative tail
  (a stable σ and exp(log_ndtr)); (iii) a POST HOC diagnostic column is added: the sign and log₁₀|G₊| of the retained
  minimiser's gap in arbitrary precision (mpmath, using that u is unimodal, checked in tests). **The registered
  verdict is the rule applied to the double-precision enclosure, as written above; the arbitrary-precision gap is
  reported beside it and does not replace it.**

(Correction, appended: the clock times written in this file were estimates and are wrong by up to an hour; the
commit times in `git log` are authoritative: registration b707e86, amendment a251e80, amendment 2 cf2d68b, result
ca6d7e6, all on 2026-09-25 between about 19:15 and 19:25 EDT.)
