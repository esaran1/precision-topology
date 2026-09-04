# Part 1a: barrier measurement — the Arrhenius premise fails

Registered in `arrhenius_prediction.md`. Data: `barrier.csv`
(13 values of a, 20 initializations each, both endpoint types;
MEP on 5 inits per cell). Energy = training BCE; barrier =
max along path − max(endpoint losses), the convention used in
`basin_profiles.csv`.

## Result: ΔE does not track findability, and MEP barriers are zero

| a | findability (fold1d_sweep) | ΔE_linear, constructed \|w₂\|=1 endpoint | ΔE_linear, found endpoint | ΔE_MEP |
|---|---|---|---|---|
| 1.02 | 0/200 | 0.020 | — (none exist) | 0.000 |
| 1.25 | 0/200 | 0.028 | — | 0.000 |
| 1.35 | 2/200 | 0.022 | — | 0.000 |
| 1.45 | 50/200 | 0.020 | 0.601 | 0.000 |
| 1.50 | 81/200 | 0.021 | 0.320 | 0.000 |
| 2.00 | 143/200 | 0.039 | 0.217 | 0.000 |
| 3.00 | 139/200 | 0.103 | 0.472 | 0.000 |

Three findings, each fatal to the registered form:

1. **Every MEP barrier is 0.000.** The string method finds a
   monotonically descending path from initialization to solution at
   every a, for both endpoint types. There is no barrier to escape —
   the solution is downhill-connected to typical initializations even
   at a = 1.02, where the rate is 0/200.

2. **ΔE_linear runs backwards.** Holding the endpoint fixed
   (constructed, |w₂| = 1 at every a) so only a varies, ΔE_linear
   *rises* with a — 0.020 at a = 1.02 to 0.103 at a = 3.0 — while
   findability rises from 0% to 70%. Arrhenius predicts the opposite
   sign. Held the other way (a = 1.5 fixed, endpoint varied), the
   **findable** solution sits behind the **higher** barrier
   (0.320 vs 0.021) and the unfindable one behind the lower.

3. **The 1065 → 10 → 1 collapse that motivated this does not
   generalize.** Those were link-setting barriers to *constructed*
   endpoints at amplification 600; the height is dominated by the
   endpoint's own weight scale (mid-path weights at half-scale produce
   huge losses), which `gap_results.md` already flagged. In the 1D
   task with the endpoint scale controlled, the effect disappears.

## Verdict

**P-1a is falsified and the Arrhenius premise with it.** Findability
here is not escape over a barrier: the minimum-energy path has no
barrier, and the linear proxy anti-correlates with the rate. There is
no ΔE to put in exp(−ΔE/T), so Parts 2, 3 and 5 as registered cannot
run — 2a's fit would be a fit to noise with the wrong sign, and 3's
temperature manipulation predicts nothing without it.

The overshoot question (1a's registered comparison) is moot in the
direction that matters: linear overshoots MEP by an *infinite* factor
(finite vs exactly zero), not a roughly constant one, so the cheap
estimator is not a usable proxy for a barrier that does not exist.

## What this leaves standing, and what it points at

The zero-basin result (T30) is unaffected — it is a measurement.
What the negative rules out is the *explanation* offered for it. A
downhill-connected solution with no basin means the failure is not
energetic but **dynamical**: SGD's trajectory does not enter the
region, though nothing blocks it energetically. Candidates the data
already hint at: the solution manifold's thinness (measure ~1e−6,
sheet thinner than one grid cell) makes it a set of near-zero capture
cross-section regardless of energy, and Adam's step size at typical
gradients may simply overshoot a sheet of that width.

Part 4 (basin volume along the manifold, why SGD lands at |w₂| ≈ 5)
does not depend on the Arrhenius form and remains well posed.

> **Terminology correction (2026-08-28).** The objects called "zero-basin
> solutions" here are **not critical points of the loss** (gradient norm
> 0.02–0.27 versus training's terminal 0.0006–0.021; loss 0.22–0.69;
> λ_min < 0 at six of eight values). Read "solution" throughout as
> "correctly-classifying parameter vector", never as "minimum of the loss".
> The measurements are unaffected; what changes is what they are about. See
> `criticality_results.md` and `terminology_correction.md`. In particular the
> statement that Ahn–Zhang–Sra does not apply was tested on one of its two
> conditions only, and is corrected there.
