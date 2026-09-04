# Registered predictions: Arrhenius account of findability

**1a.** Linear-interpolation barrier ΔE_lin overshoots the minimum-energy-path
barrier ΔE_mep by a roughly constant factor across a. Falsified if the ratio
ΔE_lin/ΔE_mep varies systematically with a (then only MEP is usable).

**1b.** Scaled initialization lowers ΔE at a = 1.02 by enough to explain
17/40 vs 0/400. Falsified if ΔE is unchanged or higher.

**2a.** log P(find) is linear in ΔE with negative slope; fitted T is constant
across a. Falsified if nonlinear, positive slope, or T drifts systematically.

**2b.** Fitted T agrees with T ∝ η/B from optimizer settings, up to one fixed
proportionality constant. Falsified if the recovered constant is not fixed.

**3a.** Onset shifts down with η and up with B, by the amount 2a's T predicts.

**3c.** η/B collapses the curves: same η/B, different η and B, same onset.

**4a.** Basin volume along the solution manifold peaks near |w₂| ≈ 5, matching
where SGD lands.

**5.** Arrhenius with the 1D-fitted temperature constant reproduces GELU's
3%/8%/17% dose-response within a factor ~2.

---

# Successor hypothesis: capture is cross-section limited (registered 2026-08-27)

Context: the Arrhenius premise is falsified (`barrier_results.md`) — MEP
barriers are exactly 0 and the linear proxy anti-correlates with the rate.
So findability is dynamical, not energetic.

**P-step (the discriminator).** If capture is limited by the solution
sheet's cross-section rather than by an energy barrier, then **reducing
the optimizer step size raises findability near the onset**, monotonically,
because a large step overshoots a thin sheet. The energetic account
predicts the opposite sign (smaller steps = lower effective temperature =
less escape), and we have shown there is nothing to escape. Falsified if
findability falls as step size falls, or is flat.

**P-ratio (the quantitative version).** The sheet's thickness in each
parameter direction, divided by Adam's typical step in that direction,
predicts which a values are findable: findability should rise where
thickness/step exceeds order 1. Falsified if the ratio is uncorrelated
with the observed rates, or ordered wrongly across a.

**P-4a.** Basin volume along the solution manifold peaks near |w2| ~ 5,
matching where SGD lands. **P-4b.** The peak sits where two effects cross:
below it, b1-perturbations destroy the solution; above it, large weights
are reached from a shrinking set of initializations. **P-4c.** The peak
moves outward with initialization scale.

---

# Edge-of-stability predictions (registered 2026-08-28, before measurement)

Literature read first: Ahn-Zhang-Sra Thm 1 (GD does not converge to stationary
points with lambda_max > 2/eta, except from a measure-zero initialization set;
deterministic GD; the paper's thesis is that unstable convergence nonetheless
happens via forward-invariant sets). Cohen et al.: lambda_max *rises to* 2/eta
and hovers there (progressive sharpening).

**P-EoS-1b.** Found solutions satisfy lambda_max * eta_eff < 2; zero-basin
constructed solutions violate it. Falsified if the two populations do not
separate at 2, or overlap substantially.

**P-EoS-1c.** lambda_max falls with |w2| along the solution manifold, crossing
2/eta_eff near |w2| ~ 5. Falsified if flat, rising, or crossing elsewhere.

**P-EoS-1d.** lambda_max of reachable solutions diverges as a -> 1+, and the a
at which it crosses 2/eta predicts the measured onset (1.30, 1.35].

**P-EoS-2b (SGD, the clean test).** For plain SGD, eta_eff = eta, so
findability should FALL monotonically as eta rises. Falsified by a rise or
non-monotonicity.

Caveat registered in advance: Cohen et al. imply lambda_max at an SGD-found
solution is partly an artifact of the optimizer that found it, so the
found-vs-constructed comparison is partly circular. The manifold sweep (1c),
where |w2| is set by construction rather than by training, is the
non-circular test.

---

# Criticality predictions (registered 2026-08-28, before measurement)

**P-crit-1a.** The zero-basin points have large ||grad L|| and are NOT critical
points; found solutions sit at/near critical points with small gradient norm.
Falsified if zero-basin gradient norms are small (comparable to found ones).

**P-crit-1b.** If lambda_min < 0 anywhere, Ahn-Zhang-Sra applies via its FIRST
condition and our earlier dismissal was incomplete; to be reported as our error.

**P-2c.** If termination is a step-budget cutoff, reached-|w2| keeps growing
with budget (2x/4x/10x) and findability improves. If a genuine attractor,
reached-|w2| saturates and findability is budget-independent.

**P-3b.** Findability = overlap between the correctly-classifying |w2| region
and the terminal-|w2| distribution. Falsified if the overlap fraction does not
track measured rates across a, lr, and init scale.

---

# Budget power-law predictions (registered 2026-08-28, BEFORE measuring onsets)

Provisional alpha from the four committed budgets (2k/4k/8k/20k, medians
3.15/7.75/17.90/48.48 at a=1.25): terminal |w2| ~ B^alpha with alpha ~ 1.19.

**P-alpha.** Terminal |w2| follows a power law in budget B with exponent
alpha, measured over a wider range (1k..160k) before any onset is measured.

**P-onset-law.** Equating reachable |w2| ~ B^alpha to required
|w2| ~ c/D(a) ~ (a-1)^{-3/2} gives (a_onset - 1) ~ B^{-2*alpha/3}.
At alpha = 1.19 this is **B^{-0.79}**. Registered prediction: the measured
onset exponent lies within +-0.15 of -2*alpha/3 using the alpha measured in
P-alpha. Falsified if outside that band, if the onset does not follow a power
law, or if alpha itself is not constant across the budget range.

**P-joint.** The one-dimensional |w2| criterion over-predicts because it
ignores (w1,b1). The joint criterion -- terminal (w1,b1,w2,b2) lands in the
correctly-classifying set -- should predict rates with substantially lower
error than 0.283 mean absolute.

**P-stall.** Stalling at loss = log 2 is a distinct mechanism from
budget-limited travel: registered as budget-INDEPENDENT (stall fraction
roughly constant in B while travel is not). Falsified if the stall fraction
falls with budget like the solve rate does.
