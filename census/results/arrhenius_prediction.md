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
