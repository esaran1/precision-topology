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
