# Design 2B: the validity boundary in κχ, redesigned (for approval; not registered)

**Why the first boundary test was unresolved (diagnosed post hoc on its runs)**
- The fast-ramp runs did not lag; they destabilised.
- At a = 1.30 the non-crossers end far from placement (median gap −2.6 to −16), with growing |w₁| and several
  windings.
- Plain SGD at a fixed lr 0.3 becomes unstable once the output scale is large, because the Hessian grows with s.
- So the redesign must keep the dynamics stable along the whole ramp.

**Setting**
- Width 1, a = 1.30 (winding −1) and a = 1.50 (winding 0).
- 40 fresh seeds per cell, full-batch SGD.
- Warm start at the population branch point θ\*(s₀) on the natural winding, with s₀ = 0.5·s\*, held for 4,000 steps.
- Then s = s₀e^{γt}, run to an end scale of s\*·(1 + 6κχ).

**Stability, checked a priori (new)**
- The learning rate for each cell is η_cell = min(0.3, 1/λ_max(H(s))) over s ∈ [s₀, s_end] along the branch, with
  λ_max from the population Hessian by continuation.
- So η·λ_max ≤ 1 on the whole ramp.
- γ is then set so that χ = γ/(η_cell λ_min(H)) gives the target κχ.
- The derivation is the discrete linear stability of gradient descent, |1 − ηλ| < 1.

**Targets.** κχ ∈ {0.02, 0.05, 0.1, 0.2, 0.3, 0.4} at each a.

**Prediction.** Lag r = κ_k χ against each seed's forced-branch switch, frozen before training as in the first
boundary test.

**Criteria, registered**
- C1: median obs/pred within [0.75, 1.25] in every cell with κχ ≤ 0.1.
- C2: outside [0.75, 1.25] in every cell with κχ ≥ 0.3.
- κχ\* (descriptive): the first target leaving the band.

**Validity, per cell**
- At least 30 of 40 runs cross.
- Median χ_own is within 30% of target.
- η_cell·λ_max ≤ 1 holds along the realised trajectory (checked from the run).

**Competing outcomes**
- C1 and C2 pass: the boundary is at κχ between 0.1 and 0.3.
- C1 fails: the law degrades already below 0.1. The boundary test hinted at this at a = 1.30 (κχ 0.23 → 1.67).
- C2 fails: the law holds to κχ 0.4.

**Compute.** 12 cells × 40 seeds, batched; smaller η lengthens the slow cells. About 1–2 hours at one worker.
