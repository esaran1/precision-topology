# Exploratory feasibility scripts for the width-2 asymmetric-window design (NOT registered)

Producers of every number labelled "exploratory" in `../width2_asym_design.md`. They used only the population
(`asym_pilot.population(0.4)`) or the training sets of ALREADY-USED T2-3 seeds 600,000–600,011
(`asym_register.training_set`; landscape and holds only). No candidate registered or pilot seed (884,xxx) was drawn,
held or trained, and no pilot was run. Every number will be recomputed and frozen by the registered pipeline if the
design is approved. `run_all.sh` reproduces everything in order (one process, nice 15, one thread; check memory first).

- `w2core.py`: width-2 loss with analytic gradient and Hessian (checked against finite differences in `a`), damped
  Newton at fixed v, exact-extrema G₊ on the asymmetric windows (`width2_geometry.gaps`), canonical form (sign flip,
  2π windings, duplicate test), the split (antisymmetric) subspace of a duplicate point, κ (P = I, reduced Hessian,
  denominator = total derivative of the branch gap along the v path), ray continuation with switch bisection.
- `w2_explore_a.py`: landscape at the held v₀: random hidden starts held by GD, then Newton; branch classes and counts
  (`a.json`: s₀ = 0.2225, equal shares; `a_s2.5398.json`: s₀ = half of D's switch; `a_s0.2225_u0.1.json`: shares
  (0.1, 0.9)).
- `w2_explore_b.py`: each class continued to its switch along the held ray (and with the step halved) and along the
  adiabatic path (reduced gradient flow of v); λ_min, split-block eigenvalues, κ by winding sum k₁ + k₂.
- `w2_explore_c.py`: free full-batch GD released from a class point on the population: crossing, lag-free switch along
  the run's own v path, κχ, lag in steps, χ_t window maximum, share at the crossing. All runs in `w2_explore_c.log`.
- `w2_explore_d.py`: landing fractions at equal shares as a function of s₀.
- `w2_explore_e.py`: own-sample duplicate copies (switch, κ, split block) and landing fractions on seeds 600,000–600,011;
  landing at shares 0.1 / 0.25 / 0.4 on the population.
- `w2_explore_f.py`: seed 600,000's unresolved starts with a 10× longer hold; own-sample two-unit copy T at shares
  (0.1, 0.9) (adiabatic switch) and landing fractions on seeds 600,000–600,005.
- `w2_explore_g.py`: pooled own-sample landing rates and the binomial chance of passing the proposed gate (arithmetic only).
