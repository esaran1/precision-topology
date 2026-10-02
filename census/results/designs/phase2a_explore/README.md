# Exploratory computations for the Phase 2A design (NOT registered)

Producers of every number labelled "exploratory" in `../phase2a_slow_sb_design.md`. They use only the population
landscape of the v2/v3 simplicity-bias benchmark (the 800 fixed points, λ = 1e−4; the benchmark has no own-sample
variation) and, for landing fractions, the **exploration seeds 2,930,000–2,930,199** (local numpy Generator draws of
the PyTorch-default init distributions; never to be registered). No candidate registered or pilot seed
(2,936,000–2,936,119; 2,936,900–2,936,959) was drawn, held or trained, and no pilot was run. Every number will be
recomputed and frozen by the registered pipeline if the design is approved. One process at a time, nice 15, one thread;
each log starts with (or contains before each job) its memory-gate line (`gate.sh`: free ≥ 25%, swap free ≥ 500 MB).

- `core.py`: training-coordinate loss/gradient/Hessian of the v3 model, branch points of `src/sb_fold.py` in training
  coordinates (idle unit exactly 0), branch labels by function-space distance (≤ 1e−3), ρ₂, G₊.
- `p2a_explore_a.py`: along M, with ρ on the whole output vector: driving rate σ = ṡ/(ρη), λ_min of P^{1/2}HP^{1/2} on
  the fixed-s tangent, χ(ρ) (floors at ≈ 3.5–4.3 as ρ → 0: the slow mode becomes the output shares), the idle unit's
  free-training block (a saddle, coupling g = ∂²L/∂W∂v ≈ −0.027 at s*), fold constant |m′c′| and ε_F per ρ. The ρ = 1 row
  is the scale-only form's metric (P = I on the fixed-s tangent): |m′c′| = 4.78e−7, Λ_F = 1.51e−3, σ_F = 0.108.
- `p2a_explore_b.py S0`: landing of random holds at s₀ (fixed-s BFGS, the Track 1A basin classifier).
  s₀ = 1.7957: M 80, L0 63, mirror-pair-only ("other", Mp) 57 of 200; s₀ = 2.5: M 99, L0 50, other 51.
- `p2a_explore_d.py`: Mp continued in s: stable ≈ 1.20–5.83 (loses stability by symmetry breaking, then folds at
  5.856); G₊ = 0 at ≈ 5.38; ρ₂ < q on its stable part.
- `p2a_explore_c.py FORM LOG2RHO [S0] [IDLE_EPS] [ETA]` (`run_c.sh` runs a list, gate before each): slow SGD from the
  exact M(s₀), η = 1; `plain` = ρ on all of v, `scale` = ρ on the output scale only. Logged in `p2a_explore_c.log`;
  JSON per run with the tracking table. The last `scale -9` entry was rerun under `/usr/bin/time -l` (peak RSS 0.85 GB,
  from ρ₂ batches of 200 rows; identical result).
- `p2a_explore_e.py`: follow check (state at 0.8 and 0.95·s_F minimised at its own s lands on M).
- `p2a_explore_f.py`: ρ = 2^−11 … 2^−14 in half-octaves; observed crossing; the proposed causal fold forecast at
  f = 0.90 and 0.95 using the frozen `src/causal_forecast.py` cutoff and extrapolation functions unchanged.

Order: a, b (1.7957 and 2.5), d, c (`run_c.sh "scale -11" "plain -11" "scale -9" "scale -13" "plain -8"`, then
`"scale -11 1.7957 1e-6" "scale -11 2.5" "scale -11 1.7957 0 0.5" "scale -15" "plain -14"`), e, f. a must precede c and f
(they read its Λ_F).
