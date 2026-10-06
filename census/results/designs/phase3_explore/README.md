# Phase 3 exploration (width 4, asymmetric windows): EXPLORATORY, NOT REGISTERED

Producers of every number labelled "exploratory" in `../phase3_width4_asym_design.md`. Data used:
- the population (`asym_register._setup()`, Δ = 0.4);
- fresh exploration own samples 7,410,000–7,410,059 (`asym_register.training_set`). The range was verified unused
  before use (`p3_seedscan.py`/`.log`): the 1C regex scan of src/, tests/, results/ and paper/, the registered and
  pilot seed constants of every test module, and the `*seed*` columns of the 14 parquet files.

No registered or pilot seed was drawn, and no pilot was run. Every number would be recomputed and frozen by a
registered pipeline. One process at a time, nice 15, one thread. Each job was preceded by a separate, logged memory
gate (`memory_gate.log`: free ≥ 25%, swap free ≥ 500 MB, disk ≥ 20 GB; all OK). Peak RSS was 0.24 GB in every job.
Total compute was about 45 min (20:56–21:41, 2026-10-05).

**Core code**
- `w4core.py`: the width-K network in the layout z = (α₁..α_K, β₁..β_K, b), v = (v₁..v_K), with analytic gradient and
  Hessian, damped Newton at fixed v, and the exact-extrema G₊ (width2_geometry's algorithm, K units) with a dense
  screen. It also has windings, the coincidence partition and branch type, the split subspace and reduced λ, κ
  (total derivative), adiabatic and ray continuation, W2-A's own-path switch and R4, and SGD training.
- `p3_check0.py`/`.log`: checks of the core code.
  - Gradient and Hessian against finite differences (1e−10).
  - With K = 2 it reproduces W2-A's frozen population numbers exactly: T switch 0.481586, κ₀ 0.04859,
    a = (−0.1390, −0.1266); D switch 5.079637, κ₀ 7.674.
  - Timings.

**Landscape and branches**
- `p3_spop4.py`, `p3_spop4_scan.log`, `p3_spop4_bisect.log`: the width-4 conditional minimiser at fixed s (BFGS,
  200 restarts per scale; NOT T2-1's validated search). With K = 2 it gives the reference losses.
  - The width-4 minimum loss equals the width-2 one to 10 digits at s = 0.2, 0.3, 0.7 and 1.5. The minimiser is a
    width-2 function with coinciding units.
  - The switch is bracketed at [0.43805, 0.43839], inside T2-1's validated [0.4371, 0.4532].
- `p3_landscape.py`: random holds at v₀ = s₀·shares, then Newton, then classes modulo windings and equal-share
  permutations.
  - `p3_land_pop.log`: box 1, shares (0.1, 0.2, 0.3, 0.4), (¼, ¼, ¼, ¼) and (0.1, 0.1, 0.4, 0.4).
  - `p3_land_pop_more.log`: box 3; s₀ = 0.35; s₀ = 0.111.
- `p3_unconv.py`/`.log`: the box-1 starts without an accepted Newton point, held 60,000 steps. 10 of 30 are then
  accepted, and the rest are near-idle-unit saddles (min |α| ≈ 0.004, λ ≲ 0).
- `p3_branches.py`: for each unplaced class, the adiabatic switch, κ₀, the four winding coefficients, λ, the minimum
  split eigenvalue along the path and the step-halved switch.
  - `p3_branches.log`: the box-1 classes. Its MULTI pass crashed in a print statement.
  - `p3_branches_multi.log`: the MULTI pass after the print fix. Results are in the `*_branches.json` files.

**Own samples and runs**
- `p3_own.py`: per exploration seed, the own copies of Q, S and T4 (Newton from the population point), their
  adiabatic switches and κ, the branch-point-start hold, and 20 random holds per seed (seeds 7,410,000–019).
  - `p3_own.log`: seeds 7,410,000–019.
  - `p3_own2.log`: seeds 7,410,020–059, branch-point checks only.
- `p3_runs.py`: one training run from the branch-point start, released with SGD. It records the observed crossing,
  the own-path switch, the full-path R4 and the CAUSAL forecast at f = 0.95 (rows < t_c only; quad extrapolation of
  log s and of each share). Every run is appended to `p3_runs.jsonl`.
  - Logs: `p3_runs_Qpop.log`, `p3_runs_pop.log` (the ρ scan on the population) and `p3_runs_seeds.log` (seeds
    7,410,000–007).
- `p3_gate.py`/`.log`: Wilson intervals and binomial gate chances (arithmetic only).
- `p3_summary.py`/`.log`: per-arm statistics, and the 1C tolerance rule applied for illustration (arithmetic only).

**Labels.** Q = population class 49 of `p3_land_pop_box3.json`, S = class 39 of the same file, and T4 = class 2 of
`p3_land_pop_0.1_0.2_0.3_0.4.json`.
