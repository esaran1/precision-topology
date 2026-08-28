# Task B: the gap without topology (Parts 1–3)

Predictions registered in `fold1d_prediction.md`. Data: `fold1d_sweep.csv`,
`fold1d_refine.csv`, `fold1d_geometry.csv`, `estimator_validation.csv`,
`estimator_anisotropy.csv`, `fold_settings.csv`.

## Part 1: the minimal fold task

`sign(|x| − 1)`, width-1 network, four parameters. Monotone `f` ⇒ monotone
logit ⇒ at most one sign change; the task needs two. Provable zero.

| | result |
|---|---|
| **P1-zero** | **exact**: 0/200 at every monotonic value (sin a ≤ 1, pwl α ≥ 0, tanh, ReLU, leaky-ReLU) |
| **P2-onset, family A** | zeros through a = **1.30**; onset (1.30, 1.35]: 2/200 at 1.35 → 15 → 50 → 81/200 at 1.5 |
| **P2-onset, family B** | **no gap**: 119/200 at α = −0.05, flat ~57% for all α < 0 |
| **P4-GELU** | 93/200 = 46.5% (registered ≥ 30% borne out) |

**P3-calibration: procedure vindicated, point guess wrong.** The
registered band [1.10, 1.30] missed by one grid step because it guessed
reachable |w₂| ≈ 20–30; measured reach is ~5–6.5. Running the registered
procedure with measured numbers — solvers satisfy |w₂|·depth ≈ 1.2, so
required |w₂| ≈ 1.2/depth(a); crossing reach ≈ 5–6 puts the onset at
a ≈ 1.4 — **inside the observed transition**. The same arithmetic
predicts family B's missing gap: its required |w₂| at α = −0.05 is ~7–10
(solvers' measured median 10.6), within reach. **One account predicts
both the presence of A's gap and the absence of B's.** The families'
link-setting difference (B onset ≈ −0.22 there, no offset here) is the
account working: what changed between settings is the reachable scale
and the required scale, not the mechanism.

## Part 2a–2b: exact geometry

Grids over [−B, B]⁴ (B = 5 and 10 reported; fractions dilute with B,
ordering and shape identical). Solution set by exact region check; basins
by SGD from 9⁴ grid initializations.

| a (sin) | solution fraction (B=5) | components | basin fraction |
|---|---:|---:|---:|
| 1.05, 1.10 | 0 in box (see below) | 0 | 0 |
| 1.25 | >0, below one cell | 8 | **0** |
| 1.35 | >0, below one cell | 12 | 0.004 |
| 1.5 | 1e−5 | 24 | 0.052 |
| 2.0 | 5e−5 | 64 | 0.502 |
| 3.0 | 1.9e−4 | 80 | 0.466 |

- **The registered volume prediction is borne out**: solution measure
  grows smoothly from the threshold; basin volume jumps at the observed
  onset. At a = 1.25–1.35, **solutions exist with (near-)zero basins** —
  possible-but-unfindable realized exactly, in four parameters.
- At a = 1.05/1.10 the solution set is **empty within both boxes** —
  because required |w₂| ≈ 58 exceeds the box. The box-dependence the task
  brief asked us to justify *is the mechanism made visible*: the solution
  set migrates to large ‖θ‖ as a → 1⁺.
- All solution sets are thinner than one grid cell in some direction
  (erosion inradius < 0.25): **thin sheets with huge catchments** —
  basin-to-solution ratios 400–10,000×.
- pwl(−0.05): solution 7.6e−4, basin 0.33 — the no-gap family has both
  large measure and large basin at reachable scales. GELU: 1e−5 / 0.17.

## Part 2c: the estimator against ground truth

- **Unbiased as a sampler**: 20-draw estimates match 300-draw truth
  within 0.006 everywhere, with binomial spread.
- **What it measures is recovery-including-background**: retrain
  recovery is floored by the from-scratch solve rate (40% at 1D a=1.5,
  floors the curves near 0.8–1.0). In the link setting backgrounds near
  the threshold were ~0, so the committed basin numbers are unaffected;
  at a = 3.0 (background 29%) correcting for it pushes *further against*
  the basin-radius account. **P-basin and P-calibration conclusions stand.**
- **Constructed-vs-found is regime-dependent, not artifactual**: at
  amplification 23 (1D, a=1.5) the retrain gap is modest (0.79 vs 0.87
  at ε=0.3) but the constructed sheet is genuinely ~3× thinner (static
  membership) and its fragile direction is the fold position b₁ (0.50 vs
  1.00 recovery). At amplification 600 (links, 1.02) that thinness is the
  two-order gap. Same anatomy, different severity.
- **Basins are strongly anisotropic** (w₁ immune at ε=1.0; b₁ fragile):
  radius and volume diverge, as suspected — which is why the exact
  volume result (2b) supplies what the radius measurements could not.

## Part 3: intermediate settings — unresolved, honestly

The three-region annulus (2D, width 2) and nested-shells (3D, width 3)
tasks could not resolve onset structure at feasible budgets. A first
design (depth 2, 2,000 steps) produced zero solves for *every*
activation including GELU and sin(3.0) — too hard to measure anything;
the piloted redesign (depth 4, 4,000 steps, wider margins) yields:
annulus — sin(1.5) 1/100, GELU 2/100, all others 0/100; shells — 0/100
everywhere. Monotonic activations are at exact zero in both (as
expected), and every solve that exists is non-monotonic, but 3 solves
total cannot locate an onset or test the calibration across dimensions.
**The cross-dimensional calibration question is left open**, with the
design that would decide it stated: order-of-magnitude larger seed
counts or longer training at these exact tasks, beyond this project's
budget envelope.

---

> **Claim-invalidating correction (2026-08-25).** The Part 2a–2b bullet
> "At a = 1.05/1.10 the solution set is **empty within both boxes** —
> because required |w₂| ≈ 58 exceeds the box … the box-dependence *is
> the mechanism made visible*" is **false and withdrawn**. Solutions
> exist inside [−5, 5]⁴ at every value tested, including **a = 1.02
> with |w₂| = 1** (`src/box_counterexample.py`, verified float64 on
> 500k+ points with strict two-sided margins). The emptiness was a
> **grid-resolution artifact**: the admissible b₂ interval has width
> |w₂|·gap(a), and a 41-point grid (step 0.25) needs |w₂| ≳ 40–456
> depending on a for any node to land inside it — which is where the
> "≈ 58" came from. Consequently every `solution_fraction` in
> `fold1d_geometry.csv` is a **lower bound at grid resolution**, not a
> measure, and the small-a entries mean "unresolvable at 41⁴", not
> "zero". The findability results (0/200 at a ≤ 1.30), the basin-volume
> jump, the solutions-with-zero-basins finding, and the provable
> monotonic zero are all unaffected — basins were measured by SGD from
> 6,561 initializations, not from this grid — and the thin-sheet
> finding is strengthened: the sheets are thinner than the grid that
> was measuring them. Full analysis: `box_emptiness_correction.md`.

> **Restatement at greater strength (2026-08-25).** With the box-emptiness
> artifact removed, the possible-versus-findable result is sharper than
> first reported. Previously the reading was that near-threshold
> solutions migrate to large ‖θ‖ — somewhere training cannot reach in
> norm. That is false. Solutions sit **inside the ordinary parameter
> region** (a = 1.02 solves with |w₂| = 1, every coordinate within
> [−5, 5], a scale training passes through constantly) and have **no
> basin whatsoever**: 0/200 SGD runs solve at a ≤ 1.30, and the exact
> basin fraction is 0 through a = 1.25. The obstruction is not distance
> or norm. It is that the solution set is a sheet of measure ~1e−6 with
> zero attracting volume, sitting in a region training visits and
> passes through without being drawn in. Analytic solution measures:
> `grid_measure_audit.md`, `solution_measure.csv`.

> **Qualification (2026-08-28).** The onset (1.30, 1.35] is specific to
> the standard configuration (Adam, lr 1e-2, 2,000 steps). A step-size
> sweep (`step_size_results.md`) finds 83/200 solves at a = 1.25 with
> Adam lr 3e-2, against 0/200 at baseline — the onset moves below 1.25.
> The monotonic zero and the existence of the gap are unaffected; the
> onset's *location* must be quoted with its optimizer configuration.
