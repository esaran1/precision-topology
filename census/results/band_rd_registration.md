# Track 3B: band task in R^d (d = 2, 4) — design and registration

Written 2026-09-25, before any registered training run. Producer: `census/src/band_rd.py`; tests:
`census/tests/test_band_rd.py`. Outputs: `census/results/band_rd/`. Numeric design checks below are labelled
**validated, not certified**.

## 0. Order of work (disclosure)

1. The design (data, model, initialisation, R^d gap, crossing rule) was fixed in code (`src/band_rd.py`, module
   docstring and `make_data_rd`, `init_rd`, `gap_rd`, `train_batch`) before any d > 1 training. This document was
   written afterwards, while the frozen inputs were being computed. After the pilot the following were added to the
   code: the budget, the secondary seed arm, `pilot_diag`, and extra descriptive columns; data, model, initialisation,
   gap and crossing rule were not changed.
2. The only d > 1 training before this registration used seeds outside the registered range:
   * unit tests on constructed initial states (seeds 889,600–889,911; placed or nearly placed starts);
   * the pilot (§3, seeds 889,000–889,011), which writes crossing counts, crossing steps and wall time only; and a
     pilot diagnosis of the pilot runs that did not cross (`pilot_stuck.csv`, final states only).
   No crossing scale |w₂|_c of any d > 1 run, pilot or otherwise, was written or read before this registration.
3. The frozen inputs (§4) were computed before this commit and are committed with it.

## 1. Design

**Data.** x = (x₁, …, x_d). x₁ and y are exactly the width-1 task: `fold1d.make_data(200, seed)` (inner
U(−0.8, 0.8) class 0; outer ±U(1.2, 2.0) class 1; 200 per class, 400 points). The noise coordinates x₂ … x_d are
i.i.d. **U(−2, 2)**, independent of x₁ and y, drawn from their own stream `numpy.random.default_rng([seed, 3])` as a
400 × 3 array of which the first d − 1 columns are used (so the d = 2 noise column is the first d = 4 noise column, and
x₁ is identical across d for a seed). y = 1 iff |x₁| > 1.

Why U(−2, 2): (i) same range as x₁, so the noise coordinates cannot be told apart from x₁ by their scale;
(ii) E[x_n²] = 4/3 is close to E[x₁²] ≈ 1.41, so gradient and Adam scales are comparable across coordinates; (iii) a box
support gives an exact R^d gap: w_noise·x_noise ranges over exactly [−N, N] with N = 2‖w_noise‖₁. The alternative
"same marginal as x₁" has a support with holes, so the gap on its support is not a simple widening.

**Model.** z = w₂·f_a(w·x + b₁) + b₂, w = (w₁, w_noise) ∈ R^d, f_a(t) = t + a sin t, a ∈ {1.30, 1.50}. Parameter
vector p = (w₁, b₁, w₂, b₂, w_noise), drawn as `torch.manual_seed(seed); torch.empty(d + 3).uniform_(-1, 1)` (float32,
then float64): the standard U(−1, 1) initialisation per coordinate, and the first four draws are exactly the width-1
initialisation of the same seed (tested). Training: Adam lr 0.01, full batch, BCE with logits, float64, all d + 3
parameters trained (the width-1 phase2b protocol with extra input weights).

**Placement gap in R^d on the continuous support.** For the inner class the pre-activation w·x + b₁ ranges over
[min(±0.8w₁) + b₁ − N, max(±0.8w₁) + b₁ + N], N = 2‖w_noise‖₁; each outer window w₁·[1.2, 2.0] + b₁ and
w₁·[−2.0, −1.2] + b₁ is widened the same way. G₊ = min_O f_a − max_I f_a, G₋ = min_I f_a − max_O f_a, with exact
extrema of f_a on intervals (`profiled_bnb._interval_extrema`: endpoints and the critical points ±acos(−1/a) + 2πk);
G = G₊ if w₂ > 0, else G₋. Checks (tests): with w_noise = 0 it equals the width-1 gap `profiled_bnb.gap` **bit for bit**;
G₋(w, b₁) = G₊(−w, −b₁); it agrees with a dense grid over the support (grid gap ≥ exact gap, difference < 2e−3 on 30
random cases); it depends on w_noise only through ‖w_noise‖₁ and decreases in it; the batched torch version agrees with
numpy to 1e−12. G1 denotes the same gap with w_noise set to 0 (the gap of the x₁ direction alone), recorded for the
description.

**Crossing.** The first step t ≥ 1 (after the t-th Adam update) with G > 0; s_c = |w₂| at that step. Checked at every
step. A run with G > 0 at initialisation is "placed at init" and has no crossing (none in the pilot).

**The conditional threshold is unchanged in the weight direction (proof).** Population: x₁ and y from the width-1
population (any distribution; the certified brackets use `width2_conditional.population()`), x_noise ~ U(−2, 2)^{d−1}
independent of (x₁, y). Write ξ = w_noise·x_noise, independent of (x₁, y). At fixed w₂ = s,

  L_d(w₁, w_noise, b₁, b₂) = E_ξ[ L₁(w₁, b₁ + ξ, b₂) ] ≥ min_b L₁(w₁, b, b₂) ≥ L₁*(s),

where L₁ is the width-1 population loss and L₁*(s) its conditional minimum. Equality in the first inequality needs ξ to
lie in argmin_b L₁(w₁, b, b₂) almost surely. That set is discrete (L₁ is real-analytic in b and not constant: it grows
without bound as |b| → ∞ because f_a(t) − t is bounded), while for w_noise ≠ 0 ξ has a density. So every conditional
minimiser of the R^d population loss has **w_noise = 0**, and its (w₁, b₁, b₂) is a width-1 conditional minimiser; with
w_noise = 0 the R^d gap is the width-1 gap. The placed/unplaced status of the conditional minimiser at every s is
therefore that of width 1, and the certified width-1 bracket carries over unchanged: a = 1.30, |w₂| ∈ (4.95, 4.9625];
a = 1.50, |w₂| ∈ (2.525, 2.5375] (`results/cond_certified_brackets.csv`, kind glob). This is a proof by reduction
(the certificate itself is the width-1 one). At w_noise = 0 the Hessian noise block is E[x_n²]·∂²L₁/∂b₁² times the
identity and the noise–(w₁, b₁, b₂) cross terms vanish (E[x_n] = 0).

It does **not** hold on a finite training sample: the sample noise coordinates correlate with the residuals, so the
own-sample minimiser has a small nonzero w_noise, and its R^d gap is reduced by the widening 2‖w_noise‖₁.

**Numeric check (validated, not certified;** `python -m src.band_rd popcheck` → `band_rd/popcheck_summary.csv`,
`popcheck_starts.csv`, `popcheck_profile.csv`). Population loss with x₁ on the population grid and the noise by tensor
Gauss–Legendre (32 nodes for d = 2, 8³ for d = 4; the inequality L_d ≥ L₁* holds for any positive quadrature rule, strictness for w_noise ≠ 0 is then generic rather than guaranteed). At both
ends of each certified bracket, for d = 2 and 4: (i) the width-1 population branch (continued from the committed switch
point z*, `lag_law.branch_point`) is unplaced at s_lo and placed at s_hi (G1 = −2.68e−4 / +1.50e−4 at a = 1.30;
−2.54e−4 / +1.36e−3 at a = 1.50); (ii) the noise block equals (4/3)∂²L/∂b₁² to relative 5e−15 and the cross terms
are < 1e−16; (iii) 14 damped-Newton starts per case in all d + 2 hidden coordinates (6 from the branch with random
w_noise of norm 0.05, 0.2, 0.5; 8 random): **no start ends below L₁*(s)**; every start that reaches L₁*(s) has
‖w_noise‖ ≤ 5e−12; the others end at higher stationary points (loss ≈ 0.344–0.376 above L₁*; gradient ≤ 8e−10), some of which
carry a large noise weight (‖w_noise‖ ≈ 0.64–0.84, loss 0.371–0.376 above L₁*, found in 7 of the 8 cases) — a
noise-direction stationary point of the population loss; (iv) the profile
min_{w₁,b₁,b₂} L(w_noise = r·u), r ∈ {0, 0.005, 0.02, 0.05, 0.1, 0.2}, increases strictly in r for every case and
direction.

## 2. Seeds

Registered seeds 880,000–880,039 (**primary**, 40 per cell; the same 40 seeds in all four (d, a) cells, a paired
design as in the width-1 work) and 880,040–880,119 (**secondary**, 80 further seeds per cell). `grep` over `src/`,
`tests/` and `results/` (including all CSVs) found no use of any seed in 880,000–899,999 before this track. Pilot seeds
889,000–889,011 are outside both ranges.

## 3. Pilot (NON-registered, disclosed) and budget

`python -m src.band_rd pilot` (12 seeds, budget 128,000, every-step crossing) → `band_rd/pilot.csv`:

| d | a | crossed | crossing steps (all) |
|---|---|---|---|
| 2 | 1.30 | 12/12 | 1,933 … 11,654 |
| 2 | 1.50 | 12/12 | 856 … 10,636 |
| 4 | 1.30 | 9/12 | 2,460 … 7,835 |
| 4 | 1.50 | 9/12 | 1,162 … 4,853 |

The 3 + 3 d = 4 pilot runs that did not cross by 128,000 steps are noise-dominated and (near-)stationary:
`pilot_stuck.csv` (`python -m src.band_rd pilot_diag`, state after 40,000 steps): ‖w_noise‖₂ = 0.63–2.01 against
|w₁| = 0.03–0.66, |w₂| = 0.11–2.04, G1 < 0; the largest change of |w₁|, ‖w_noise‖, |w₂| over the last 36,000 steps is
≤ 0.002 for four of them and 0.05 / 0.18 for two (seeds 889,011 and 889,005 at a = 1.30). The unit has fitted the sample
through the noise coordinates (the finite-sample counterpart of the noise-direction stationary point in §1 (iii)).

**Budget: 64,000 steps** (2× the width-1 standard 32,000; 5.5× the largest pilot crossing step, 11,654; no pilot run
crossed between 11,654 and 128,000). The instruction "budget sufficient for ≥ 90% crossing" **cannot be met at d = 4 by
any budget** on the pilot's evidence (9/12 = 75%, not budget-limited). Consequence, stated before running: a d = 4 cell
may have fewer than 30 crossings of 40 and then be UNRESOLVED; the secondary arm (§5) is registered for that reason
(the precedent is Track 3A's primary/secondary design). Batching is validated before the runs
(`python -m src.band_rd validate` → `validate_batching.csv`: pilot seed 889,000 trained alone vs inside the 12-seed
batch for 3,000 steps in all four cells); STOP if any parameter differs by more than 1e−10. Result (before this
commit): largest parameter difference 7.1e−15 (not bitwise; summation order), identical crossing steps (1,856 and 2,273
in the two cells where the seed crossed within 3,000 steps) and identical |w₂| at the crossing: PASS.

## 4. Frozen inputs (committed with this registration)

* `band_rd/width1_reference.json` — width-1 free-training reference from committed data
  (`results/phase2b_checkpoints.csv`, budget 32,000, seeds 0–39, crossing residual r = |w₂|_c/s_mid − 1 against the
  certified bracket midpoint s_mid). SHA-256 `da0f85fde9eeb7ae3a36a8a64ffefc6a66a86f2a9e99928c418f75b3ca879a61`.

  | a | crossing runs | s_lo | s_mid | Q1 | median | Q3 | fraction s_c ≥ s_lo |
  |---|---|---|---|---|---|---|---|
  | 1.30 | 38/40 | 4.95 | 4.95625 | 0.035215 | 0.095937 | 0.153509 | 0.842 |
  | 1.50 | 38/40 | 2.525 | 2.53125 | 0.072484 | 0.126303 | 0.184891 | 0.974 |

* `band_rd/own_x1_frozen.csv` — the width-1 own-sample threshold (`own_threshold.own_threshold`, the committed
  global conditional search) of each primary seed's x₁ sample, both a (80 pairs). SHA-256 in
  `band_rd/own_x1_frozen.csv.sha256` (`b4c9abcb13116674b260f0fe0caa8e0fb3e20ca08dfcf7c106b3ee47a188fd2d`); all 80 defined.

`run` and `score` refuse to proceed if either hash changes.

## 5. Registered predictions and rules

Cells: (d, a) ∈ {2, 4} × {1.30, 1.50}. **Primary**: the 40 primary seeds per cell; rules P1, P2a, P2b. **Secondary**:
the 120 seeds pooled (primary + secondary) per cell; rules P1 and P2a only (own thresholds are frozen only for the
primary seeds). Primary verdicts are the registered result; secondary verdicts are reported beside them, never in place
of them. A **width-1 control** (d = 1, same code, same 40 primary seeds) is run and reported descriptively; it is not
scored and does not enter any verdict.

Crossing runs exclude runs placed at initialisation. In every rule, a cell with **fewer than 30 crossing runs is
UNRESOLVED**.

* **P1 (at or above the threshold).** PASS iff the fraction of crossing runs with s_c ≥ s_lo (4.95 at a = 1.30,
  2.525 at a = 1.50) is ≥ 0.90.
* **P2a (lag in the width-1 range).** Let r_pop = s_c/s_mid − 1. PASS iff Q1 ≤ median(r_pop) ≤ Q3 with the width-1
  quartiles above (a = 1.30: [0.035215, 0.153509]; a = 1.50: [0.072484, 0.184891]; inclusive). A FAIL is reported with
  its side (above / below).
* **P2b (lag law with width-1 κ).** For each crossing run: growth = log(s_c/s_{c−100})/100; relax =
  0.01·λ_min(D^{−1/2} H_sig D^{−1/2}) (`residual_timescale.relax_rate`), where H_sig is the (w₁, b₁, b₂) block of the
  Hessian of the run's own-sample loss at its branch (damped Newton in all hidden coordinates (w₁, b₁, b₂, w_noise) at
  w₂ fixed, from the crossing state) and D is Adam's bias-corrected v̂ of (w₁, b₁, b₂) at the crossing; χ = growth/relax;
  κ_k = `lag_law.kappa_k` with the width-1 population landscape (`lag_law/kappa.csv`: H, θ*′, ∇G) and the median Adam
  preconditioner, on the run's winding k (canonical b₁, `lag_law.canonical_b1`); pred = κ_k·χ. Observed: r_own =
  s_c/s_own − 1 with s_own the frozen x₁-sample own threshold. Usable runs: finite χ > 0, converged branch
  (gradient < 1e−8), finite s_own. PASS iff |median(r_own) − median(pred)| ≤ max(0.01, 0.25·|median(pred)|) over usable
  runs. UNRESOLVED with < 30 usable runs or with > 20% of crossing runs unusable. (With d = 1 these are exactly the
  width-1 quantities of `residual_timescale` / `lag_law`, tested.) For orientation: width-1 κ_k (median P) is +7.61 at
  a = 1.30, k = −1 and +4.05 at a = 1.50, k = 0; the width-1 free-training comparison (`lag_law/compare_arms.csv`,
  4b control) had pred 0.030 / obs 0.031 at a = 1.30 and 0.063 / 0.066 at a = 1.50.

**What the registered rules expect, and what would falsify it.** The claim under test is: in R^d the unit finds the x₁
direction, and training crosses the (unchanged) threshold from above with a width-1-sized lag.
* P1 at a = 1.30 is at risk **even without noise coordinates**: the width-1 reference fraction is 0.842 < 0.90, because
  finite-sample own thresholds fall below the population threshold for some seeds. P1 is registered as specified by the
  author; a FAIL at a = 1.30 with a fraction near 0.84 would reproduce width-1 behaviour, not reveal a d-effect (the
  d = 1 control on the same seeds is reported beside it). At a = 1.50 the width-1 reference is 0.974.
* Competing prediction C1 (**noise delays placement**): if w_noise must decay before G > 0 and decays slowly relative to
  the growth of |w₂|, crossings come late: P2a FAILS above Q3 and P2b FAILS with r_own − pred > tol.
* Competing prediction C2 (**early placement**): crossings well below the threshold (P1 fails at a = 1.50 or P2a fails
  below Q1) — would contradict the threshold carrying over to R^d.
* Competing prediction C3 (**the direction is not found**): runs stuck at the noise-direction stationary point; with
  < 30 crossings the cell is UNRESOLVED (pilot: 25% of d = 4 runs).

**Validity checks.** (V1) frozen-input hashes (run and score); (V2) the R^d gap recomputed in numpy at every recorded
crossing state must be > 0, else the cell is STOP (all rules); (V3) batching validation above (STOP before running if it
fails); (V4) branch convergence (P2b UNRESOLVED if > 20% of crossing runs unusable). Every rule and check is exercised
on constructed pass and fail cases in `tests/test_band_rd.py`.

## 6. Registered descriptives (not scored): how the unit finds the direction

Per cell (primary and pooled), from `band_rd/scored_runs.csv`, `descriptives.csv` and the per-cell traces
(`traces_{arm}_d{d}_a{a}.npz`, every 10 steps: step, |w₁|, ‖w_noise‖₂, |w₂|, G, G1):
* ρ = ‖w_noise‖₂/|w₁| at initialisation, at the first step with |w₂| ≥ 0.5·s_mid, and at the crossing (quartiles);
* the first recorded step with ρ < 0.1, relative to the crossing step (median ratio; fraction of crossing runs in which
  it precedes the crossing);
* the number of runs crossing with non-negligible noise weight, **ρ_c ≥ 0.05**, and the largest ρ_c;
* the widening at the crossing relative to the projected inner half-width, 2‖w_noise‖₁/(0.8|w₁|);
* the noise delay: steps and relative scale between the first step with G1 > 0 (x₁ direction placed) and the crossing
  (G > 0);
* non-crossing runs: count, how many end with ρ > 1 (noise-dominated), median final |w₁|, ‖w_noise‖, |w₂|;
* winding counts, mirror-branch count, the branch's own w_noise norm, and λ_min(full)/λ_min(signal block).

## 7. Commands

    python -m src.band_rd reference | own | popcheck | pilot | pilot_diag | validate   # before this commit
    python -m src.band_rd run      # after this commit: primary (d = 2, 4, control d = 1), then secondary
    python -m src.band_rd score
