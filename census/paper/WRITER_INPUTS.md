# Writer inputs

Everything below comes from this repository. Each item names its ledger ID (`CLAIMS.md`),
the artifact that holds the number and the script that produces it. Every number first
stated here is checked by `src/verify_ledger.py` (0 findings as of 2026-09-23). "Certified Ĝ"
means the value in `results/ghat_certified_all.csv`, column `Ghat_certified` (§2).

**This is the single final handoff (2026-09-23).** It supersedes earlier versions and any
number in the draft that conflicts with it.

### Still unresolved

1. **Ren & Lim's venue** (the only open item). `ren2026low` is cited in its arXiv version (2606.31856). The ICML 2026 /
   PMLR 306 listing could not be confirmed on PMLR: volume 306 does not exist there yet. Switch to
   the PMLR entry once it appears (§14).
Nothing else is open. Every item in §15 is resolved: either a committed script now produces the
number, or the claim has been removed from every paper-facing file.

### Decisions applied, and what the draft must change

1. **Registration tally (§9).** The headline is **151 predictions scored by their registered rules:
   66 PASS, 47 FAIL, 8 PARTIAL, 30 UNRESOLVED.** The **13 verdicts assigned post hoc** in the census
   are listed separately (3 PASS, 6 FAIL, 4 PARTIAL), each with its row.
2. **E-2 failed and Block E is negative per its registration (§8).** Report the 0/37 against 33/37
   contrast as a measurement, not as a passed causal test.
3. **Numbers at a = 1.30 (§4).** **R_glob = 0.2153; the certified median crossing is 0.2338.**
   The draft's 0.2332 used the old Ĝ.
4. **Distance to constructed solutions (§5).** "At most 0.8% closer" and "no point comes meaningfully
   closer" are withdrawn. The minimum is at initialisation in 114 of 400 runs, and the largest
   approach is 23.4%.
5. **Margin (§11).** The population statement is **17 of 81 found solutions (21%) at a = 1.5 have
   margin below the construction's 0.0384 (minimum 0.0017).** "6 of 24" and "5 of 20" are withdrawn.
6. **Preconditioned distance.** Withdrawn everywhere. The distance finding is Euclidean only.
7. **Metric check (§13).** Print "76% of the error improvement occurs where no run solves". Do not
   print "the crossings do not even overlap".
8. **Phase 1 population (§7).** Use one population throughout.
9. **Ĝ enclosure (§2).** The enclosure is now globally certified; R is unchanged.
10. **Optimiser in T1 (§1).** Every link run used Adam, not SGD.
11. **Bibliography (§14).** 32 entries: 31 verified, 1 in its arXiv version, none failed.

---

## 1. Training losses and configurations

### Width-1 fold task (T43, T52, T55, T57)

- **Model**: `N(x) = w₂ f_a(w₁x + b₁) + b₂`, with `f_a(t) = t + a sin t` and one logit
  (`src/fold1d.py:58-60`).
- **Loss**: binary cross-entropy on the logit, **mean over the batch**
  (`F.binary_cross_entropy_with_logits`, default reduction; `fold1d.py:83`):
  `L = (1/n) Σ [log(1 + e^{N(xᵢ)}) − yᵢ N(xᵢ)]`.
- **Labels**: `y ∈ {0, 1}`. The inner window `I = [−0.8, 0.8]` is **class 0**; the outer window
  `O = ±[1.2, 2.0]` is **class 1** (`fold1d.py:23-38`). With ±1 labels this is the same logistic
  loss.
- **Data**: 200 points per class, uniform on each window (400 total), full batch.
- **Training**:
  - Adam, lr 1e-2, 2,000 steps (`fold1d.py:26-27, 80`).
  - Initialisation `U(−1, 1)⁴`.
  - float32 in the main sweep. Block F, barrier and Hessian code use float64.
- **Budget experiments** (T42/T43) use the same loss with the step budget varied.
- **Other optimisers**:
  - AdamW lr 1e-2 with weight decay 0.01 in Block C (`blockC_equivalence.py:36, 60`).
  - Block F uses AdamW with PyTorch's default weight decay, which is also 0.01
    (`blockF_optimisers.py:42`).
  - SGD lr 0.3 in both.
- **"Solved"**: correct sign everywhere on a dense grid of both windows (4,001 inner and 2×2,000
  outer points; `fold1d.solves`).

### Link setting, width 3 (T1)

- **Model**: constant-width MLP with a **two-logit** head (`src/models.py:49-57`).
- **Loss**: softmax cross-entropy, **mean** reduction (`F.cross_entropy`, `src/train.py:181-182`).
- **Labels**: `{0, 1}` int64, one per solid torus. Class 0 is the core in the `z = 0` plane
  (`src/data.py:123-139`).
- **Data**: 1,000 train and 1,000 eval points per class (2,000 + 2,000), tube radius 0.2.
  Train seed `10,000 + s`, eval seed `20,000 + s`.
- **Training (six of the seven strata)**: full batch, **Adam** lr 1e-2, 2,000 steps, float32,
  deterministic CPU, PyTorch default initialisation (`train.py:23-24, 160`).
- **Protocol stratum (360 runs)**: minibatch Adam lr 1e-3, batch 128, up to 800 epochs,
  patience 150. It early-stops on **evaluation** accuracy, which is the set it is then scored
  on (`src/author_protocol.py:47-50, 33`). See §15.
- **"Separated"**: final eval accuracy = 1.0 on the 2,000-point eval set (`perfect_eval`;
  `zero_decomposition.py:26`).
- **Correction**: T1 read "under SGD"; it now reads **Adam**. Every stratum trains with Adam.

### ℝ⁵ linked spheres (T72, T75)

- **Model**: `Linear(5,5) → activation` five times (width 5, depth 5), then `Linear(5,1)`.
- **Loss**: binary cross-entropy with logits, **mean** (`src/blockA5d.py:199`). This is the same
  function class and loss as Ren–Lim's two-logit cross-entropy.
- **Labels**: sphere A = 0, sphere B = 1.
- **Training**: full-batch Adam, lr 1e-3, constant, float32, one CPU thread. Budgets of 1k / 4k /
  16k / 64k steps are checkpoints of one 64,000-step run.
- **Data**: Ren–Lim's generator (reimplemented call for call), targeted thickening ρ = 0.5,
  10,000 training points.
  - **k = 10**: 10 copies; held-out set of 20,000 uniform points.
  - **k = 1** (1 copy): held-out sets of 20,000 uniform points, 20,000 stratified points
    (closest 10% of each core), and 200,000 uniform points (counted only when the uniform set
    has zero errors).
- **"Perfect"**:
  - k = 10: zero uniform errors.
  - k = 1: zero uniform **and** zero stratified errors (registered).

---

## 2. Ĝ: the certified value R uses (T64, T65, T76)

- **Which point R uses**: the **lower end** of the enclosure. This is the gap attained at an
  actual placement, found by the grid search with exact x-extrema (`kappa_certify.certify_exact`
  → `Ghat_lo`, stored as `Ghat_certified`; `src/session_artifacts.py:335-337`;
  `R = |w₂|·Ghat_certified/2`, `src/r50_fit.py:47-54`). Because it is attained, it is a valid
  lower bound on the supremum.
- **Gap evaluation**: exact in x. The extrema of `f_a(w₁x+b₁)` on each window are the endpoints
  plus the roots of `1 + a cos(w₁x+b₁) = 0` (`src/exact_extrema.py`). The gap is oriented:
  `G = max(min_O φ − max_I φ, min_I φ − max_O φ)`.
- **Analytic compact domain** (T76, `src/ghat_bnb.py` docstring):
  - `f_a(u) − f_a(t) ≥ (u − t) − 2a` for `u ≥ t`. Comparing `x = −2.0` with `0.8`, and `−0.8`
    with `2.0`, shows both orientations are negative once `|w₁| > a/1.4`.
  - `G` is invariant under `b₁ ↦ b₁ + 2π` (φ shifts by a constant) and under
    `(w₁, b₁) ↦ (−w₁, −b₁)` (`f_a` is odd).
  - `G = 0` at `w₁ = 0`.
  - Hence **Ĝ(a) = sup over w₁ ∈ (0, a/1.4], b₁ ∈ [0, 2π)**, which lies inside the writer's
    `w₁ ∈ (0, a]`, `b₁ ∈ [0, 2π)`. Every certified argmax satisfies `w₁ ≤ a/1.4`.
- **The original zoom certificate did not give a global upper bound** (T76).
  - It searched `w₁ ∈ [−3, 3]`, `b₁ ∈ [π−6, π+6]` on a square grid (sup metric), with
    `h₀ = 0.05` and `h ÷ 5` per round, stopping when `L·h/2 < 10⁻³·Ĝ`.
  - After the first pass it refined only a ±12h box around the coarse argmax.
  - Its Lipschitz constant `L = (1+a)·max(1, max|x|) = 2(1+a)` gave slack `(1+a)h`. That
    under-states the step: moving `(w₁, b₁)` by `(δw, δb)` moves `max_I φ` by up to
    `(1+a)(0.8δw + δb)` and `min_O φ` by up to `(1+a)(2δw + δb)`, so `G` moves by up to
    **`(1+a)(2.8δw + 2δb)`**.
  - Its lower end is unaffected.
- **Global certificate now used for the enclosure**: branch and bound over
  `(0, a] × [0, 2π]` (`src/ghat_bnb.py` → `results/ghat_bnb.csv`).
  - Cells are sup-norm with half-widths `(h_w, h_b)`, starting at 0.05 and halving each round.
  - Each cell's centre is evaluated exactly, with upper bound
    `G(centre) + (1+a)(2.8h_w + 2h_b)`.
  - A cell is discarded when its upper bound ≤ the best attained value.
  - It stops when the largest surviving bound is within 10⁻³ (relative) of the best value.
  - All 13 values of a converged in 10–19 rounds.

| a | Ĝ used for R (attained) | global lower | global upper | rel. width | argmax (w₁, b₁) |
|---:|---:|---:|---:|---:|---|
| 1.02 | 0.0016267 | 0.0016267 | 0.0016277 | 0.056% | (0.2254, 2.9724) |
| 1.05 | 0.0063601 | 0.0063601 | 0.0063638 | 0.059% | (0.3526, 2.8767) |
| 1.10 | 0.0176720 | 0.0176734 | 0.0176888 | 0.087% | (0.4901, 2.7729) |
| 1.15 | 0.0319184 | 0.0319204 | 0.0319519 | 0.099% | (0.5903, 2.6969) |
| 1.25 | 0.0665056 | 0.0665073 | 0.0665731 | 0.099% | (0.7383, 2.5839) |
| 1.30 | 0.0861018 | 0.0861030 | 0.0861703 | 0.078% | (0.7967, 2.5390) |
| 1.35 | 0.1069136 | 0.1069141 | 0.1069828 | 0.064% | (0.8482, 2.4994) |
| 1.40 | 0.1287719 | 0.1287725 | 0.1288427 | 0.055% | (0.8940, 2.4639) |
| 1.45 | 0.1515453 | 0.1515456 | 0.1516890 | 0.095% | (0.9354, 2.4318) |
| 1.50 | 0.1751262 | 0.1751257 | 0.1752720 | 0.084% | (0.9730, 2.4025) |
| 1.60 | 0.2243602 | 0.2243673 | 0.2245195 | 0.068% | (1.0391, 2.3508) |
| 2.00 | 0.4410339 | 0.4410336 | 0.4413848 | 0.080% | (1.2265, 4.0810) |
| 3.00 | 1.0522978 | 1.0522967 | 1.0532331 | 0.089% | (1.4641, 2.0084) |

- The value used for R is within **0.0084%** of the global lower end at every a and never exceeds
  the global upper end. **No R, verdict or figure changes.**
- **Suggested wording**: "Ĝ(a) is enclosed to within 0.1% by a branch-and-bound certificate over
  the compact domain w₁ ∈ (0, a/1.4], b₁ ∈ [0, 2π); R uses the attained lower end."
- The old table in `ghat_unification.md` listed restricted values in its "certified" column. It
  is corrected.

---

## 3. Conditional search (T58, T60; deviation in T74)

**Frozen Block B procedure** (`src/blockB_landscape.py:32-39, 46-146`):

- **Problem**: at fixed `w₂`, minimise the population logistic loss over `(w₁, b₁, b₂)`.
  The population is 800 deterministic points: 400 `linspace` on I and 200 per side on ±[1.2, 2.0].
  Everything is float64.
- **Optimizer**: Adam, lr 3e-2, dropping to 5e-3 at half the steps.
- **Restarts and initialisation**:
  - `R_glob`: 50 restarts. `R_solve`: 24.
  - Each restart draws `w₁ ~ U[−2, 2]`, `b₁ ~ U[−4, 4]`, `b₂ ~ U[−|w₂|, |w₂|]` (seeded).
- **Two stages**:
  - Screen every restart for **600 steps**.
  - Discard degenerate runs: `|loss − log 2| < 1e-4` or `|w₁| < 1e-3` (the constant predictor).
  - Keep the best **8**, warm-start each for **3,000 steps**, and discard degenerate runs again.
  - The minimiser is the lowest loss.
- **Domain**: unconstrained. The class gap is evaluated on a 4,001-point dense grid.
- **Stopping rule**: a fixed number of steps; no convergence test. "Converged at 3,000"
  is checked empirically: gaps settle by 3k, not by 900.
- **Switch location**: monotone bracketing in `|w₂|`, coarse 0.5 then fine 0.05, over
  `|w₂| ∈ [1.5, 11]`.
  - `R_glob` is the first `w₂` where the minimiser's gap > 0.
  - `R_solve` is the first where it solves.
  - The a = 1.35–1.60 `R_glob` windows were refined to 0.01 (`blockB_fine_windows.csv`).

**Scale-equivariant deviation (T74; `src/blockB_scaled.py`; log in `blockB_scaled_validity.md`)**:

- **Why**: at a = 1.01–1.04 the raw starts collapse. At a = 1.02, 48 of 50 screening starts went
  degenerate, and a start at the certified optimum also collapsed for R ≥ 0.35.
- **Change**: only the coordinates. `w₁ = s·u`, `b₁ = c + s·v`, logit `W·(f(c+sσ) − f(c))/ε^{1+1/q} + b₂′`
  with `W = w₂ε^{1+1/q}` and `s = ε^{1/q}` (q = 2 for the sin family, `c = π`).
  - Restarts, screen/keep, steps, schedule, degeneracy rule and data are all unchanged.
  - The frozen random box is read in `(u, v, b₂′)`.
  - The R grid is fixed at family A's resolution at a = 1.30: ΔR = 0.0215 coarse, 0.00215 fine.
- **Gates at a = 1.30** (scaled against frozen, tolerance one grid step):

| family | threshold | difference |
|---|---|---:|
| A | R_glob, R_solve | −0.00166 each |
| q2 | R_glob, R_solve | +2.4e-7, +1.3e-7 |
| q1 | R_glob, R_solve | +1.6e-8, +1.9e-8 |

---

## 4. Per-a table under the certified Ĝ (T77)

`src/writer_inputs.py` → `wi_per_a_certified.csv`, `wi_crossing_runs.csv`.

- `R = |w₂|·Ĝ_cert/2` exactly. R is linear in Ĝ, and switch points and crossings are recorded in
  `|w₂|`.
- A **crossing** is the first training step whose oriented dense-grid gap is > 0.
- Crossing data come from `phase2b_checkpoints.csv`: Adam lr 1e-2, 200 points per class, 40 seeds
  per cell, budgets 2k / 4k / 8k / 16k / 32k.

| a | Ĝ | R_glob | R_solve | crossing R, median of 5 budget medians | pooled per-run median [95% CI] | crossing \|w₂\| | crossings / runs | offset over R_glob |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1.30 | 0.0861018 | 0.2153 | 0.3078 | 0.2338 | 0.2341 [0.2321, 0.2386] | 5.432 | 145 / 200 | +8.6% |
| 1.35 | 0.1069136 | 0.2165 | 0.3074 | 0.2378 | 0.2381 [0.2363, 0.2427] | 4.448 | 154 / 200 | +9.8% |
| 1.40 | 0.1287719 | 0.2189 | 0.3058 | 0.2421 | 0.2421 [0.2404, 0.2468] | 3.761 | 162 / 200 | +10.6% |
| 1.45 | 0.1515453 | 0.2197 | 0.3069 | 0.2460 | 0.2460 [0.2448, 0.2518] | 3.246 | 170 / 200 | +11.9% |
| 1.50 | 0.1751262 | 0.2233 | 0.3065 | 0.2498 | 0.2498 [0.2488, 0.2562] | 2.853 | 172 / 200 | +11.9% |
| 1.60 | 0.2243602 | 0.2300 | 0.3085 | 0.2580 | 0.2580 [0.2575, 0.2654] | 2.299 | 171 / 200 | +12.2% |

- **Crossings per budget** are the same at every a from 8k upward: 36–37 at 8k, 37 at 16k and
  38 at 32k, all out of 40. At 2k and 4k:

| a | 2k | 4k |
|---:|---:|---:|
| 1.30 | 4 | 30 |
| 1.35 | 10 | 33 |
| 1.40 | 17 | 34 |
| 1.45 | 24 | 34 |
| 1.50 | 26 | 34 |
| 1.60 | 26 | 33 |

- **Spread across the six a**: **CV(R) = 0.0324**, CV(|w₂|) = 0.2821. The draft's 0.0152 / 0.1508
  used only three values of a and the old Ĝ.
- **At a = 1.30: R_glob = 0.2153 and the certified median crossing is 0.2338.** The draft's
  "0.2332" used the old Ĝ.
- **Claim 13's "crossing step 1,860 to 6,839"** is a = 1.30 at budget 8,000 (36 runs). Across all
  five budgets the range is 1,860 to 27,778 (`wi_crossing_steps.csv`).
- **Interval**: 4,000-sample bootstrap of the pooled per-run median. The table's headline
  statistic is the median of the five per-budget medians, which is the registered unit.
- `R_glob` in this table comes from the coarse `blockB_switches.csv` (0.05 grid). The 0.01-grid
  refinement is in `blockB_fine_windows.csv` and is regenerated exactly by its producer (§15).

---

## 5. Branch tracking (T80, T67)

- **Distance to the conditional branch (T80)**:
  - For each training checkpoint, warm-start the Block B minimisation from the checkpoint's
    `(w₁, b₁, b₂)` at its own `|w₂|` (Adam 3e-2 → 5e-3, 3,000 steps, on the run's own 400
    training points).
  - The distance is `‖θ_ckpt − θ_min‖₂ / ‖θ_min‖₂` over `(w₁, b₁, b₂)` (`src/blockC_adiabatic.py:40-47`).
  - Data: 30 runs at a ∈ {1.30, 1.40}, 408 checkpoints.

| | n | Q1 | median | Q3 | max |
|---|---:|---:|---:|---:|---:|
| before crossing | 197 | 0.147 | **0.834** | 1.068 | 9.18 |
| after crossing | 211 | 0.0010 | **0.0017** | 0.0033 | 0.0082 |

- The registered criterion (median < 0.2) is met overall (0.0065) and after crossing, **not
  before**. Training follows the branch once placement exists.
- **Distance to constructed solutions (T67, corrected)** is a different quantity: the Euclidean distance
  in raw `(w₁, b₁, w₂, b₂)` to the nearest of 2,034–2,768 verified constructed `|w₂| = 1` solutions.
  - The logged checkpoints start **after one Adam step**. The true initialisation is regenerated with
    `torch.manual_seed(seed)` and `U(−1,1)⁴`, and reproduces step 1 exactly in 400 of 400 runs
    (`src/discrepancies.py pathwise` → `discrepancy_pathwise_init.csv`).
  - Median distance at initialisation **5.2259**; minimum along the path (step 0 included) **4.8755**;
    at the end **49.7**.
  - **The minimum is at initialisation in 114 of 400 runs.** The median approach is 0.84%. 112 runs
    get more than 5% closer, 41 more than 10%, and the largest approach is **23.4%**.
  - **Print**: "trajectories end about ten times further from the constructed solutions than they
    start (median 49.7 against 5.2); the closest approach is a median 0.8% nearer than the start and
    at most 23% nearer."
  - Withdrawn: "at most 0.8% closer", "no point comes meaningfully closer", and "minimum at step 0
    in 0 of 400", which was a vacuous check because step 0 was never logged.

---

## 6. H1 and H2 (T70) — the math note is attached as Appendix A

- **H1 (localisation)**:
  - Claim: Ĝ(a) is attained with limit coordinates in `C = {1 ≤ |u| ≤ 2.2, |v| ≤ 2}`.
  - Evidence: every certified argmax for a ∈ [1.02, 1.6] lies in C, at **|u| = 1.341–1.594,
    |v| = 1.021–1.196**.
  - The global branch and bound agrees: every argmax has `w₁ ≤ a/1.4`.
- **H2 (nondegeneracy)**:
  - Claim: at `R_glob^∞` the minimiser of `L_∞` is unique on its branch with a positive-definite
    Hessian, and the gap crosses zero transversally.
  - Evidence at `W = 2R/K = 0.690`: minimum Hessian eigenvalue **0.1470**; `d(gap)/dW = 1.656`;
    the minimiser moves ≤ **0.0036** per 0.005 step in W across the crossing.
- Both are verified numerically, not proved. H2 was checked at `R_glob^∞` only.
- The rates the proposition predicts match the measured ones: log-log slope 0.9926 for κ, 0.9472
  for `R_glob`.

---

## 7. Phase 1 population (T57)

`wi_phase1_reconciliation.csv`: 12 values of a × 200 seeds × 2 precisions, Adam lr 1e-2, 2,000
steps.

| precision | runs | solved | failed | placement | bias | disagreements with `solves()` |
|---|---:|---:|---:|---:|---:|---:|
| float32 | 2,400 | 430 | 1,970 | **1,664** | **306** | 0 |
| float64 | 2,400 | 440 | 1,960 | 1,633 | 327 | 0 |
| both | **4,800** | **870** | 3,930 | 3,297 | 633 | 0 |

- The draft's "4,800 runs, 870 solved, 1,664 + 306 failures" takes the first two numbers from the
  pooled population and the last two from float32 only.
- Either use float32 throughout (2,400 = 430 + 1,664 + 306) or both precisions throughout
  (4,800 = 870 + 3,297 + 633).
- Region-wide certification of solved runs is pooled: 866 of 870 at h = 4e-4 and the last 4 at
  h = 4e-5.

---

## 8. E-2, re-scored under its registered rule (T59, T78)

**Registered rule, verbatim** (`results/blockE_redesign.md`, first added in commit `4abecfa`,
2026-09-21 22:28:57 −0400; line 63 and lines 72–74):

> | `hold_high` | run to placement, then scale to `1.15 R_glob` and hold | **placement KEPT**, >= 9/10 |

> **Falsifier for the primary pair**: if `hold_low` keeps placement, or `hold_high`
> loses it, in more than 1 of 10 seeds, the threshold does not govern stability and
> Block E is reported as negative.

**Measured** (`wi_e2_rescore.csv`):

- hold_high kept placement in **33 of 37** runs that reached the intervention (0.892). It lost
  placement in 4 of 37 (10.8%): seeds 9, 25, 36 and 38, each 50 steps after intervening.
- hold_low kept 0 of 37.

**Verdict**:

- **E-2 FAILS** (0.892 < 0.9).
- **The falsifier fires** (4/37 > 1 of 10), so as registered Block E's primary result is
  **negative**, narrowly.
- The earlier PASS rested on the Fisher contrast (p = 1.2e-16), which the rule does not use.
- **For the paper**: the contrast 0/37 against 33/37 may be reported as a measurement. It must not
  be called a passed registered test, and the claim that the threshold "governs stability,
  causally" must be dropped or marked as not supported by the registration.
- Fig 10(b) shows the 9/10 level.

---

## 9. Registration census (T78)

`results/registration_census.csv` has one row per registered prediction. It is built by
`src/registration_census.py`; the tally is in `registration_tally.csv`.

**Headline — print this:** of the **151 registered predictions scored by their registered rules,
66 passed, 47 failed, 8 partially passed and 30 were not resolved.**

**Listed separately — 13 verdicts assigned post hoc in the census.** These predictions were
registered but never scored against their rule, or the rule leaves room for judgement. The census
scored them against the registered criterion after the fact. They are **3 PASS, 6 FAIL,
4 PARTIAL** and are not in the headline:

| row | block | registration | verdict | why it is a judgement call |
|---|---|---|---|---|
| E-cold_low | E | `blockF_lag_prediction.md` | PASS | freeze-low arm registered before the pilot; blockE_redesign scores only the prospective pair |
| loc-2 | localization 08-22 | `localization_prediction.md` | FAIL | fails as registered; borne out only in the corrected \|lk\| form (arguably PARTIAL) |
| thr-P5 | threshold 08-22 | `threshold_prediction.md` | PASS | families agree on a common derivative value, which the registration said would count against (arguably FAIL) |
| amp-2a-match | early census 08-23 | `amplification_prediction.md` | FAIL | stated reason retracted in T28; never re-scored |
| gelu-up | early census 08-23 | `gelu_scale_prediction.md` | FAIL | 17.0% sits on the edge of the stated 3–17% band |
| collapse-link | collapse 09-12 | `collapse_prediction.md` | FAIL | registered biconditional violated once the corrected Item 2 passes; never re-scored |
| metric-1a | metric artifact 09-12 | `metric_artifact_prediction.md` | PARTIAL | non-directional: outcome classes without a predicted class; intermediate mapped to PARTIAL |
| metric-2 | metric artifact 09-12 | `metric_artifact_prediction.md` | PARTIAL | non-directional criterion; family B qualifies and fails |
| rc-1d | R-collapse 09-12 | `r_collapse_prediction.md` | PARTIAL | only the parametric criterion was registered (strictly FAIL); superseded by the third-optimizer test |
| B-spin | B | `blockB_prediction.md` | FAIL | scored pass later withdrawn; the registered falsifier fired (alternative: UNRESOLVED) |
| B-solve | B | `blockB_prediction.md` | FAIL | scored in the census against the registered 15% tolerance; every a lies outside |
| B-CV | B | `blockB_prediction.md` | PASS | scored in the census against the registered CV ≤ 0.15 |
| S-3 | scaling limit | `scaling_limit_prediction.md` | PARTIAL | passes for R_glob, fails for R_solve (strictly FAIL for R_solve) |

| scope | PASS | FAIL | PARTIAL | UNRESOLVED | total |
|---|---:|---:|---:|---:|---:|
| **scored by registered rules (headline)** | **66** | **47** | **8** | **30** | **151** |
| assigned post hoc (listed separately) | 3 | 6 | 4 | 0 | 13 |
| all | 69 | 53 | 12 | 30 | 164 |

**Corrections within the 151**, each scored against its own registration:

- **E-2**: PASS → **FAIL** (33/37 < 9/10). **Block E is negative per its registration**
  (the falsifier fired; §8).
- **P-SGD**: PASS → PARTIAL. Its registered rule scores it DIRECTIONAL-ONLY (outside the band by
  0.0356).
- **P-null**: PASS → UNRESOLVED. The test did not execute.
- **P-threshold**: its 2026-09-13 upgrade to PASS had never reached the totals.

The earlier "47 + 17 = 64" covered only the 2026-09-12 summary plus the v2 blocks. The per-block
breakdown of all 164 is in the CSV's `block` column.

---

## 10. Block F (T63)

- **Growth rate**: for each run, the OLS slope of `|w₂|` against step, over trace points within
  ±400 steps of the first gap > 0 (trace every 25 steps). The per-optimiser rate is the median
  over runs that crossed (`blockF_optimisers.py:69-74`).
- **Settings**: a = 1.25, 60 seeds, 20,000 steps, float64; Adam and AdamW lr 1e-2, SGD lr 0.3.

| optimiser | crossed | median rate |
|---|---:|---:|
| Adam | 57 | 0.002471 |
| AdamW | 60 | 0.001908 |
| SGD | 43 | 0.000631 |

- **The spread is 3.92×** (Adam/SGD, measured medians).
- The **~2.8×** (2.76×) is a different quantity: the ratio of the **pre-measurement** rates
  written into the registration (0.00207 / 0.00075, `blockF_lag_prediction.md:29-33`). Use 3.92×
  for the measurement.
- **SGD miss of 6.19 percentage points**:
  - Registered: predicted offset = Adam's offset × (rate_SGD / rate_Adam), tolerance ±4 pp
    (`blockF_lag_prediction.md:44-52`).
  - The offset is median crossing R / R_glob(1.25) − 1.
  - SGD measured 8.647%; predicted 9.627% × 0.255 = 2.458%. The miss is **6.19 pp**.
  - With the registered rates instead, the miss is 5.16 pp; it fails either way.
  - These offsets use the restricted Ĝ. Offsets are ratios at a fixed a, so a common Ĝ cancels
    exactly.

---

## 11. Exclusion table (T41)

- **"Found" configurations**: the first solving seed of 0–39 under **Adam, lr 1e-2, full batch,
  2,000 steps** (`src/exclusion_table.py:80-81`). The seeds are a = 1.45 → 2 and
  a = 1.5, 2.0, 3.0 → 1.
- **The 41 images** is the number of images on each string-method path for the minimum-energy
  barrier.
  - Settings: 400 steps, lr 5e-3, endpoints pinned (`src/barrier.py:25-28`).
  - 5 initialisations per row × 12 rows = 60 paths.
  - **The barrier is 0.000 on every path.** Whether the string converged doesn't matter: a path with
    zero barrier at every image bounds the barrier at those images.
  - It is not a CIFAR or MNIST count. Now in the ledger (T41) and checked by the verifier.
  - `barrier.csv` is a different experiment (paths from initialisation to a solution; maximum MEP
    barrier 0.0957). It covers **13** values of a: the script tries 15, and a = 1.20 and 1.60 have
    no solving endpoint (`discrepancy_mep.csv`).
- **Margin (population statement)**: float32 phase-1 solvers (the same training as "found"; margin =
  the exclusion table's `margin_of`) against the construction's margin (`src/discrepancies.py margins`
  → `discrepancy_margin.csv`):

| a | solvers | min margin | median | constructed margin | solvers below it |
|---:|---:|---:|---:|---:|---:|
| 1.45 | 50 | 0.0033 | 0.0479 | 0.0325 | 18 |
| 1.50 | 81 | 0.0017 | 0.1118 | 0.0384 | 17 |
| 2.00 | 143 | 0.1502 | 0.7738 | 0.1149 | 0 |
| 3.00 | 139 | 0.1209 | 1.6120 | 0.3329 | 1 |

  - **Print**: "at a = 1.5, 17 of 81 found solutions (21%) have a smaller margin than the constructed
    solution (0.0384; minimum 0.0017)".
  - The table's single "found" row at a = 1.5 has margin 0.0771, which is larger. Attach "smaller" to
    the population only.

---

## 12. ℝ⁵ and MNIST configurations

**ℝ⁵ (T72, T75)**: see §1.

| | k = 10 | k = 1 |
|---|---|---|
| optimiser | full-batch Adam, lr 1e-3 | full-batch Adam, lr 1e-3 |
| steps per budget | 1k / 4k / 16k / 64k, checkpoints of one 64k run | same |
| training points | 10,000 (1,000 per copy) | 10,000 |
| held-out | 20,000 uniform | 20,000 uniform + 20,000 stratified (+ 200,000 uniform when uniform errors = 0) |
| seeds | 20 per cell | stage 1: 100–119; pilot 0–9 |
| depth | 5 | 5 (chosen by the capacity pilot) |

**MNIST (T48)**, `src/mnist_fold.py`:

- **Architecture**: 784 → 256 → [w] → 128 → 10. ReLU everywhere except `f_a` at the bottleneck.
- **Training**: Adam lr 1e-3, **batch 128**, minibatch with the last partial batch dropped,
  softmax cross-entropy (mean).
- **Budgets**: counted in **steps**. Pilots used **2,000 steps** (about 4.3 epochs at 468
  steps/epoch); the long-budget pilot used **8,000 steps** (about 17.1 epochs).
- **Data**: train 60,000 (normalised (x − 0.1307)/0.3081); test 10,000. Training accuracy is
  measured on the first 10,000.
- **Pilot counts**: 90 (widths 4/6/8 × 6 a × 5 seeds) + 75 (widths 1/2/3 × 5 a × 5 seeds; no
  a = 0.5) + 45 long-budget = 210 runs. **All 210 are regenerated exactly** by the committed driver
  `src/mnist_fold_driver.py` (maximum difference 1e-16; `mnist_fold_driver_check.csv`). T48's
  "6 widths × 6 a × 5 seeds" is corrected.
- **Other MNIST configurations**:
  - T34 bottleneck sweep: 784 → 128 → w → 128 → 10, 3 epochs, batch 256, lr 1e-3.
  - T35 control search: width 256, depths 4/8/12, 15 epochs, batch 256, lr 1e-3.

---

## 13. Figures

**Manifest.** `paper/figures/*` is produced by `paper/make_figures.py`, and provenance is written
to `paper/figures/PROVENANCE.md` and `manifest.csv` on every run. Every figure uses the certified Ĝ
where R appears.

| file | one-line description | ledger |
|---|---|---|
| `fig1_setting.pdf` | `f_a` for a = 0.9, 1.05, 2.0 with fold depth marked; two trained networks (a = 1.5 solves, a = 1.0 does not) | T37, T56 |
| `fig2_exclusions.pdf` | sharpness, distance from initialisation, logit margin and MEP barrier against a, found versus unreached | T41 |
| `fig3_r_collapse.pdf` | P(solve) against R (pooled); three optimisers at a = 1.25; AUC of R versus \|w₂\| per family | T52, T55 |
| `fig4_metric_check.pdf` | **regenerated (certified Ĝ)**: binary solve rate and mean eval errors against R, with 10–90% bands computed from the data | T53, T79 |
| `fig5_budget_law.pdf` | onset ε against budget (Adam); fitted α against window budget | T43, T52 (3c: α drift) |
| `fig6_four_family.pdf` | onset exponent against 1/β, four families | T44 |
| `fig7_family_b.pdf` | family A versus B onsets; median \|w₂\| and w₂·Ĝ_norm/2 against Ĝ_norm | T62 |
| `fig8_link.pdf` | GELU separation rate against initialisation scale, with the monotonic floor | T29, T1 |
| `fig9_expressivity.pdf` | ℝ⁵, k = 1: fraction perfect against a per budget, threshold a = 1, ReLU/GELU | T75 |
| `fig10_mechanism.pdf` | **new, main-text mechanism figure**: (a) conditional branch gap against R at a = 1.30 with R_glob, R_solve, 10 training trajectories and 145 crossings; (b) retention of placement under hold_low, hold_high and noise floor, with the registered 9/10 level | T59, T60, T77, T78 |

Each PDF has a PNG preview beside it. The legacy figures in `results/figures/` are not for the
paper:

| file | description | producer |
|---|---|---|
| `metric_check.png` | older Fig 4 (**regenerated with the certified Ĝ**) | `src/figures.py` |
| `fig1_four_family.png`, `beta_law.png` | onset exponent against 1/β | `src/figures.py`, `src/beta_law_figure.py` |
| `fig2_budget_law.png` | ε_onset against budget | `src/figures.py` |
| `fig3_zero_to_hundred.png` | solve rate against budget, a = 1.25 | `src/figures.py` |
| `fig4_margin_shift.png` | CIFAR-10 test-error advantage against epochs | `src/figures.py` |
| `fig5_cross_optimizer.png` | Adam versus SGD onset exponents (hardcoded values) | `src/figures.py` |
| `saturation_by_output_distance.png`, `vector_collisions_by_output_distance.png`, `training_dynamics.png` | early census diagnostics | `src/report.py` |
| `followup_accuracy_collision_dynamics.png` | accuracy, collision and saturation against step | `src/followup.py` |
| `projections/*.png` (6) | 3D projections of the Hopf-link input and width-8 final layers | `figs_tmp.py` (repo root, temporary) → `src/projection_figures.py` |

**Fig 10 data** (`src/mechanism_figure_data.py`):

- **The branch** is the frozen Block B minimisation at each `|w₂|` from 2.0 to 10.0 in steps of
  0.25 (`mechanism_branch_a130.csv`). Its gap first turns positive at `|w₂|` = 5.00
  (R = 0.2153 = R_glob), and its minimiser first solves at 7.25 (Block B's finer grid: 7.15).
- `phase2b_conditional.csv` was **not** used: it comes from the under-converged 600/900-step runs
  (`paper_claims_delta.md` R2).
- **Trajectories**: 10 runs at budget 32,000, every 25th step (`mechanism_trajectories_a130.csv`).

**Old Ĝ**: Fig 4 and `results/figures/metric_check.png` were the only figures still on the old Ĝ;
both are regenerated. The binning sensitivity of Fig 4's bands (T79):

| bin width | binary 10–90% | continuous 10–90% | disjoint? |
|---:|---|---|---|
| 0.020 | 0.332–0.455 | 0.056–0.306 | yes |
| 0.025 (Fig 4) | 0.330–0.429 | 0.055–0.342 | **no (overlap 0.012)** |
| 0.030 | 0.327–0.417 | 0.058–0.304 | yes |
| 0.040 | 0.324–0.426 | 0.060–0.307 | yes |
| 0.050 | 0.326–0.426 | 0.062–0.316 | yes |

- The **76%** (76.2%: 117.1 of 153.6 errors removed across 1,815 runs with R ≤ 0.30, none of
  which solve) holds whatever the binning.
- **Use that sentence and drop "do not even overlap".**
- The registered verdict (width ratio 2.90, "intermediate") is unchanged.

---

## 14. Bibliography

**`paper/references.bib`**: 32 entries. Every entry was checked on 2026-09-23 against a fetched
page, and that page's URL is in a `% verified:` comment above the entry. The verification log is
`paper/references_verification.csv`, one row per key with every discrepancy noted.

- The list is the writer's candidates plus the 8 sources already read from PDFs in
  `related_work.md`. Duplicates are merged: Hanin–Sellke, Guss–Salakhutdinov, Ahn–Zhang–Sra and
  Soudry et al.
- **Venue versions are used where verifiable**:
  - JMLR: Soudry et al. 2018, JMLR 19(70):1–57; Naitzat et al. 2020, JMLR 21(184):1–40.
  - ICLR: Johnson 2019; Lyu & Li 2020; Park et al. 2021; Cohen et al. 2021; Cai 2023;
    Kim et al. 2024.
  - NeurIPS: Montúfar et al. 2014; Chen et al. 2018; Garipov et al. 2018; Chizat et al. 2019;
    Dupont et al. 2019; Sitzmann et al. 2020 (pp. 7462–7473); Ziyin et al. 2020 (pp. 1583–1594).
  - COLT: Telgarsky 2016 (PMLR 49:1517–1539); Eldan & Shamir 2016 (PMLR 49:907–940); Kidger &
    Lyons 2020 (PMLR 125:2306–2327).
  - ICML: Raghu et al. 2017 (PMLR 70:2847–2854); Shalev-Shwartz et al. 2017 (PMLR 70:3067–3075);
    Draxler et al. 2018 (PMLR 80:1309–1318); Ahn et al. 2022 (PMLR 162:247–257); Li et al. 2023
    (PMLR 202:19460–19470); Jin et al. 2023 (PMLR 202:15200–15238).
  - BMVC: Misra 2020, via the BMVA archive.
- **arXiv only** (no venue version): Hanin & Sellke 2017, Hendrycks & Gimpel 2016, Ramachandran et
  al. 2017, Guss & Salakhutdinov 2018, Jacot et al. 2021, Kaplan et al. 2020.
- **Failed verification: none.**
- **Partial: `ren2026low`.** It is cited as arXiv 2606.31856; PMLR volume 306 does not exist yet.
- **Corrections to the candidate list**:
  - **Jacot et al.**: the full title is "Saddle-to-Saddle Dynamics in Deep Linear Networks: Small
    Initialization Training, Symmetry, and Sparsity".
  - **Draxler et al.**: pages are 1309–1318 per PMLR; the arXiv journal-ref's 1308–1317 is wrong.
  - **Years change with the venue version**: Soudry 2017 → 2018, Lyu & Li 2019 → 2020, Chizat 2018 →
    2019, Eldan & Shamir 2015 → 2016, Misra 2019 → 2020.
  - **Author forms as printed by the venue**: "Li'Ang Li"; "Simon Shaolei Du"; "David K. Duvenaud";
    "Andrew G. Wilson"; "Lénaïc Chizat".
  - **Kaplan et al.**: ten authors (Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford,
    Wu, Amodei).
- **Verification route**: openreview.net and dblp.org served bot challenges to automated fetches,
  so the ICLR entries were verified on iclr.cc / proceedings.iclr.cc. I re-checked three entries by
  hand: Draxler, Jin and Jacot.
- **Anonymity**: `references.bib` names the Ren–Lim authors, as any citation does. It is not part of
  the supplementary.

---

## 15. Discrepancies found while assembling this — all resolved

| discrepancy | resolution | producer / where |
|---|---|---|
| T67 "distance at initialisation" was step 1; "minimum at step 0 in 0 of 400" was vacuous; "at most 0.8% closer" | **Recomputed**: true initialisation regenerated and checked; minimum at initialisation in 114/400; largest approach 23.4%. T67 and claim 34 restated; verifier check replaced | `src/discrepancies.py pathwise`, `discrepancy_pathwise_init.csv` |
| Margin: "6 of 24, min 0.0054" versus "5 of 20, min 0.0038"; neither had a producer | **Recomputed**: 17 of 81 at a = 1.5 (min 0.0017); 18 of 50 at a = 1.45. Replaced in VERIFIED_NUMBERS, results_draft, exclusion_table.md and T37; `margin_gap_results.md` marked superseded | `src/discrepancies.py margins`, `discrepancy_margin.csv` |
| Preconditioned distance: "survives" versus "does not survive"; no producer | **Removed** from VERIFIED_NUMBERS and results_draft; marked withdrawn in exclusion_table.md and interpretive_audit.md | — |
| MEP "converged" versus "not converged"; "13 values of a" versus 15 | **Resolved**: not converged, and the zero-barrier result doesn't need convergence; `barrier.csv` has 13 values of a (the script tries 15; 1.20 and 1.60 have no endpoint). barrier.py docstring and interpretive_audit.md corrected | `src/discrepancies.py mep`, `discrepancy_mep.csv` |
| "SGD" in the criticality.py docstring and exclusion_table.md | **Corrected** to Adam (T1 was corrected earlier) | — |
| fold1d registered "10,000 fresh points"; code uses a 400-point sample plus dense grids | **Recorded as a protocol deviation** in fold1d_results.md and T30; no number depends on it | — |
| AdamW weight decay: explicit 0.01 (Block C) versus default (Block F) | **Same value**: the verifier checks that PyTorch's AdamW default is 0.01 | `verify_ledger.py` |
| MNIST pilots: "6 widths × 6 a × 5 seeds" = 180 versus 165 rows; no driver committed | **Driver committed**; it regenerates all 210 pilot runs exactly; T48 counts corrected | `src/mnist_fold_driver.py`, `mnist_fold_driver_check.csv` |
| Protocol stratum early-stops on the evaluation set it is scored on | **Disclosed** in VERIFIED_NUMBERS §3 (the separation count is 0 regardless) | — |
| a = 1.60 row of `blockB_fine_windows.csv` looks hand-entered | **Regenerated** by its producer: \|w₂\| values match the committed row to 3e-14; only the R columns had been rounded to five decimals (< 1e-6). The file is replaced with the exact producer output | `session_artifacts.blockB_fine_windows` |
| Phase 1 registration cited as `7db80cf` | **Corrected** to `ab7556d` | — |
| Metric check: no producer; binary 90% crossing 0.452 not reproducible; slopes −1,649 / −136 | **Script committed**: reproduces 0.055, 0.307, 76%, −1,649 and −136 exactly under the old Ĝ; certified values in VERIFIED_NUMBERS and T53; "disjoint" withdrawn | `src/metric_check.py`, `metric_check.csv` |
| Claim 13: "crossing step 1,860 to 6,839" with no stated population; claim 14 on three values of a | **Corrected**: that range is a = 1.30 at budget 8,000 (1,860–27,778 across all budgets); claim 14 restated on six values of a with the certified Ĝ | `writer_inputs.py`, `wi_crossing_steps.csv` |
| `ghat_unification.md` "certified" column held restricted values | **Corrected** | — |

---

## Appendix A — the math note (`results/scaling_proposition.md`, verbatim)

# Proposition: the scaling reduction of the 1D fold task

Checks: `src/scaling_proposition.py` → `results/scaling_proposition_checks.csv`
(39 checks, all pass). Check IDs in brackets.

## Setting

`f_a(t) = t + a sin t`, `a = 1 + ε`, `ε > 0`. Width-1 network
`N(x) = w₂ f_a(w₁x + b₁) + b₂` on windows `I = [−0.8, 0.8]` (label 0),
`O = ±[1.2, 2.0]` (label 1); `|x| ≤ 2` on `I ∪ O`.
Limit coordinates: `w₁x + b₁ = π + √ε σ(x)`, `σ(x) = ux + v`.
Compact placement set `C = {1 ≤ |u| ≤ 2.2, |v| ≤ 2}`, so `|σ| ≤ S = 6.4` on `I ∪ O`.
`h(σ) = −σ + σ³/6`, `r(σ) = σ³/6 − σ⁵/120`.
Class gap `G(φ) = max(min_O φ − max_I φ, min_I φ − max_O φ)` for a function `φ`
of `x`; `Ĝ(a)` its supremum over placements; `K = sup_{(u,v)} G(h∘σ) = 0.579454926`
(certified, T64). `R = |w₂| Ĝ(a)/2`.

## Hypotheses

- **H1 (localisation).** For `ε` small, `Ĝ(a)` is attained at a placement whose
  limit coordinates lie in `C`, up to the symmetries `t ↦ t + 2π`
  (`f_a(t + 2π) = f_a(t) + 2π`) and `σ ↦ −σ`, `x ↦ −x`.
  *Evidence:* every certified argmax for `a ∈ [1.02, 1.6]` lies in `C`, at
  `|u| = 1.341–1.594`, `|v| = 1.021–1.196` [H1(·)].
- **H2 (nondegeneracy).** At `R_glob^∞` the minimiser of `L_∞` below is unique on its
  branch, with positive-definite Hessian, and the class gap of the minimiser
  crosses zero with nonzero derivative in `R`.
  *Evidence:* at `W = 2R/K = 0.690`: min Hessian eigenvalue `0.1470` [6b];
  `d(gap)/dW = 1.656` [6a]; minimiser moves ≤ `0.0036` per 0.005 step in `W`
  across the crossing [6c].

## Statement

Under H1 and H2, as `ε → 0`:

1. `f_a(π + s) = π − εs + (1+ε)s³/6 − (1+ε)s⁵/120 + O(s⁷)`.
2. `f_a(π + √ε σ) = π + ε^{3/2} h(σ) + ε^{5/2} r(σ) + ρ_ε(σ)`, with
   `|ρ_ε(σ)| ≤ ε^{7/2}(|σ|⁵/120 + (1+ε)|σ|⁷/5040)`.
3. With `b₂' = b₂ + w₂π`, the logit is `b₂' + w₂ε^{3/2}h(σ) + w₂ε^{5/2}(r(σ) + ε^{-5/2}ρ_ε)`.
4. `Ĝ(a) = K ε^{3/2}(1 + δ_ε)`, `|δ_ε| = O(ε)`; hence `w₂ε^{3/2} = (2R/K)/(1 + δ_ε)`.
5. Let `L_a(u, v, b₂'; R)` be the population logistic loss at output weight `w₂`
   and `L_∞(u, v, b₂'; R) = E ℓ(b₂' + (2R/K) h(σ(x)), y)`. Then, uniformly on
   `C × [−B, B]`,
   `|L_a − L_∞| ≤ (2R/K)·(ε M' + |δ_ε| H)/(1 + δ_ε) = O(Rε)`,
   with `M' = sup_{|σ|≤S} |r| + ε(S⁵/120 + S⁷/5040)` and `H = S + S³/6`.
   `L_∞` depends on the output weight only through `R`, and on the task only
   through the windows.
6. Each switch point of the conditional landscape (`R_glob`, `R_solve`), expressed
   in `R`, converges to the corresponding switch point of `L_∞` at rate `O(ε)`.

## Proof

**1.** `f_a(π + s) = π + s + (1+ε) sin(π + s) = π + s − (1+ε) sin s` [1b]; insert
the sine series [1].

**2.** Substitute `s = √ε σ`. The `ε^{3/2}` coefficient is `h` [2a], the `ε^{5/2}`
coefficient is `r` [2b], and no lower power of `√ε` occurs [2c]. The remainder is
`−ε^{7/2}σ⁵/120 + (1+ε)R₇(s)` with `|R₇(s)| ≤ |s|⁷/5040` (alternating series),
which gives the bound. It holds with the true remainder at 3.9–6.6% of the bound
for `ε ∈ {0.3, 0.1, 0.02, 0.001}` in 60-digit arithmetic [R(·)].

**3.** Algebra from 2.

**4.** For fixed `(u, v)`, `G` is 1-Lipschitz in the sup-norm of `φ` on `I ∪ O`
(each of its terms is a min or max of `φ`), so by 2,
`|G(f_a∘σ) − ε^{3/2}G(h∘σ)| ≤ 2ε^{5/2}M_ε`, `M_ε = sup_{|σ|≤S}|r| + ε(S⁵/120 + (1+ε)S⁷/5040)`,
uniformly on `C`. Taking sups over `C` (H1 for the left side) gives
`|Ĝ(a)/ε^{3/2} − K| ≤ 2εM_ε`. Measured sup deviation: `1.236, 0.4765, 0.1008,
0.02548` at `ε = 0.3, 0.1, 0.02, 0.005`, ≈ `5ε`, against the bound ≈ `94ε` [4(·)].
The rate is `O(ε)`; the constant in the bound is conservative by about 20×.

**5.** `ℓ(z, y) = log(1 + eᶻ) − yz` has `|∂_z ℓ| = |sigmoid(z) − y| ≤ 1`, so
`|L_a − L_∞| ≤ E|z_a − z_∞|`. By 3–4,
`z_a − z_∞ = (2R/K)[(1+δ_ε)^{-1} ε(r + ε^{-5/2}ρ_ε) − δ_ε(1+δ_ε)^{-1} h]`, bounded by
the stated expression. Verified on 1,500 random points of `C × [−3, 3]` at
`ε ∈ {0.1, 0.02, 0.005}`, `R ∈ {0.2, 0.3}`; measured sup `|L_a − L_∞|` scales as
`ε` [5(·)]. `L_∞` involves `w₂` only through `2R/K`.

**6.** Steps 2–5 hold also for derivatives in `(u, v, b₂')` (the remainder is
smooth in `σ`, and `σ` is affine in `(u, v)`), so `∇L_a → ∇L_∞` at rate `O(Rε)`
on `C`. By H2 and the implicit function theorem, the minimiser of `L_a` on its
branch is within `O(ε)` of that of `L_∞`, uniformly for `R` near `R_glob^∞`. The
class gap of the minimiser is then within `O(ε)` of its limit, and because the
limit gap crosses zero transversally (H2), its zero `R_glob(a)` is within `O(ε)`
of `R_glob^∞`. The same argument applies to `R_solve` wherever its defining
crossing is transversal. ∎

## Connection to measured rates

- `(κ(a) − κ₀)/κ₀` against `ε` for `ε ≤ 0.1`: log-log slope **0.9926** [M1].
- `R_glob(a)/R_glob^∞ − 1` over the six measured `a`: log-log slope **0.9472** [M2].

Both are the `O(ε)` rate of 4 and 6.

## The q-families

For `|x| ≤ 1` the constructed families are `f(x) = (1−a)x + a sgn(x)|x|^{q+1}/(q+1)`.
With `a = 1 + ε`, `x = ε^{1/q}σ`:

    f = ε^{1+1/q} [ h_q(σ) + ε g_q(σ) ],   h_q = −σ + sgn(σ)|σ|^{q+1}/(q+1),   g_q = sgn(σ)|σ|^{q+1}/(q+1)

**exactly**, with no higher terms, while `ε^{1/q}|σ| ≤ 1` [q=·]. So steps 2–6 hold
with `β = 1 + 1/q` in place of `3/2` and remainder exactly `ε g_q`. For `q = 2`,
`√2·h₂(τ/√2) = h(τ)` [q=2 vs sin]: the `q = 2` limit is the sin-family limit up to
scale, which is why their onsets differ by a constant factor (T44).

## Limits of the statement

- **Cross-family test (post-registration finding, `crossfamily_results.md`).**
  For the q-families the reduction holds only while `ε^{1/q}·max|σ| ≤ 1`, because
  they are polynomial only for `|t| ≤ 1`. The registered cross-family test ran
  at `ε ≥ 0.3`, where 25–34% of window points fall on the linear continuation.
  There, q2 matched family A's `R_glob` at all six `a` but **not** its `R_solve`
  (below A by 0.012–0.051). So at finite `ε` the reduction does **not** fix the
  solve threshold across families whose folds agree only near the fold.
  Statement 6 is claimed only inside the reduction's domain.

- **Cross-family follow-up, inside the domain (`crossfamily_followup_results.md`;
  registered after X1 failed).** At `a` = 1.01–1.04, with 0% of window points on
  q2's linear continuation, q2 matches family A on **both** `R_glob` and
  `R_solve` at all four `a` (Y1 passes). Inside its domain, the reduction therefore
  predicts both thresholds across equivalent families. The agreement is at grid
  resolution (`ΔR = 0.00215`), with both families within one step of the limit.
  A registered negative control (Z1, made after Y1) shows that this resolution is
  enough to discriminate: q1, whose limit is not a rescaling of `h`, sits 35–36 grid
  steps from A on `R_glob` and 5 on `R_solve` at the same `a`. So Y1's agreement
  reflects the reduction, not the grid. X1 stays failed as registered.

- H1 and H2 are verified numerically, not proved.
- Step 6 is conditional on H2 at each switch point used; H2 was checked at
  `R_glob^∞` only.
- The remainder constants are explicit but conservative; the rate, not the
  constant, is what the measurements confirm.
