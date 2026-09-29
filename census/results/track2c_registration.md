# Test 2C registration: the band task in R^d against each seed's own-sample R^d switch, prospectively, 2026-09-29

**Status: written before any registered run.** No registered seed has been searched or trained, and no crossing, gap or
placement of any 2C training trajectory has been evaluated.
- This text is committed first.
- It becomes the registration when the frozen inputs of §9 are committed with the SHA-256 manifest
  `results/track2c/registration.sha256`, which includes this file's hash (the registration commit).
- That commit is timestamped with OpenTimestamps (`results/track2c/registration_stamp.txt` and its `.ots` proof).

## 1. Design reference

- Approved design: `results/designs/2C_band_own_sample_design.md` (committed in `c3840fe`, wording `4b4739c`). The
  author approved it as designed on 2026-09-29. Nothing in it is changed here.
- **The design's conditional clause is resolved: Test 2A PASSED** (`results/track2a_writer_inputs.md`: A1 Spearman
  0.932, A2 fraction within 10% 0.773, A3 median ratio 1.058). So χ uses the 2A rule: **P is frozen at t_sw**, the first
  step at which |w₂| reaches the occupied branch's switch, which here is s_own,d. The rule point is Track A's (the last
  upward passage of 0.5·s_own,d).
- Where the page is silent, the convention of Track 3B (`results/band_rd_registration.md`) or Test 2A
  (`results/track2a_registration.md`) is used. Every such choice is listed in §7. None of them changes the meaning of a
  criterion.
- Code: `src/track2c.py`. It reuses Track 3B's data, model, gap, branch solver and training formulation
  (`src/band_rd.py`), Track A's rule point (`src/track_a.py`), the width-1 own-threshold search
  (`src/own_threshold.py`), the width-1 κ_k (`src/lag_law.py`) and the relaxation rate (`src/residual_timescale.py`).
- Tests: `tests/test_track2c.py`. They cover every decision rule on constructed PASS, FAIL and UNRESOLVED cases, the
  bootstrap rule, the rule point and t_sw, the own-sample switch search and its validation rules, and that the prediction
  step evaluates no gap.

## 2. What the test is about

- Track 3B's registered lag prediction failed in R^d (d = 2: median residual 0.15/0.19 against a predicted 0.024/0.049).
- A POST HOC decomposition attributed the excess to a static finite-sample offset. The own-sample R^d threshold sits
  12% (d = 2) and 40% (d = 4) above the x₁-sample own threshold, because the sample minimiser keeps a small noise
  weight. Measured from each run's own R^d branch switch, the lag followed the width-1 law (obs/pred 0.84–1.10).
- 2C tests that decomposition **prospectively**. Each seed's own-sample R^d switch is computed and frozen before
  training. The lag is predicted from pre-crossing information only.

## 3. Setting and seeds

- **Task and model: Track 3B unchanged.**
  - x₁ and y are the width-1 task, `fold1d.make_data(200, seed)`: 400 points.
  - x₂…x_d are i.i.d. U(−2, 2) from `numpy.random.default_rng([seed, 3])`.
  - d ∈ {2, 4} and a ∈ {1.30, 1.50}: four cells.
  - One unit: z = w₂·f_a(w·x + b₁) + b₂, f_a(t) = t + a sin t.
  - p = (w₁, b₁, w₂, b₂, w_noise) ~ U(−1, 1)^{d+3}, float32 then float64. It is drawn from a local torch Generator
    seeded with the seed, bit-identical to `band_rd.init_rd` (tested), so no global RNG state changes.
- **Training:**
  - free Adam, lr 0.01, full batch, BCE with logits, float64, all d + 3 parameters (`band_rd.train_single`'s
    formulation, reproduced bit for bit, tested);
  - budget **64,000 steps** (3B's).
- **Crossing (3B's rule, every step):** the first step t ≥ 1 with G > 0. G is the exact R^d gap on the continuous
  support (`band_rd.gap_rd`), oriented by w₂. A run with G > 0 at initialisation is placed at init and has no crossing.
  s_obs = |w₂| at the crossing.
- **Seeds 2,030,000–2,030,059** (60 seeds), the same 60 seeds in all four cells.
  - Checked 2026-09-29: none of the integers 2,030,000–2,030,099 appears, with or without thousands separators, in any
    text file of the repository (3,810 files scanned, all of `precision-topology/` except `.git` and `.venv`).
  - Non-registered seeds used before this registration: 889,000 (3B's disclosed pilot seed; timing only, no training)
    and 880,000–880,005 (3B's primary seeds; a search diagnostic on committed 3B data, no training; §8).

## 4. The prediction

**s_pred = s_own,d·(1 + κ_k·χ)**, per run.

**s_own,d: frozen per seed and cell before any training (`track2c.own_switch_rd`).**
- The placement switch of the seed's own-sample conditional minimiser in all hidden coordinates (w₁, b₁, b₂, w_noise)
  at w₂ = s, noise weights included.
- Placement is 3B's: G₊ > 0. Only w₂ > 0 is searched: w₂ → −w₂ with (w, b₁) → (−w, −b₁) leaves the loss and the
  placement unchanged.
- **Search at one scale (validated, not certified):**
  - 200 restarts of batched BFGS (`width2_conditional.bfgs_batch`, gradient tolerance 1e−8, at most 3,000
    iterations) on the loss with b₂ profiled exactly. Restarts that do not converge (gradient above 1e−6) or are not
    finite are not eligible.
  - Starts: w₁ ~ U(−W, W), b₁ ~ U(0, 2π), and w_noise ~ U(−cW, cW)^{d−1} with c = 1 for even restarts and 0.1 for odd
    ones. W is the width-1 bound `profiled_bnb.w_bound` of the seed's x₁ sample.
  - The 5 lowest distinct eligible candidates are polished by damped Newton in all hidden coordinates
    (`band_rd.branch_rd`). The retained minimiser is the lowest, with the constant predictor included (unplaced).
- **Bracket:** own_threshold's convention. Start at the seed's x₁-only own threshold. Step by 0.1 (down while placed,
  up while unplaced) until the status changes, then bisect to hi − lo ≤ 0.01. s_own,d is the midpoint.
- **Validation at both bracket ends:**
  - (i) restart ladder: a 200-restart and an 800-restart search on independent streams give the same retained loss
    within 1e−9 and the same status;
  - (ii) independent search: CMA-ES (`src/cmaes.py`, 20 starts in the full box, 300 generations) on (w₁, b₁, w_noise)
    with b₂ profiled. It must not be lower than the retained loss by more than 1e−9;
  - (iii) audit: no candidate of the other status within 1e−9 of the retained loss (the eligible 800-restart
    candidates within 1e−7, and the polished ones);
  - and the status must be unplaced at s_lo and placed at s_hi.
- Random streams: `numpy.random.default_rng([seed, d, 100a, 1, evaluation index])` for the bracket and
  `[seed, d, 100a, 2, end, j]` for validation.
- **A seed-cell whose s_own,d is not validated has no prediction.** Its run is trained and counted (§6, V3).

**x₁-only own threshold s_own,x1 (C3's comparator; frozen):** 3B's convention, `own_threshold.own_threshold` on the
seed's x₁ sample (the width-1 global conditional search; bracket width 0.01), the bracket midpoint.

**κ_k: the width-1 κ_k, frozen** (`kappa_k.csv`): `lag_law.kappa_k(H, θ*′, ∇G, P_median, k)` with the width-1
population landscape of `lag_law/kappa.csv` at the run's a and its median Adam preconditioner. This is 3B's registered
P2b constant. k = −3…3 is tabulated; for orientation, κ = +7.61 at a = 1.30, k = −1 and +4.05 at a = 1.50, k = 0.

**χ: pre-crossing information only** (`track2c.predict_run`; s_t = |w₂(t)|):
- **t_sw** = the first step t ≥ 1 with s_t ≥ s_own,d (2A's P-freeze rule).
- **Rule point t_R** = Track A's `rule_step`: the last upward passage of 0.5·s_own,d before t_sw.
- **Occupied branch:** damped Newton in all hidden coordinates at w₂ = w₂(t_R), from the state at t_R, on the seed's own
  sample (`band_rd.branch_rd`; converged if the gradient is below 1e−8). It is then continued in s to s_{t_sw}, in
  log-steps of at most 1%, each solved from the previous point. The continuation is lost if a solve does not converge
  or the branch moves by more than 0.5 in any coordinate (3B's post hoc convention).
- **Winding k:** 3B's rule on the rule-point branch: k = round((canonical b₁ − b₁\*)/2π). Canonical b₁ = b₁·sign(w₂),
  and b₁\* is the width-1 switch point's b₁ (`lag_law/kappa.csv`).
- **P frozen at t_sw:** P = 1/(√v̂(t_sw) + 1e−8), v̂ the run's bias-corrected Adam second moment of (w₁, b₁, b₂).
- **relax** = 0.01·λ_min(P^{1/2} H_sig P^{1/2}) (`residual_timescale.relax_rate`). H_sig is the (w₁, b₁, b₂) block of
  the own-sample Hessian on the continued branch at s_{t_sw}.
- **growth** = log(s_{t_sw}/s_{t_sw−100})/100 (3B's window; Track A's min(100, t_sw) if t_sw < 100).
- **χ** = growth/relax. r_pred = κ_k·χ and s_pred = s_own,d·(1 + r_pred).
- **No prediction** (counted, not replaced): s never reaches s_own,d; no converged rule-point branch; continuation lost;
  winding outside the table; relax ≤ 0; or s_pred not finite and positive.
- Observed lag: r_obs = s_obs/s_own,d − 1.

**Keeping crossings out of the predictions** (Track A / 2A):
- `train` runs the full budget **without evaluating any gap or placement** (tested).
- It saves each parameter path (`results/track2c/paths/`, untracked) and records its SHA-256 with the predictions.
- `predictions.csv` and its SHA-256 are committed before `observe` runs.
- `observe` asserts the registration manifest, the committed predictions hash and every path hash. Then it evaluates G at
  every saved step. Training is deterministic, so this is every-step detection.

## 5. Registered criteria, per cell (`track2c.score_cell`)

On the cell's **scored runs**:

| criterion | statistic | PASS if |
|---|---|---|
| **C1** | median of \|log(s_obs/s_pred)\| | ≤ 0.05 |
| **C2** | median of r_obs/r_pred (signed lags) | ∈ [0.75, 1.25], inclusive |
| **C3** | D = \|log(s_obs/s_pred)\| − \|log(s_obs/s_own,x1)\| | upper end of the 95% percentile bootstrap interval of mean D < 0 |

- **Bootstrap (C3):** 10,000 resamples of the scored runs with replacement, `numpy.random.default_rng(2030000)`, the same
  seed in every cell. The interval is the 2.5th and 97.5th percentiles of the resampled means.
- **Scored run:** a crossing run whose t_sw is strictly before its crossing step, with a prediction and a finite x₁-only
  threshold.
- **Signed lags:** a run observed below s_own,d has r_obs < 0 and a negative ratio.
- **Numbers** are compared as computed in float64, with no rounding.

## 6. Validity, per cell

All three are required. If any fails, **C1–C3 of that cell are UNRESOLVED**.
- **V1** (design): at least 30 crossing runs. Crossing runs exclude runs placed at initialisation.
- **V2** (design): ρ = ‖w_noise‖₂/|w₁| at the crossing state is < 0.05 in at least 90% of crossing runs.
- **V3** (2A's convention, §7 item 9): scored runs are at least 90% of crossing runs. This means the prediction existed
  before the crossing for at least 90% of them.

**Other scoring rules:**
- A statistic that cannot be computed (no scored run; C3 with fewer than 2) is UNRESOLVED.
- **Outcome per cell:** PASS if C1, C2 and C3 pass; UNRESOLVED if any is UNRESOLVED; otherwise FAIL, naming each failing
  criterion.
- There is no pooled verdict across cells.

## 7. Details the page left open (the choices made, fixed now)

1. **Seeds:** the same 60 seeds in all four cells (3B's paired design).
2. **Protocol, crossing rule and budget:** 3B's (§3). The design says "as Track 3B".
3. **"The validated conditional search":**
   - the search, bracket and validation of §4, following the conventions of `act_general.validate_point` (ladder,
     independent CMA-ES, audit, tie 1e−9) and `own_threshold` (0.1 steps, width 0.01, midpoint);
   - placement is 3B's G > 0, the rule that defines the crossing.
4. **x₁-only own threshold:** 3B's `own_x1` convention (`own_threshold.own_threshold`, midpoint).
5. **κ:** 3B's P2b κ_k (width-1 population landscape, median Adam preconditioner, frozen). k comes from the branch
   occupied at the rule point, which is pre-crossing (3B took it at the crossing).
6. **χ:** 3B's P2b χ (growth over 100 steps; relax = lr·λ_min of the preconditioned own-sample signal Hessian block).
   Every quantity is taken at t_sw, not at the crossing. The branch is the one occupied at the rule point (Track A's
   branch rule), continued to s_{t_sw}.
7. **t_sw:** the first step t ≥ 1 with s_t ≥ s_own,d. This is 2A's rule with the occupied branch's switch replaced by
   s_own,d, as the brief states.
8. **C2's lag ratio:** r_obs/r_pred with r_obs = s_obs/s_own,d − 1 and r_pred = κ_k·χ, signed (GELU registration's
   convention).
9. **Runs with no pre-crossing prediction** (t_sw at or after the crossing, or no prediction):
   - The design requires χ from pre-crossing information only. Such a run cannot be scored. It is unscored, counted and
     reported by seed (2A).
   - To keep the scored set from being selected by the outcome, V3 requires scored runs to be at least 90% of crossing
     runs. This follows 2A's V2 (t_sw before the crossing in ≥ 90%) and 3B's P2b guard (unresolved if more than 20% of
     crossing runs are unusable).
   - A validation failure of s_own,d counts here, never as a silent exclusion.
10. **C3's comparator:** the page's "x₁-only own threshold" is used literally, s_own,x1. A stricter comparator,
    s_own,x1·(1 + κ_k·χ) (the x₁-only threshold plus the same lag, 3B's registered P2b form), is reported beside it as a
    registered descriptive and never replaces it.
11. **Bootstrap:** the GELU registration's rule (10,000 resamples, percentile interval, one fixed seed), seed 2030000.
12. **ρ:** 3B's definition, ‖w_noise‖₂/|w₁| at the crossing state. "< 0.05" is strict.
13. **Degenerate statistics and the outcome rule:** 2A / GELU conventions (§6).

## 8. A-priori risks, stated now (from non-registered seeds and committed 3B data only)

- **Mirror branches.** On a finite sample, the conditional minimiser has two mirror branches, (w₁, b₁) and ≈ (−w₁, b₁)
  at w₂ > 0. Their losses differ, and so do their R^d switches.
  - Diagnostic on 3B's primary seeds 880,000–880,005, all four cells (`diag_mirror_3b.csv`; NON-REGISTERED, no
    training): the two mirrors' switches differ by factors 0.67–1.21.
  - On the 22 of these seed-cells whose 3B run crossed, the global switch (the design's s_own,d) equals the switch of
    the branch the run followed (3B's post hoc `s_branch`) within 0.5% in 16. In the other 6 the run sat on the
    higher-loss mirror, and the global switch was off by factors 0.818–1.211.
  - The switch of the minimiser restricted to the run's mirror matched in 22 of 22 (ratios 0.9990–1.0006).
  - **Consequence, stated before running:** if a similar share of 2C runs (about a quarter) occupy the other mirror, their
    errors against s_own,d are 10–20%.
    - C1 and C2 are medians. They pass only if the runs on the global mirror are a clear majority and follow the
      decomposition.
    - C3 is a mean, so it absorbs these errors in full.
    - A FAIL driven by mirror choice would be reported as a FAIL of the registered prediction.
  - **Registered descriptive (not scored):**
    - For every seed-cell, the other mirror's own-sample switch is frozen too (`mirror_switches`: the mirror-restricted
      search, bracketed like s_own,d, not validated).
    - Each run's mirror at its rule point is recorded before observation.
    - A "mirror-matched" prediction is computed before observation with the same rules and the switch of the
      rule-point mirror. The design's s_own,d is used when that mirror is the global minimiser's.
    - Its C1–C3 statistics are reported beside the registered verdicts and never replace them.
- **Crossing rate at d = 4.** 3B crossed 29/40 (primary) and 90/120, 93/120 (pooled) at d = 4. At 73–78%, 60 seeds give
  an expected 44–47 crossings against V1's 30.
- **ρ at crossing.** 3B had no crossing run with ρ ≥ 0.05 among its 407 crossing runs (d = 2, 4; primary and
  secondary arms pooled; largest 0.048). V2 is not expected to bind, but 0.048 is close to 0.05.
- **V3.** A run that crosses before reaching s_own,d (below its frozen switch) is unscored. The mirror effect makes this
  possible when a run's mirror has the lower switch. In the diagnostic, 2 of the 6 mismatched runs followed a branch
  whose switch was below s_own,d (s_own,d/s_branch = 1.21, 1.20). The other 4 were above it (0.82–0.90), so they cross
  well above s_own,d. If more than 10% of a cell's crossing runs cross below s_own,d, the cell is UNRESOLVED.
- **Cost.** Timing on seed 889,000 (`timing.csv`): 395 s per seed for all four cells, including the x₁-only threshold,
  the validated R^d switch and the other mirror's switch. That is about 6.6 h for 60 seeds at one worker. Training is
  240 runs × 64,000 steps.

## 9. Frozen values (computed before any training; in the registration commit)

To be filled in from `results/track2c/frozen_seeds.csv` and `kappa_k.csv` before the registration commit.

## 10. Commands

    python -m src.track2c freeze      # per seed: x₁-only threshold, validated s_own,d, other-mirror switch
    python -m src.track2c kappa       # frozen width-1 κ_k table
    python -m src.track2c hashes      # registration manifest (after this file's §9 is final)
    python -m src.track2c train       # after the registration commit: 240 runs, no gap evaluated
    python -m src.track2c finalize    # predictions.csv + predictions.sha256 (committed before observe)
    python -m src.track2c observe     # every-step detection, scores -> scores.json, observed_runs.csv
