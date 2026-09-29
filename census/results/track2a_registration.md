# Test 2A registration: Adam per-run ordering at an unseen activation value (a = 1.85)

**Status: written before any registered run.** No run at a = 1.85 has been trained. No gap, placement or crossing of
any training trajectory at a = 1.85 has been evaluated. This file becomes the registration when it is committed together
with the frozen inputs of §5 and their SHA-256 hashes (`results/track2a/registration.sha256`). That commit is
timestamped with OpenTimestamps (`results/track2a/registration_stamp.txt` and its `.ots` proof).

## 1. Design reference

- Approved design: `results/designs/2A_adam_ordering_design.md` (committed in `c3840fe`), approved by the author.
- The one number the design left open was supplied by the author: **A2 threshold = 0.75**. Nothing else in the design
  is changed.
- Code: `src/track2a.py`, a thin layer over the Track A pipeline `src/track_a.py` (registered in `fcd2e46`).
- Tests of every decision rule, on constructed cases: `tests/test_track2a.py`.

## 2. Where the rule comes from (stated plainly)

- Track A (registered, a = 1.65) froze Adam's preconditioner P at the rule point (the last upward passage of
  0.5·s\*_frozen). Its per-run ordering criterion failed: Spearman 0.25 < 0.5.
- **The t_sw rule was chosen POST HOC on those a = 1.65 runs** (`src/track_a_diag.py`, variant `p_switch`; writer
  inputs `results/track_a_writer_inputs.md` §5). There, P frozen at t_sw gave Spearman 0.95 and 91% of runs within 10%.
- **It is tested prospectively here**, at a value of a with no training or crossing data, on 80 fresh seeds, with every
  rule and threshold fixed before training.

## 3. Setting and why a = 1.85

- Width 1, f_a(t) = t + a sin t, **a = 1.85**.
- Checked on 2026-09-28 against every CSV under `results/` (all columns scanned): a = 1.85 appears only in
  `corner_tracking.csv` (column `a`, 1 row; a landscape table, no training) and in `search_anneal.csv` (column
  `parameter`, 12 rows; a depth-3 annealing search, not width-1 training). No `src` file trains width 1 at 1.85.
- **Seeds: 1,850,000–1,850,079** (80 fresh seeds). No seed in this range appears in any results CSV, JSON, log or
  Markdown file, or in `src/` or `tests/`.
- Protocol (Track A's, unchanged): torch Adam, lr 0.01, other settings default; sample `fold1d.make_data(200, seed)`
  (400 points), float64; initialisation `torch.manual_seed(seed)` then uniform(−1, 1)⁴; budget 32,000 steps;
  every-step detection with `phase2b_ordering.state` on the dense 4,001-point windows. The observed crossing is the
  first placed step; s_obs = |w₂| there.

## 4. The pipeline: Track A unchanged except the P-freeze step

`src/track2a.py` loads `src/track_a.py` as a separate module instance. The file must be byte-identical to its
registered version (SHA-256 `06d21cf36511a1797df1f8d20691c1d6f5f8d29bcbc812780b1ddc2a46121b09`, asserted at load).
Only its data constants are rebound: a = 1.85, the 80 seeds, the output directory `results/track2a/`, Adam only.

Per run (Track A's rules R0–R5, with R3 replaced):
- **R0 (frozen per seed, before training):** s\*_frozen = the switch of the landscape branch on the seed's own sample
  (Newton from the landscape switch point onto the own sample, then continuation); the seed's global own-sample
  threshold (`own_threshold`, bracket 0.01; descriptive only).
- **R1 rule point t_R:** the last upward passage of 0.5·s\*_frozen before s_t = |w₂(t)| first reaches s\*_frozen
  (`track_a.rule_step`). No rule point if s never reaches s\*_frozen.
- **R2 branch rule:** the branch occupied at t_R (Newton at fixed s = s_{t_R} from the run's state, continued on the own
  sample over [0.3, 1.7]·s\*_frozen, spacing 0.002·s\*_frozen). s\*_run = its switch nearest s\*_frozen. No
  information after t_R is used.
- **R3′ (the one change) P-freeze step:** **t_sw = the first step at which |w₂| ≥ s\*_run** (`track2a.t_switch`).
  P = 1/(√v̂ + ε) with v̂ the run's bias-corrected Adam second moment at t_sw. It is read from the output-scale
  trajectory in real time; no gap is evaluated. Implementation: `predict_one` reads v̂ only at t_R, so v̂[t_R] is
  replaced by v̂[t_sw] and `predict_one` runs unchanged (exactly the post hoc diagnostic's construction).
- **R4 primary prediction (trajectory-integrated):** Track 1's linear-response recursion from t_R with
  δ₀ = θ_{t_R} − θ\*(s_{t_R}), m₀ = the run's Adam first moment at t_R, **P frozen at P(t_sw)**, H(s_t) and θ\*(s_t)
  of the occupied branch along the run's own s_t, Adam's bias correction 1 − β₁^t. Predicted crossing: first step with
  G(θ\*(s_t) + δ_t) > 0 (exact extrema on the continuous windows). r_pred = s_t/s\*_run − 1 there. No prediction if the
  iterated linear map has spectral radius > 1 (checked every 10th step) or sup|δ| > 1.
- **R5 secondary (closed form):** r_cf = κ_k·χ with κ_k from the landscape at s\*_pop (H, θ\*′, ∇G of the principal
  copy) and P = P(t_sw); χ = (ṡ/s\*_run)/(η·λ_min(P^{1/2}H_pop P^{1/2})), ṡ = (s_{t_sw} − s_{t_sw−100})/100.
  **Reported, not scored.**
- **Observed lag:** r_obs = s_obs/s\*_run − 1.
- **Descriptive only, never scored:** the Track A rule (P frozen at t_R) is computed on the same runs, for comparison.

**Keeping the crossing out of the predictions** (as in Track A): `train` runs the full budget without evaluating any gap
or placement, saves each parameter path (`results/track2a/paths/*.npz`, untracked) and records its SHA-256 with the
predictions. The predictions are committed with their SHA-256 before `observe` runs. `observe` asserts the registration
hashes, the committed predictions hash and each path's hash, then evaluates `state` at every saved step. Training is
deterministic, so this is every-step detection.

## 5. Frozen before any training (the registration commit)

- The validated switch and κ at a = 1.85 (`results/track2a/landscape.json`): continuation in a at fixed s from the
  certified a = 1.60 switch point (`lag_law/kappa.csv`), then continuation in s on the 800-point population objective;
  s\*_pop = the root of G(θ\*(s)). Validated by the existing conditional search `own_threshold.global_min`: the global
  conditional minimiser must be unplaced at 0.995·s\*_pop, and placed at 1.005·s\*_pop on the continued branch (up to
  the mirror and 2π) (`track_a.landscape`, unchanged). If this validation fails, 2A stops before registration.
  Values: §9.
- The winding rule: the copy with canonical b₁ ∈ (−π, π], orientation w₂ > 0 (`track_a.principal_copy`; defined in
  Track A from existing crossings at other a).
- Each seed's s\*_frozen and global own threshold (`results/track2a/frozen_seeds.csv`).
- The code and its hash. `results/track2a/registration.sha256` lists the SHA-256 of: `src/track2a.py`,
  `src/track_a.py`, `src/linear_response.py`, `src/fold1d.py`, `src/phase2b_ordering.py`, `src/own_threshold.py`,
  `src/width2_conditional.py`, `tests/test_track2a.py`, this file, `results/track2a/landscape.json` and
  `results/track2a/frozen_seeds.csv`. `train` and `observe` assert every one of them before running.

## 6. Registered criteria (`track2a.score_2a`)

On the **scored runs** only, with pred = r_pred (primary, P = P(t_sw)) and obs = r_obs:

| criterion | statistic | PASS if |
|---|---|---|
| **A1** | Spearman(pred, obs) (average ranks for ties) | ≥ 0.5 |
| **A2** | fraction of scored runs with \|obs/pred − 1\| ≤ 0.10 | ≥ **0.75** |
| **A3** | median of obs/pred | ∈ [0.9, 1.1] (inclusive) |

**Scored run:** a run that (i) crossed within the budget, (ii) has t_sw defined and **strictly before** its crossing
step, and (iii) has a finite primary prediction.

**Validity** (all three required; otherwise **every criterion is UNRESOLVED**):
- V1: at least 60 of the 80 runs cross.
- V2: t_sw strictly precedes the crossing in at least 90% of crossing runs. A crossing run whose t_sw is undefined
  (no rule point, no occupied-branch switch, or s never reaches s\*_run) counts as not preceding. Crossing runs without
  t_sw before the crossing are unscored, counted and reported by seed.
- V3: the rule point t_R strictly precedes the crossing in every scored run.

**Other scoring rules:**
- Runs without a primary prediction (R1, R2 or R4 fails) are counted and not replaced.
- A criterion whose statistic cannot be computed (Spearman with fewer than 3 scored runs; A2 or A3 with no scored run)
  is UNRESOLVED. The design is silent on this degenerate case; this is its only possible completion, fixed now.
- **Outcome:** PASS if A1, A2 and A3 all pass; UNRESOLVED if any is UNRESOLVED; otherwise FAIL, naming each failing
  criterion.
- Numbers are compared as computed in float64, with no rounding before comparison.

## 7. Competing predictions and falsifiers

| outcome | reading |
|---|---|
| PASS (A1, A2, A3) | Track A's Adam L3 failure at 1.65 was a measurement-point artifact; the corrected rule, fixed before training, predicts per-run Adam lags at an unseen a. |
| A1 fails | Per-run variation is set by more than P at t_sw. The t_sw rule does not carry over from 1.65. |
| A2 fails (A1, A3 pass) | Ranking and typical magnitude are right, but per-run precision is worse than 10% in more than a quarter of runs. |
| A3 fails | The magnitude is off at this a. |
| UNRESOLVED | Too few crossings, t_sw too often at or after the crossing, or a scored run whose rule point is not before its crossing. The test says nothing about the rule. |

- **Competing rule (descriptive, not scored):** Track A's P at the rule point. At 1.65 it gave Spearman 0.25.
- **Falsifier of the post hoc claim:** any single FAIL. A failure stays a failure; post hoc readings may be placed beside
  it, labelled POST HOC.

## 8. A-priori risks, stated now

- At 1.85, 0.5·s\*_pop ≈ 0.65 lies inside the initial |w₂| range (uniform on [0, 1]). The rule point is the last upward
  passage (Track A's R1), which is robust to this.
- Track A's Adam runs crossed 76 of 80 at 1.65 within 32,000 steps; 1.85 has a smaller switch. Validity V1 is not
  expected to bind, but this is not known.
- At 1.65, t_sw preceded the crossing in 74 of 74 runs with a prediction. A negative lag (crossing before s\*_run) would
  make a run unscored under V2.

## 9. Frozen values (filled in by the registration commit, before any training)

See `results/track2a/landscape.json` and `results/track2a/frozen_seeds.csv`.
