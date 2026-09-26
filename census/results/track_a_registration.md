# Track A registration: the lag law at an unseen activation value (a = 1.65)

**REGISTERED.** This file, `src/track_a.py`, `tests/test_track_a.py` and the frozen inputs below are committed in one
commit before any registered run. Their SHA-256 hashes are listed in §6. No registered run has been trained. No gap or
placement of any training trajectory at a = 1.65 has been evaluated.

## 1. Why

Reviewers call the lag law post hoc. One asks whether the winding index and the preconditioner were fixed before any
crossing data were seen. In Track 1A they were not. This test fixes everything first, at a value of a with no crossing
data.

## 2. The value of a

**a = 1.65.** The author's spec named a = 1.40, but his decision was to use a value with no existing crossing data.

| candidate | status |
|---|---|
| 1.40, 1.35 | Width-1 training crossing data exist (`wi_crossing_runs.csv`, `crossing_audit_runs.csv`, `blockC_adiabatic.csv`, …). Excluded. |
| 1.55, 1.70 | Appear in `onset_curves.csv`, `sgd_onset_curves.csv` or `onset_bootstrap_outcomes.csv`: placement rates of training runs by budget, which is crossing information. Excluded. |
| 1.65 | Appears only in `corner_tracking.csv` (a landscape table; no training) and in `search_anneal.csv` (`parameter` column of a depth-3 annealing search; not width-1 training). |

- The candidates were checked against every CSV under `results/` with an a/parameter column. No `src` file trains width
  1 at a = 1.65.
- Fresh seeds: 1,650,000–1,650,079 (80 seeds), used for both optimisers. They appear nowhere in the repository.

## 3. Frozen landscape (`results/track_a/landscape.json`; `track_a.landscape`)

- **No certified glob bracket exists at 1.65.** The switch is validated by branch continuation plus the existing
  conditional search `own_threshold.global_min`, the search behind every own-sample threshold in the paper.
- **Construction.** Newton continuation in a at fixed s from the certified 1.60 switch point (`lag_law/kappa.csv`),
  then continuation in s on the 800-point population objective. s\*_pop is the root of G(θ\*(s)), found by bisection.
  **s\*_pop = 1.81002.**
- **Validation.** The global conditional minimiser has G = −0.00231 at 0.995·s\*_pop and G = +0.00228 at 1.005·s\*_pop.
  At both scales it lies on the tracked branch up to the mirror x → −x and 2π. The Hessian is positive definite.
- **Winding rule, from the landscape alone:** the copy with canonical b₁ ∈ (−π, π], taken with orientation w₂ > 0.
  - This rule was defined using existing data at OTHER a. All 1,750 existing crossings at a = 1.30/1.45/1.50/1.60 have
    canonical b₁ ∈ (−2.47, −2.22). The rule therefore reproduces 1A's measured windings: k = −1 relative to the
    retained minimiser at 1.30, and k = 0 at 1.45–1.60.
  - At 1.65 the continued branch is already on that copy: b₁ = −2.2026, shift 0.
- **Landscape quantities at s\*_pop:** θ\*, H, θ\*′ and ∇G on the principal copy. κ_SGD(1.65) = 3.047. For Adam, κ is
  computed per run with the shape of that run's rule-point preconditioner. κ does not depend on P's scale.

## 4. Protocol (free training)

- **Adam:** torch Adam, lr 0.01, other settings default (the Task B protocol).
- **SGD:** plain torch SGD, lr 0.3, no momentum (the `sgd_own` protocol).
- **Common to both:**
  - Sample: `fold1d.make_data(200, seed)` (400 points), float64.
  - Initialisation: `torch.manual_seed(seed)`, then uniform(−1, 1)⁴.
  - Budget: 32,000 steps.
  - Detection: every step, with `phase2b_ordering.state` on the dense 4,001-point windows. The observed crossing is the
    first placed step, and s_obs = |w₂| at that step.
- **Compute.** One process per optimiser, at most two in total, with the coordinator's permission. Each process is
  independent and deterministic and writes its own parts file.
- **Keeping the crossing out of the predictions.**
  - `train` runs the full budget without evaluating any gap or placement. It saves the parameter path (`paths/*.npz`,
    untracked; each file's SHA-256 is recorded in the predictions) and computes the predictions.
  - Predictions are committed and hashed.
  - Only then does `observe` evaluate `state` at every saved step. Before doing so it asserts the committed hash and
    each path hash. Training is deterministic, so this is exactly every-step detection.

## 5. Rules (all frozen now; per run, only information up to the rule point plus the run's s_t = |w₂(t)| trajectory)

- **R0. Frozen per seed, before training** (`results/track_a/frozen_seeds.csv`):
  - s\*_frozen: the switch of the landscape branch on the seed's own sample. Obtained by Newton from the landscape switch
    point onto the own sample, then continuation.
  - The seed's global own-sample threshold (`own_threshold` rule, bracket width 0.01). It is descriptive only and does
    not enter L1–L3.
- **R1. Rule point.** t_R is the step at which s_t last passes 0.5·s\*_frozen upward before s_t first reaches
  s\*_frozen (the first step t_R ≥ 1 with s_t ≥ 0.5·s\*_frozen for all t_R ≤ t ≤ t_top). This is the author's "when the
  run first reaches 0.5 of its branch switch", made robust to the initial transient: at 1.65, 0.5·s\* ≈ 0.9 lies inside
  the initial |w₂| range (uniform on [0, 1]), so a literal "first reaches" would often be step 1. It uses the
  output-scale trajectory only. There is no rule point if s never reaches s\*_frozen.
- **R2. Branch rule.** The branch occupied at t_R: Newton at fixed s = s_{t_R} from the run's state (w₁, b₁, b₂),
  continued on the own sample (grid [0.3, 1.7]·s\*_frozen, spacing 0.002·s\*_frozen).
  - s\*_run is its switch nearest s\*_frozen. All lags are measured against s\*_run.
  - No information after t_R is used to identify the branch.
- **R3. Preconditioner rule.** Adam: P_R = 1/(√v̂ + ε) at t_R. SGD: P = I.
- **R4. Trajectory-integrated prediction.** This is the Track 1 linear-response recursion, started at t_R.
  - Initial state: δ₀ = θ_{t_R} − θ\*(s_{t_R}), and m₀ = the run's Adam first moment at t_R.
  - **P is frozen at P_R.** H(s_t) and the exact θ\*(s_t) path come from the occupied branch along the run's own s_t.
    Adam's bias correction 1 − β₁^t and momentum β₁ = 0.9 are included. SGD has P = I and no momentum.
  - Prediction: the first step with G(θ\*(s_t) + δ_t) > 0, using exact extrema on the continuous windows. Then
    r_traj = s_t/s\*_run − 1 at that step.
  - No prediction if the iterated linear map has spectral radius > 1 (checked every 10th step) or sup|δ| > 1.
- **R5. Closed form.** r_cf = κ·χ.
  - κ = λ_min·[∇G·(PH)⁻¹θ\*′]/[∇G·θ\*′] from the landscape at s\*_pop (H, θ\*′, ∇G), with P = P_R (Adam) or I (SGD).
  - χ = (ṡ/s\*_run)/(η·λ_min(P^{1/2}H_pop P^{1/2})).
  - ṡ = (s_{t_sw} − s_{t_sw−100})/100, where t_sw is the first step with s_t ≥ s\*_run. This uses the output-scale
    trajectory only.
- **Observed lag.** r_obs = s_obs/s\*_run − 1.

## 6. Registered criteria (per optimiser; relative only)

| criterion | statement | PASS if |
|---|---|---|
| **L1** | median over runs of r_obs/r_traj | ∈ [0.90, 1.10] |
| **L2** | median over runs of r_obs/r_cf | ∈ [0.80, 1.20] |
| **L3** | Spearman(r_traj, r_obs) over runs | ≥ 0.5 |

- The Spearman of r_cf with r_obs is reported as secondary.
- **Validity (per optimiser):**
  - At least 60 crossings within the budget.
  - Each criterion needs at least 60 crossing runs that have its prediction. Otherwise that criterion is UNRESOLVED.
  - Runs whose rule point is not strictly before their crossing are excluded and counted.
  - Runs with no prediction (R2 or R4 fails) are counted and not replaced.
- **A-priori risk, stated now.** In the existing SGD runs at 1.30 and 1.50, 30 of 40 crossed at each a; the others
  stalled on the small-|w₂| plateau. 80 seeds therefore give an expected ~60 SGD crossings, so SGD validity is uncertain
  by design. Adam (Task B) crossed 80 of 80 at 1.45 and 1.60.
- Implementation: `track_a.score_opt`, tested on constructed pass, fail and unresolved cases in `tests/test_track_a.py`.

## 7. Frozen files and hashes

These are listed in `results/track_a/registration.sha256`, which is written by this commit:
- `src/track_a.py`
- `tests/test_track_a.py`
- `results/track_a/landscape.json`
- `results/track_a/frozen_seeds.csv`
