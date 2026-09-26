# Track 3B writer inputs: band task in R^d (d = 2, 4)

All numbers below come from committed producers: `census/src/band_rd.py` (commands `score` and `posthoc`), with outputs
in `census/results/band_rd/`. Registration: `census/results/band_rd_registration.md`.

## 1. What was committed before any comparison

| commit | content |
|---|---|
| f19535b | REGISTRATION: design, proof that the threshold carries over, population check (validated, not certified), seeds, pilot disclosure, budget 64,000, rules P1 / P2a / P2b, validity checks, descriptives, tests. Frozen inputs: `width1_reference.json` (SHA-256 da0f85fd…), `own_x1_frozen.csv` (SHA-256 b4c9abcb…, 80/80 defined), batching validation (max difference 7e−15). |
| ca62e36 | RESULT: runs, scoring, registered verdicts and descriptives. |
| e51c0bb | POST HOC (labelled): residual against each run's own R^d branch switch. |

Disclosed ordering caveat. The design was fixed in code before any d > 1 training, but the registration document was
written after a 12-seed pilot, which recorded crossing counts and steps but no crossing scales. The pilot changed only
the budget (64,000) and led to adding the secondary arm, because ≥ 90% crossing turned out not to be reachable at
d = 4.

## 2. Design (short)

- **Features.** x₁ is exactly the width-1 task: `fold1d.make_data(200, seed)`, 400 points. The coordinates
  x₂…x_d are i.i.d. U(−2, 2) and independent of the label.
- **Model and training.** A single unit z = w₂ f_a(w·x + b₁) + b₂ with w ∈ R^d. Initialisation is U(−1, 1) per
  coordinate, and its first four draws equal the width-1 initialisation of the same seed. Training is Adam at lr 0.01,
  full batch.
- **Placement gap.** The gap is taken on the continuous support. Every class window is widened by N = 2‖w_noise‖₁,
  and extrema are exact. When w_noise = 0 it equals the width-1 gap bit for bit (tested).
- **The threshold carries over (proof).** The R^d population loss equals E_ξ L₁(w₁, b₁ + ξ, b₂), where
  ξ = w_noise·x_noise is independent of the label. This is at least L₁*, with equality only when w_noise = 0. So the
  conditional minimiser has w_noise = 0, and the certified width-1 bracket carries over exactly: a = 1.30 in
  (4.95, 4.9625], a = 1.50 in (2.525, 2.5375].
- **Numeric check** (validated, not certified; `popcheck_summary.csv`). At both bracket ends, for d = 2 and d = 4:
  - no start ends below L₁*;
  - every start that reaches L₁* has ‖w_noise‖ ≤ 5e−12;
  - the noise Hessian block equals (4/3)·∂²L/∂b₁² to within 5e−15;
  - the profile in ‖w_noise‖ increases strictly;
  - there is a higher stationary point carrying a noise weight: ‖w_noise‖ ≈ 0.64–0.84, loss 0.37 above L₁*.

## 3. Registered results (primary: 40 seeds 880,000–880,039 per cell; `verdicts.csv`, `verdicts.json`)

| d | a | crossings | P1: fraction s_c ≥ s_lo (≥ 0.90) | P2a: median r_pop, width-1 IQR | P2b: median r_own vs pred κχ (tol) |
|---|---|---|---|---|---|
| 2 | 1.30 | 36/40 | 0.972 **PASS** | 0.182 vs [0.035, 0.154] **FAIL (above)** | 0.147 vs 0.024 (0.010) **FAIL** |
| 2 | 1.50 | 36/40 | 0.972 **PASS** | 0.207 vs [0.072, 0.185] **FAIL (above)** | 0.187 vs 0.049 (0.012) **FAIL** |
| 4 | 1.30 | 29/40 | **UNRESOLVED** (< 30) | **UNRESOLVED** | **UNRESOLVED** |
| 4 | 1.50 | 29/40 | **UNRESOLVED** | **UNRESOLVED** | **UNRESOLVED** |

**Secondary arm** (registered; 120 seeds pooled; P1 and P2a only; reported beside the primary, not in place of it):

| d | a | crossings | P1 | P2a |
|---|---|---|---|---|
| 2 | 1.30 | 112/120 | 0.946 PASS | 0.189 FAIL (above) |
| 2 | 1.50 | 112/120 | 0.964 PASS | 0.226 FAIL (above) |
| 4 | 1.30 | 90/120 | 1.000 PASS | 0.528 FAIL (above) |
| 4 | 1.50 | 93/120 | 1.000 PASS | 0.553 FAIL (above) |

**Validity.** The recomputed R^d gap was > 0 at every recorded crossing. Every branch converged, so every d = 2
crossing run was usable for P2b. No run was placed at initialisation.

**Width-1 control** (descriptive, not scored; d = 1, same seeds, same code). All 40 runs crossed in each cell.
- P1 fractions: 0.875 at a = 1.30 (it would fail P1, as the width-1 reference 0.842 anticipated) and 0.975 at a = 1.50.
- Median r_pop: 0.055 and 0.088, both inside the IQR.
- Lag law: r_own 0.0298 vs pred 0.0289, and 0.0622 vs 0.0626, both within tolerance.

So on these fresh seeds the width-1 lag law reproduces, and the R^d excess is not a seed effect.

## 4. Registered descriptives: how the unit finds the direction (`descriptives.csv`; primary cells unless noted)

- **The noise weight collapses early.** ρ = ‖w_noise‖₂/|w₁|:
  - at initialisation, median 1.23 (d = 2) and 2.29 (d = 4);
  - at |w₂| = 0.5·s_mid, median 0.014 (d = 2) and 0.028–0.030 (d = 4);
  - at the crossing, median 0.012 (d = 2) and 0.020–0.021 (d = 4).

  In every crossing run ρ falls below 0.1 before the crossing, at a median 3–9% of the crossing time.
- **No run crosses with a non-negligible noise weight.** Zero runs have ρ_c ≥ 0.05; the largest ρ_c is 0.034 in the
  primary cells and 0.048 in the pooled cells.
- **The residual noise weight still matters.** Its widening at the crossing is a median 3% (d = 2) and 7–8% (d = 4) of
  the projected inner half-width. Every crossing comes after the x₁ direction alone is placed (G1 > 0), never on the
  same step:
  - d = 2: median delay 295 / 143 steps, +13% in scale;
  - d = 4: 770 / 361 steps, +36–39%.
- **Non-crossing runs are noise-dominated.** Every one ends with ρ > 1: 4 of 40 at d = 2 and 11 of 40 at d = 4. Their
  median final values are ‖w_noise‖ ≈ 2.1–2.3, |w₁| ≈ 0.5–1.4 and |w₂| ≈ 0.06–0.08. This matches the noise-carrying
  stationary point found in the population check. Pooled: 8/120 at d = 2 and 30 and 27 of 120 at d = 4.
- **Windings and branch geometry match width 1.**
  - Windings as in width 1: k = −1 at a = 1.30 and k = 0 at a = 1.50, with one exception per d = 2 cell.
  - The branch's own w_noise has median norm 0.010–0.021.
  - λ_min(full)/λ_min(signal block) = 0.999, so the noise block does not set the relaxation rate.

## 5. POST HOC (labelled; after the registered result; `posthoc_summary.csv`, `posthoc_scored_runs.csv`)

For each crossing run, the switch of its own-sample R^d branch was computed: continuation in all hidden coordinates,
using the R^d gap.

- **The R^d own-sample switch sits above the x₁-sample own threshold.** Median ratio 1.118–1.120 at d = 2 and
  1.401–1.412 at d = 4. The x₁-direction (G1) switch of the same branch equals the x₁ own threshold (ratio
  0.9996–0.9997).
- **Measured against that switch, the lag law holds in every cell,** within the registered tolerance (obs/pred in
  brackets):

  | d | a = 1.30: median r_branch vs pred | a = 1.50: median r_branch vs pred |
  |---|---|---|
  | 2 | 0.025 vs 0.024 (1.06) | 0.054 vs 0.049 (1.10) |
  | 4 | 0.019 vs 0.017 (1.09) | 0.038 vs 0.036 (1.07) |
  | 4, pooled | 0.014 vs 0.017 (0.84) | 0.033 vs 0.034 (0.96) |
  | 1, control | 0.030 vs 0.029 | 0.063 vs 0.063 |

- **Interpretation (post hoc).** The excess residual in R^d is a static finite-sample offset. On 400 points the loss
  minimiser keeps a small noise weight, and on the continuous support that weight widens the class windows and raises
  the sample's own threshold. The dynamic lag behind that threshold is width-1-sized and follows κχ. This decomposition
  was not registered.

## 6. Writer section

**Outcome: P1 at d = 2 (PASS, both a; secondary PASS in all four cells)**
- Say: "In R² with a label-independent noise coordinate, 97% of crossing runs placed the unit at or above the certified
  width-1 threshold (35/36 at both a; the pooled 120 seeds: 95% and 96%; d = 4 pooled: 100%)."
- Do not say: that P1 passed at d = 4 in the registered primary arm. It was UNRESOLVED with 29 crossings; the 100%
  figure is the secondary arm.

**Outcome: P2a and P2b at d = 2 (FAIL, above), P2a in the secondary arm (FAIL, above, all four cells)**
- Say: "The registered prediction that the crossing lag stays in the width-1 range failed. Crossings in R^d came later.
  The median residual against the population threshold was 0.18–0.21 at d = 2 (width-1 upper quartile 0.15 / 0.18)
  and 0.53–0.55 at d = 4 (pooled). Against the x₁-sample own threshold, the median residual was 0.15 / 0.19, against a
  predicted lag of 0.024 / 0.049."
- Say (clearly labelled post hoc): "A post hoc decomposition attributes the excess to the finite sample's own R^d
  threshold, not to a longer lag. The sample minimiser keeps a noise weight of about 1–2% of |w₁|, which widens the
  class windows on the continuous support and raises the sample threshold by 12% (d = 2) and 40% (d = 4). Measured from
  that threshold, the lag matches the width-1 lag law (obs/pred 0.84–1.10 across cells)."
- Do not say: "training crosses above the threshold with a lag in the width-1 range". That was the registered claim, and
  it failed.
- Do not say: that the lag law was confirmed in R^d as a registered result. The branch-switch comparison is post hoc.
- Do not say: that the conditional threshold moves in R^d. The population threshold provably does not (§2). What moves
  is the finite-sample own threshold.
- Do not present the post hoc offset as a law in d or n. Only d = 2, d = 4 and n = 400 were measured.

**Outcome: d = 4 primary UNRESOLVED**
- Say: "At d = 4 the registered primary cells were UNRESOLVED (29 of 40 crossings, below the registered 30). A
  registered secondary arm of 120 seeds crossed in 90 and 93 runs."
- Say: "About a quarter of d = 4 runs (11/40; 30 and 27 of 120) never place the unit. They converge to a noise-dominated
  state (‖w_noise‖ > |w₁|) that matches a noise-direction stationary point of the population loss."
- Do not say: that a longer budget would have resolved d = 4. The pilot ran to 128,000 steps without further crossings.

**Outcome: how the unit finds the direction (descriptive)**
- Say: "The unit finds the x₁ direction early. The noise-to-signal weight ratio falls from about 1–2 at initialisation
  to about 0.01–0.03 by half the threshold scale, and below 0.1 within the first 3–9% of the time to crossing. No run
  crossed with a noise ratio of 5% or more."
- Say: "Runs that fail to find the direction never cross: 4/40 at d = 2 and 11/40 at d = 4 end noise-dominated."
- Do not say: that the noise weight goes to zero before the crossing. It settles at the finite-sample value (about
  0.01–0.02), and that residual is what delays the crossing.

**On the reviewer's inner-ball-versus-shell task.**
- Do not say: that this track ran or tested it. It replaces that task with the band task in R^d, because a single unit
  cannot solve inner-ball-versus-shell: projections of the shell cover the ball's interval.

## 7. Files

- **Producer:** `census/src/band_rd.py`. Tests: `census/tests/test_band_rd.py` (20 tests).
- **Registered outputs:** `census/results/band_rd/`: `runs.jsonl`, `scored_runs.csv`, `verdicts.csv`, `verdicts.json`,
  `descriptives.csv`, `run.log`, `score.log`.
- **Design and pilot:** `popcheck_*.csv`, `pilot.csv`, `pilot_stuck.csv`, `validate_batching.csv`,
  `width1_reference.json`, `own_x1_frozen.csv`.
- **Post hoc:** `posthoc_summary.csv`, `posthoc_scored_runs.csv`, `posthoc_branch_switch.csv`.
- **Per-step traces:** `traces_*.npz`, 135 MB. They are not committed; `run` regenerates them, and their SHA-256 are in
  `traces_sha256.txt`. `describe_cell` reads them, for the ρ < 0.1 timing only.
