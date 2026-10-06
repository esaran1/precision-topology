# Phase 3 registration: the lag law at width 4 on the asymmetric windows (arms Q, S, T4)

**STATUS: NOT YET A REGISTRATION. STOPPED BEFORE THE REGISTRATION COMMIT for one decision of the author (§10, D1:
T4's hold length, which touches T4's gate).** The text was committed before the pilot (747d61e). Done: the scan, the
landscape, the Q and S pilot, the Q and S freeze and the author's budget condition (§13). Not done: the T4 pilot, the
T4 freeze, the summary, the manifest.

- **Not done:** no registered seed has been trained or observed. The Q and S seeds (7,430,000–119; 7,431,000–119) were
  held and classified at the freeze (release information only, §4); no T4 seed (7,432,000–709) has been drawn.
- **Before any registered training:** the registration commit is pushed and its OpenTimestamps proof is obtained
  (`results/phase3/registration_stamp.txt` and its `.ots`). `run` refuses to start without the proof file.

## 0. The finding reported first: random holds collapse (exploratory; the registered table in §13)

- At s₀, random holds (U(−1, 1)⁹) reached a genuine four-unit branch **0 of 400** times on the exploration's own
  samples. All 170 of the 200 accepted population landings were coinciding-unit points (3+1, 2+2, or the duplicate):
  width-2 or width-1 functions.
- The width-4 conditional minimiser at fixed s is the width-2 function (loss equal to 10 digits; switch bracket
  [0.43805, 0.43839], inside T2-1's).
- Genuine three- and four-function copies appear only from wide holds (U(−3, 3)⁹), at most 4/150 each.
- **So Q is a MECHANISM test from a branch point.** It asks whether a four-unit branch's signed lag is forecast once a
  run is on that branch. It does not claim that training from initialisation reaches such a branch.
- The registered T4 random holds (710 seeds) give this finding's registered count: the landing table by type, the
  genuine four-unit landings and the landings on each seed's own Q and S points (`frozen.json`,
  `collapse_random_holds_T4`; §13). It is descriptive and is reported first in the results.

## 1. Design reference

- **Approved page:** `results/designs/phase3_width4_asym_design.md` (commit bd2c48d), with the author's decisions of
  2026-10-06. Everything on it is binding here.
- **Author's condition (binding, 2026-10-06):** after the pilot sets each arm's ρ, every Q and S registered seed's
  budget is computed from its frozen t\*. If MORE THAN 24 seeds in either arm would exceed the 10⁶-step cap, Phase 3
  STOPS before registering, and the counts and the distribution of B are reported (`phase3_w4.budget_condition`; §13).
- **Code:** `src/phase3_w4.py`. Its numerics promote `results/designs/phase3_explore/w4core.py`, with W2-A's Newton
  rule. **Tests:** `tests/test_phase3_w4.py`.
- **Machinery mirrored:** W2-A (`results/width2_asym_registration.md`: setting, hold, Newton classification, windings,
  κ, adiabatic switch validation, own-path switch, R4, follow check, V1–V7 with its change 3) and Phase 1C
  (`results/phase1c_registration.md`: the strictly causal forecast, its guard and NaN recomputation, C1–C4, S, the
  scored set, misses, the 90% cutoff rule). 2A-PS (`results/phase2a_ps_registration.md`): training to t_c with the
  state hashed, the release frozen, the scan, the stamp.
- **Exploration** (`results/designs/phase3_explore/`, README): the population and the exploration seeds
  7,410,000–059 only.
- **Width-2 reproduction (tested):** at K = 2 the code reproduces W2-A's frozen population T and T′ switches to the
  last digit, κ₀ and the winding coefficients to 1e−9, and W2-A's own-path switch and R4 step for step.

## 2. Setting and seeds

- W2-A's setting at width 4: a = 1.30, Δ = 0.4; I = [−0.8, 0.8] (class 0), O = [−2.0, −1.2] ∪ [1.2, 2.4] (class 1).
- N(x) = Σ₁⁴ v_i f(α_i x + β_i) + b, f(t) = t + a·sin t. Hidden z = (α₁..α₄, β₁..β₄, b), output v = (v₁..v₄),
  s = ‖v‖₁. Each seed's own 400-point sample `asym_register.training_set(seed)`; mean BCE with logits; float64.
- G₊ = min_O φ − max_I φ: the exact-extrema enclosure with K units (width2_geometry's algorithm), screened by a dense
  grid. Observation and hold: placed iff the enclosure's lower end > 0. Branch points, switches, forecasts: placed iff
  the midpoint > 0.
- s₀ = 0.5·s_pop2 = **0.2225397** (T2-1's bracket, `asym_scores.json`). Held output v₀ = s₀·**(0.1, 0.2, 0.3, 0.4)**.
- Release: full-batch SGD without momentum, η = 0.03 on z and ρη on v.
- **Seeds:** Q 7,430,000–119 (120); S 7,431,000–119 (120); T4 7,432,000–709 (710). Pilot: 7,439,000–009 for Q and S;
  **a separate T4 pilot range, 7,439,100–299**, drawn in order until 10 releases land on T4 (§8, §10 item 1).
- **Unused-seed check** (`phase3_w4.scan`, `results/phase3/seed_scan.json`): no number 7,430,000–7,432,999 or
  7,439,000–999 (also written with `_` or `,`) in any text file under src, tests, results, paper, notes,
  independent, data or dist (Phase 3's own files and the approved page excepted); no overlap with any SEEDS\* /
  PILOT\_SEEDS\* constant of any src module; disjoint from the exploration seeds.

## 3. Arms, holds, release, gates

**Copies** (population points at v₀: damped Newton from the exploratory class points, 6 decimals, `POP_START`;
`results/phase3/landscape.json`). Each is checked for type, partition and unplaced status. Its switch must be
within 0.5% of the page's value and κ₀ must have the page's sign; otherwise the landscape STOPs.

| arm | role | copy (type; partition) | start before the hold | η, ρ start | seeds | gate |
|---|---|---|---|---|---|---|
| **Q** | headline, MECHANISM test | four-function (1111:++--; {0},{1},{2},{3}); switch 0.482, κ₀ > 0 | θ\*_Q,pop(s₀) (branch point) | 0.03, 2⁻¹³ | 120 | ≥ 96/120 on Q AND no run with G > 0 at the start or any hold step |
| **T4** | control | W2-A's T embedded (31:+-; {0},{1,2,3}); switch 0.762 | random hold: z ~ U(−1, 1)⁹, the first draw of numpy `default_rng(seed)` | 0.03, 2⁻¹⁰ | 710 | ≥ 60 runs on T4, regardless of G in the hold |
| **S** | sign test | three-function (121:+-+; {0,2},{1},{3}); switch 0.440, κ₀ < 0 | θ\*_S,pop(s₀) (branch point) | 0.03, 2⁻¹² | 120 | ≥ 96/120 on S AND no run with G > 0 in the hold |

- **Hold.** v fixed at v₀; z by full-batch GD at lr 1.0 for W steps; G at the start state and after every step.
  Q and S: W = max(4000, ⌈25/λ_min⌉), λ_min of the full z-Hessian at the seed's own copy at s₀ (W2-A's rule); 4000 if
  the own copy has no accepted point. **T4: W = max(8000, ⌈25/λ_min⌉)** (author's decision D1, 2026-10-06; §10); 8000
  if the own copy has no accepted point.
- **Release classification** (W2-A's Newton rule + the partition + the activity test). Damped Newton at v₀ from the
  release state, accepted iff max|∇| < 1e−8 and H_zz positive definite. The run is **on copy c at winding k ∈ ℤ⁴** iff:
  - the Newton point is within 1e−6 (sup) of shift(θ\*_own,c(s₀), v₀, k);
  - the release state is within 1e−3 of it;
  - the Newton point's coincidence partition equals the copy's (units coincide iff α equal and β equal mod 2π, within
    1e−6);
  - **every unit is active** (below).
  
  Otherwise it is on **neither**; an unconverged Newton is neither. Such runs are counted, never replaced.
- **Activity test** (Phase 3 only; introduced after the 2A-PS post hoc diagnosis, 2f0fb33). Unit i is active iff
  |v_i| ≥ θ·s, **θ = 1e−8** (inclusive, scale-free). A release on a copy with an inactive unit is classified
  "neither (inactive unit)" and counted. Any scale direction â is built from active units only.
- **Windings.** κ_k = κ₀ + a·k (exact). The table is ‖k‖₁ ≤ 3 (129 windings). The winding is fixed AT RELEASE by the
  classification (W2-A's condition 1); κ_k, the own-path branch, R4 and the follow check take it from that record. A
  run at a winding outside the table gets no forecast (counted).
- **Gate failure:** the arm is "UNRESOLVED (gate)" and stops; nothing of it is scored.

## 4. Frozen before training (the registration commit)

Per seed (`phase3_w4.freeze_seed`; `results/phase3/frozen_parts.jsonl`, summary `frozen.json`):
- **The arm's own copy point** θ\*_own(s₀): Newton from the population copy on the own sample; accepted, of its type
  and partition, unplaced. λ_min (full and reduced) and W.
- **The hold and the release**, run at the freeze: the release classification, the winding, G in the hold, the
  activity test, and the release state with its SHA-256. `run` re-asserts the hash. For T4 seeds, the landing is also
  classified against the seed's own Q and S points (descriptive; the collapse table).
- **The copy in full** (Q and S: every seed; T4: the seeds released on T4):
  - **the validated switch** along the adiabatic reference (the reduced gradient flow of v, RK4 in s at 0.5% of s,
    z\* by Newton at every stage; the first sign change of the branch gap, bisected). Validation: step halving agrees
    to 1e−6; no turning point; reduced λ_min > 0 along the path; for coinciding groups the split block PD along the
    path and the partition kept throughout; G(s₀) < 0 and the switch goes to placed; decided at (1 ± 1e−3)·s_switch;
    ∇G one-sided agreement ≤ 1e−5; Newton accepted at the switch; λ_min(switch) > 0; κ finite; the drive integral
    finite and positive;
  - **κ₀ and a** at the switch: κ = λ_min·[∇_zG·H⁻¹θ\*′]/[∇_zG·θ\*′ + ∂_vG·v′] (W2-A's total derivative; λ_min reduced
    by every group's split directions); a_i = −2π·λ_min·(H⁻¹∇_zG)_b·v′_i/[dG/ds];
  - **the follow point** at 0.8·s_switch on the reference path (step halving 1e−6, partition kept), with ∂z\*/∂v;
  - **the drive integral** ∫ds/D, D = −sign(v)·∇_vL(z\*(v), v) (trapezoid on the RK4 points; the last piece to the
    switch).
- **Budgets:** t\* = (1/ρη)·∫ds/D and **B = min(⌈1.5·t\*⌉ + 3000, 10⁶)**. A seed with ⌈1.5·t\*⌉ + 3000 > 10⁶ is
  **over cap: counted and NOT trained**. Hard per-run caps: 10⁶ steps and RSS 1 GB.
- A copy that fails any check is invalid; a run on it gets no forecast (counted).
- **Also frozen:** ρ per arm, τ_cross and τ_lag per arm, the pilot median χ at t_sw (V6), f = 0.95, θ, every tolerance,
  and the code (manifest, §12).

## 5. The causal forecast and the order of the steps

- **Cutoff (a stopping time):** t_c = the first t ≥ 1 with s_t ≥ 0.95·s_switch(frozen). Training from the frozen
  release stops at t_c (or at B). The state at t_c and the v prefix (rows 0 … t_c) are hashed.
- **Forecaster** (`phase3_w4.forecast_w4`, through `run_forecast`): it receives GuardedArray views of the v path
  (cutoff t_c) and of the z path (row 0, the release, only). With causal_forecast's frozen rule (25aac9e):
  - quad extrapolation of log s over the last ⌈0.05·t_c⌉ steps (at least 50), anchored, up to 1.6·s_switch or B;
  - **each unit's share** u_i = |v_i|/s extrapolated by the same family (`extrapolate_share`), clipped at 0 and
    renormalised (‡); the signs of v_{t_c−1}; no forecast if a sign changes in the window;
  - the own-path switch (W2-A's, change C) and R4 (W2-A's) along [visible, extrapolated], from the frozen copy at the
    release winding → t_sw,fc, t_fc, lag_fc = t_fc − t_sw,fc, r_fc = s_fc/s_sw,fc − 1.
- **Checks:** every guard's last row read is < t_c (`run_forecast`); a NaN recomputation (the same forecaster,
  unguarded, every row ≥ t_c set to NaN) must be identical (`nan_recompute_identical`).
- **Order** (`src/phase3_w4.py`):
  1. `scan`, `landscape`, `pilot` (pilot seeds only), `freeze` (all registered seeds; no training), `summary` (the gates
     at the freeze, the budgets, the author's condition), `manifest`; the registration commit; `stamp`.
  2. After the push and the OpenTimestamps proof: `run` per arm (training to t_c; the forecast; no observation).
  3. `finalize`: `forecasts.sha256` over the runs files (every seed once, no observed key). **Committed before
     observation.**
  4. `observe`: retraining from the release (deterministic), every-step detection, the t_c state and the v prefix
     asserted bit for bit, continued until the crossing, t_c and the actual own-path switch are all decided (or B).
     Then the registered validity inputs and the follow check.
  5. `score` → `scores.json`.

## 6. Criteria, scored runs, validity (1C; `phase3_w4.score_arm`)

**Gate first** (§3).

**Base (crossing runs):** on the arm's copy at release and trained (valid copy, winding in the table, within the cap),
follows its branch, crossed within B, finite r_cf and r_obs. **Scored:** the base with t_c STRICTLY before the
crossing. A scored run without a forecast (t_fc or t_sw,fc undefined) is a **miss**.

lag_fc = t_fc − t_sw,fc and lag_obs = t_obs − t_sw (the own-path switch on the actual v path), signed, in steps;
r_obs = s_obs/s_sw − 1.

| | statistic | PASS if |
|---|---|---|
| **C1** | \|t_fc − t_obs\| ≤ τ_cross (inclusive) | in ≥ 80% of the scored runs (a miss is not within) |
| **C2** | \|lag_fc − lag_obs\| ≤ τ_lag (inclusive) | in ≥ 80% of the scored runs (a miss is not within) |
| **C3** | median of r_obs/r_fc over the scored runs with a forecast and **\|lag_fc\| ≥ 6** | ∈ [0.9, 1.1] (closed) |
| **C4** | D = \|t_fc − t_obs\| − \|t_sw,fc − t_obs\|, paired, over the scored runs with a forecast | upper end of the 95% percentile bootstrap interval of mean D < 0 |
| **S** (arm S) | lag_fc < 0 and lag_obs < 0 | in ≥ 80% of the eligible scored runs: forecast and **\|lag_fc\| ≥ 6**, plus every miss (not both negative) |

- τ_cross, τ_lag per arm from the pilot: 1.5 × the pilot q90, rounded up to a multiple of 5 (§8).
- **C4:** 10,000 resamples from `numpy.random.default_rng(7430000)` (a local generator); 2.5th and 97.5th percentiles.
  An interval above 0 or containing 0 is FAIL. Fewer than 2 runs: UNRESOLVED.
- **Not computable:** C1/C2 with no scored run, C3 with no eligible finite ratio, S with no eligible run: UNRESOLVED.
- **Outcome per arm:** gate fails → "UNRESOLVED (gate)"; validity fails → "UNRESOLVED (validity)"; otherwise PASS iff
  every criterion passes (Q, T4: C1–C4; S: C1–C4 and S), UNRESOLVED if any is, otherwise FAIL naming each failure.
  Float64, no rounding.

**Validity** (all required; any failure makes every criterion UNRESOLVED):

| | condition (W2-A's V1–V7 with its change 3; over the scored runs, misses included) |
|---|---|
| V1 | ≥ 60 scored runs |
| V2 | t_sw strictly precedes the crossing in ≥ 90% of on-copy crossing runs with κ > 0 (none: holds) |
| V3 | η·λ_min(switch) ≤ 0.5 in ≥ 80% |
| V4 | median \|κ_k/(η·λ_min)\| ≥ 10 steps |
| V5 | q90 \|κ_kχ\| at t_sw ≤ 0.1 |
| V6 | \|median χ at t_sw / the arm's pilot median − 1\| ≤ 0.30 |
| V7 | q90 of max χ_t over 0 ≤ t < t_sw with s_t ≥ 0.8·s_switch ≤ 0.25 (χ_t with the reduced λ_min at z\*(v_t)) |
| follow | the follow check (below): a run that does not follow is not scored (counted; it reduces V1) |
| cutoff | t_c strictly before the crossing in ≥ 90% of the base (no cutoff = not before) |
| NaN | the NaN recomputation identical in EVERY run with a cutoff |

**Follow check** (partition kept). At t₀.₈ (the first step with s_t ≥ 0.8·the copy's frozen switch, where its follow
point is): Newton at v_{t₀.₈} from the run's z; the copy's point = Newton from the frozen follow point carried by one
tangent predictor at the release winding. Follows iff both are accepted, they agree within 1e−6 (sup), and the run's
Newton point keeps the copy's partition. One state, no gap; 0.8 < 0.95, so it reads a state before t_c.

## 7. Descriptive (never a verdict)

- **The random-hold collapse table** (T4's 710 random holds; §0), reported first.
- S on Q and on T4; L1–L5 on r_fc (W2-A's with the forecast in place of r_traj; L2 with r_cf; L4 against the
  population switch of the copy; L5 with \|t_fc − t_sw\| ≥ 6); the horizon t_obs − t_c; T4's criteria without the
  runs with G > 0 in the hold; counts of over-cap, aborted, not-following, late-cutoff and missed runs; κ₀ < 0 counts.

## 8. Pilot rules (pilot seeds only; before the registration)

- **ρ per arm** (the page's rule, W2-A's T rule from the page's ρ): run the on-copy pilot runs to the own-path switch
  only (no gap of a state after release); the value is V7's window max χ_t. Keep ρ if q90 ≤ 0.1, else halve and rerun,
  at most three halvings, then STOP. No value: STOP. Starts: Q 2⁻¹³, S 2⁻¹², T4 2⁻¹⁰ (the page's table, ‡).
- **Pilot seeds:** Q and S on 7,439,000–009 (branch-point starts, as W2-A piloted every arm on one range). T4 on
  7,439,100 onward, IN ORDER until 10 releases land on T4 (every drawn seed counted; at most 200).
- **τ and V6 at the chosen ρ:** each on-copy pilot run goes through the registered `run_one` (training to t_c, the
  forecast, the NaN check) and `observe_one`. τ_cross = ⌈1.5·q90 \|t_fc − t_obs\|⌉₅ and
  τ_lag = ⌈max(1.5·q90 \|lag_fc − lag_obs\|, 1)⌉₅ (Phase 1A's `_tolerances`). The pilot median χ at t_sw is frozen
  for V6.
- A STOP halts Phase 3 before registration and the author is asked.

## 9. Disclosures

- **A MECHANISM test (Q, S).** Q and S start from a branch point. Random holds do not reach a four-unit branch (§0).
- **The activity test was introduced after the 2A-PS post hoc diagnosis** (2f0fb33: a 1.2e−93 unit passed the rule
  v ≠ 0 and leaked the scale direction). θ = 1e−8 was set then (‡). It applies to Phase 3 only. At release v = v₀,
  whose smallest share is 0.1, so it cannot fire on a Phase 3 release; it is a guard, tested on constructed cases.
- **Set after exploratory data (‡ on the page):** the shares (0.1, 0.2, 0.3, 0.4); s₀ = 0.5·s_pop2; the ρ table;
  θ; the budget rule; the per-unit share extrapolation; the τ rule.
- **Q's C4: a KNOWN RISK, registered UNCHANGED (author, 2026-10-06).** In the exploration (8 seeds) Q's C4 interval
  was [−113, −6]. **On the pilot (9 Q runs at the registered ρ = 2⁻¹⁴ and τ) C4 FAILED: interval [−74.9, +29.0].**
  Q's crossing errors were all LATE (18–101 steps) while its lag errors were ≤ 4 steps; with lags of 41–213 steps (one
  −80) the no-lag forecast is often about as close. C4, f and every other criterion are registered unchanged; this
  result is recorded here before any registered run.
- **T4's hold length (D1, author's decision 2026-10-06).** W2-A's rule would give W = 4,000 steps for every T4 copy.
  The registration uses W = max(8000, ⌈25/λ⌉) for T4 ONLY. Both landing rates — **53/400 at 8,000 and 46/400 at
  4,000; gate power 0.955 vs 0.624 at the Wilson low** — were measured on the exploration seeds 7,410,000–019 BEFORE
  any T4 pilot or registered draw. The registered landing count is in §13.
- **Exploratory runs** (8 seeds per arm; the page's table): q90 crossing error Q 33, S 17, T4 7.6 steps; q90 window χ
  Q 0.108 (above 0.1 at 2⁻¹³), S 0.133 (at 2⁻¹²), T4 0.075; C4 intervals Q [−113, −6], S [−57, −25],
  T4 [−153, −126]. These are exploration seeds, never registered seeds.
- **The pilot** fixes ρ, τ and V6 only. Its C1–C4 at its own τ are reported as a disclosure (`pilot.json`).

## 10. Details the page left open

### D1 (DECIDED BY THE AUTHOR 2026-10-06: APPROVED; touches T4's gate): T4's hold length

- **The issue.** The page sets T4's gate (≥ 60 of 710; P = 0.95 at the Wilson low) on the exploration's landing rate
  53/400, measured with random holds of **W = 8000** steps (p3_own). W2-A's rule, W = max(4000, ⌈25/λ⌉), gives
  **W = 4000** for every T4 copy (λ ≈ 0.0097).
- **Measured on the exploration seeds 7,410,000–019** (the same 400 draws, the registered classifier; no registered or
  pilot seed): **53/400 land on T4 at W = 8000** (the exploration's count, reproduced exactly) and **46/400 at W = 4000**.
  P(≥ 60 of 710) at the Wilson low: **0.955 at 8000, 0.624 at 4000** (1.000 and 0.997 at the point estimates).
- **Resolution (APPROVED by the author 2026-10-06):** T4's hold W = max(**8000**, ⌈25/λ_min⌉) — the hold length on
  which the page's gate power rests (`W_FLOOR`; `test_hold_length_rule_and_the_proposed_t4_floor`). Q and S keep
  W2-A's rule (unchanged; their pilot and freeze were done before the decision).
- The rejected alternative: W2-A's rule as written (W = 4000; gate power ≈ 0.62 at the Wilson low).
- The decision was made before any T4 pilot or registered T4 seed was held or drawn; the T4 pilot and freeze ran
  after it.

### Implementation only

1. **The T4 pilot range** (7,439,100–299, drawn in order until 10 land on T4). "10 seeds per arm" read as 10 on-copy
   pilot runs for T4: at the exploration's landing rate (0.13), 10 pilot seeds give about 1.3 on-copy runs, too few
   for a q90. Q and S share 7,439,000–009 (W2-A precedent).
2. **Hold W per arm:** W2-A's rule at the arm's own copy for Q and S; T4 with the 8000 floor (D1; §3, §9).
3. **Holds at the freeze.** Every release (Q, S, T4) is computed and hashed at the freeze, so each gate is known
   before the registration (2A-PS froze its releases). The hold is release information only.
4. **T4 copies in full only for seeds released on T4.** The others never use the switch. Every T4 seed's own T4, Q and
   S points are frozen (the collapse table).
5. **The random draw for T4:** numpy `default_rng(seed).uniform(−1, 1, 9)` (the exploration's p3_own draw; a local
   generator). W2-A drew from a torch generator; the width-4 layout differs.
6. **Population copies** from the exploratory class points at 6 decimals (Q: box-3 class 49; S: box-3 class 39; T4:
   box-1 class 2), checked as in §3.
7. **Memory:** the branch along a v path is computed in one pass with O(1) storage (`StreamBranch`; W2-A's semantics,
   tested equal at K = 2), so a 10⁶-step run fits the 1 GB cap. The v path and the z path to t_c are kept in memory
   only; their hashes are stored (no path files).
8. **Observation stops** once the crossing, t_c and the actual own-path switch are all decided (or at B). No recorded
   quantity depends on later steps.
9. **The per-run RSS cap:** the process's current RSS checked every 20,000 steps; a run that exceeds 1 GB is stopped,
   counted ("aborted: per-run RSS cap") and, like an over-cap seed, not scored. Peak RSS in every exploratory and test
   run was ≤ 0.36 GB.
10. **The C4 bootstrap generator:** `default_rng(7430000)`, the first registered seed (as W2-A, 1C, 2A-PS).
11. **τ's rounding:** Phase 1A's `_tolerances` (τ_lag at least 5).
12. **The κ₀-sign check of the landscape** (Q > 0, S < 0, T4 > 0, the page's signs): a reproduction check only.
13. **Machine rules:** one process, nice 15, one thread; the memory gate (free ≥ 25%, swap free ≥ 500 MB; waits) and the
    disk check (≥ 20 GB, else STOP) as a separate logged step before every job and inside each job between seeds
    (`results/phase3/memory_gate.log`); stop above 3 GB peak RSS.
14. **Ledger:** every Phase 3 artifact is a producer in `verify_ledger.PRODUCERS`.

## 11. Competing outcomes

| outcome | reading |
|---|---|
| Q PASS | A four-unit branch's signed lag is forecast causally, once the run is on that branch (a mechanism test). |
| Q FAIL with T4 PASS | The law holds only on embedded width-2 branches. |
| S PASS | The sign of the lag (a crossing before the switch) is forecast. |
| C4 FAIL (falsifier) | The forecast is no better than the no-lag forecast. |
| C1 fails, C2 passes | The switch is mis-extrapolated, not the lag. |
| C3 fails | The scale of the lag is off by more than 10%. |
| UNRESOLVED (gate) | The start does not reach the copy (Q, S: < 96/120 or a hold with G > 0; T4: < 60 of 710). |
| UNRESOLVED (validity) | Too few scored runs, cutoffs too late, V1–V7, or a forecast that read past its cutoff. |

A failure stays a failure. Post hoc readings may be placed beside it, labelled POST HOC.

## 12. Frozen files and hashes

`results/phase3/registration.sha256` (`phase3_w4.manifest`) lists: this file and the approved page; the import
closure of `src/phase3_w4.py`; `tests/test_phase3_w4.py` and `tests/test_causal_forecast.py`;
`results/phase3/{seed_scan.json, landscape.json, pilot_frozen_parts.jsonl, pilot_parts.jsonl, pilot.json,
frozen_parts.jsonl, frozen.json}`; `results/asym_scores.json`; W2-A's `landscape.json` (the K = 2 test); and the
exploration's `w4core.py`, `p3_activity.py`, `test_p3_activity.py`, `p3_budget.py`, `p3_runs.py`, `p3_runs.jsonl`,
`p3_own.py` and README. `run`, `observe` and `score` assert every hash, that the manifest and the stamp are
committed and unmodified, and that the OpenTimestamps proof exists.

## 13. Scan, landscape, pilot and freeze (2026-10-06; no registered seed trained)

**Scan** (`seed_scan.json`): every Phase 3 range unused (no hit for 7430, 7431, 7432, 7439; 61 SEEDS\*/PILOT\_SEEDS\*
constants, no overlap; disjoint from the exploration).

**Landscape** (`landscape.json`, population): Q 1111:++-- switch 0.482210, κ₀ +0.01651; S 121:+-+ switch 0.439780,
κ₀ −0.01201; T4 31:+- switch 0.761909, κ₀ +0.2470. All validated, within 0.5% of the page, κ₀ signs as the page.

**Pilot, Q and S** (`pilot.json`, `pilot_parts.jsonl`, `pilot_frozen_parts.jsonl`; seeds 7,439,000–009):

| arm | on copy | q90 window χ (ρ) | ρ | τ_cross (q90 \|e_cross\|) | τ_lag (q90 \|e_lag\|) | V6 median χ | disclosure at the pilot's τ |
|---|---|---|---|---|---|---|---|
| Q | 9/10 (7,439,001: no own Q point) | 0.117 (2⁻¹³), 0.059 (2⁻¹⁴) | 2⁻¹⁴ | 125 (82.6) | 5 (2.4) | 0.0311 | C1–C3 PASS, **C4 FAIL [−74.9, +29.0]**, S 1/9 |
| S | 10/10 | 0.139 (2⁻¹²), 0.070 (2⁻¹³) | 2⁻¹³ | 50 (33.1) | 5 (1.0) | 0.0449 | C1–C4 PASS ([−71.7, −31.0]), S 10/10 |

- Q's crossing errors are all LATE (18–101 steps) while its lag errors are ≤ 4: the extrapolated switch, not the lag,
  carries the error. With Q's lags of 41–213 steps (one −80), the no-lag forecast is often as close, so **C4 fails on
  the pilot**. This is the page's "Q's C4 risk", now seen on pilot seeds at the registered ρ. Every pilot run: the
  cutoff before the crossing, the NaN recomputation identical, the follow check passed.
- T4: not piloted (D1).

**Freeze, Q and S** (`frozen_parts.jsonl`; no training; 407 s and 389 s):

| arm | seeds | own point / full valid | on copy at release | hold G > 0 | gate | κ₀ < 0 | windings | over cap | B (min / median / q90 / max) |
|---|---|---|---|---|---|---|---|---|---|
| Q | 120 | 115 / 112 | 115 | 0 | **PASS** (≥ 96) | 18/112 | all (0,0,0,0) | **0** | 44,030 / 198,853 / 348,694 / 441,271 |
| S | 120 | 120 / 119 | 120 | 0 | **PASS** (≥ 96) | 114/119 | all (0,0,0,0) | **0** | 30,522 / 73,960 / 119,989 / 150,664 |

- Invalid copies: Q 5 without an accepted own point of the type and partition, 3 without a validated switch; S 1
  without a validated switch. Counted, never scored.
- **The author's budget condition: PASS** (0 over-cap seeds in Q and in S; ≤ 24 allowed). t\*: Q 27,353–292,180,
  S 18,348–98,443.
- W: Q 4000–27,324, S 4000–12,648 (W2-A's rule).

## 14. Compute and machine rules

- The page's estimate: about 13 h of training and observation in jobs of at most 4 h (Q 6 h, S 2.5 h, T4 1.5 h), plus
  the freeze and the pilot.
- One process, nice 15, one thread; the memory gate before every job and between seeds; disk ≥ 20 GB; peak RSS ≤ 3 GB;
  per-run caps 10⁶ steps and 1 GB.
- **Process note (a lapse, recorded):** before the S freeze the separate memory gate was CHAINED in the same background
  command as the job (after the Q freeze), not run as its own step. It passed (logged), and the freeze also gated
  before every seed inside the job.

## 15. Reproduce

```
python -m src.phase3_w4 gate scan;      python -m src.phase3_w4 scan
python -m src.phase3_w4 gate landscape; python -m src.phase3_w4 landscape
python -m src.phase3_w4 gate pilot;     python -m src.phase3_w4 pilot
python -m src.phase3_w4 gate freeze;    python -m src.phase3_w4 freeze Q|S|T4; python -m src.phase3_w4 summary
python -m src.phase3_w4 manifest        # the registration commit; then: stamp, commit, push, OpenTimestamps
python -m src.phase3_w4 run Q|S|T4; python -m src.phase3_w4 finalize     # commit runs_*.jsonl + forecasts.sha256
python -m src.phase3_w4 observe Q|S|T4; python -m src.phase3_w4 score
```
