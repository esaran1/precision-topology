# Prediction-inputs audit: W2-A, GELU-T, Test 2A, Track A (a = 1.65)

**AUDIT, 2026-09-30, requested by the author.** The question is what information the registered trajectory-integrated
prediction (r_traj) and closed-form prediction (r_cf) use. The prompt was that L1 and L3 came out at 1.000 in W2-A
(and 1.000 / 0.999 in GELU-T). **Nothing registered was changed.** No code, frozen file, prediction, observation,
verdict or writer input was touched: every registered predict function was imported and called unchanged.

- Producer: `src/prediction_inputs_audit.py`.
- Counts: `results/prediction_inputs_audit.json`. Every number below is there; the ledger regenerates the file from
  the committed CSVs and replay rows and requires equality.
- Replay rows: `results/prediction_inputs_audit/replay_{w2a,gelu,track_a,track2a}.jsonl` (MASKED and CAUSAL) and
  `results/prediction_inputs_audit/truncated_{w2a,gelu,track_a,track2a}.jsonl` (TRUNCATED).
- Tests: `tests/test_prediction_inputs_audit.py`.
- Ledger: `verify_ledger.prediction_inputs_audit_checks`.

The CAUSAL replay (§8) is a **POST HOC diagnostic**. It replaces no registered prediction or verdict.

**Method: four lines of evidence**
1. **Code.** Every input of every prediction is listed with the `file:line` that reads it (§3–§5). Every `file.py:N`
   cited in this file is checked by `code_refs_check` and the ledger to still contain the cited code.
2. **Committed data.** For each scored run, the committed CSVs (`observed_runs*.csv`) give the steps each prediction
   reads (t_traj, t_sw) and the observed crossing step t_obs. Track A's t_top is recomputed from the saved path.
3. **MASKED replay.** Each path's SHA-256 was asserted against its committed row before use; every path matched.
   - The registered predict function is called on the run's own path with every hidden coordinate after release set
     to NaN (Track A and 2A: every step except the rule point). For W2-A and GELU-T one step is kept: the follow-check
     step t₀.₈, whose hidden state the registered follow check reads (a scored-set input, §1).
   - Adam's moments are NaN except the rows the registration names (t_R; for 2A also t_sw).
   - **Evidence.** A prediction that reproduces its committed value under MASKED used no hidden-parameter state after
     release or the rule point, and no gap of such a state (NaN would propagate into it). A comparison with NaN
     evaluates False rather than propagating, so the replay backs the code listing (1) rather than replacing it; the
     listing shows no such comparison on a masked row.
   - **Adam moments.** Track A and 2A save θ only, so the moments were rebuilt by replaying the registered Adam run
     with a local torch Generator. The replayed θ equals the saved path **bit for bit** in 74/74 (Track A) and 75/75
     (2A) runs.
4. **TRUNCATED replay.** As MASKED, and in addition **every row after the claimed last-read step L** (hidden and
   output) set to NaN; L_traj for r_traj, L_cf for r_cf (§7). A prediction that reproduces reads nothing after L. The
   code reads step L itself (s_traj = s[t_traj], s[t_sw] in ṡ, s[t_top] in the rule point). So L is the last step
   read, and the counts of steps read at or after the crossing below are **not undercounts**.

**Definitions**
- A prediction **reads step t** if its value can depend on the output weights (or the state) at step t. Whole-path
  scans whose value cannot depend on later steps (the rule point's search for t_top, GELU-T's and Track A's segment
  end T_end) are listed, but they read only up to the step their value depends on (TRUNCATED confirms this).
- The recursion stops at its predicted crossing t_traj (`linear_response.py:333`, `width2_asym.py:793`). So r_traj
  reads the output path from 0 (or t_R) **through t_traj**, plus whatever its other inputs read (s_switch for W2-A T
  and T′; t_R for Track A and 2A).
- **"At the crossing"** means step t_obs itself. **"After"** means t > t_obs. The output weights at step t_obs are
  computed from the state at t_obs − 1; condition (iii) as posed counts them as read at the crossing.
- Counts are over the **scored** runs of each arm or optimiser. Track A has no `scored` column; its set is rebuilt as
  registered (crossed, rule point strictly before the crossing, finite r_traj and r_obs): 74 Adam and 63 SGD runs,
  equal to `scores.json`.

---

## 1. Findings in brief

Conditions: **(i)** no hidden-parameter state after the rule point / release is used; **(ii)** no gap evaluation of
any training state is used; **(iii)** nothing read at or after the observed crossing is used. For (iii) the count is
over **all** inputs of the prediction (the "last read" is the latest step any input reads).

| Test / arm | Prediction | (i) | (ii) | (iii): runs reading step ≥ t_obs (at / after; max, median steps after among "after") |
|---|---|---|---|---|
| **W2-A T** (98) | r_traj | **confirmed** (MASKED 98/98) | **confirmed** | **VIOLATED in 98/98**: 95 at, 3 after (max 3, median 1). The recursion alone: 97 at, 1 after (1 step). The own-path switch (s_switch) adds 2 runs, read 1 and 3 steps after. |
| | r_cf | confirmed | confirmed | **VIOLATED in 4/98**: t_sw = t_obs in 2, after in 2 (1 and 3 steps). 94/98 clean. |
| **W2-A D** (120) | r_traj | confirmed (120/120) | confirmed | **confirmed.** Last step read 1–6 before t_obs (median 4). |
| | r_cf | confirmed | confirmed | **confirmed.** t_sw 161–289 steps before t_obs. |
| **W2-A T′** (120) | r_traj | confirmed (120/120) | confirmed | **VIOLATED in 120/120**, all after: 10–83 steps (median 43.5). The recursion reads v through t_obs (118) or t_obs − 1 (2); its s_switch is the lag-free switch on the run's own v path, which lies **after** the crossing in 120/120. |
| | r_cf | confirmed | confirmed | **VIOLATED in 120/120**: t_sw 10–83 after (median 43.5); the ṡ window straddles the crossing in 120. |
| **GELU-T random** (80) | r_traj | **confirmed** (MASKED 80/80) | confirmed | **VIOLATED in 80/80**: 36 at, 44 after (max 191, median 1). |
| | r_cf | confirmed | confirmed | **VIOLATED in 13/80**: t_sw after t_obs in all 8 κ < 0 runs (20–310 steps, median 84) and in 5 κ > 0 runs (≤ 11). The ṡ window lies entirely after the crossing in 4, straddles it in 9. |
| **GELU-T branch** (80) | r_traj | **confirmed** (MASKED 80/80) | confirmed | **VIOLATED in 80/80**: 42 at, 38 after (max 19, median 1). |
| | r_cf | confirmed | confirmed | **VIOLATED in 7/80**: both κ < 0 runs (26 and 60 after) and 5 κ > 0 runs (≤ 13); the window straddles in 7. |
| **Test 2A** (75) | r_traj, r_cf | **VIOLATED in 75/75 through the optimiser state (by design).** P = P(t_sw) is Adam's bias-corrected second moment of the hidden coordinates at t_sw, 30–488 steps after t_R (median 164), before the crossing in 75/75. No hidden-parameter **value** after t_R is read (MASKED 75/75). | confirmed | r_traj: **VIOLATED in 10/75**: 2 at, 8 after (max 53, median 1). The recursion alone: 3 at, 7 after (1–4); the rule-point definition reads s up to t_top ≥ t_obs in 1 (53 after). r_cf: **VIOLATED in 1/75** (the same t_top read, 53 after); its t_sw is 8–122 before. |
| **Track A Adam** (74) | r_traj | confirmed: the rule-point θ, m and v̂ only (MASKED 74/74) | confirmed | **VIOLATED in 34/74**: 1 at, 33 after (max 140, median 19). The recursion alone: 1 at, 32 after (max 66, median 16); t_top ≥ t_obs in 4 (max 140). |
| | r_cf | confirmed | confirmed | **VIOLATED in 4/74**, through the rule point only (t_top 4/74, max 140 after; t_R → P_R and s\*_run). t_sw and the ṡ window: 18–123 before in 74/74. |
| **Track A SGD** (63) | r_traj | confirmed (63/63) | confirmed | **VIOLATED in 5/63**: 1 at, 4 after (max 181). The recursion alone: 1 at, 0 after; t_top ≥ t_obs in 4 (max 181). |
| | r_cf | confirmed | confirmed | **VIOLATED in 4/63**, through the rule point only (t_top, max 181 after). t_sw 61–93 before in 63/63. |

**Scored-set membership (not a prediction input).** In W2-A and GELU-T the registered follow check reads the run's
**hidden state at t₀.₈** (`width2_asym.py:1324`, `gelu_transfer.py:829`). That is a hidden-parameter state after
release.
- It was before the crossing in every scored run. The margins before t_obs are 3,032–7,594 steps (T), 968–2,167 (D),
  7,821–12,603 (T′), 1,378–4,266 (GELU-T random) and 1,482–4,266 (GELU-T branch).
- It excluded **0** on-branch runs in every arm.
- So condition (i) holds for the prediction **values**. It does not hold for the rule that decides which runs are
  scored (it reads one hidden state after release, before the crossing, and excluded no run).
- Track A's scored set requires t_R < t_obs, which uses t_obs; that is a scoring rule, not a prediction input.

**Gaps (condition ii).** No prediction evaluates the gap of any training state.
- The only gaps evaluated in r_traj, t_sw and s_switch are those of **predicted** states z\*(v_t) + δ_t
  (`width2_asym.py:792`, `gelu_transfer.py:873`, `track_a.py:336`) and of **branch points** z\*(v_t)
  (`width2_asym.py:760`, `width2_asym.py:767`). Neither is a state of the run. MASKED would turn any gap of a
  post-release training state into NaN.
- The hold (before release, v fixed) evaluates the gap of every hold state (`width2_asym.py:1261`,
  `gelu_transfer.py:768`). This enters the gate (W2-A D and T′; GELU-T branch) and the descriptive sensitivity
  analyses only. Hold G > 0 occurred in 1 run of T, 0 of D, 0 of T′, 6 of GELU-T random and 0 of GELU-T branch.

**The four items asked**
- **(a) r_traj's recursion reads s_t / v_t at or beyond t_obs** (recursion alone, scored runs): W2-A T 97 at / 1 after
  (1 step); D 0 / 0; T′ 118 at / 0 after; GELU-T random 36 / 44 (max 191, median 1); branch 42 / 38 (max 19,
  median 1); Test 2A 3 / 7 (max 4, median 1); Track A Adam 1 / 32 (max 66, median 16); SGD 1 / 0. With all of r_traj's
  inputs: the table above.
- **(b) The lag-free switch / t_sw after t_obs, κ < 0 runs.** W2-A T′ 120/120 (10–83 steps after, median 43.5); W2-A T
  has 1 scored κ < 0 run (seed 884149), switch 3 steps after. GELU-T random 8/8 κ < 0 runs (20–310, median 84), branch
  2/2 (26 and 60). In GELU-T, t_sw enters r_cf and V7 only (r_traj's s_switch is frozen). In W2-A T and T′ it also
  enters r_traj, r_obs, L4/L5's comparators and V7, through s_switch.
- **(c) r_cf's ṡ window and χ at t_sw.** χ = (ṡ/s_switch)/(η·λ_min) with ṡ = (s[t_sw] − s[t_sw − w])/w, w = min(100,
  t_sw) (`width2_asym.py:469`, `gelu_transfer.py:185`, `track_a.py:322`). λ_min and κ are frozen (W2-A: κ₀ + a·k with
  k fixed at release). So in every run with t_sw ≥ t_obs, **χ at t_sw uses post-crossing output weights**: s[t_sw]
  always, the whole window when t_sw − 100 ≥ t_obs, and in W2-A T and T′ also s_switch (bisection on v_{t_sw−1} →
  v_{t_sw}). Counts: W2-A T 4 (window straddles 4, entirely after 0), D 0, T′ 120 (straddles 120, entirely after 0);
  GELU-T random 13 (straddles 9, entirely after 4), branch 7 (straddles 7); Test 2A 0 (t_sw), but 1 through the rule point;
  Track A 0 (t_sw), but 4 + 4 through the rule point.
- **(d)** The output path after the crossing belongs to the same coupled run (§6).

## 2. Why L1 = 1.000 and L3 ≈ 1.000 (W2-A T and T′; GELU-T)

r_traj and r_obs are two readouts of the **same** s path, divided by the **same** s_switch:

- r_traj = s_{t_traj}/s_switch − 1 (`width2_asym.py:1396`, `gelu_transfer.py:882`);
- r_obs = s_{t_obs}/s_switch − 1 (`width2_asym.py:1644`, `gelu_transfer.py:1070`).

So **whenever t_traj = t_obs, r_traj ≡ r_obs**: the same array entry, not two close numbers. (In the CSVs the two r
columns agree to ≤ 6.8e−12 relative in those runs; the residue is the 16-decimal rounding of r values near 1e−5.)

| arm | scored | t_traj = t_obs (s_traj ≡ s_obs) | other runs: r_obs/r_traj < 1 | > 1 | observed lag t_obs − t_sw (steps): min / median / max | registered L1 median, L3 |
|---|---|---|---|---|---|---|
| W2-A T | 98 | **97** | 1 | 0 | −3 / 24 / 75 | 1.000000000000049, 0.99992 |
| W2-A D | 120 | 0 | 0 | 120 | 161 / 206 / 289 | 1.0210, 0.9973 |
| W2-A T′ | 120 | **118** | 2 | 0 | −83 / −43.5 / −10 | 1.0000000000000038, 0.99999 |
| GELU-T random | 80 | 36 | 34 | 10 | −310 / 39 / 61 | 1.0000000000000087, 0.9986 |
| GELU-T branch | 80 | 42 | 33 | 5 | −60 / 41.5 / 61 | 1.000000000000006, 0.9988 |
| Track A Adam | 74 | 1 | 32 | 41 | — | 1.0651, 0.2485 |
| Track A SGD | 63 | 1 | 0 | 62 | — | 1.0555, 0.9945 |
| Test 2A | 75 | 3 | 7 | 65 | — | A3 1.0579, A1 0.9322 |

- **Why the median is exactly 1.** In T and T′ more than half the scored runs have t_traj = t_obs. In GELU-T the
  middle of the sorted ratios falls inside the identity block (random: 34 below, 36 at 1, 10 above).
- **Why L3 ≈ 1.** The same identity gives Spearman ≈ 1: most pairs are equal.
- **What the prediction actually supplies.** The substantive content of r_traj is the **step** t_traj. The recursion
  landed on the observed step exactly in 97/98 (T) and 118/120 (T′) runs, where the observed lag is 24 and −43.5 steps
  (medians). The scale is then read off the run's own s path.
- **What drives the step.** The recursion (`width2_asym.py:798`) is the first-order (in δ) form of the hidden update
  that training performs (`width2_asym.py:1248`). It runs about the run's own branch z\*(v_t) and is driven by the
  run's own v path. That is how the registration defines it (§5: "Track A's R4 recursion", "on the run's own v path";
  §1 C: "T's and T′'s predictions are conditional on that path"). It is not a gap of the training state.
- **The two detectors differ.** The predicted crossing is the first t with the enclosure **midpoint** of the
  predicted state > 0 (`width2_asym.py:293`); the observed crossing is the first t with the enclosure **lower end** of
  the actual state > 0 (`width2_asym.py:284`, `width2_asym.py:1580`). In GELU-T the prediction uses the exact-extrema
  midpoint (`gelu_transfer.py:873`) and the observation `phase2b_ordering.state`'s placement (`gelu_transfer.py:1035`).
- **Is the landing on t_obs a read of the crossing?** Not directly: the recursion never evaluates the actual state
  after release (MASKED). The CAUSAL diagnostic (§8), which removes every output weight at or after t_obs, leaves
  t_traj unchanged in **every** W2-A run and in 160/160 GELU-T runs. But the v path before t_obs is itself a function of
  the hidden trajectory before the crossing (§6), so this is path-conditioned tracking, not a forecast from release.
- **What L1 and L3 therefore measure.** In these arms, L1 and L3 at 1.000 measure whether the linearised hidden
  dynamics, driven by the run's own output path, reproduce the crossing step. They cannot be read as an independent
  forecast of the crossing scale. Conclusions beyond that are for the author.

## 3. W2-A: every input, by prediction

Code: `src/width2_asym.py` (registration 28b2432; predictions 2c630f2). `predict_one` starts at `width2_asym.py:1350`.

**r_traj (primary; L1, L3, L4, L5)**

| input | class | read at |
|---|---|---|
| the seed's own training sample (x, y) | frozen before training | `width2_asym.py:1283` |
| occupied copy c and winding k, from the release state (Newton at v₀) | release state | `width2_asym.py:1288`, `width2_asym.py:1292` |
| frozen copy point z_c(s₀), shifted to k and anchored at v₀ | frozen | `width2_asym.py:1367` |
| η of the arm | frozen | `width2_asym.py:1358` |
| v path V = P[:, VI], s = ‖v‖₁ | output path | `width2_asym.py:1359`, `width2_asym.py:1360` |
| own branch z\*(v_t) by warm-started Newton at every v_t, t = 0 … max(t_traj, t_sw) | output path | `width2_asym.py:738`, `width2_asym.py:739` |
| δ₀ = z_release − z\*(v₀), with z_release = P[0][ZI] | release state | `width2_asym.py:1390`, `width2_asym.py:783` |
| recursion δ_{t+1} = (I − ηH_t)δ_t − (z\*(v_{t+1}) − z\*(v_t)), t < t_traj | output path | `width2_asym.py:798` |
| hit test: enclosure midpoint of the **predicted** state z\*(v_t) + δ_t at v_t; returns at the hit | gap of a non-training point | `width2_asym.py:792`, `width2_asym.py:793` |
| s_traj = s[t_traj] | output path, at step t_traj | `width2_asym.py:1396` |
| s_switch (denominator): T, T′ = own-path switch (below); D = frozen | output path (T, T′) / frozen (D) | `width2_asym.py:1370` / `width2_asym.py:1373` |
| hidden coordinates P[t][ZI], t ≥ 1 | **not read** (except t₀.₈, follow check, `width2_asym.py:1324`) | MASKED 98/98, 120/120, 120/120 |

**t_sw and s_switch for T and T′ (change C, "lag-free switch on the run's own v path")**

| input | class | read at |
|---|---|---|
| own branch z\*(v_t), t = 1 … t_sw | output path | `width2_asym.py:757`, `width2_asym.py:760` |
| placement of the **branch point** z\*(v_t) (enclosure midpoint) | gap of a non-training point | `width2_asym.py:760` |
| bisection on v_{t_sw−1} → v_{t_sw} (branch-gap root); returns at t_sw | output path, gap of branch points | `width2_asym.py:761`, `width2_asym.py:767`, `width2_asym.py:772` |

**r_cf (L2)**: κ_k = κ₀ + a·k (frozen, `width2_asym.py:1361`); λ_min at the copy's frozen switch
(`width2_asym.py:1362`); η; s_switch and t_sw as above; ṡ = (s[t_sw] − s[t_sw − w])/w with w = min(100, t_sw)
(`width2_asym.py:469`, `width2_asym.py:1379`).

**Also read (not a prediction input)**
- Follow check: the hidden state at t₀.₈ (`width2_asym.py:1324`).
- V7: χ_t on 0 … t_sw (`width2_asym.py:1381`, `width2_asym.py:1383`).
- `s_min_after_release`: the whole path (`width2_asym.py:1365`; descriptive only, used by no verdict).

**Last step read vs the crossing, per arm (scored runs)**

| arm | r_traj recursion: at / after (max) | own-path switch t_sw ≥ t_obs: at / after (range after) | r_traj, all inputs (max(t_traj, t_sw)): at / after (max) | r_cf window: ends at/after; entirely after |
|---|---|---|---|---|
| T | 97 / 1 (1) | 2 / 2 (1–3) | 95 / 3 (3) | 4; 0 |
| D | 0 / 0 (last read 1–6 before) | — (frozen switch; t_sw 161–289 before) | 0 / 0 | 0; 0 |
| T′ | 118 / 0 (2 read to t_obs − 1) | 0 / **120** (10–83, median 43.5) | 0 / **120** (83) | 120; 0 |

- **Cross-checks.** `branch_steps_computed − 1 = max(t_traj, t_sw)` in 98/98, 120/120 and 120/120 runs (read from
  predictions.csv). TRUNCATED at L_traj = max(t_traj, t_sw) reproduces r_traj in 98 (T), 120 (D) and
  120 (T′) runs (§7).
- **(b) T′.** The negative lag puts the lag-free switch after the crossing in **120/120** runs.
  - Steps read after t_obs: **10–83, median 43.5**. v_{t_obs} … v_{t_sw} are read by `OwnBranch.ensure` and the
    segment bisection.
  - Through s_switch, these steps enter r_traj, r_obs, r_cf, L4/L5's comparators and V7's window.
  - The registration discloses that the predictions are "conditional on that path" (§1 C). It does not say that for
    T′ the switch lies after the crossing.
  - T has 1 scored run with κ < 0 (seed 884149). Its switch is at t_obs + 3, so its prediction reads 3 steps after the
    crossing.
  - Existence of a prediction for T′ also requires the own branch to survive to t_sw, which is after the crossing. The
    branch was lost in 0 scored runs.

## 4. GELU-T: every input (`src/gelu_transfer.py`; registration 736b6bf, predictions 8d39e75)

`predict_one` starts at `gelu_transfer.py:838`.

**r_traj**

| input | class | read at |
|---|---|---|
| copy at release | release state | `gelu_transfer.py:814` |
| frozen s_switch, κ, λ_min(s_switch) | frozen | `gelu_transfer.py:848` |
| static own-sample branch grid θ\*(s), H(s) from the frozen copy point at s₀ (no path input) | frozen before training | `gelu_transfer.py:854` |
| s = \|w₂\| path | output path | `gelu_transfer.py:846` |
| z_rel = Wp[0, [0, 1, 3]] (δ₀) | release state | `gelu_transfer.py:864` |
| segment end T_end = first s_t outside the grid (loop bound) | output path | `gelu_transfer.py:865`, `gelu_transfer.py:866` |
| θ\*(s_t), H(s_t) for t < T_end | output path | `gelu_transfer.py:868`, `gelu_transfer.py:869` |
| recursion, stopping at the first hit | output path | `linear_response.py:317`, `linear_response.py:332`, `linear_response.py:333` |
| hit test: exact-extrema midpoint of the **predicted** state | gap of a non-training point | `gelu_transfer.py:873` |
| stability radius over recursion indices ≤ t_traj | output path | `gelu_transfer.py:876` |
| s_traj = s[t_traj] | output path, at t_traj | `gelu_transfer.py:882` |
| hidden coordinates Wp[t, [0, 1, 3]], t ≥ 1 | **not read** (except t₀.₈, `gelu_transfer.py:829`) | MASKED 80/80, 80/80 |

**r_cf**
- κ, λ_min: frozen (`gelu_transfer.py:848`).
- t_sw = first s_t ≥ frozen s_switch (`gelu_transfer.py:849`).
- ṡ window (`gelu_transfer.py:184`, `gelu_transfer.py:185`, `gelu_transfer.py:850`).

**Other reads**
- Follow check: the hidden state at t₀.₈ (`gelu_transfer.py:829`).
- V7: χ_t on 0 … t_sw (`gelu_transfer.py:858`).
- `w2_min_after_release`: the whole path (`gelu_transfer.py:847`; descriptive only).
- Hold gaps: `gelu_transfer.py:768` (gate).

**About T_end.** `th_seg` and `Hs` are materialised for every step before T_end. T_end is ≥ 12,894 in every scored
run, far past t_obs. The **value** depends only on s₀ … s_{t_traj}: `simulate` returns at the hit and the radius check
uses recursion indices ≤ t_traj (`gelu_transfer.py:876`). TRUNCATED at L_traj = t_traj (so T_end = t_traj + 1)
reproduces t_traj, s_traj and r_traj in 80 (random) and 80 (branch) runs.

| arm | r_traj: at / after (max; median among after) | (s_traj − s_obs)/s_switch among "after": median / max | r_cf t_sw > t_obs: κ < 0 / κ > 0 (max after) | window entirely after |
|---|---|---|---|---|
| random | 36 / **44** (191; 1) | 9.8e−5 / 0.0188 | **8/8** / 5 (310) | 4 |
| branch | 42 / **38** (19; 1) | 9.0e−5 / 0.0024 | **2/2** / 5 (60) | 0 |

- t_traj ≥ t_obs in **160/160** GELU-T runs.
- **(b) κ < 0 runs.** r_traj does not use t_sw in GELU-T; its s_switch is frozen. r_cf does: its ṡ window ends at the
  frozen-switch step, which lies after the crossing in all 10 κ < 0 runs (20–310 steps after). In these runs t_sw also
  lies after t_traj (14–119 steps, random; 19–41, branch).

## 5. Test 2A and Track A (`src/track_a.py`, unchanged in 2A; `src/track2a.py`)

`predict_one` starts at `track_a.py:293`; 2A calls it with v̂(t_sw) in row t_R (`track2a.py:119`, `track2a.py:225`).

| input | class | read at |
|---|---|---|
| landscape κ inputs (H_pop, θ\*′, ∇G at s\*_pop) | frozen | `track_a.py:294`, `track_a.py:323` |
| rule point t_R: last upward passage of 0.5·s_frozen **before t_top** (t_top = first s_t ≥ s_frozen), which reads s₁ … s_{t_top} | output path | `track_a.py:96`, `track_a.py:102`, `track_a.py:302` |
| θ_{t_R} → Newton → occupied branch, s\*_run (static grid, no path input) | rule-point state | `track_a.py:300`, `track_a.py:306`, `track_a.py:309`, `track_a.py:310` |
| P_R = 1/(√v̂_{t_R} + ε) (Track A, Adam) | rule-point (optimiser) state | `track_a.py:315` |
| **P = 1/(√v̂_{t_sw} + ε) (2A: v̂ at t_sw written into row t_R)** | **optimiser state after the rule point** | `track2a.py:122`, `track2a.py:225` |
| t_sw = first s_t ≥ s\*_run; ṡ window | output path | `track_a.py:318`, `track_a.py:322`; `track2a.py:218` |
| δ₀ = θ_{t_R} − θ\*(s_{t_R}); m₀ = Adam m at t_R; bias correction from the step count K | rule-point state; K is not a state | `track_a.py:332`, `track_a.py:335`, `track_a.py:337` |
| s path t_R … t_traj (θ\*(s_t), H(s_t)); P frozen | output path | `track_a.py:326`, `track_a.py:331`, `track_a.py:334` |
| hit test: gap of the **predicted** state | gap of a non-training point | `track_a.py:336`; `linear_response.py:332` |
| s_traj = s[t_R + t_hit] | output path | `track_a.py:346` |
| θ at any t ≠ t_R | **not read** (MASKED 74/74, 63/63, 75/75) | — |

| | r_traj recursion: at / after (max; median) | t_top ≥ t_obs (max after) | r_traj, all inputs: at / after (max) | r_cf, all inputs: after (max) | t_sw vs t_obs | t_R vs t_obs |
|---|---|---|---|---|---|---|
| Track A Adam (74) | 1 / **32** (66; 16) | 4 (140) | 1 / **33** (140) | 4 (140) | all before (18–123) | all before (78–744) |
| Track A SGD (63) | 1 / 0 | 4 (181) | 1 / **4** (181) | 4 (181) | all before (61–93) | all before (205–647) |
| Test 2A (75) | 3 / **7** (4; 1) | 1 (53) | 2 / **8** (53) | 1 (53) | all before (8–122) | all before (41–604) |

- **Test 2A, condition (i).** Its one registered change freezes P at t_sw, which is after the rule point in **75/75**
  runs (30–488 steps; median 164). v̂ is the bias-corrected EMA of the squared gradients of the hidden coordinates
  over steps 1 … t_sw. It is a function of the hidden-parameter trajectory up to t_sw, though not a parameter value.
- **The rule point.** Its search reads s up to t_top. In 4 + 4 Track A runs and 1 2A run t_top lies after the
  crossing, so identifying t_R reads post-crossing output weights. The value of t_R is unchanged when every output
  weight at or after t_obs is replaced (§8: 74/74, 63/63, 75/75).

## 6. (d) The output path belongs to the same coupled run, before and after the crossing

In every test the output weights are trained **jointly** with the hidden weights by one gradient step on the full
parameter vector:
- W2-A: `width2_asym.py:1248`, `q = q − lr·∇L(q)`, with ρη on v;
- GELU-T: one torch SGD optimiser over (w₁, b₁, w₂, b₂), `gelu_transfer.py:783` to `gelu_transfer.py:791`;
- Track A and 2A: one torch optimiser, `track_a.py:282` to `track_a.py:286`.

So v_{t+1} (w₂ at t + 1) is computed from the gradient at the full state (z_t, v_t).
- **After the crossing.** Every output weight after the crossing step is a function of the hidden state at and after
  the crossing. Whatever the placed hidden state does to ∇_vL enters the output path from step t_obs + 1 on. Output
  weights **at** t_obs are computed from the state at t_obs − 1, before the crossing.
- **Before the crossing.** The same holds at every step: v_1 … v_{t_obs} are functions of the hidden states
  z_0 … z_{t_obs − 1}, which are after release. Condition (i) is confirmed in the sense audited here: no hidden
  coordinate after release (Track A, 2A: after t_R) is read directly (MASKED). It is **not** the case that the
  predictions are independent of the post-release hidden trajectory: they read it through the output path, as the
  registration's "conditional on that path" states.

This is what the code does. The size of the post-crossing part of the effect on the predictions is what the causal
diagnostic measures (§8). This audit does not estimate the pre-crossing part.

## 7. Evidence from the replays

**MASKED** (hidden coordinates after release / t_R set to NaN; t₀.₈ kept for the follow check): runs in which every
registered prediction field reproduces its committed value.

| arm | runs | t_traj, s_traj, r_traj | t_sw, s_switch, ṡ, r_cf | V7 window max | follows_branch | path SHA-256 verified |
|---|---|---|---|---|---|---|
| W2-A T | 98 | 98 | 98 | 98 | 98 | 98 |
| W2-A D | 120 | 120 | 120 | 120 | 120 | 120 |
| W2-A T′ | 120 | 120 | 120 | 120 | 120 | 120 |
| GELU-T random | 80 | 80 | 80 | 80 | 80 | 80 |
| GELU-T branch | 80 | 80 | 80 | 80 | 80 | 80 |
| Track A Adam | 74 | 74 (and t_R, s\*_run 74) | 74 | — | — | 74 (Adam replay bit-identical 74) |
| Track A SGD | 63 | 63 (t_R, s\*_run 63) | 63 | — | — | 63 |
| Test 2A | 75 | 75 (t_R, s\*_run, t_sw 75) | 75 | — | — | 75 (Adam replay bit-identical 75) |

**TRUNCATED** (MASKED, and every row after L set to NaN). L_traj: W2-A max(t_traj, t_sw) (D: t_sw < t_traj in 120/120,
so L_traj = t_traj); GELU-T t_traj; Track A max(t_traj, t_top); 2A max(t_traj, t_top, t_sw). L_cf: W2-A and GELU-T
t_sw (predict_one without the recursion); Track A and 2A max(t_top, t_sw). Runs in which **every** field of the
prediction reproduces:

| arm | runs | r_traj fields at L_traj | L_traj ≥ t_obs: at / after (max) | r_cf fields at L_cf | L_cf ≥ t_obs: at / after (max) |
|---|---|---|---|---|---|
| W2-A T | 98 | 98 | 95 / 3 (3) | 98 | 2 / 2 (3) |
| W2-A D | 120 | 120 | 0 / 0 | 120 | 0 / 0 |
| W2-A T′ | 120 | 120 | 0 / 120 (83) | 120 | 0 / 120 (83) |
| GELU-T random | 80 | 80 | 36 / 44 (191) | 80 | 0 / 13 (310) |
| GELU-T branch | 80 | 80 | 42 / 38 (19) | 80 | 0 / 7 (60) |
| Track A Adam | 74 | 74 | 1 / 33 (140) | 74 | 0 / 4 (140) |
| Track A SGD | 63 | 63 | 1 / 4 (181) | 63 | 0 / 4 (181) |
| Test 2A | 75 | 75 | 2 / 8 (53) | 75 | 0 / 1 (53) |

The L ≥ t_obs columns equal the "all inputs" counts of §1, from the replay instead of the CSVs.

## 8. POST HOC diagnostic: the predictions with nothing read at or after the crossing

**The change.** The registered predict function (unchanged), with two input changes:
- hidden coordinates masked as in MASKED;
- the output weights at every t ≥ t_obs replaced by the linear extrapolation from t_obs − 1, using the mean increment
  over t_obs − 101 → t_obs − 1.

**What this changes.**
- Every input the prediction reads is then from steps ≤ t_obs − 1, or extrapolated from them. The diagnostic is well
  defined for every scored run (t_obs ≥ 101 in all of them; every run gave a prediction).
- Choosing the cut uses t_obs, so this is a sensitivity diagnostic, not a prediction rule.
- r_obs is recomputed with the causal s_switch (the same comparator on both sides); the registered-comparator ratio is
  also given in the JSON.

| arm | runs | t_traj changed | r_traj equal to registered | max / median rel. change of r_traj | max rel. change of s_switch | median r_obs/r_traj (causal; same s_switch) | Spearman (causal) | registered L1 median, L3 |
|---|---|---|---|---|---|---|---|---|
| W2-A T | 98 | 0 | 0 | 6.9e−3 / 2.9e−5 | 1.6e−7 | **0.99997** | **0.99992** | 1.000000, 0.99992 |
| W2-A D | 120 | 0 | 120 | 3.6e−15 / 1.4e−15 | 2e−16 | 1.02095 | 0.99730 | 1.02095, 0.99730 |
| W2-A T′ | 120 | 0 | 0 | 5.0e−4 / 4.3e−4 | 8.8e−7 | **1.0000068** | **0.99999** | 1.000000, 0.99999 |
| GELU-T random | 80 | 0 | 0 | 0.35ᵃ / 6.2e−5 | 0 (frozen) | **0.99996** | **0.99859** | 1.000000, 0.9986 |
| GELU-T branch | 80 | 0 | 0 | 0.35ᵃ / 4.9e−5 | 0 (frozen) | **0.99996** | **0.99876** | 1.000000, 0.9988 |
| Track A Adam | 74 | 1 | 41 | 1.4e−2 / 1.2e−15 | 2e−16 | 1.0651 | 0.2489 | 1.0651, 0.2485 |
| Track A SGD | 63 | 0 | 62 | 3.3e−4 / 4.4e−16 | 2e−16 | 1.0555 | 0.9945 | 1.0555, 0.9945 |
| Test 2A | 75 | 0 | 65 | 4.4e−2 / 4.4e−16 | 2e−16 | 1.0579 (A3) | 0.9302 (A1) | 1.0579, 0.9322 |

**Notes**
- ᵃ Seed 876014 (both GELU-T arms), whose registered r_traj is 1.3e−6: the absolute change is 4.6e−7. The next
  largest relative changes are 5.0e−3 (random) and 2.4e−3 (branch). GELU-T's causal Spearman equals the registered L3
  to the digits shown: t_traj is unchanged in 160/160 and the ranks are preserved.
- **Unchanged inputs.** The rule points (Track A, 2A), the t_sw of W2-A D and 2A, and the run's s_switch where
  t_sw < t_obs are reproduced exactly by the causal replay.
  - W2-A T: t_sw is unchanged in 98/98 and s_switch in 94/98.
  - W2-A T′: t_sw is unchanged in 115/120; s_switch changes in all 120, by ≤ 8.8e−7 relative.
  - Track A: t_R is unchanged in 74/74 and 63/63 although t_top ≥ t_obs in 4 + 4 runs; 2A: 75/75.
  - GELU-T: s_switch is frozen; t_sw is unchanged in 77/80 (random) and 79/80 (branch), ṡ and r_cf in 67/80 and 73/80.
- **Where the causal step lands.** In the causal replay t_traj still falls at or after t_obs in 98/98 (T), 118/120
  (T′), 80/80 (GELU-T random), 80/80 (GELU-T branch), 33/74 (Track A Adam) and 10/75 (2A).
  At those steps the causal prediction reads only extrapolated output weights.
- **The same diagnostic from other cuts** (e.g. the release, or t₀.₈, with an extrapolation of v over thousands of
  steps) would be a forecast of the output path, not a small change. It was not computed.

## 9. Reproduce

```
python -m src.prediction_inputs_audit replay track_a|track2a|w2a|gelu      # MASKED + CAUSAL; needs the untracked paths
python -m src.prediction_inputs_audit truncated track_a|track2a|w2a|gelu   # TRUNCATED; needs the untracked paths
python -m src.prediction_inputs_audit summarize                           # -> results/prediction_inputs_audit.json
```

Each replay asserts every path's hash and is resumable. Run as one process at nice 15 with one thread, behind the
memory gate. The replay rows are committed, so `summarize` and the ledger need no path.
