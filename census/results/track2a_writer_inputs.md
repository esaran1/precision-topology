# Test 2A: Adam per-run ordering at an unseen activation value (a = 1.85) — writer inputs

**REGISTERED** (`results/track2a_registration.md`). Design: `results/designs/2A_adam_ordering_design.md` (author-approved;
A2 threshold 0.75 supplied by the author). Code: `src/track2a.py` over the unchanged `src/track_a.py`. Outputs:
`results/track2a/`. Ledger block: `verify_ledger.track2a_checks`.

## 1. Order of events

| commit | content |
|---|---|
| `0f21102`, `647772e`, `eba559e` | Implementation, registration text, tests of every decision rule (20 constructed cases). Nothing trained. |
| `9fd1f32` | **Registration.** Landscape at 1.85, per-seed frozen inputs, code and text hashed (`registration.sha256`, 11 files). No run trained. |
| `91a85cb` | OpenTimestamps proof of `9fd1f32` (`registration_stamp.txt` + `.ots`). |
| `42316d9` | 80 Adam runs trained to the 32,000-step budget with no gap or placement evaluated. Predictions committed; `predictions.csv` SHA-256 `428a58dbbe5c197a4b2f55a990b059227c344b318f5c97485afd8d64e0faf873`; each run's parameter path hashed there. |
| this commit | `observe`: asserted the registration hashes, the committed predictions hash and all 80 path hashes, then every-step detection; scored. |

- **The t_sw rule was chosen POST HOC on the a = 1.65 Track A runs** (`track_a_writer_inputs.md` §5). 2A tests it
  prospectively: new a, 80 fresh seeds (1,850,000–1,850,079), every rule and threshold fixed before training.
- a = 1.85 had no training or crossing data (it appeared only in a landscape table and a depth-3 annealing search).
- Landscape at 1.85 (frozen; registration §9): s\*_pop = 1.29718, validated by the conditional search (unplaced at
  0.995·s\*_pop, placed at 1.005·s\*_pop, same branch). No certified bracket exists at 1.85.

## 2. Verdicts (registered; `results/track2a/scores.json`)

**Validity: valid.**
- V1: 79 of 80 runs crossed (≥ 60).
- V2: t_sw strictly preceded the crossing in 78 of 79 crossing runs (98.7% ≥ 90%). The other one (seed 1,850,066) has
  no t_sw: its branch rule failed (no minimum at the rule point). It is unscored.
- V3: the rule point preceded the crossing in all 75 scored runs.
- Scored: 75 runs. Unscored: 1 non-crossing run (seed 1,850,015, never reaches the rule point), 1 crossing run without
  t_sw (above), and 3 crossing runs with t_sw before the crossing but no primary prediction (the recursion had no hit
  or was unstable: seeds 1,850,004, 1,850,049, 1,850,068). Not replaced.

| criterion | registered threshold | result | verdict |
|---|---|---|---|
| **A1** Spearman(pred, obs) | ≥ 0.5 | **0.93** | **PASS** |
| **A2** fraction with \|obs/pred − 1\| ≤ 0.10 | ≥ 0.75 | **0.773** (58 of 75) | **PASS** |
| **A3** median obs/pred | ∈ [0.9, 1.1] | **1.058** | **PASS** |

**Outcome: PASS on all three.**

- A2's margin is small: 57 of 75 runs within 10% is the least that passes; 58 were.
- Of the 17 runs outside 10%, 15 have obs/pred > 1.1 and 2 have obs/pred < 0.9.

**Secondary (registered as reported, not scored):** the closed form κχ with the same P(t_sw): Spearman 0.63, 38 of 75
(0.51) within 10%, median obs/pred 0.963.

**Descriptive (not scored):**
- Median lags over scored runs: r_obs 0.136, r_pred 0.128. r_obs q10–q90 0.111–0.163; r_pred q10–q90 0.099–0.157.
- obs/pred q10–q90: 1.00–1.125.
- Median steps from t_sw to the crossing: 56. From the rule point to the crossing: 221.
- The winding rule agreed with the occupied branch in every run with a branch (78 of 78).
- **Track A's rule on the same runs** (P frozen at the rule point; computed, never scored): Spearman 0.36, 13 of 75
  (0.17) within 10%, median obs/pred 1.31. This is the rule whose per-run ordering failed at 1.65.

## 3. Writer section

### Say / Do not say, for every possible outcome (as fixed before scoring)

**PASS (A1, A2, A3 all pass).** ← **this occurred**
- Say: "In a second registered test, at an activation value with no prior training data (a = 1.85, 80 fresh seeds), we
  froze Adam's preconditioner at the first step the output scale reaches the occupied branch's switch — a rule chosen
  post hoc on the a = 1.65 runs and fixed before training here. The lag law then predicted each Adam run's lag: per-run
  Spearman 0.93, 77% of runs within 10%, median observed/predicted 1.06 (all three registered criteria pass; 75 scored
  runs of 80)."
- Do not say: that the rule was chosen without data (it was chosen on the a = 1.65 runs; 2A is prospective for it).
  Do not say that Track A's registered Adam L3 failure is overturned: it stands, and 2A is a new registered test of a
  new rule. Do not say that the test covers other widths, activations, optimisers or learning rates. Do not say "every
  run within 10%": 58 of 75 were, just above the 0.75 threshold. Do not present the closed form (Spearman 0.63, 51%
  within 10%) as passing: it was registered as secondary and not scored.

**A1 FAIL only (A2, A3 pass).** (did not occur)
- Say: "With P frozen at t_sw, the typical Adam lag and the per-run precision were predicted, but the per-run ranking
  failed its registered threshold (Spearman < 0.5)."
- Do not say: that the lag law predicts per-run Adam ordering, or that the post hoc a = 1.65 result replicated.

**A2 FAIL only (A1, A3 pass).** (did not occur)
- Say: "The per-run ranking and the typical magnitude passed, but fewer than 75% of runs fell within 10%."
- Do not say: that the lag law predicts each Adam run's lag to within 10%.

**A3 FAIL only (A1, A2 pass).** (did not occur)
- Say: "The per-run ranking passed, but the median observed/predicted lag was outside [0.9, 1.1]."
- Do not say: that the magnitude of Adam's lag is predicted at a = 1.85.

**More than one criterion fails.** (did not occur) Name each failing criterion with its number; the registered test
failed.

**UNRESOLVED.** (did not occur)
- Say: "The registered validity conditions were not met; the test is unresolved and says nothing about the rule."
- Do not say: any of the A1–A3 numbers as a result (they may be listed as descriptive, labelled as such).

## 4. Files

- `results/track2a_registration.md`
- `results/track2a/`:
  - `registration.sha256`, `registration_stamp.txt`, `registration_stamp.txt.ots`
  - `landscape.json`, `landscape.log`, `frozen_seeds.csv`, `freeze.log`
  - `predictions.csv`, `predictions.sha256`, `train.log`
  - `observed_runs.csv`, `scores.json`, `observe.log`
- Parameter paths `paths/*.npz` are untracked. Their SHA-256 hashes are in `predictions.csv`.
