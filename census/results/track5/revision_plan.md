# Track 5: revision plan for the 10-page limit (PREPARATION; author-approved 2026-09-30)

**Status.** This is a plan only. Nothing under `paper/` was edited. The revision itself waits for the reviews (early
November). The revised PDF is due 27 November. The main text must be at most 10 pages, and the title stays unchanged:
"From class means to margins: predicting hidden-unit placement in width-one networks".

**What the plan does:**
- adds **one** new section on registered predictions beyond width-one sine: GELU-T, Test 2C and W2-A;
- places Tests 2A and 2B in the existing lag-law section (§5.1);
- places Track 1 (WP-39), the Track 4 certificate checks, the census update and math note §14–§17;
- gives a page budget.

**Rules followed:**
- Every number carries a source key (§0) naming the committed file it comes from.
- Anything not yet known is marked **PENDING**.
- The precise path-conditioning wording is left as the placeholder **[pending the prediction-inputs audit]**, because
  the audit of exactly which inputs each prediction uses is running now.

---

## 0. Sources (keys used in this file and in `reply_drafts.md`)

| key | file (relative to `census/` unless absolute) |
|---|---|
| [P] | Main-text PDF used for the page map: `/Users/Evan/Downloads/Precision_Topology_ICLR (18).pdf` (56 pp.; main text pp. 1–9). **Not in the repo; see PENDING P1.** Cited as [P p:lines]. |
| [S] | `dist/supplementary/results/Precision_Topology_ICLR (16).pdf`, the file named in the task. It is the **supplementary material** (5 pp.: link, linked spheres, MNIST, CIFAR-10, full census), not the main text. |
| [2A] | `results/track2a_writer_inputs.md` |
| [2B] | `results/track2b_writer_inputs.md`; design `results/designs/2B_boundary_design.md` [2Bd] |
| [GT] | `results/gelu_transfer_writer_inputs.md`; design `results/designs/GELU_transfer_design.md` [GTd] |
| [2C] | `results/track2c_writer_inputs.md` |
| [W2] | `results/width2_asym_writer_inputs.md`; design `results/designs/width2_asym_design.md` [W2d] |
| [TA] | `results/track_a_writer_inputs.md` (Track A, a = 1.65) |
| [WP37] [WP38] [WP39] | `paper/WRITER_INPUTS_WP37_track_T.md`, `paper/WRITER_INPUTS_WP38_boundary_and_adam.md`, `paper/WRITER_INPUTS_WP39_fold_posthoc.md` |
| [T4] | `results/track4_writer_inputs.md` §3 (discussion-phase addendum); `results/certificate_audit.md` "Track 4" [CA]; `results/certificate_checks/track4_summary.json` [T4J] |
| [MN §n] | `results/math_note_v2.md`, section n |
| [WP-n] | `paper/WRITER_INPUTS_v4_patch.md` **as committed**: read at `190253d` and regenerated in `ee0b0cb` (`git show HEAD:./paper/WRITER_INPUTS_v4_patch.md`). Any later working-copy edits by other agents were not used. |
| [CEN] | `results/registration_tally.csv`, `results/registration_census.csv` and `results/census_relevance_counts.csv` **as committed at HEAD**: census commit `ee0b0cb`, which adds W2-A; the previous round was `1a29593`. |
| [SB] | `results/sb_fold_report.md` (Track 1A, POST HOC) |
| [STAMP] | `results/{track2a,track2b,track2c,gelu_transfer,width2_asym}/registration_stamp.txt` with their `.ots` proofs. The Bitcoin block numbers come from the OpenTimestamps-upgrade commits `549f44f`, `2115c6b` and `190253d`. |

---

## 1. Summary of the revision

| change | where | size |
|---|---|---|
| **New §6, "Registered predictions beyond width-one sine"**: GELU-T (both arms), Test 2C (a = 1.50 PASS; a = 1.30 UNRESOLVED), W2-A (arms T, D, T′), each with its limits stated plainly | after §5; the old §6–§8 become §7–§9 | ≈ 1.0 page, including a new Table 3 |
| **Test 2A** (Adam per-run ordering at a = 1.85, PASS) | §5.1, after the a = 1.65 test; extra column in Table 2 | ≈ +3 lines net |
| **Test 2B** (validity boundary: C1 PASS, C2 UNRESOLVED) | §5.1, replacing the boundary paragraph now in §6 | about 0 net (moved and rewritten) |
| Lag law as a theorem for fixed-P GD (math note §15) | §5.1 sentence; Table 1 row | +3 lines |
| Theorem 1 compactness (math note §17) | §3 sentence; Table 1 | +1 line |
| Track 4 certificate checks | §4 sentence; Table 1 row (author decision, PENDING P5); AI Use addendum; Appendix E.1 | +1 line in the main text |
| Track 1 (fold, POST HOC, one sentence, WP-39) | new §7 ("Where the account applies") | +3 lines |
| Census update, and OpenTimestamps for this phase's registrations | §9 (Methodology); Appendix U; supplement §5 and its Table 1 | +3 lines |
| Abstract and introduction brought in line | abstract; the contribution bullets | +5 lines |
| Old §6 compressed (width-two, practical-activation and Track T paragraphs) | new §7 | −6 lines |
| Constant-lag carry-over paragraph moved | §5.1 → Appendix V.7 | −5 lines |

---

## 2. Page budget

### 2.1 The submitted main text, measured

- The main text runs from p. 1 to p. 9, line 465, of [P].
- Each page has 54 numbered lines, so the budget is 10 × 54 = 540 lines.
- The submitted text uses 466 lines, about **8.63 pages**. That leaves **74 lines, about 1.37 pages**, below the
  10-page limit.
- Figures and tables are counted by the line numbers they occupy. The 18 inter-section blank lines are assigned to the
  section that follows them.

| § | title (submitted) | lines in [P] | lines | pages | figures / tables inside |
|---|---|---|---|---|---|
| — | Title and abstract | p1:000–030 | 31 | 0.57 | — |
| 1 | Introduction (with the contribution bullets) | p1:031 – p2:065 | 35 | 0.65 | — |
| 2 | Setting and the margin identity | p2:066–099 + Fig. 1 p3:108–129 | 56 | 1.04 | Fig. 1 (22 lines) |
| 3 | The fixed-scale dichotomy | p2:100–107 + p3:130–150 + Fig. 2 p4:162–186 | 54 | 1.00 | Fig. 2 (25) |
| 4 | Certificates and the scaling law | p3:151–161, Table 1 p4:187–199, p4:200–215, Fig. 3 p5:216–240, p5:241–249 | 74 | 1.37 | Table 1 (13), Fig. 3 (25) |
| 5 | Training tracks the minimizer (prospective thresholds; §5.1 the tracking lag; the a = 1.65 test; post hoc Adam; learning rate and interventions) | p5:250 – p7:374, Table 2 p8:378–393 | 141 | 2.61 | Fig. 4 (26), Fig. 5 (27), Table 2 (16) |
| 6 | Where the account applies | p7:375–377 + p8:394–421 | 31 | 0.57 | — |
| 7 | Related work | p8:422 – p9:452 | 31 | 0.57 | — |
| 8 | Methodology and limitations | p9:453–465 | 13 | 0.24 | — |
| — | free on p. 9 | p9:466–485 | 20 | 0.37 | — |
| | **total** | | **486 = 9 pages** | | |

§5 in detail:

| part | lines in [P] | lines |
|---|---|---|
| Prospective thresholds | p5:250–262 | 13 |
| §5.1 derivation (Eqs. 6–7) | p5:264–269, p6:296–311 | 22 |
| Registered test at a = 1.65 | p6:313–323 | 11 |
| Post hoc Adam diagnosis | p7:351–356 | 6 |
| Constant-lag carry-over; SGD most accurate | p7:357–361 | 5 |
| Learning rate and interventions | p7:363–372 | 10 |

Old §6, the paragraph that this plan compresses or moves:

| paragraph | lines in [P] | lines |
|---|---|---|
| Scope sentence and the boundary test (becomes the 2B text in §5.1) | p7:375–377, p8:394–401 | 11 |
| Width two | p8:402–408 | 7 |
| GELU, SiLU and Mish; the band task | p8:409–413 | 5 |
| Linear-plus-slab (Track T) | p8:414–420 | 7 |

**Assumption, PENDING P2.** The reproducibility statement, the AI Use statement and the references stay outside the
10-page count, as the submitted layout suggests: they begin on p. 10 after the main text ends on p. 9. Check this
against the ICLR 2027 author guide.

### 2.2 Budget of the revision, in lines (54 per page)

| item | + / − lines |
|---|---|
| New §6: 34 lines of text (about 3 introduction, 8 GELU-T, 8 Test 2C, 10 W2-A, 5 on limits) and Table 3 with caption (about 16) | **+50 to +54** |
| §5.1: the 2A paragraph (+6); the post hoc Adam paragraph cut from 6 to 3 lines (−3); the theorem sentence from math note §15 (+3) | +6 |
| §5.1: the 2B paragraph, moved from old §6 (+8 in §5.1, −8 in §7) | 0 |
| Table 2: a third column for 2A | 0 |
| Abstract (+3); introduction bullet and closing paragraph (+2) | +5 |
| §3: compactness sentence (+1); §4: Track 4 sentence (+1); Table 1: two new rows and edited rows (+3) | +5 |
| New §7: width-two paragraph 7 → 3, practical activations 5 → 3, Track T 7 → 4; Track 1 sentence +3 | −6 |
| §9: census headline and OpenTimestamps (+3) | +3 |
| **Mandatory cut C1:** the constant-lag carry-over paragraph, p7:357–361, moved to Appendix V.7 | −5 |
| **Net** | **+58 to +62 lines ≈ 1.07–1.15 pages** |
| Room before the limit | 74 lines ≈ 1.37 pages |
| **Margin** | **12–16 lines ≈ 0.22–0.30 page** |

These are line estimates. LaTeX float placement can move a figure by a quarter page, so the reserves in §2.3 are
listed in the order to use them.

### 2.3 What must be cut or moved, and the reserves

**Mandatory:**
- **C1.** Move "Post hoc, with nothing fitted to these runs, a constant lag carried over … consistent with the stale
  rule-point preconditioner" (p7:357–361) to Appendix V.7, which already carries the same statistics.
- **C2.** Shorten the post hoc Adam paragraph (p7:351–356) to two sentences:
  - the registered L3 failure stands;
  - the stale rule-point preconditioner was diagnosed post hoc, and the rule it suggested was then registered as
    Test 2A.
  - The full author wording (WP-38 §2) moves to Appendix V.7.
- **C3.** Compress old §6 as in §2.2. The width-two, practical-activation and Track T failures each keep one sentence
  with their verdicts, and the details are already in Appendices N, X, Y and in [WP37].

**Reserves, in order of use, if the draft overflows:**
- **R1.** Move the second and third sentences of the related-work paragraph on staged dynamics (p9:432–441, about 6
  lines) to Appendix T. **−6 lines.**
- **R2.** Figure 4: keep panel (a) only and move panel (b), the error bars by predictor, to Appendix I. **About −10
  lines.** The caption keeps the P1 and P2b numbers.
- **R3.** Figure 2: reduce it to the a = 1.30 panel and state the a = 1.50 bounds in the caption. **About −8 lines.**
- **R4 (last resort).** Merge Table 2 into Table 3 as one "registered lag tests" table. **About −6 lines.**

### 2.4 Figures and tables that change

| item | change | producer / status |
|---|---|---|
| Table 1 (status of claims) | Edited rows: "Lag law" (proved for fixed-P GD without momentum, math note §15; for Adam a first-order approximation); "The a = 1.65 test" (add "a = 1.85, Adam, the t_sw rule: registered, PASS"); "Small-scale selection" (compactness proved on the population objective, math note §17). New rows: "Validity range κχ ≤ 0.1: registered (2B C1 PASS); above 0.1 unresolved"; "Transfer tests (GELU-T, W2-A, 2C): registered; each conditional on a warm start or slowed SGD, see §6". The "sharp limiting threshold" row changes only on the author's decision (P5). | text only |
| Table 2 (registered lag test at a = 1.65) | Add a column "Adam, a = 1.85 (2A)" with the registered numbers in §4.1 below. Rows with no committed value show "—". | [2A] |
| **Table 3 (new, in new §6)** | One row per arm: setting, start, optimiser (η, ρ), n scored / n on-branch, L1, L2, L3, L4, L5 and the verdict for GELU-T primary and control and W2-A T, D and T′. Test 2C gets one row per a = 1.50 cell (C1, C2, C3) and one line stating that the a = 1.30 cells are UNRESOLVED (V3). The draft is in §3.4. | numbers from [GT], [W2], [2C]; a table producer is **PENDING P6** |
| Figure 5 (lag against χ) | **Unchanged.** It stays the 36 free-training arms. The new registered points go to Figure 15. | — |
| Figure 15 (Appendix V, observed/predicted against χ) | Add the registered 2B cells (the six C1 cells; the others marked invalid), GELU-T (2 arms), W2-A (3 arms) and the 2C a = 1.50 cells. Caption: the registered and post hoc markers stay distinct. | regeneration **PENDING P6** |
| Figure 16 (Appendix Y.3, scoreboard of registered value predictions) | Add the new registered value predictions: 2A A3; GELU-T L1/L2 (2 arms); W2-A L1/L2 (3 arms); 2C C1/C2 at a = 1.50. The a = 1.30 cells of 2C are shown as unresolved, not as points. Update "21 of 27" in the text. | regeneration **PENDING P6** |
| Supplement §5 and Table 1 (census by block) | New blocks (2A, 2B, GELU-T, 2C, W2-A); totals as in §5.3 below | [CEN] (`ee0b0cb`; `writer_patch_census_by_block.csv`) |
| Appendix U, Table 34 (central failures) | The count stays 33 at HEAD ([CEN]; commits `1a29593` and `ee0b0cb`, "Central failures 33 … unchanged"). The totals sentence is updated (§5.3). | [CEN] |
| AI Use statement, the checker paragraph | Its last sentence is replaced by the Track 4 addendum (§5.2). | [T4] |

---

## 3. The new section: §6 "Registered predictions beyond width-one sine"

**Placement.** After §5 "Training tracks the minimizer". Old §6 "Where the account applies" becomes §7 and keeps the
failures.

**Opening paragraph** (about 3 lines):
- All three tests were registered after the submission. Each was frozen and hashed, with an OpenTimestamps proof,
  before any registered seed was trained, and each prediction was committed before any crossing was evaluated.
- Each tests the lag law r = κχ, with κ computed from that setting's own landscape and no fitted constant, in a setting
  where the earlier registered training predictions failed or were unresolved: Track 3A at GELU, T2-3 at width 2 and
  the band-task P2a.
- It states in one clause what all three share: training is started on or near a branch (a warm start at held output
  scale, for GELU-T and W2-A) or is slow (full-batch SGD with a small or slowed output learning rate). §6 therefore
  tests tracking, not branch discovery from initialisation.

**Required disclosure in every verdict sentence (the rules of the writer inputs):**
- GELU-T: V7's window, the follow check's state condition and the primary gate's handling of G during the hold were
  fixed after the pilot, on pilot seeds only, before registration [GT §1].
- W2-A: five registration details were changed after the pilot, on pilot seeds only, before any registered seed was
  touched [W2 §2 "Rules for every arm"].
- 2C: V3 was added at registration by analogy with 2A. It was not on the reviewed design [2C §5].

### 3.1 GELU-T: the lag law at GELU under full-batch SGD (both arms)

**Exact claim.** Registered before training, with no fitted constant, the lag law predicts where full-batch SGD crosses
at GELU (width 1) after a warm start at held output scale:
- **Primary arm (random hidden start):**
  - median observed/predicted lag 1.000 (L1), closed form 0.977 (L2), per-run Spearman 0.999 (L3);
  - the prediction beats the population switch (L4: mean D −0.104 [−0.121, −0.088]);
  - it beats a lag-free crossing at each run's own branch switch (L5: −0.0031 [−0.0036, −0.0026]);
  - 80 of 80 runs scored [GT §3].
- **Mechanism control (start at the population branch point):** L1 1.000, L2 0.985, L3 0.999, L4 −0.103
  [−0.119, −0.087], L5 −0.0026 [−0.0029, −0.0023]; 80 of 80 scored [GT §3].

**Registered verdicts:** PASS (primary, the headline) and PASS (control). All seven validity conditions hold in both
arms [GT §3].
- Registration `736b6bf`; OpenTimestamps `d7996d7`; predictions `8d39e75` (predictions.csv SHA-256 `218309c5…`);
  scoring `58aa960`.
- Gates: primary 80 of 80 on a branch at release (38 target, 42 mirror); control 80 of 80 on target [GT §3].
- The registered descriptive sensitivity analysis excludes the 6 runs with G > 0 at initialisation or in the hold.
  It leaves 74 scored runs with L1 1.000, L2 0.977 and L3 0.999: still PASS [GT §3].

**Limits, stated plainly:**
- **Warm start at held output scale.** w₂ is held at +s₀ = 0.5·s_pop (s_pop = 6.645633) while (w₁, b₁, b₂) relax by
  GD (lr 0.3) for at least 4,000 steps. Training then runs from a Newton-checked point on the run's own-sample branch
  [GTd]. The test says nothing about reaching a branch from initialisation.
- **Slow full-batch SGD.** η = 0.03, ρ = 1 (the pilot rule was inactive: pilot q90 of κχ 0.0037) [GT §3], 40,000
  steps [GTd].
  - The earlier registered GELU test used free Adam at lr 0.01 from initialisation. Its primary arm was UNRESOLVED and
    its 200-seed secondary arm FAILED (above-threshold fraction 0.477) [P p55 Table 44]. Those verdicts stand.
- **The lag is small.** The median observed lag r_obs is 0.00275, about 40 steps [GT §3]. L5 is what shows the lag is
  resolved against a lag-free prediction. L4 mostly reflects the sample variation of the own-branch switch [GT §5].
- **The arms are not independent.** In 8 seeds the random-start hold reached the same release point as the
  branch-point start bit for bit (152 distinct paths of 160) [GT §3].
- **One optimiser, one learning rate.** No registered η-invariance result exists for SGD or GELU [GT §5].
- **Path-conditioning.** r_traj is Track A's R4 recursion from release on the occupied branch [GTd "Predictions"]. It
  integrates along the run's own output-scale path; the closed form r_cf = κ_seed·χ is evaluated at t_sw.
  **Exact inputs: [pending the prediction-inputs audit].**
  - Question for the audit: in the 8 + 2 runs with κ < 0 the crossing came before t_sw [GT §3]. Does χ at t_sw, for
    these runs, use post-crossing information?
- Branch selection is not predicted. A prediction exists only for the branch copy occupied at release [GTd].

**Do not say:** "proved for GELU"; "holds for Adam"; "holds at any learning rate"; that the 3A failure is overturned;
that negative-κ runs were excluded (they were scored) [GT §2].

### 3.2 Test 2C: the band task in R^d against each seed's own-sample R^d switch

**Exact claim (a = 1.50).** In a prospective, registered test:
- each seed's own-sample R^d switch was frozen before training;
- that switch, plus the width-1 lag law, predicted the crossing scale of free Adam training (lr 0.01, 64,000 steps,
  every-step detection) to a median |log error| of 0.0076 at d = 2 and 0.0091 at d = 4 (C1);
- the observed/predicted lag ratio was 0.940 and 1.071 (C2);
- the prediction beat the noise-blind x₁-only threshold, with 95% intervals of the mean log-error difference
  [−0.158, −0.121] and [−0.403, −0.332] (C3) [2C §3].
- The prediction uses:
  - κ_k frozen from width 1 (4.046 at a = 1.50);
  - χ from pre-crossing information, with P frozen at t_sw, the 2A rule [2C §2].

**Registered verdicts** (per cell) [2C §3]:
- **PASS** at a = 1.50, d = 2 and d = 4 (C1, C2, C3 each);
- **UNRESOLVED** at a = 1.30, d = 2 and d = 4, on validity V3 (pre-crossing prediction in 50/57 = 0.877 and
  41/51 = 0.804 of crossing runs, below the registered 0.90);
- registration `b22ebd0`; OpenTimestamps `906c07b`; predictions `d8d3569` (SHA-256 `27c0b676…`); scoring `b863a54`.

**The frozen offset** (a prospective statement of the static shift): the median s_own,d/s_own,x1 is 1.075 and 1.078
at d = 2, and 1.359 and 1.355 at d = 4 [2C §2]. The population threshold does not move with d. What moves is the
finite-sample threshold [2C §5].

**Limits, stated plainly:**
- **The a = 1.30 cells are UNRESOLVED, not failed and not passed.**
  - V3 was added at registration, by analogy with 2A, and was not on the reviewed design [2C §5].
  - The scored-run statistics there (median error 0.0032 and 0.0068; lag ratio 0.956 and 1.010) are reported, not
    scored [2C §3].
- **Adam, free training from initialisation, one learning rate, one unit.**
  - The own-sample switches are validated, not certified: 240 of 240 seed-cells passed the restart ladder, CMA-ES and
    the audit [2C §2].
- **Per-run ordering was not a criterion.** Spearman was 0.31–0.67 on the scored runs [2C §3].
- **Mirror branches.** 17 of the 26 unscored crossing runs had settled on the other mirror. The mirror-matched
  prediction is a registered descriptive, never scored [2C §4–5].
- **The earlier registered band-task result stands.** The population-referenced lag test failed (Track 3B P2a:
  0.182 and 0.207, above the width-1 range, at d = 2; [P p55 Table 45]).
- **Path-conditioning.** 2C's prediction is closed-form type, s_pred = s_own,d·(1 + κ_k·χ), with χ at the Track A rule
  point and P at t_sw [2C §2]. **Whether any input is read after the crossing: [pending the prediction-inputs audit].**

**Do not say:** that 2C passed as a whole; that a = 1.30 failed or "would have passed"; that the mirror-matched
prediction is the registered test; that the population threshold moves with d [2C §5].

### 3.3 W2-A: the lag law at width 2 on the asymmetric windows (arms T, D, T′)

**Exact claim.** Registered before training, with no fitted constant: at width 2 on the asymmetric windows (a = 1.30,
Δ = 0.4), full-batch SGD crosses at the occupied branch's own switch plus the predicted lag, after a warm start at held
output scale. [W2 §3]

| arm | start | η / ρ | on-branch / scored | L1 | L2 | L3 | L4 mean D [95%] | L5 mean D [95%] (n) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **T** (headline) | random hidden start, v held at s₀·(0.1, 0.9) | 0.03 / 2⁻¹⁰ | 98 of 200 on T / 98 | 1.000 | 1.009 | 1.000 | −0.109 [−0.125, −0.094] | −0.00121 [−0.00129, −0.00113] (92) | **PASS** |
| **D** (control) | branch-point start, duplicate branch | 0.3 / 1 | 120 / 120 | 1.021 | 1.008 | 0.997 | −0.077 [−0.088, −0.065] | −0.0307 [−0.0314, −0.0300] (120) | **PASS** |
| **T′** (sign test, κ < 0) | branch-point start | 0.03 / 2⁻¹² | 120 / 120 | 1.000 | 0.993 | 1.000 | −0.062 [−0.070, −0.054] | −0.00088 [−0.00095, −0.00082] (120) | **PASS** |

(η/ρ from [W2 §3 "ρ (from the pilot)"] and [W2 §5]; starts from [W2d] and [W2 §2].)

- **T′:** every one of the 120 scored runs crossed **before** its own switch, as predicted by κ < 0. The median signed
  lag was −43.5 steps [W2 §3].
- Registration `28b2432`; OpenTimestamps `7800b48` (Bitcoin attestation recorded in `190253d`); predictions `2c630f2`
  (SHA-256 `e78d85c8…`); scoring `7eb4538`.
- All seven validity conditions hold in every arm [W2 §3].

**Limits, stated plainly:**
- **Warm start at held output scale.** v is held at s₀ = 0.5·s_pop2 = 0.2225 while the hidden coordinates relax by GD
  (lr 1.0) for at least 4,000 steps, and full-batch SGD follows [W2d "Setting"].
- **Slowed output learning rate.** The output weights move at ρη with ρ = 2⁻¹⁰ (T) and 2⁻¹² (T′) at η = 0.03. D runs
  at η = 0.3, ρ = 1 [W2 §3, §5].
  - T2-3, full-speed Adam at width 2, **stays FAIL** [W2 §5]: the median crossing was 3.29× the threshold, above the
    registered bound of 1.25 [P p42 Table 29].
- **Only T starts at random, and fewer than half its runs reached T.**
  - 98 of 200 reached T. The rest: 15 were on T′ and 87 on neither (58 D, 24 Newton not accepted, 3 D′, 2 off the
    frozen copy) [W2 §3].
  - D and T′ are branch-point starts. Branch selection from initialisation is not predicted.
- **Five registration details were changed after the pilot**, on pilot seeds only, before any registered seed was
  touched: D's start and gate, T's 200 seeds, V4/V5 on magnitudes, D′'s L4 comparator, and κ at the actual winding
  pair [W2 §2].
  - Disclosure B: the total-derivative κ was adopted after exploratory population runs gave the wrong sign with
    GELU-T's form [W2d "κ"].
- **The lags are small at T and T′.** Median r_obs was 0.00116 (24 steps) and −0.00084 (−43.5 steps) [W2 §3].
- **Path-conditioning (disclosure C).**
  - T's and T′'s lag-free switch is read from each run's own output-weight (v) path, "v only, before any gap is read".
    Their predictions are conditional on that path [W2d "Frozen"; W2 §5].
  - R4 integrates along the run's own v_t [W2d "Predictions"]. D's switch is frozen.
  - Predicted and observed crossing steps are equal in 97 of 98 T runs and 118 of 120 T′ runs [W2 §3].
  - **Exact inputs, and what the step-level agreement does and does not show: [pending the prediction-inputs audit].**
  - Question for the audit: T′ runs cross before t_sw, so what is χ at t_sw for L2?
- **Erratum.** Registration §14 says 840 copy points. The frozen file has 920, all accepted. The error is in the text
  only, and no frozen value, rule or verdict is affected [W2 §6]. This goes in the appendix.

**Do not say:**
- that T2-3 is overturned or "explained";
- that the switch was predicted in advance (it is read from the run's own v path);
- that the law holds for Adam or at any learning rate;
- that random starts usually reach T;
- that D or T′ is the headline;
- that the negative-κ explanation was found on registered seeds [W2 §2].

### 3.4 Draft Table 3 (for the new section; the numbers are the registered values above)

Caption draft:
> "Registered transfer tests of the lag law, each frozen and OpenTimestamped before training, with predictions committed
> before any crossing was evaluated. L1: median observed/predicted, trajectory-integrated, required [0.90, 1.10]. L2:
> closed form κχ, [0.80, 1.20]. L3: per-run Spearman ≥ 0.5. L4 and L5: 95% bootstrap upper end of mean
> log-error difference < 0 against the population switch and against a lag-free crossing at the run's own switch. Every
> run starts on or near a branch: a warm start at held output scale (GELU-T, W2-A) or free training in the band task
> (2C). SGD rates as listed. [path-conditioning clause: pending the prediction-inputs audit]"

Rows:
- GELU-T primary;
- GELU-T control;
- W2-A T;
- W2-A D;
- W2-A T′;
- 2C d = 2, a = 1.50 (C1 0.0076, C2 0.940, C3 [−0.158, −0.121]);
- 2C d = 4, a = 1.50 (0.0091, 1.071, [−0.403, −0.332]);
- one line: "2C at a = 1.30 (d = 2, 4): UNRESOLVED (validity V3: 0.877, 0.804 < 0.90)."

### 3.5 Closing paragraph of the new section (about 5 lines): what the three tests do and do not establish

- **They establish** that, given a run that is on a tracked branch and moving slowly (κχ well below 0.1: q90 0.0039 for
  GELU-T, and 0.00167 / 0.0359 / 0.00142 for W2-A T / D / T′ [GT §3; W2 §3]), the width-1 lag law with its own
  landscape's κ predicts the crossing outside width-one sine.
- **They do not establish:**
  - prediction from initialisation;
  - branch selection;
  - Adam at width 2 or at GELU;
  - full-speed training.
  The earlier failures in those regimes stand (§7).

---

## 4. Tests 2A and 2B in the lag-law section (§5.1)

### 4.1 Test 2A: Adam per-run ordering at a = 1.85 (after the a = 1.65 paragraph)

**Exact claim.**
- In a second registered test, at an activation value with no prior training data (a = 1.85, 80 fresh seeds), Adam's
  preconditioner was frozen at t_sw, the first step the output scale reaches the occupied branch's switch.
- That rule was chosen post hoc on the a = 1.65 runs and fixed before training here.
- The lag law then predicted each Adam run's lag: per-run Spearman 0.93 (A1 ≥ 0.5), 58 of 75 runs within 10% = 0.773
  (A2 ≥ 0.75), median observed/predicted 1.058 (A3 ∈ [0.9, 1.1]) [2A §2–3].

**Registered verdicts: A1, A2, A3 PASS; validity valid.**
- 79 of 80 crossed; 75 scored.
- Registration `9fd1f32`; OpenTimestamps `91a85cb`; predictions `42316d9` (SHA-256 `428a58db…`); scoring `252f8b3`
  [2A §1].

**Limits, stated plainly:**
- **The rule was not chosen blind.** It came from the a = 1.65 runs. 2A is prospective for it.
- **Track A's registered Adam L3 FAIL (Spearman 0.25) stands.** 2A is a new registered test of a new rule [2A §3].
- **A2's margin is small.** 57 of 75 is the least that passes; 58 did [2A §2].
- **The switch at 1.85 is validated, not certified** (s\*_pop = 1.29718) [2A §1].
- **One width, one activation family, Adam at one rate.**
- **The closed form is secondary.** Spearman 0.63, 51% within 10%, median 0.963; it was not scored and is not reported
  as passing [2A §2].
- **Path-conditioning.** The prediction is the trajectory-integrated recursion along the run's own output-scale path,
  with P frozen at t_sw [2A §3]. **Exact inputs: [pending the prediction-inputs audit].**

**Table 2, new column "Adam, a = 1.85 (2A)"** [2A §2]:

| row | value |
|---|---|
| crossers / initialised | 79/80 |
| runs scored | 75 |
| observed median lag | 0.136 |
| predicted median lag, integrated | 0.128 |
| median per-run ratio, integrated | 1.058 |
| closed form | "0.963 (secondary, not scored)" |
| runs within 10% | 77% (58/75; required ≥ 75%) |
| rank correlation | 0.93 (pass) |
| predicted median lag (closed form); median absolute lag error | "—": no committed value in [2A]. Do not compute one for the paper without a producer. |

**Replacement for the post hoc Adam paragraph (C2), two sentences:** "The registered Adam ordering criterion failed
(Spearman 0.25). Post hoc, the preconditioner frozen at half the switch scale was stale by the crossing; the pre-crossing
rule this suggested (P at the first step the output scale reaches the occupied branch's switch) was then registered at
a = 1.85 and passed (above)." The author's full wording (WP-38 §2) moves to Appendix V.7.

### 4.2 Test 2B: the validity boundary in κχ (it replaces the boundary sentences of old §6, p8:394–401)

**Exact claim (the replacement sentence of [2B], used as written):**
> "In a registered redesign of the boundary test (width-1 SGD forced ramps with the learning rate set a priori for
> linear stability along the ramp, a = 1.30 and 1.50, 40 seeds per cell), the lag stayed within 25% of the predicted
> κχ in every cell with κχ ≤ 0.1 (cell medians 0.99–1.21); at κχ = 0.2 it exceeded the prediction by 41–56%, and the
> cells at κχ ≥ 0.3 could not be scored because the realised dynamics left the stability range the design required."

**Registered verdicts:**
- **C1 PASS.** Six cells with κχ ≤ 0.1: 1.046 / 1.102 / 1.209 at a = 1.30 and 0.990 / 1.046 / 1.139 at a = 1.50.
- **C2 UNRESOLVED.** All four κχ ≥ 0.3 cells are invalid: η_cell·λ_max > 1 on the realised trajectory in 39, 40, 17
  and 40 of 40 runs.
- κχ\* = 0.2 at both a, descriptive only [2B].
- Registration `0708ae9`; OpenTimestamps `349911f`; predictions `08057a9`; scoring `f8f9329`.

**Keep beside it** (WP-38, still true): across the 36 free-training arms, arm medians of κχ are 0.007–0.082 [WP38 §1].

**Limits, stated plainly:**
- **Forced ramps, not free training.** The output scale is imposed as s = s₀e^{γt} after a warm start at the
  population branch point θ\*(s₀), s₀ = 0.5·s\*, held for 4,000 steps [2Bd].
- **The upper side of the boundary is not registered.**
  - The first boundary test's B1 and B2 stay UNRESOLVED [WP38 §1].
  - The a = 1.50, κχ = 0.02 cell is 32 crossing runs, the seeds with the smaller switches [2B].
- **The design did not keep the fast cells stable.** The a-priori rule bounds λ_max on the branch, not off it. Post hoc,
  every violation happened after the ramp had passed the switch [2B].

**Do not say:**
- "the boundary test passed";
- "the registered test locates the boundary between 0.1 and 0.3";
- "the law fails at κχ ≥ 0.3" as a registered result;
- "accurate to 25% up to κχ = 0.2" [2B].

### 4.3 The lag law as a theorem (math note §15; one sentence in §5.1, after Eq. 7)

Draft: "For gradient descent with a fixed preconditioner and no momentum, tracking and the lag r = κχ + O(χ²) are
theorems with explicit constants, and in free training the lag is learning-rate invariant to leading order [MN §15,
Theorems L1–L2, Corollary L3]; Adam's moving preconditioner and momentum are not covered, and the constants have not
been enclosed on the sine-family branches [MN §15.5]."

The existing sentence "The continuous-time theory supplies no tracking guarantee for discrete Adam or SGD" (p6:301)
must be edited: it stays true for Adam, but not for fixed-P discrete GD, which is plain SGD.

Table 1's "Lag law" row changes to match.

---

## 5. Where the other items go

### 5.1 Track 1: fold, POST HOC, one sentence (WP-39)

- **Where:** new §7 "Where the account applies", after the compressed width-two and practical-activation sentences.
  It explains, post hoc, why fast training falls outside the account.
- **Text, used as written** [WP39]:
  > "Post hoc, locating each run's branch by validated continuation shows that slowly driven training (GELU) crosses at
  > its branch's ordinary switch (median ratio 1.0017), whereas fast training (width 2, and the linear-plus-slab task)
  > leaves its branch well before the switch or fold and tracks no branch, crossing a median 2.79 (width 2) and 2.17
  > (linear-plus-slab, relative to the linear branch's fold) times later."
- **Also state:** the registered gate to a prospective fold test FAILED, so no fold test was registered [WP39; SB
  "Gate to 1B: FAIL"].
- **Do not say** [WP39]:
  - that training follows its branch to a fold;
  - that the fold-delay law (math note §14) is confirmed;
  - that this was registered;
  - anything about SiLU or Mish.

### 5.2 Track 4 certificate checks

- **AI Use statement:** replace the checker paragraph's last sentence, "Outer exclusion and the finite and limiting
  solve brackets were not independently checked; only the searches support them." (p10:523–524), with the addendum in
  [T4 §3], used as written:
  > "It also re-derives the outer exclusion by a fresh Arb branch and bound (41 of 41 A-intervals; one convexity lemma is
  > used as stated in the math note, not machine-checked), and confirms the signs at both ends of every solve bracket
  > (the limit bracket and 12 finite-a ends at a = 1.30–1.60). It does not check that the sign changes exactly once
  > inside each bracket."
- **§4, p3:160–161,** "Outer exclusion and the finite and limiting solve brackets remain search-certified only": change
  it to the same content in one sentence. The margins are ≥ 1.006·10⁻⁴ on the annulus and ≥ 1.649·10⁻⁴ on the solve
  bracket [CA; T4J].
- **Appendix E.1:** add the three Track 4 checks to the list of independent checks, and delete "Independent outer
  exclusion and finite and limiting solve checks were not run because of their projected cost" (p22:1152–1154).
- **Table 1, "Sharp limiting threshold":** whether the row may now read "independently re-checked (one convexity lemma
  as stated)" is an **author decision, PENDING P5**. The Track 4 checks cover the outer-exclusion link that the sharp
  interval needed [P p22:1150–1154; CA]. WP-17's older "do not say the sharp value is independently verified" predates
  Track 4.
- **Do not say** [T4 §3]:
  - that the checker verifies the finite-a margin enclosures as numbers (only their signs);
  - that it verifies Theorem G's finite-ε parts;
  - that it verifies the convexity and localisation lemmas.

### 5.3 Census update

- **At HEAD** (census commit `ee0b0cb`, which adds W2-A; ledger 0 findings per its commit message) [CEN]:
  - **278 predictions: 256 scored by registered rules (135 PASS / 66 FAIL / 8 PARTIAL / 47 UNRESOLVED), plus 22 post
    hoc (3 / 6 / 13 / 0).** Over all 278: 138 PASS / 72 FAIL / 21 PARTIAL / 47 UNRESOLVED.
  - The submitted census had 236 = 214 (100 / 66 / 8 / 40) + 22 [S §5].
  - Central failures: 33 (31 registered, 2 post hoc), unchanged.
  - By registered unit: 338 (169 / 99 / 13 / 57).
  - The discussion phase added 42 rows: 35 PASS, 7 UNRESOLVED, 0 FAIL.
    - Round 2026-09-29, 27 rows: 2A 3; 2B 2; GELU-T 10; 2C 12 [WP-4; WP-14].
    - Round 2026-09-30, 15 rows: W2-A 5 per arm × 3 arms, all PASS [CEN, `ee0b0cb`].
  - The headline **278** is resolved: it was PENDING when this plan was drafted and was committed in `ee0b0cb`.
- **Where it goes:**
  - §9 (old §8, Methodology), p9:458–460: one sentence with the headline and the 33 central failures.
  - Appendix U, p47:2536–2538: the totals sentence.
  - Supplement §5 and Table 1 [S pp. 2–3]: new blocks and totals.
  - The relevance classification of the new blocks is post hoc (WP-14 rule, by topic) [WP-14].
- **OpenTimestamps.**
  - §9, p9:455–456, and supplement §5 ("Timestamps are repository commits, not an external registry") gain one clause:
    the five registrations of this phase (2A, 2B, GELU-T, 2C, W2-A) also carry OpenTimestamps proofs attested in
    Bitcoin block headers [STAMP].
  - The earlier registrations remain repository commits only. **Do not** extend the claim to them.

### 5.4 Math note §14–§17

| § | content (status) [MN] | main text | appendix |
|---|---|---|---|
| §14 Fold tracking: Theorem F (Riccati normal form) | Theorem F PROVED. The n-dimensional reduction is derived, with the reduction step cited, and **not proved**. The discrete map is checked numerically only. | **None.** Track 1's WP-39 sentence needs no theorem, and WP-39 forbids saying the fold-delay law is confirmed by training. | New short Appendix V.9 "Fold tracking (theory)", with the status per part and the WP-39 post hoc numbers beside it. The full proof stays in the supplementary math note. |
| §15 The lag law as a theorem: L1, L2, Corollary L3 | PROVED for fixed-P GD without momentum. Not Adam; constants not enclosed for the sine family. | One sentence in §5.1 (§4.3 above); Table 1 row "Lag law". | Appendix V.1 gains the theorem statements with hypotheses (L0–L3), and V.1's "first-order approximation" wording is reconciled. The proofs go in the supplementary math note. |
| §16 Theorem G (local to global near the limit) | PROVED for 0 < ε ≤ ε₀ = 1.4·10⁻¹², s-form only. The R-form (c₁) is not covered. | Table 1, "Branch scale law, Equation (5)": add "global for ε ≤ 1.4·10⁻¹² (s-form)". | Appendix F, after "Finite-a continuation" (p24:1266–1273), whose sentence "No explicit ε₀ is claimed" must change: ε₀ is now explicit and tiny. |
| §17 Small-scale compactness, width 1 | PROVED on the 800-point population. CHECKED (not proved) on continuous windows. FAILS on training samples, proved. Theorem 1's width-1 conclusion does not need it. Width 2 not covered. | §3, p3:137–138, "Small-scale compactness remains assumed": replace with one sentence carrying these statuses. Table 1 row "small-scale selection". | Appendix A.2, first paragraph (p16:836–840). |

---

## 6. Other text that must change for consistency (short list for the writer)

- **Abstract, last sentence** (p1:026–028): "outside this setting, registered predictions fail or remain unresolved,
  and agreement conditioned on the eventual branch is explanatory". This is no longer accurate.
  - Replace it with a sentence stating that registered tests at GELU, at width 2 and in R^d passed (R^d at a = 1.50
    only), each for runs started on a branch or trained slowly.
  - Free, fast training outside width-one sine still fails or is unresolved.
  - Branch selection is not predicted.
- **Abstract:** add 2A's registered Adam ordering result, one clause, stating that the rule was chosen on the a = 1.65
  runs.
- **Introduction, p1:052–053:** "At width two it predicts the conditional landscape, but training need not follow that
  landscape." Qualify it with W2-A (slowed SGD, warm start) and keep T2-3's failure.
- **Contribution bullet 5** (p2:062–063): add "registered transfer tests beyond width-one sine".
- **Old §6 sentence, p8:406–408:** "Agreement on a tracked branch outside width-one sine is post hoc, and branch
  selection could not be predicted in advance". The first clause is superseded by §6. The second clause stays true.
- **§9 (old §8) last paragraph, p9:461–463:** "The revised Adam measurement rule and the suggested predicted-lag
  boundary also await prospective support." Both now have registered support: 2A PASS, and 2B C1 PASS on the lower
  side only. The upper side is still unresolved.
- **Appendix V.7, p52:2793–2798:** "we name it as the rule for a future registered test". Add the 2A outcome.

---

## 7. PENDING items

| id | item | blocks |
|---|---|---|
| P1 | **Which PDF was submitted.** The repo file named in the task, `dist/supplementary/results/Precision_Topology_ICLR (16).pdf`, is the 5-page supplementary. The page map above uses `/Users/Evan/Downloads/Precision_Topology_ICLR (18).pdf`, the latest main-text build on disk (26 Sep 05:38), which is not in the repo. The author should confirm it matches the OpenReview PDF; line numbers shift otherwise. | §2 |
| P2 | Whether the AI Use statement and the reproducibility statement are outside the 10-page count in ICLR 2027, and whether 10 pages is the limit for this revision (the task says so; the CFP was not checked here). | §2 |
| P3 | **RESOLVED** during this task: census headline 278 committed in `ee0b0cb` (15 W2-A rows, central / threshold, all PASS). It is kept here as a record; nothing is pending. | — |
| P4 | **The prediction-inputs audit.** Every "[pending the prediction-inputs audit]" above; the path-conditioning clause of Table 3's caption; the two questions on negative-κ runs, where χ at t_sw comes after the crossing (GELU-T 8 + 2 runs; all 120 W2-A T′ runs); what W2-A's step-level agreement (97 of 98 T, 118 of 120 T′) shows. | §3, §4.1, replies (c), (g) |
| P5 | The author's decision on Table 1's "Sharp limiting threshold" status after Track 4. | §5.2 |
| P6 | Producers, not written or run here (no compute in this task): Table 3, and the regeneration of Figures 15 and 16 with the new registered points. Each new main-text number must pass `verify_ledger` before the PDF. | §2.4 |
| P7 | The reviews themselves (early November). The allocation of the ~0.25-page margin, and which reserves R1–R4 are used, depend on what the reviewers ask for. | §2.3 |
| P8 | Independent verification of the OpenTimestamps attestations (`ots verify` on the five `.ots` files). The block numbers quoted are from the upgrade commit messages `549f44f`, `2115c6b` and `190253d`. | §5.3; reply (b) |

---

## 8. Do-not-claim list for the whole revision (collected from the writer inputs)

- Do not claim that a later pass overturns an earlier failure:
  - T2-3 FAIL, Track A Adam L3 FAIL, Track 3A secondary FAIL, band P2a and the Track 1 gate FAIL all stand
    [W2 §5; 2A §3; GT §2; WP39].
- Do not claim prediction from initialisation or prediction of branch selection. GELU-T and W2-A start at held output
  scale; only W2-A's T arm starts at random, and 98 of 200 of its runs reached T [W2 §3].
- Do not claim Adam, other learning rates or other optimisers for GELU-T and W2-A [GT §5; W2 §5].
- Do not claim that 2C passed as a whole, or anything at a = 1.30 beyond UNRESOLVED [2C §5].
- Do not claim a registered upper validity boundary (2B C2 UNRESOLVED) [2B].
- Do not claim that the t_sw rule was chosen without data [2A §3].
- Do not claim that the fold-delay law is confirmed by training [WP39].
- Do not claim that every certificate is independently verified, or that margins are verified as numbers [T4 §3].
- Do not claim external timestamps for registrations before this phase [STAMP].
- Do not print census numbers other than those committed in `ee0b0cb` or later, and re-read `registration_tally.csv` at
  revision time, since the census may grow.
