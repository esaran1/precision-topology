# Track 5: reply drafts for the standard objections (PREPARATION; author-approved 2026-09-30)

**Status.** These are drafts, written before any review has arrived (reviews are expected in early November). They
must be matched to the reviewers' actual wording and trimmed to the OpenReview length limits.

**Sources.**
- Every number carries a source key, defined in `results/track5/revision_plan.md` §0.
- [P p:lines] is the main-text PDF and [S] the supplementary.
- Placeholders:
  - **[pending the prediction-inputs audit]** marks wording that waits for the audit now running;
  - **PENDING** marks anything not yet committed.

**Shape of each reply.** Objection, then reply, then numbers and commits, then **Do not claim**.

**Prior objection and rebuttal notes found and reused:**
- `results/rejection.md` (2026-09-11), R1–R6. It was written for an earlier framing of the paper (budget law,
  constructed families). Only its R6 answer carries over to reply (a): conceded on scope, disputed on value.
- `results/review_objection1.md`: the amplification measure, family B. It was upheld in part and changed a committed
  claim. It is useful in reply (b) as a record that upheld objections change claims. Its content now lives in
  Appendix Q only.
- `paper/gap_list.md` (2026-09-11) describes a pre-submission draft structure and is obsolete.
- WP-29 (the committed writer patch) lists "open items for the rebuttal". The discussion phase has since tested
  several of them: the outer-exclusion check (Track 4) and another width-2 variant (W2-A).

---

## (a) "This is a toy setting; everything is width-1 sine."

**Reply.**
- **Conceded on scope.** The certified results (switch brackets, the scaling reduction, the margin identity) are for a
  width-one sine unit. The minimal setting is what makes exact Hessians, global conditional-loss certificates and an
  independent ball-arithmetic re-check possible [P p1:050–051].
- **What has changed since submission.** Registered tests of the lag law now pass outside width-one sine:
  - at GELU (width 1, SGD, warm start);
  - at width 2 on the asymmetric windows (slowed SGD, warm start);
  - in the band task in R^d at a = 1.50 (Adam, free training).
- The certificates remain width-one sine only. The new tests use validated switches, not certified ones.

**Numbers and commits.**
- GELU-T primary: L1 1.000, L2 0.977, L3 0.999, 80 of 80 scored; registration `736b6bf` [GT §3].
- W2-A T: L1 1.000, L2 1.009, L3 1.000, 98 scored; registration `28b2432` [W2 §3].
- 2C at a = 1.50: C1 0.0076 / 0.0091; registration `b22ebd0` [2C §3].

**Do not claim:**
- relevance to practical networks;
- that the certificates extend beyond width-one sine;
- that the new tests remove the "toy" objection. They are still one or two units on synthetic windows.

---

## (b) "Cherry-picking / garden of forking paths."

**Reply.**
- **Every registered prediction is in the census, including failures and unresolved outcomes.** At HEAD (census
  commit `ee0b0cb`) the census holds 278 predictions [CEN]:
  - 256 scored by their registered rules: 135 PASS, 66 FAIL, 8 PARTIAL, 47 UNRESOLVED;
  - 22 with post hoc verdicts: 3 / 6 / 13 / 0;
  - 72 failures in all (66 registered, 6 post hoc), 33 of them central (31 registered, 2 post hoc; the relevance
    classification is itself post hoc, by topic) [CEN; WP-14].
  - At submission it was 236 [S §5]. The 42 added rows are the discussion phase's five tests.
  - A later analysis never replaces a registered verdict. T2-3, Track A's Adam L3 and the Track 1 gate stay failed
    (reply (j)).
- **This phase's five registrations are externally timestamped.**
  - Each of 2A, 2B, GELU-T, 2C and W2-A was frozen and hashed (code, tests, registration text, frozen inputs) before
    any registered seed was trained.
  - Each registration commit carries an OpenTimestamps proof, now attested in Bitcoin block headers [STAMP]:
    - 2A `91a85cb`: blocks 969098, 969100, 969137;
    - GELU-T `d7996d7` and 2B `349911f`: 969103, 969137;
    - 2C `906c07b`: 969143, 969146, 969170, 969173;
    - W2-A `7800b48`: 969332.
  - The block numbers come from commits `549f44f`, `2115c6b` and `190253d`. Independent `ots verify` is PENDING.
  - **Registrations before this phase are repository commits only** [P p9:455–456].
- **Predictions were committed before any crossing was evaluated.** Training evaluated no gap. `observe` asserted the
  registration hashes, the committed predictions hash and every path hash before reading any gap:
  - 2A `42316d9`; 2B `08057a9`; GELU-T `8d39e75`; 2C `d8d3569`; W2-A `2c630f2` [2A §1; 2B; GT §1; 2C §1; W2 §1].
- **Design review.** Each test had a one-page design approved by the author before implementation:
  - 2A, 2B and 2C: `c3840fe`;
  - GELU-T: `4ecc97d`, from draft `5e6323f`;
  - W2-A: `907e225`, from draft `ccce4fa` [GT §1; W2 §1; 2C §1].
  - Every decision rule was tested on constructed PASS, FAIL and UNRESOLVED cases before registration:
    2A 20 tests, 2C 33, GELU-T 42, W2-A 49 (commits `eba559e`, `8dcc690`, `9b2be59`, `596942c`).
- **The forks we took are disclosed.**
  - 2A's t_sw rule was chosen on the a = 1.65 runs [2A §1].
  - GELU-T fixed three details after its pilot, on pilot seeds only [GT §1].
  - W2-A changed five details after its pilot, on pilot seeds only, and adopted the total-derivative κ after
    exploratory runs [W2 §2; W2d].
  - 2C added validity V3 at registration [2C §5].
- **Why this phase has no FAIL.** The 42 committed rows of this phase are 35 PASS, 7 UNRESOLVED and 0 FAIL
  [WP-14; CEN `ee0b0cb`].
  They were designed after post hoc analyses showed where the law applies: slow runs on a tracked branch. They test
  that regime prospectively, and they do not sample training conditions at random. The failures of the fast regime
  came earlier and stay in the census.

**Do not claim:**
- that the whole census is externally timestamped;
- that the discussion-phase tests were chosen without looking at earlier data;
- that the relevance classification was registered;
- that 278 is final (re-read the census at reply time);
- that "no FAIL in this phase" is evidence of generality.

---

## (c) "Circularity: the prediction uses the outcome."

**Reply structure** (the wording in brackets waits for the audit):
1. **What is frozen before training.** Landscape, branch switches, κ, winding and branch rules, the preconditioner rule
   and every decision rule are hashed in `registration.sha256`, then OpenTimestamped [2A §1; GT §1; 2C §1; W2 §1]. No
   frozen quantity comes from a registered run.
2. **What each prediction reads from the run itself.**
   - The trajectory-integrated prediction integrates the linearised recursion along the run's own output-scale path
     (W2-A: the output-weight path v_t) [GTd; W2d].
   - Adam's preconditioner is frozen at t_sw, the first step the output scale reaches the frozen switch [2A §3; 2C §2].
   - In W2-A's T and T′ arms, the lag-free switch itself is read from the run's own v path, "v only, before any gap is
     read", so those predictions are conditional on that path (disclosure C) [W2d; W2 §5].
   - **[pending the prediction-inputs audit: an exact list, per test, of every run-dependent input and the last step
     it is read at.]**
3. **What is never read.** Training and prediction evaluated no placement gap. The observe step checked every committed
   hash before it evaluated one [2A §1; GT §1; 2C §1; W2 §1]. **[pending the prediction-inputs audit: whether any
   input read up to the predicted crossing carries information about the placement event itself.]**
4. **Tests that rely less on the path.**
   - L2, the closed form κχ at t_sw, passes wherever L1 does: GELU-T 0.977 / 0.985; W2-A 1.009 / 1.008 / 0.993
     [GT §3; W2 §3].
   - L5 compares with a lag-free crossing at the run's own switch, so it tests the lag itself [GT §3; W2 §3].
   - 2C's prediction is closed-form type, s_own,d·(1 + κχ), with χ from pre-crossing information [2C §2].
   - **[pending the audit: for runs with κ < 0 (all 120 W2-A T′ runs; 8 + 2 GELU-T runs) the crossing precedes t_sw
     [GT §3; W2 §3]. State what χ at t_sw uses there.]**
5. **What this is not.** It is not a forecast of a whole trajectory from initialisation. The paper already says so for
   the post hoc linear-response analysis [P p49:2642–2644].

**Numbers to have at hand.**
- W2-A: predicted and observed crossing steps are equal in 97 of 98 T runs and 118 of 120 T′ runs; D's observed step
  trails the prediction by 1–6 steps on a ~200-step lag [W2 §3].
- GELU-T: equal in 36 of 80 and 42 of 80 runs (primary, control), within 3 steps in 68 and 75, and the observed step is
  never after the predicted one [GT §3]. A reviewer may read step-level agreement as circularity. The reply must use
  the audit's finding, not an argument.

**Do not claim:**
- "the prediction uses only pre-crossing information" for the trajectory-integrated form (until the audit says so);
- "the switch is predicted in advance" for W2-A T and T′;
- that step-level agreement proves the mechanism.

---

## (d) "The lag law is just linear response; it is trivial."

**Reply.**
- **Agreed that the mechanism is standard.** First-order slow-manifold tracking is Fenichel's correction, and the
  paper says so [P p5:266–268; p9:442]. We do not claim it as new.
- **What is not automatic:**
  1. **A no-fit constant.** κ is computed from the landscape (Hessian, branch tangent, gap gradient) before any
     comparison, and it predicts registered lags:
     - SGD at a = 1.65: 1.056 / 1.024, Spearman 0.995 [TA §2];
     - GELU-T and W2-A: medians 0.977–1.021 across L1/L2 [GT §3; W2 §3].
  2. **The sign.** On W2-A's T′ branch κ < 0, and all 120 scored runs crossed **before** their own switch, as
     predicted (median −43.5 steps) [W2 §3]. A generic "training lags the minimiser" account predicts the wrong
     sign.
  3. **The reference matters.** Against the global own-sample threshold the closed form puts 31/36 arm medians within
     10%; against the tracked branch's switch, 36/36 [P p6:303–306].
  4. **A theorem with a scope.** For fixed-P GD without momentum, tracking and r = κχ + O(χ²) are proved with explicit
     constants. In free training the lag does not vanish as η → 0 (Corollary L3) [MN §15]. The registered free-Adam
     test found the lag unchanged over η = 0.01–0.0025 [P p53 Table 42].
  5. **Where it breaks.** The law is validated up to κχ = 0.1 (reply (f)). It fails or does not apply in fast training:
     T2-3, Track 3A, Track T [P §6].

**Do not claim:**
- novelty of linear response or of slow-manifold tracking;
- that the theorem covers Adam or momentum (it does not [MN §15.5]);
- that the theorem's constants are enclosed for the sine-family runs (not done [MN §15.5]).

---

## (e) "Does it hold for Adam / other optimisers?"

**Reply.**
- **SGD.** At a = 1.65 all three registered criteria pass: 1.056, 1.024, Spearman 0.995 [TA §2]. GELU-T, W2-A and 2B
  are SGD too [GT; W2; 2B].
- **Adam, typical lag.**
  - At a = 1.65: 1.065 and 1.045, both PASS [TA §2].
  - 2C, Adam at a = 1.50: PASS [2C §3].
  - The earlier registered fresh-sample test of the fitted Adam timescale relationship (a = 1.45, 1.60; a fitted
    relationship, not the no-fit law) passed [P p40 Table 27].
- **Adam, per-run ordering.**
  - The registered criterion **failed** at a = 1.65: Spearman 0.25 [TA §2].
  - Post hoc, the preconditioner frozen at half the switch was stale [WP38 §2].
  - The pre-crossing rule this suggested (P at t_sw) was then registered at an unseen a = 1.85 and **passed**:
    Spearman 0.93, 58/75 within 10% (0.773 ≥ 0.75), median 1.058. Registration `9fd1f32` [2A §2].
- **Other optimisers.** AdamW appears only in the band-targeted placement-midpoint equivalence (Appendix P), not in any
  lag-law test [P p43–44]. Adam forced ramps fail even post hoc: warm-up at a stationary point shrinks v̂ 8–88-fold
  [P p53].
- **The theorem** covers fixed-P GD without momentum only [MN §15].

**Do not claim:**
- that Track A's L3 failure is overturned (it stands; 2A is a new test of a new rule);
- that the t_sw rule was chosen blind;
- AdamW, momentum SGD or any optimiser other than Adam and plain SGD for the lag law;
- that 2A's margin is comfortable (57 of 75 was the minimum; 58 passed) [2A §2].

---

## (f) "What is the validity range?"

**Reply.**
- **Registered lower side (2B C1 PASS).** In a redesigned registered test (width-1 SGD forced ramps, learning rate set
  a priori for linear stability, 40 seeds per cell), the lag stayed within 25% of κχ in every cell with κχ ≤ 0.1.
  Cell medians were 0.99–1.21 [2B].
- **Upper side not registered (2B C2 UNRESOLVED).** All four κχ ≥ 0.3 cells were invalid: η·λ_max > 1 on the realised
  trajectory in 39, 40, 17 and 40 of 40 runs [2B].
  - Descriptively, the lag exceeded the prediction by 41–56% at κχ = 0.2 (κχ\* = 0.2 at both a), and by 1.98–5.10× in
    the κχ ≥ 0.3 cells, which are invalid [2B].
  - Post hoc, the fast cells became unstable after the switch, where λ_max off the branch exceeds the a-priori bound
    [2B].
- **Where the other tests sit.** The 36 free-training arms have median κχ 0.007–0.082 [WP38 §1]. The q90 of κχ at t_sw
  is 0.0039 for GELU-T, and 0.00167, 0.0359 and 0.00142 for W2-A's T, D and T′ [GT §3; W2 §3]. Every registered pass
  lies well inside κχ ≤ 0.1.
- The first boundary test stays UNRESOLVED (B1, B2) [WP38 §1].

**Numbers and commits.** Registration `0708ae9`, predictions `08057a9`, scoring `f8f9329` [2B].

**Do not claim:**
- "the boundary lies between 0.1 and 0.3" as a registered result;
- "the law fails at κχ ≥ 0.3" as registered;
- "accurate to 25% up to κχ = 0.2" (the κχ = 0.2 medians are 1.41 and 1.56);
- that 2B is free training (forced ramps after a held warm start) [2Bd].

---

## (g) "Beyond the paper's activations and widths?"

**Reply.**
- **GELU (GELU-T), PASS in both arms** (registration `736b6bf`). Full-batch SGD at η = 0.03 after a warm start with w₂
  held at 0.5·s_pop [GTd].
  - Primary: L1 1.000, L2 0.977, L3 0.999, L4 and L5 pass, 80 of 80 scored.
  - Control: 1.000 / 0.985 / 0.999 [GT §3].
- **Width 2, asymmetric windows (W2-A), PASS in T, D and T′** (registration `28b2432`). Warm start with v held at
  s₀ = 0.2225, then full-batch SGD with the output learning rate slowed to ρη, ρ = 2⁻¹⁰ (T) and 2⁻¹² (T′); D at η 0.3,
  ρ 1 [W2d; W2 §3].
  - T: 1.000 / 1.009 / 1.000; D: 1.021 / 1.008 / 0.997; T′: 1.000 / 0.993 / 1.000 [W2 §3].
  - Only 98 of 200 random starts reached T. Five registration details were changed after the pilot, on pilot seeds
    only [W2 §2–3].
- **Band task in R^d (2C).** PASS at a = 1.50 for d = 2 and 4; UNRESOLVED at a = 1.30 on validity V3 (0.877, 0.804 <
  0.90), a condition added at registration [2C §3, §5].
- **Path-conditioning.** GELU-T and W2-A integrate along each run's own output-scale or output-weight path. W2-A's T and
  T′ switch is read from that path [W2d]. **[pending the prediction-inputs audit]**
- **What still fails** [P §6]:
  - free, fast training outside width-one sine: T2-3 (Adam, width 2) FAIL, median crossing 3.29× the threshold;
  - Track 3A GELU, SiLU and Mish (Adam from initialisation): primary UNRESOLVED, secondary FAIL;
  - the width-2 no-gating prediction: FAIL;
  - SiLU and Mish show the wrong lag sign post hoc and were not retested [P p55].
- **Branch selection** is not predicted anywhere.

**Do not claim:**
- general activations, widths above 2, depth, or Adam at GELU or at width 2;
- prediction from initialisation or of branch selection;
- that T2-3 or Track 3A is overturned;
- that 2C passed at a = 1.30.

---

## (h) "Relevance to real networks / simplicity bias?"

**Reply (honest negatives).**
- **Track T** (linear-plus-slab, width-4 tanh, registered `91ef9cf`) is **UNRESOLVED** on both validity conditions:
  22 of 40 runs crossed (30 required), and the median χ at crossing was 7.43 against a bound of 0.06. Would-be verdicts
  were FAIL on both criteria (0.727 of crossings at or above the switch; median crossing 2.86× the switch) [WP37].
  - The fixed-scale landscape does show a single reproducible switch under weight decay [WP37].
  - Transfer to the benchmark is not established.
- **Track 1** (POST HOC). The registered gate to a prospective fold test **FAILED**, so no fold test was registered
  [WP39; SB]. Linear-plus-slab late crossers cross a median 2.17× the linear branch's fold and are never on a branch.
  GELU crosses at its branch's switch (median ratio 1.0017) [WP39].
- **MNIST** did not produce the setting the test needed. Monotone activations beat non-monotone at width one (0.8658
  vs 0.7715, p = 0.0002). This is reported as a failure to obtain the setting [S §3].
- **CIFAR-10** measures sensitivity of an activation comparison to training length (750.4 → 150.6 errors). It does not
  test the mechanism [S §4].

**Do not claim:**
- that the account explains simplicity bias;
- that it predicts feature changes in practical networks;
- that Track T or Track 1 supports transfer;
- that a slower Track T would have passed (not run [WP37]).

---

## (i) "Which claims are certified and which are only validated?"

**Reply.** The paper's Table 1 separates the statuses [P p4 Table 1].
- **Certified and independently re-checked** in 80-bit Arb ball arithmetic by a checker that shares no code with the
  searches [P p10:516–524]:
  - the finite-a placement brackets (a = 1.30–1.60);
  - the Ĝ enclosures;
  - the limit-switch bracket and its localisation;
  - K;
  - the first-order Krawczyk boxes;
  - the PD and ring certificates.
- **New since submission (Track 4)** [T4 §3; CA; T4J]:
  - Outer exclusion is re-derived by a fresh Arb branch and bound: 41 of 41 A-intervals pass, margins
    ≥ 1.006·10⁻⁴ (annulus) and ≥ 1.649·10⁻⁴ (solve bracket). One convexity lemma is used as stated, not
    machine-checked.
  - The signs at both ends of every solve bracket are confirmed: the limit and 12 finite-a ends.
  - It does not check that the sign changes exactly once inside each bracket.
- **New theorems** [MN §15–§17]:
  - the lag law for fixed-P GD without momentum (§15);
  - global uniqueness of the continued branch for ε ≤ 1.4·10⁻¹², s-form (§16);
  - small-scale compactness on the 800-point population; it fails on samples, proved (§17).
- **Validated, not certified:**
  - the switches at a = 1.65 and 1.85 [TA §1; 2A §1];
  - GELU's switches [GTd];
  - width-2 thresholds [W2d];
  - the own-sample R^d switches [2C §2];
  - the lag-law constants (numerical, not interval) [P p48:2586–2587].

**Do not claim:**
- that every certificate is independently verified;
- that margins are verified as numbers (only their signs);
- Theorem G's finite-ε parts;
- that the convexity and localisation lemmas are machine-checked;
- that the sharp limiting value is "independently verified" until the author decides (PENDING, `revision_plan.md` P5).

---

## (j) "What failed?"

**Reply (verdicts stand; later results do not replace them).**
- **T2-3** (width 2, asymmetric windows, Adam, registered amendment 1, `2505b2d`): **FAIL** [CEN; P p42 Table 29].
  - 93.7% of crossings are at or above the threshold, but the median is 3.29× it, against the registered bound 1.25.
  - The follow-ups T2-3b and T2-3c are UNRESOLVED, and T2-3d FAILED.
  - W2-A, a different slow SGD design, does not overturn it [W2 §5].
- **Track 1 gate: FAIL** (POST HOC analysis of existing runs). The v3 late crossings sit at a median 2.17× the fold
  (15% band required), so no prospective fold test was registered [SB; WP39].
- **Track A, Adam L3 at a = 1.65: FAIL**, Spearman 0.25 < 0.5 (registration `fcd2e46`) [TA §2; CEN].
  - The post hoc diagnosis and 2A's registered pass with a new rule stand beside it, not in place of it [2A §3].
- **The rest of the record.** 72 failed predictions in the committed census (66 registered, 6 post hoc), 33 of them
  central; the full list is in the supplementary [CEN; WP-14; S §5.1].
  - Also: Track 3A secondary FAIL at GELU, SiLU and Mish [P p55].
  - The width-2 no-gating prediction FAIL [P p43].
  - Track T UNRESOLVED [WP37].
  - 2B C2 and 2C at a = 1.30 UNRESOLVED [2B; 2C].

**Do not claim:**
- "the central claims never failed";
- that a failure was "explained away";
- that any failure's verdict changed after the fact.

---

## (k) (likely) "The new passes rely on engineered conditions: warm starts, slowed learning rates."

**Reply.**
- **Yes, by design, and stated as such.** The account is claimed only where a run tracks a branch and grows slowly
  relative to relaxation [P p8:394; 2B].
- The new tests put training in that regime on purpose, to test tracking outside width-one sine:
  - a warm start at held output scale (GELU-T; W2-A) [GTd; W2d];
  - slow full-batch SGD (η = 0.03; W2-A's output learning rate slowed by 2⁻¹⁰ and 2⁻¹²) [GT §3; W2 §3].
- They do not test whether ordinary training reaches that regime. Where it does not, the registered results are failures
  or unresolved (reply (j)).
- The only random-start arm at width 2 had 98 of 200 runs on the target branch [W2 §3].
- Path-conditioning: **[pending the prediction-inputs audit]**.

**Do not claim:**
- that the conditions are typical of practical training;
- that W2-A says anything about full-speed width-2 training.

---

## PENDING items that affect these replies

| id | item | replies |
|---|---|---|
| P3 | RESOLVED: headline 278 committed in `ee0b0cb` during this task. Re-read it at reply time. | (b), (j) |
| P4 | The prediction-inputs audit: path-conditioning wording; negative-κ runs where χ at t_sw comes after the crossing; what step-level agreement shows | (c), (g), (k) |
| P5 | The author's decision on the sharp limiting threshold's status after Track 4 | (i) |
| P8 | `ots verify` of the five OpenTimestamps proofs (block numbers are from the upgrade commit messages) | (b) |
| — | The actual reviews (early November): match, merge or drop drafts; respect length limits | all |
