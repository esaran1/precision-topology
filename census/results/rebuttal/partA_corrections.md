# Part A, list 3: corrections to the submitted paper (READ-ONLY audit, 2026-11-14)

Scope: every sentence or number in the submitted paper that a result committed after the tag `paper-submitted-v5`
(5f8bd29, 2026-09-26 04:09 EDT; tag object 04:52 EDT) changes or withdraws. Nothing was computed, run, tested or
committed for this file; no other file was edited.

Severity: **WITHDRAW** = the claim no longer holds; **CORRECT** = a number or the wording must change;
**SCOPE** = the claim holds but must be narrowed or qualified. "Upgrade" or "downgrade" says which way the later
result moves the claim.

## Summary

**Count by severity (31 items):** WITHDRAW 8 · CORRECT 11 · SCOPE 12. In addition, 1 flagged item that is inferred,
not established by a later result (F1, not counted), and a list of submitted claims that were checked and need no
correction (§5).

**The 5 most important items**
1. **The a = 1.65 per-run claims (S1–S5).** They stand as registered, but they are path-conditioned, not forecasts.
   Phase 1B (221c8ce) re-scored them with a strictly causal forecaster. SGD fails C3 (median r_obs/r_fc 1.216) and
   Adam fails C1 and C2 (39/74, 43/74). "Every run within 10% under SGD" must not be read as a forecast.
2. **The Adam t_sw rule (C7, C8).** The submitted paper names it as "the rule for a future registered test". Test 2A
   registered it and PASSED as registered (A1 0.932, A2 0.773, A3 1.058). The pass does not hold as a causal forecast
   (Phase 1B: A1 0.388, A2 0.214, A3 1.354). The registered causal Adam forecast at a = 1.77 PASSED (9d9449e).
3. **What lies outside width-one sine (W7, W8).** "Outside this setting, registered predictions fail or remain
   unresolved" and "agreement on a tracked branch outside width-one sine is post hoc" are no longer true. Registered
   tests later passed at GELU, at width 2, at width 4 and in R^d (a = 1.50). Phase 1C replicated the GELU and width-2
   arms causally on fresh seeds.
   Each pass is for runs started on a branch or trained slowly; branch selection is still not predicted.
4. **Theorem 1, small-scale compactness (W1, W2, S8, S9, S10).** The paper says it "remains assumed". It is now
   PROVED at width 1 on all three objectives (math note §21, 526b606), with the alias forced on the 800-point
   quadrature. At width 2 it is proved on the symmetric continuous windows only (§22, 7da9bec). On the asymmetric
   windows the minimiser is not attained, and on the quadratures it is not proved.
5. **The independent certificate checks (W3–W6, C2).** "Outer exclusion and the finite and limiting solve brackets
   remain search-certified only" is withdrawn. Track 4 (69af96f) checked them independently: 41/41 outer-exclusion
   intervals, the limit solve bracket, and 12/12 finite-a solve ends. The sharp R∞_glob is still not fully
   independently verified, but for a narrower reason.

Next in weight: equation (5) and the global threshold (S7, C1); the width-2 values below the ramp limit (C10); the
census totals (C11).

## 1. Sources and how the submitted text was read

- **The tag tree has no main-text .tex and no compiled main PDF.**
  - `git ls-tree -r paper-submitted-v5` lists no `*.tex`, `*.bbl` or `*.sty`. Its only PDFs are figures and
    `paper/sources/*`. Its `paper/` folder holds writer inputs (`WRITER_INPUTS*.md`, `results_draft.md`, ...), not the
    paper.
  - The 5-page supplement built from the tag is `dist/supplementary_v5/supplement.pdf` (untracked; built 2026-09-26
    05:29).
- **The main-text quotes below** come from `~/Downloads/Precision_Topology_ICLR (18).pdf`: 56 pp., pdfTeX, created
  2026-09-26 05:38 EDT, 46 min after the tag. It is the same file `results/track5/revision_plan.md` uses for its page
  map ([P]). **Confirmed by the author 2026-11-14:** sha256
  `2648fb643d7b1366ea14a276d2968f0cab2af2d1c14bfb95ac4ff16667872ca6`, identical to the OpenReview download of
  submission 54786 (hash recomputed here with `shasum -a 256`; matches). This closes PENDING P1 of `revision_plan.md`.
- **Quoting.** Text was extracted with pypdf. Spacing was restored and math was transcribed (e.g. "s → 0"); the words
  are unchanged.
  - Locations are PDF page and section.
  - Margin line numbers `[P pN:lines]` are given only where `revision_plan.md` had already mapped them.
- **Later sources read:**
  - results: `track5/revision_notes.md`, `phase1b_results.md`, `prediction_inputs_audit.md`, `phase1c_results.md`,
    `trackA_results.md` (a = 1.77), `phase2a*_results.md`, `phase2b_results.md`, `phase3_results.md`,
    `trackL_results.md`, `final_posthoc.md` (983b594), `ramp_crossover_check.md` (09fef81),
    `theorem1_hypotheses.md` (addenda), `certificate_audit.md` (Track 4, NaN audit), `track4_writer_inputs.md`,
    `track2b_writer_inputs.md`, `track2c_writer_inputs.md`, `uniform_eps/U2_design.md`;
  - `math_note_v2.md` §§15–23;
  - `git log` after the tag (212 commits);
  - `git diff paper-submitted-v5 HEAD -- src/verify_ledger.py`.

## 2. WITHDRAW (8)

**W1. Small-scale compactness "remains assumed" (upgrade).**
- **Submitted:** main text §3, p3 [P p3:137–138]: "For the sine family, attainment and large-scale compactness are
  proved. Small-scale compactness remains assumed, since fixed-s localization gives no common compact set as s → 0."
- **Changed by:** math note §17 (c3afcf2) and §21 (526b606); `theorem1_hypotheses.md` addendum (bbe46cb).
  - Width 1: PROVED on all three objectives.
  - Continuous windows: s₁ ≥ 0.643 and |w₁| ≤ W ≤ 11.93 at every tabulated a (Arb-certified constants).
  - Float 800-point quadrature, in exact rationals: s₁ ≥ 3.7455·10⁻¹⁹, W ≤ 1.3621·10¹⁹.
  - Exact linspace population: the alias is forced; every admissible W ≥ 623,614.
  - Training samples: compactness fails (Prop. C′), but the minimisers are still unplaced (C3).
- **Replacement:** "For the sine family at width one, attainment and large- and small-scale compactness are proved on
  the continuous windows and on the 800-point quadrature; on the quadrature the compact set must contain the aliased
  class-mean maximizer, so the s → 0 limit there is the alias, which is unplaced."

**W2. "A common compact set is not proved" (upgrade).**
- **Submitted:** App. A.2, p16 [P p16:836…]: "At small scale, W(s) is finite for each s > 0 but diverges as s → 0;
  a common compact set is not proved."
- **Changed by:** §21, 526b606 (as W1).
- **Replacement:** "The fixed-s localization bound W(s) diverges as s → 0, but a common compact set is proved by a
  separate argument (math note §21): |w₁| ≤ 11.93 for s < 0.643 on the continuous windows, and |w₁| ≤ 1.3621·10¹⁹ for
  s < 3.7455·10⁻¹⁹ on the quadrature as evaluated."
  - Keep the next sentences: quadratic growth near the aliases is still not examined.

**W3. "Remain search-certified only", main text (upgrade).**
- **Submitted:** main text §4, p3 [P p3:160–161]: "Outer exclusion and the finite and limiting solve brackets remain
  search-certified only."
- **Changed by:** Track 4, 69af96f (`certificate_audit.md` "Track 4"; `certificate_checks/track4_summary.json`).
  - Outer exclusion: 41/41 pass (40 annulus sub-intervals, margin ≥ 1.006·10⁻⁴; solve bracket ≥ 1.649·10⁻⁴).
  - Limit solve bracket: both ends pass.
  - Finite-a solve ends: 12/12 pass at a = 1.30–1.60.
- **Replacement:** "Outer exclusion (41 of 41 intervals) and the signs at both ends of every solve bracket are also
  independently re-derived; that each sign changes exactly once inside its bracket, and one convexity-in-A lemma, are
  not machine-checked."

**W4. AI Use statement, same claim (upgrade).**
- **Submitted:** p10 [P p10:523–524]: "Outer exclusion and the finite and limiting solve brackets were not
  independently checked; only the searches support them."
- **Changed by:** 69af96f; replacement text fixed in `track4_writer_inputs.md` §3 addendum (a2060b2).
- **Replacement (as written there):** "It also re-derives the outer exclusion by a fresh Arb branch and bound (41 of 41
  A-intervals; one convexity lemma is used as stated in the math note, not machine-checked), and confirms the signs at
  both ends of every solve bracket (the limit bracket and 12 finite-a ends at a = 1.30–1.60). It does not check that
  the sign changes exactly once inside each bracket."

**W5. App. E.1, same claim (upgrade).**
- **Submitted:** p22 [P p22:1150–1154]:
  - "The outer-exclusion certificates and the solve brackets are verified by the original searches only."
  - "Independent outer exclusion and finite and limiting solve checks were not run because of their projected cost;
    no verification is claimed for them."
- **Changed by:** 69af96f (1.93 h of single-process CPU in total).
- **Replacement:** list the three Track 4 checks with their margins (W3), and delete the "not run" sentence.

**W6. App. F, the four-link chain described as unchecked (upgrade).**
- **Submitted:** p24:
  - "…using the outer-exclusion link that has not been independently checked, gives R∞_glob ∈ [0.1985926, 0.1985927]."
  - "The same search-certified four-link argument, whose outer exclusion and solve bracket have not been
    independently checked, establishes global uniqueness on the solve bracket A ∈ [1.05875, 1.06]."
- **Changed by:** 69af96f.
  - The limit solve bracket's Krawczyk zero is checked at both ends, and all four links are checked for every A in
    the bracket.
  - Not checked: transversality d(margin)/dA ∈ [0.532, 0.536], and uniqueness of the sign change.
- **Replacement:** "…using the outer-exclusion link (independently re-checked) …". Also: "… whose outer exclusion
  and end signs are independently re-checked, but whose transversality is search-certified only …".

**W7. "Agreement … outside width-one sine is post hoc" (first clause).**
- **Submitted:** main text §6, p8 [P p8:406–408]: "Agreement on a tracked branch outside width-one sine is post hoc,
  and branch selection could not be predicted in advance (Appendices N and X)."
- **Changed by:** registered tests after the submission:
  - GELU-T: registration 736b6bf, result 58aa960; PASS in both arms.
  - W2-A, width 2, asymmetric windows: 28b2432 → 7eb4538; PASS in arms T, D and T′.
  - Phase 3, width 4: 7a9d791 → 5a3cb7a; Q, S and T4 PASS.
  - Test 2C, R^d: b22ebd0 → b863a54; PASS at a = 1.50, UNRESOLVED at a = 1.30.
  - Phase 1C, causal replication on fresh seeds: 6433edc → 6522a15; W1, G, T and T′ PASS.
  - Caveats:
    - GELU-T, W2-A and Phase 3 hold each run at a fixed output scale before release (Phase 3: "mechanism tests from a
      branch point"; 0/710 random holds land on a genuine four-unit point). Test 2C is free Adam training and passes at
      a = 1.50 only.
    - The GELU-T and W2-A predictions read the run's own output path at or after the crossing (prediction-inputs audit
      432a5a2: r_traj in 80/80, 80/80, 98/98 and 120/120 runs; W2-A T′'s switch lies after the crossing in 120/120).
    - The Phase 1B causal re-score nevertheless passes them (GELU-T both arms, W2-A T and T′; D has no tolerance).
- **Replacement:** "For runs started on a branch or trained slowly, registered tests at GELU, at width two and four,
  and in R^d (a = 1.50) passed, and a causal replication on fresh seeds passed; branch selection could not be
  predicted in advance."
  - The second clause stays.

**W8. Abstract, last sentence.**
- **Submitted:** p1 [P p1:026–028]: "This quantitative predictive link is established for width-one sine networks that
  reach a branch and grow slowly relative to relaxation; outside this setting, registered predictions fail or remain
  unresolved, and agreement conditioned on the eventual branch is explanatory."
- **Changed by:** as W7.
- **Replacement:** "Registered tests establish this link for width-one sine networks, and, for runs started on a branch
  or trained slowly, at GELU, at width two and four, and in R^d (a = 1.50); free fast training outside width-one sine
  fails or is unresolved, and branch selection is not predicted."
  - If the a = 1.65 clause stays in the abstract, it must carry the qualifier of S1.

## 3. CORRECT (11)

**C1. "No explicit ε₀ is claimed" (upgrade, with status).**
- **Submitted:** App. F, "Finite-a continuation", p24 [P p24:1266–1273]: "… so its transversal zero moves by O(ε). No
  explicit ε₀ is claimed."
- **Changed by:**
  - Theorem G, math note §16 (e469035): the continued branch is the unique global minimiser and
    s_glob = A_ε ε^{−3/2} for 0 < ε ≤ **1.4·10⁻¹²**, s-form only. It is proved in mpmath interval arithmetic. Its
    finite-ε parts are not independently checked (`certificate_audit.md`); its outer-margin input is.
  - Theorem U1, §20 (042691a): the same conclusions for 0 < ε ≤ **4.1·10⁻⁶**, with mixed status:
    - the ring and PD links are proved;
    - the switch Krawczyk is certified in interval arithmetic;
    - the ε-uniform outer exclusion is **producer-standard**: float branch and bound with a 10⁻⁹ allowance and
      τ = 10⁻⁷, **not yet Arb-checked** (U3).
  - The R-form (c₁) is not covered by either theorem.
- **Replacement:** "For 0 < ε ≤ 4.1·10⁻⁶ the continued branch's switch is the global threshold (math note §20; its
  ε-uniform outer exclusion is a producer-standard float certificate not yet independently checked; with every link
  proved or certified, ε₀ = 1.4·10⁻¹², §16). The R-form is not covered."
  - **Superseded 2026-11-14 by the author's decision:** the main text keeps the submitted wording and claims no explicit ε₀; the appendix states Theorem G (ε₀ = 1.4·10⁻¹², s-form) with its exact check status (outer margin re-checked in ball arithmetic; finite-ε parts checked by the producing code only). 4.1·10⁻⁶ is not quoted. See `partB_input_blocks.md` block 2.
  - Do **not** cite U2 (see S7).

**C2. Why the sharp R∞_glob is "not yet fully independently verified" (reason changes; status holds).**
- **Submitted:** App. E.1, p22: "The sharper interval [0.1985926, 0.1985927] needs the Krawczyk zero to be the global
  switch, which uses outer exclusion. It is therefore certified but not yet fully independently verified."
- **Changed by:** 69af96f. The outer-exclusion link is now independently checked. What remains unchecked
  (`certificate_audit.md`, "What remains unchecked"):
  - the annulus ring and PD runs, whose centre is a parameter their JSON files do not record;
  - the convexity-in-A lemma, used as stated;
  - transversality.
- **Replacement:** "Its outer-exclusion link is now independently re-checked; the interval is still not fully
  independently verified, because the convexity-in-A lemma is used as stated and the annulus ring and PD runs do not
  record their centre."
  - Table 1's "Sharp limiting threshold" wording is author decision P5 (revision_plan §5.2).
  - `track4_writer_inputs.md` says: do not say the sharp value is independently verified.

**C3. "No tracking guarantee for discrete … SGD" (upgrade, partial).**
- **Submitted:** §5.1, p6 [P p6:301]: "The continuous-time theory supplies no tracking guarantee for discrete Adam or
  SGD."
- **Changed by:** math note §15 (2e936fd, 389fd0d). Theorems L1 and L2 and Corollary L3 are PROVED for preconditioned
  GD with a fixed P and no momentum:
  - r = κχ + O(χ²) + τ with explicit constants;
  - lag → κχ̄ + O(χ²) as η → 0.
  - Not covered: Adam (moving v̂, momentum). The constants are not enclosed for the sine family, so the theorems are not
    a certified statement about the a = 1.30–1.60 runs.
- **Replacement:** "For full-batch gradient descent with a fixed preconditioner and no momentum, the first-order lag
  holds with explicit error bounds (math note §15); Adam is not covered, and the constants are not enclosed for the
  sine family."

**C4. Table 1, "Lag law" (upgrade, partial).**
- **Submitted:** p4: "Lag law | First-order approximation, derived post hoc."
- **Changed by:** §15 (as C3).
- **Replacement:** "Derived post hoc; a theorem for fixed-P GD without momentum (constants not enclosed for the sine
  family); for Adam a first-order approximation."

**C5. What the a = 1.65 predictions read (main text; disclosure incomplete).**
- **Submitted:** §5.1, p6 [P p6:313–323]: "Branch and preconditioner predictions use information available there; the
  trajectory-integrated calculation also uses the measured output-scale path."
- **Changed by:** the prediction-inputs audit (432a5a2; `prediction_inputs_audit.md` §§1, 5, 8).
  - **The rule-point search.** It reads s up to t_top, the first step with s ≥ s_frozen. t_top lies after the observed
    crossing in 4/74 Adam runs (up to 140 steps) and 4/63 SGD runs (up to 181 steps). The value of t_R is unchanged when
    every output weight at or after the crossing is replaced (74/74, 63/63).
  - **r_traj.** It reads the path through its own predicted step, which is at or after the observed crossing in 34/74
    Adam runs (1 at, 33 after, up to 140 steps) and 5/63 SGD runs.
  - **Replacing post-crossing outputs** by extrapolation leaves the medians unchanged: Adam 1.0651, Spearman 0.2489;
    SGD 1.0555, Spearman 0.9945.
- **Replacement:** "Branch and preconditioner predictions use the state at that point; the trajectory-integrated
  calculation uses the run's measured output-scale path up to its predicted crossing, which lies at or after the
  observed crossing in 34 of 74 Adam and 5 of 63 SGD runs."

**C6. The same disclosure in App. V.7.**
- **Submitted:** p51: "Branch and preconditioner rules use information through this point. The trajectory-integrated
  model then uses the measured s_t path with P frozen at the rule point."
- **Changed by:** 432a5a2 (as C5).
- **Replacement:** "…rules use the state at this point; locating the point reads s up to t_top, which follows the
  crossing in 4 Adam and 4 SGD runs without changing t_R. The trajectory-integrated model then uses the measured s_t
  path through its predicted crossing (at or after the observed one in 34/74 Adam, 5/63 SGD runs)."
  - Keep "No rule point was at or after an observed crossing": it holds (t_R is 78–744 steps before the crossing for
    Adam and 205–647 for SGD).

**C7. The t_sw preconditioner rule, "for a future registered test".**
- **Submitted:** main text p7 [P p7:351–356] and App. V.7, p52 [P p52:2793–2798]: "A pre-crossing rule chosen after
  seeing these runs, taking the preconditioner at the first step the output scale reaches the occupied branch's
  switch, gives 0.95 (91% within 10%); we name it as the rule for a future registered test."
- **Changed by:**
  - Test 2A, a = 1.85 (registration 9fd1f32, predictions 42316d9, scored 252f8b3): PASS as registered. A1 Spearman
    0.932; A2 58/75 = 0.773 within 10%; A3 median ratio 1.058.
  - Phase 1B (221c8ce): **not a causal forecast.** P at t_sw lies after the causal cutoff. It is 30–488 steps after
    t_R, before the crossing in 75/75 (audit 432a5a2). Causally re-scored with P at t_c − 1, 2A fails A1, A2 and A3
    (0.388, 0.214, 1.354; n 56, 19 misses) and C1, C2 and C3 (55/75, 55/75, 1.354).
  - The registered strictly causal Adam per-run forecast at a = 1.77 (173a612 → 9d9449e): PASS. C1 93/98, C2 93/98,
    C3 0.940 in [0.90, 1.10], C4 [−59.3, −47.4].
- **Replacement:** "…; registered at a = 1.85, it passed its criteria (Spearman 0.93; 77% within 10%), but it reads
  the optimizer state after the forecast point and does not hold as a causal forecast; a strictly causal Adam forecast
  registered at a = 1.77 passed (95% of runs within 10 steps)."
- "Pre-crossing" is accurate; "causal" or "advance" must not be used.

**C8. "Await prospective support".**
- **Submitted:** §8, p9 [P p9:461–463]: "The revised Adam measurement rule and the suggested predicted-lag boundary
  also await prospective support."
- **Changed by:**
  - 252f8b3 and 221c8ce (as C7).
  - Test 2B (0708ae9 → f8f9329): C1 PASS in all six cells with κχ ≤ 0.1 (medians 0.990–1.209, band [0.75, 1.25]);
    C2 UNRESOLVED (all κχ ≥ 0.3 cells invalid: η·λ_max > 1 in 39, 40, 17 and 40 of 40 runs).
- **Replacement:** "The revised Adam rule passed its registered test but not as a causal forecast; a causal Adam rule
  passed at a = 1.77. The lower side of the predicted-lag boundary (κχ ≤ 0.1) passed a registered test; its upper side
  is unresolved."

**C9. "These three cells suggest, post hoc, κχ ≲ 0.1".**
- **Submitted:** §6, p8 [P p8:394–401]: "These three cells suggest, post hoc, a condition on predicted lag, κχ ≲ 0.1,
  rather than on χ alone."
- **Changed by:** Test 2B (f8f9329; `track2b_writer_inputs.md`). The 2B replacement sentence is already drafted in
  revision_plan §4.2.
- **Replacement:** "A registered redesign confirmed the lower side: the lag stayed within 25% of κχ in every cell with
  κχ ≤ 0.1; cells at κχ ≥ 0.3 could not be scored."
- Do not say the boundary is located between 0.1 and 0.3: C2 is UNRESOLVED, and κχ* = 0.2 is descriptive only.

**C10. Width 2, asymmetric windows: "unplaced minima at s ≤ 0.4217" (downgrade of wording; verdict unchanged).**
- **Submitted:** App. N.1, p41: "A validated search finds unplaced minima at s ≤ 0.4217 and placed minima at
  s ≥ 0.5623 on a 17-scale grid, with bisection brackets ∈ [0.4371, 0.4532]."
- **Changed by:** math note §22.6 (7da9bec) and the ramp-limit crossover check (09fef81, `ramp_crossover_check.md`).
  - Below s_ramp ∈ [0.1576358, 0.1576364] the landscape's infimum is a ramp limit, which lies below the registered
    retained loss:
    - by **1.144·10⁻³ at s = 0.1**;
    - by **3.352·10⁻⁴ at s = 0.13335**.
  - So at those two grid points the registered "conditional minimiser" is not the infimum. At s = 0.1 the committed
    point is a partial escape (α₁ = 174.9, weight 0.0245).
  - The infimum there is not attained: proved on the continuous windows (K12); argued only on the 800-point set.
  - It is unplaced (|c̄| = 2.71 and 2.01 > a/1.4).
  - T2-1's status at every scale, its single switch and its PASS are unchanged.
  - Every registered width-2 landscape quantity lies above s_ramp; the closest is W2-A's s₀ = 0.2225 = 1.41·s_ramp.
- The submitted PDF prints no retained-loss value at these scales; only this sentence is affected.
- The wording "starts near the width-2 conditional minimiser at its initial scale" does **not** occur in the submitted
  PDF or the tag tree. T2-3's "matched initialization" (Table 29) starts all runs below s_ramp, but no submitted
  sentence places them near a minimiser.
- **Replacement:** "A validated search finds unplaced retained points at s ≤ 0.4217 (below s ≈ 0.158 the infimum is an
  unplaced ramp limit that the retained points do not attain) and placed minima at s ≥ 0.5623…"

**C11. Census totals (numbers).**
- **Submitted:**
  - App. U, p47 [P p47:2536–2538]: "The full census has 236 predictions: 214 scored by registered rules (100 passed,
    66 failed, 8 partial, 40 unresolved) and 22 assigned post hoc (3 passed, 6 failed, 13 partial)."
  - Supplement §5, p2: "There are 236 predictions. Of these, 214 are scored by their registered rules: 100 passed, 66
    failed, 8 partially passed and 40 unresolved."
  - Supplement Table 1, total rows "100 66 8 40 214" and "236".
- **Changed by:** the census commits 1a29593 and ee0b0cb, and the ledger's expected values (the only expected values
  changed in `verify_ledger.py` since the tag):
  - 236 → **278** predictions;
  - registered 214 → **256**: 135 pass, 66 fail, 8 partial, 47 unresolved;
  - post hoc 22 (3/6/13/0), unchanged;
  - by unit 296 → 338 (169/99/13/57);
  - central failures **33** (31 + 2), unchanged.
- **Caveat:** the census at HEAD stops at round 2026-09-30 (W2-A). Phase 1C, 2A, 2B, 2A-PS, 2A-PS2, Phase 3, Track A
  (a = 1.77) and Track L are registered but **not yet in the census**. Adding them is an author counting decision.

## 4. SCOPE (12)

**S1. Abstract, the a = 1.65 sentence.**
- **Submitted:** "At an activation value with no prior crossing data, its registered test gives median per-run
  observed/predicted ratios of 1.07 under Adam and 1.06 under SGD; individual predictions are accurate under SGD,
  where every run is within 10%, but not under Adam, where 22% are."
- **Changed by:** Phase 1B (221c8ce; `revision_notes.md` item 1). The frozen causal forecaster at f = 0.90 gives:
  - SGD: C1 63/63 and C2 63/63 pass; C3 median 1.216 FAIL; the registered L1 recomputed with r_fc fails (1.216).
  - Adam: C1 39/74 and C2 43/74 FAIL (8 misses); C3 median 1.158.
  - The registered verdicts stand: SGD L1–L3 PASS; Adam L1 and L2 PASS, L3 FAIL.
- **Replacement:** "…its registered, path-conditioned test gives median per-run ratios of 1.07 (Adam) and 1.06 (SGD);
  every SGD run is within 10% given its measured output-scale path; these are not causal forecasts."
- Optionally add: Phase 1C's registered causal width-1 SGD arm at a = 1.58 passed (C1, C2 84/84; median r_obs/r_fc
  1.169 within ±0.20).

**S2. Introduction, the a = 1.65 sentence.**
- **Submitted:** p1: "At a = 1.65, with no prior crossing data, the test confirms typical lag under both optimizers and
  individual ordering under SGD; Adam's ordering criterion fails. … The original Adam metric came from measured runs;
  the new test freezes its measurement rule before training."
- **Changed by:** 221c8ce (as S1).
- **Replacement:** add "given each run's measured output-scale path (not a causal forecast)" after "confirms".

**S3. Contribution bullet 4.**
- **Submitted:** p2: "Prospective tests predict width-one crossings and typical Adam and SGD lag, while rejecting
  Adam's individual ordering."
- **Changed by:** 221c8ce.
- **Replacement:** "Registered tests predict width-one crossings and, conditional on the measured output-scale path,
  typical Adam and SGD lag, while rejecting Adam's individual ordering."

**S4. Table 1, "The a = 1.65 test" row.**
- **Submitted:** p4: "Registered; uses each run's measured output-scale growth; accurate per run under SGD, in the
  median under Adam."
- **Changed by:** 221c8ce.
- **Replacement:** append "; path-conditioned, not a causal forecast (post hoc causal re-score: SGD median ratio 1.22,
  Adam C1 and C2 fail)."

**S5. The a = 1.65 results text and Table 2.**
- **Submitted:**
  - §5.1, p6: "For both versions and both optimizers, the median per-run observed/predicted ratio is within about 7% of
    1 (Table 2). Under SGD every predicted run is within 10% and the rank correlation is 0.995."
  - Table 2, p8 (1.065 / 1.056; 22% / 100%; 0.25 / 0.995).
  - App. V.7, p51–52: "SGD's per-run integrated ratios have 10th–90th percentiles 1.04–1.06; every predicted run is
    within 10%, with median absolute lag error 0.005."
- **Changed by:** 221c8ce; 432a5a2.
- **Replacement:** keep every registered number. Add one sentence: "These predictions read each run's output-scale
  path up to the predicted crossing; with that path replaced by a strictly causal forecast (post hoc), the SGD median
  ratio is 1.216 and Adam fails both per-run criteria (39/74, 43/74)."

**S6. §6, "The predictive link is established for width-one sine networks …".**
- **Submitted:** p7–8 [P p7:375–377]: "The predictive link is established for width-one sine networks on the interval
  geometries tested under Adam and SGD, with prospective own-sample tests at a = 1.30, 1.45, 1.50, 1.60 and the lag
  test at a = 1.65."
- **Changed by:** 221c8ce (the a = 1.65 lag test is path-conditioned); W7's tests (the scope is now wider).
- **Replacement:** "… and the path-conditioned lag test at a = 1.65; registered causal forecasts passed at a = 1.58
  (SGD) and a = 1.77 (Adam)."

**S7. Equation (5) and the global threshold (holds; the branch scoping must be kept).**
- **Submitted:**
  - p4, Eq. (5): "Along the continued branch, the switch scale is s_br = A* ε^{−3/2}(1 + c_s ε + O(ε²)), …"
  - "At every certified activation value, the global threshold lies within 3% of this branch switch, and in the
    certified limit the two coincide."
  - Table 1: "Branch scale law, Equation (5) | Proved for the continued branch; tied to the global threshold by
    certificates at finite activation values."
- **Changed by:** §16 (e469035) and §20 (042691a): see C1. **U2** (3ce551d, 0f3b252, a139513, a565687;
  `uniform_eps/U2_design.md` §§8–9):
  - a windowed producer run over ε ∈ [0, 0.6] (1,080/1,080 units and 27/27 slab switches pass);
  - **PRODUCER ONLY, NOT INDEPENDENTLY CHECKED.** The checker is projected at ≥ 76 h at 3 workers; a cluster package
    with an OTS proof has been prepared.
- **Replacement:** keep the submitted scoping to s_br. If a global statement is added, restrict it to
  0 < ε ≤ 4.1·10⁻⁶ (producer-standard outer exclusion), or to ε ≤ 1.4·10⁻¹² (all links proved or certified), in the
  s-form only.
  - **Do not** state the global law for ε ∈ (0, 0.6] on the strength of U2.
  - **Superseded 2026-11-14 by the author's decision:** the main text keeps the submitted wording and claims no explicit ε₀; the appendix states Theorem G (ε₀ = 1.4·10⁻¹², s-form) with its exact check status (outer margin re-checked in ball arithmetic; finite-ε parts checked by the producing code only). 4.1·10⁻⁶ is not quoted. See `partB_input_blocks.md` block 2.

**S8. Theorem 1 and Table 1, "small-scale selection (Theorem 1): Proved".**
- **Submitted:** p3, Theorem 1 ("Assume … that conditional minimizers remain, as s → 0, in a common compact set …");
  p4, Table 1, "Proved." under "Proofs apply under their stated hypotheses."
- **Changed by:** §21 (W1); §22 (7da9bec); bbe46cb.
  - Width 1: the hypothesis is proved on all three objectives.
  - Width 2, symmetric continuous windows: compactness proved (K11; s ≤ s₂, s₂ = 0.01 at a = 1.30). The limit is the
    placed cancelling pair (G₊ ≥ 0.887 at a = 1.30). The maximiser's uniqueness is certified.
  - Width 2, quadratures: not proved (aliasing; K10).
  - Width 2, asymmetric continuous windows: the hypotheses **fail** (sup Δμ = +∞; the infimum is not attained for
    s ≤ 0.0076 at a = 1.30; K12).
  - Still open: quadratic growth at the aliases; Corollary 1(ii) on the asymmetric windows.
  - The submitted PDF contains **no** "all hypotheses proved / checked" wording. The row may stay "Proved".
- **Replacement (optional):** "Proved; its small-scale compactness hypothesis is proved at width one, and at width two
  on symmetric continuous windows." Do not say "all hypotheses of Theorem 1 are proved".

**S9. Introduction, "At width two it predicts the conditional landscape".**
- **Submitted:** p1 [P p1:052–053]: "The selection criterion applies beyond width one under explicit hypotheses. At
  width two it predicts the conditional landscape, but training need not follow that landscape."
- **Changed by:** §22 (the hypotheses are proved on symmetric continuous windows and fail on asymmetric ones); W2-A
  (a registered pass for slowed SGD from a held branch); T2-3 still FAILS.
- **Replacement:** "At width two its hypotheses are proved on symmetric continuous windows, where it predicts the
  conditional landscape; training follows that landscape in a registered slowed test from a branch, but not at full
  speed."

**S10. App. N.1, "The small-scale criterion predicts a conditional placement switch".**
- **Submitted:** p41, on the asymmetric windows.
- **Changed by:** §22.4–22.5 (7da9bec).
  - Theorem 1 does not apply there: attainment fails (S = +∞) and minimisers are not attained (K12).
  - The unplaced-at-small-scale side still holds by Proposition K13 (s_u = 0.0202 at a = 1.30).
  - The large-s side is T2-1's registered numerical result.
- **Replacement:** "Conditional minimizers are provably unplaced at small scale (Proposition K13), although Theorem 1's
  attainment hypothesis fails on these windows; a validated search locates the switch."

**S11. The linear-plus-slab paragraph, "the transfer to this benchmark is not established".**
- **Submitted:** §6, p8 [P p8:414–420].
- **Changed by:** results on the same simplicity-bias benchmark (the Track 1A fold, 5c21832, post hoc):
  - Phase 2A (569b836 → 7b15d9f): registered PASS. One landscape; runs released on branch M; slow, output-scale-only
    rule; fold-crossing forecast; 22/22 rates.
  - 2A-PS (c819290 → d6fb90e): UNRESOLVED (validity).
  - 2A-PS2 (a1bd625 → 4d10abc): FAIL H (F and E_seed pass).
  - Final post hoc checks (983b594): change no verdict. The per-seed slopes do not trend toward 2/3, and no landscape
    quantity frozen before training separates crossers from non-crossers at Holm 0.05.
  - The submitted registered training test stays UNRESOLVED.
- **Replacement:** "… so transfer under standard training is not established; in a registered slow-tracking test on
  one fixed landscape the fold-crossing forecast passed, and the per-seed-sample follow-ups were unresolved and
  failed."

**S12. App. Y.1, "explanatory only".**
- **Submitted:** p55: "In these activations the branch-conditioned account is explanatory only."
- **Changed by:** GELU-T (736b6bf → 58aa960: PASS in both arms; SGD, warm start at s₀ = 0.5·s_pop); Phase 1B causal
  re-score PASS; Phase 1C arm G PASS (6522a15).
- **Replacement:** "For SiLU and Mish the branch-conditioned account is explanatory only; for GELU, a registered test
  from a branch-held start passed (see …)."
- The submitted registered Track 3A verdicts for GELU are unchanged.

## 5. Flagged, not counted, and checked with no correction

**F1 (inferred, not established by a later result; author to decide).** The 36-arm trajectory-integrated agreement
uses the same recursion construction that the prediction-inputs audit examined.
- **Where it appears:**
  - §5.1, p6: "integrating the linearized dynamics along measured output-scale paths brings every defined arm median
    within 5%";
  - Fig. 5's grey crosses;
  - App. V.3, Table 36;
  - Table 39: "Integrated, tracked branch 1,708 99.9%".
- **What the audit found for that construction:** r_traj reads the run's own s path through the predicted step. In the
  W2-A and GELU-T arms, r_traj ≡ r_obs whenever t_traj = t_obs (§2 of the audit).
- **What is not known:** these 36 arms were neither audited nor re-scored causally. V.3 already says "not a forecast of
  an entire trajectory from initialization".
- **Suggestion only:** if any revised text calls these agreements predictive, qualify them as path-conditioned.

**Checked; the submitted text holds (later results only add to it):**
- The ledger: no printed value changed after the tag except the census counts (C11). The digit-stability
  regenerations after the tag are all marked count-only, with delta 8.4·10⁻¹⁴ unchanged.
- The NaN audit (`certificate_audit.md`, 2026-10-19):
  - the faulty helpers first appear in bd1083e (2026-10-18) and are in no paper-submitted* tag;
  - no earlier certificate check is affected;
  - E.1's sentence on the voided-and-rerun NaN check holds.
- The final post hoc checks (983b594) concern Phase 2A and 2A-PS2, both registered after the submission. They
  contradict no submitted claim.
- Track G (§19; 9188ea0) finds no sharp grokking prediction. The paper only cites lazy-to-rich grokking as related
  work (§7), so nothing changes.
- Track L (R FAIL, N PASS, as predicted) and Phase 2B (primary PASS; arm 2b's R_ℓ-acc a narrow pass, +0.0038,
  `revision_notes.md` item 3) are not in the submitted paper.
- These still hold, with additions available for a revision:
  - V.7's "No rule point was at or after an observed crossing";
  - L.2's disclosure that χ is measured at the crossing;
  - "Beyond width-one sine, prospective branch selection remains open";
  - p8's GELU, SiLU and Mish and band-task sentences (add GELU-T; add Test 2C: PASS at a = 1.50, UNRESOLVED at
    a = 1.30);
  - Y.2's "This branch-conditioned decomposition was not registered" (Test 2C later registered an own-sample R^d
    switch);
  - App. A.2 and K on the width-2 continuous-window statement (now proved by K11; the maximiser's uniqueness is
    certified, in [1.784407, 1.800158]);
  - the 33 central failures.

**`results/track5/revision_plan.md` is out of date on four points.** Use this list instead where they differ.
1. It proposes reporting Test 2A as "registered at a = 1.85 and passed" with no causal caveat (§2.4, §4.1, §6). That
   was written before Phase 1B (C7).
2. It gives §17's status ("CHECKED (not proved) on continuous windows"), which §21 supersedes (W1).
3. It proposes "global for ε ≤ 1.4·10⁻¹²" for Table 1. Keep the submitted Table 1 wording instead (no explicit ε₀ in
   the main text). **Superseded 2026-11-14 by the author's decision:** the main text keeps the submitted wording and claims no explicit ε₀; the appendix states Theorem G (ε₀ = 1.4·10⁻¹², s-form) with its exact check status (outer margin re-checked in ball arithmetic; finite-ε parts checked by the producing code only). 4.1·10⁻⁶ is not quoted. See `partB_input_blocks.md` block 2.
4. Its PENDING P1 (which PDF was submitted) is closed: the author confirmed the hash on 2026-11-14 (§1).
