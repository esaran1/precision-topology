# Part B: input blocks for the writing model (rebuttal revision, 2026-11-14)

**How to use this file.** Four blocks (switch, scaling law, lag, scope) and a glossary. Each block has *Say* (what
the text may state, with every number) and *Do not say* (claims that are withdrawn, unproved or not causal). Write only
what *Say* licenses. Every number carries a ledger tag:

- `[L: "label"]`: a check of that name in the project ledger (`src/verify_ledger.py`), which recomputes the value from
  the committed artifact. The ledger runs with 0 findings.
- `[L: RB-nn]`: a check in the ledger's rebuttal section (`rebuttal_input_checks`), added for this revision, under
  the same rule.
- `[L: submitted]`: a number printed in the submitted PDF (sha256 `2648fb64…72ca6`) and verified by the ledger at
  submission. It is unchanged since then.
- `[Reg: file §n]`: a design constant (learning rate, seed count, tolerance, cutoff) fixed in that registration
  document before its runs. It is a setting, not a measurement.

Status words, used exactly: **proved** (a hand proof, possibly with certified constants); **computer-assisted**
(a certificate whose decisions are interval comparisons); **independently re-checked** (re-verified by the separate
ball-arithmetic checker, which shares no code with the producer); **registered** (prediction and scoring rule
committed and timestamped before the scored runs); **strictly causal** (each forecast reads only data from before a
declared cutoff step t_c, enforced in code and checked by recomputing with every later value set to NaN); **post hoc**
(analysis after the data existed).

**The cutoff f.** Every causal forecast below stops reading the run at the first step where the output scale reaches
f times a frozen switch scale, then extrapolates. Give f with every forecast result, and report the f = 0.90 step
failures next to the passes whenever the registered f was 0.95 (block 3, table 3.2).

---

## Block 1: the switch (Theorem 1 after the compactness proof)

### Say

- **Statement (unchanged in content from the submitted Theorem 1).** Fix the output scale s and minimise the logistic
  loss over the hidden parameters and the output bias. Every small-scale limit point of these fixed-scale minimisers
  maximises the class-mean gap Δμ, and among those it minimises the output variance. If every such point has a
  strictly negative data gap and a positive maximal data gap is attained, the fixed-scale minimisers are **unplaced
  for all sufficiently small s and placed for all sufficiently large s** `[L: submitted]`.
- **What changed: the small-scale compactness hypothesis is now proved at width one** (it was "assumed" in the
  submission). On the continuous windows: for s < s₁, every fixed-scale minimiser has |w₁| ≤ W, with certified
  constants s₁ = 0.675, W = 10.81 at a = 1.30 and s₁ = 0.643, W = 11.93 at a = 1.60
  `[L: "§21 K s1 continuous box a=1.30"]` `[L: "§21 K Wc continuous box a=1.30"]`
  `[L: "§21 K s1 continuous box a=1.60"]` `[L: "§21 K Wc continuous box a=1.60"]`. It is also proved on the 800-point
  quadrature, where the compact set must contain an aliased maximiser at very large input weight. That maximiser is
  also unplaced, so the width-one conclusion is unchanged.
- **The small-scale limit, located.** On the continuous windows, every class-mean maximiser has
  |w₁| ∈ [1.784407, 1.800158] `[L: "§21 K kept span lo"]` `[L: "§21 K kept span hi"]` (certified), and the maximiser is
  unique (certified later, with the width-two work) `[L: "§22 W2 uniqueness ok"]`. It is the ramp-plus-cosine feature, and
  it is unplaced.
- **The sine-specific bracket needs no compactness** (as submitted): at a = 1.30, an unplaced candidate beats every
  placed one for s < 0.342, and placed solutions beat every data-unplaced one for s > 163.8
  `[L: "A1 s0(1.30)"]` `[L: "A1 s1_data(1.30)"]`.
- **Proof idea (five lines).**
  1. Profile out the output bias. At small s the profiled loss is log 2 − (s/4)Δμ(θ) + (s²/8)Var(u_θ) + O(s⁴)
     (equation 3).
  2. Compactness (now proved at width one) makes the minimisers converge along subsequences. The expansion then forces
     each limit to maximise Δμ first and to minimise Var among the maximisers.
  3. On symmetric windows the linear ramp adds nothing to Δμ. So the maximiser uses a steep input weight, and endpoint
     pairs give G ≤ 2a − 2.8|w₁| < 0: it is unplaced.
  4. At large s, exponential bounds on the logistic loss make the worst-case data gap dominate. A placed configuration
     with positive gap is attained, so large-s minimisers are placed.
  5. Between the two regimes the minimiser changes placement. The certificates locate where (block 2).
- **The width-two statement (proved; new).**
  - Symmetric continuous windows: fixed-scale minimisers exist and stay bounded at small scale (W₂ ≤ 8.84 for
    s ≤ 0.0066, over the tabulated a) `[L: "§22 W2 bound max over a"]` `[L: "§22 W2 s2 min over a"]`. The two units
    cancel each other's linear ramp, and the small-scale limit is placed. **Width two therefore has no small-scale
    unplaced regime on symmetric windows.**
  - Asymmetric windows (offset 0.4): at small scale the infimum of the loss is approached by a **ramp limit** (one unit
    escapes to a linear term) and is **not attained**. The small-scale behaviour is unplaced, and Theorem 1's
    small-scale part holds after rescaling.
  - The ramp limit matters only at very small scale: it beats the retained width-two solution only below
    s_ramp ∈ [0.1576358, 0.1576364] `[L: "RC RC s_ramp lo"]` `[L: "RC RC s_ramp hi"]`. Every registered width-two
    landscape quantity lies above it; the closest is 1.41·s_ramp `[L: "RC RC s0 over s_ramp"]`. This is a validated
    float search, not a certificate.

### Do not say

- "All hypotheses of Theorem 1 are proved." At width two, compactness on the 800-point quadratures, the large-scale
  side on asymmetric windows, and Theorem 1(ii) on asymmetric windows are not proved.
- "Theorem 1 gives a unique switch" or "a converse holds." It is a sufficient criterion. Uniqueness of the sign
  change is shown only near the limit (block 2).
- "The small-scale minimiser is the cosine at α\*" without "on the continuous windows". On the 800-point quadrature
  the literal s → 0 limit is the aliased weight.
- "Width-two minimisers exist at small scale on asymmetric windows." The infimum is not attained.
- "The minimiser crosses the placement bound at the switch scale" or "training crosses at the switch scale" (the
  mechanism figure). The minimiser falls below the bound a/1.4 well before the switch (at R/R_glob ≈ 0.54 and 0.76),
  and the bound is necessary, not sufficient `[L: submitted]`. Training crosses after the switch, by the lag (block 3).

---

## Block 2: the scaling law (equation 5)

### Say

- **Equation (5), unchanged:** along the continued branch, s_br = A\* ε^{−3/2}(1 + c_s ε + O(ε²)), with
  A\* ∈ [0.68125, 0.6875] and c_s = 0.66215 `[L: "A* bracket lower"]` `[L: "A* bracket upper"]` `[L: "A'/A* lo"]`.
  Here ε = a − 1.
- **Its scope is the branch** (as submitted). It is proved for the continued branch. It is tied to the global
  switch by separate certificates at each tested activation value, where the global threshold lies within 3% of the
  branch switch `[L: submitted]`.
- **Main text: keep the submitted wording**, including "No explicit ε₀ is claimed." Claim no value of ε₀ in the main
  text. At most, add a pointer to the appendix statement below.
- **Appendix only: Theorem G (new), with its exact check status.** For 0 < ε ≤ ε₀ = 1.4·10⁻¹² `[L: "§16 eps0"]`, the
  continued branch is the unique global fixed-scale minimiser, and its gap changes sign exactly once. This is
  computer-assisted, and it holds in s-units (the R-form is not covered). State both parts of its status, every time
  the theorem is stated:
  - Its outer-exclusion margin (≥ 1.0024·10⁻⁴) was **independently re-checked** in ball arithmetic on all 40
    sub-intervals: minimum 1.006·10⁻⁴ `[L: "T4 outer annulus sub-intervals passed"]`
    `[L: "T4 outer annulus min certified margin"]`.
  - Its finite-ε parts (the ε-inflated Krawczyk boxes, 110 branch sub-intervals `[L: "§16 subintervals"]`, and the
    finite-ε localisation) are checked **by the producing code only**. No independent check was run on them.
- **The first-order coefficient**, in R units c₁ = 0.28523 `[L: "c1 lo"]`, is confirmed by certified thresholds at
  ε = 0.08–0.12, with all ten endpoint certificates independently re-checked `[L: submitted]`. The limiting
  geometric constant predicts measured constants within 0.808% across five new geometries, with nothing fitted
  `[L: "G-4 worst error (%)"]`.
- **Corrected status line for the certificates.** The submission said that outer exclusion and the finite and limiting
  solve brackets were "search-certified only". All three have since been independently re-checked: the limit chain's
  outer exclusion (40/40 sub-intervals), the limit solve bracket, and 12/12 finite-a solve ends
  `[L: "T4 outer annulus sub-intervals passed"]` `[L: "T4 limit solve ends pass"]` `[L: "T4 finite solve ends pass"]`.

### Do not say

- Any explicit ε₀ in the main text. In the appendix, any value of ε₀ other than 1.4·10⁻¹². In particular, do not
  print 4.1·10⁻⁶. Its outer exclusion was never re-checked in ball arithmetic.
- "Theorem G is independently verified" or "certified and independently checked" without saying that its finite-ε
  parts were checked by the producing code only.
- "Equation (5) holds uniformly in ε", "the branch is global for all tested a", or anything about ε ∈ [0, 0.6]. The
  run over that range was not independently checked, and it is not part of this revision.
- "Theorem G is fully independently re-checked." Only its outer-margin input was re-checked.
- "Equation (5) predicts the global switch at the tested a" without "tied by the per-a certificates".

---

## Block 3: the lag

### 3.1 The law, and what is proved

- **Say.** Near the switch, training follows the fixed-scale minimiser with a relative lag r = (s_c − s\*)/s\* ≃ κχ.
  Here χ = (ṡ/s\*)/(ηλ_min) is the growth-to-relaxation ratio, and κ comes from the loss landscape at the switch
  (equation 7) `[L: submitted]`.
- **Say (new theorem).** For gradient descent with a fixed preconditioner and no momentum, r = κχ + O(χ²) + τ is
  proved with explicit constants. The lag is independent of the learning rate in free training. This matches the
  registered finding, in the submission, that the lag does not vanish as the learning rate decreases
  `[L: submitted]`.
- **Say (limits of the theorem).** Adam's moving preconditioner, momentum and folds are not covered. The sine-family
  constants are not enclosed, so the theorem is not a certified statement about the a = 1.30–1.60 runs.

### 3.2 Causal support: every strictly causal forecast, with its cutoff

Every forecast predicts each run's crossing step and its signed lag from the run's own output-scale path before t_c.
"Crossing" means the step within τ_cross steps (in at least 80% of scored runs); "lag" means the lag within τ_lag
steps (in at least 80%); "median ratio" means the median of observed over forecast lag, which must fall inside the
band.

**Inside the width-one sine network (the home setting):**

| Setting | Status | f | Crossing | Lag | Median ratio (band) | Verdict |
|---|---|---|---|---|---|---|
| a = 1.58, SGD, 100 seeds | registered, causal | 0.90 (set after seeing the estimation seeds) | 84/84 | 84/84 | 1.169 ([0.80, 1.20]; band set after seeing a bias of 1.172) | **PASS** `[L: RB-01]` |
| same, at f = 0.95 | descriptive | 0.95 | 84/84 | 84/84 | 1.094 | n/a `[L: RB-02]` |
| a = 1.77, Adam, 100 seeds | registered, causal | 0.90 (fixed at registration) | 93/98 | 93/98 | 0.940 ([0.90, 1.10]; 0.040 inside the edge, a registered known risk) | **PASS** `[L: RB-07]` |
| same, at f = 0.95 | descriptive | 0.95 | 92/98 | 92/98 | 0.970 | n/a `[L: RB-07]` |
| a = 1.65, SGD, 63 scored | causal re-score, **post hoc** | 0.90 | 63/63 | 63/63 | 1.216 ([0.80, 1.20]) | **FAIL** (median ratio) `[L: RB-08]` |
| a = 1.65, Adam, 74 scored | causal re-score, **post hoc** | 0.90 | 39/74 | 43/74 | 1.158 | **FAIL** (crossing, lag) `[L: RB-08]` |
| a = 1.65, at f = 0.95 | descriptive | 0.95 | SGD 63/63; Adam 34/74 | SGD 63/63; Adam 46/74 | SGD 1.123; Adam 1.067 | n/a `[L: RB-08]` |
| a = 1.85, Adam (registered ordering test), 75 scored | causal re-score, **post hoc** | 0.90 | 55/75 | 55/75 | 1.354 | **FAIL** `[L: RB-09]` |
| same, at f = 0.95 | descriptive | 0.95 | 61/75 | 60/75 | 1.209 | n/a `[L: RB-09]` |

**Outside it (registered, causal; f = 0.95 for all three, chosen after the pilot):**

| Setting | Crossing at f = 0.95 | Lag | Median ratio (band) | Verdict | Crossing at f = 0.90 |
|---|---|---|---|---|---|
| width-1 GELU, SGD, warm start at half the population switch scale, 80 seeds | 74/79 | 74/79 | 0.995 ([0.85, 1.15]) | **PASS** `[L: RB-03]` | **9/79 (fails)**; lag 74/79 |
| width-2 sine, output learning rate ÷1024, 200 seeds (101 reached the tested branch) | 101/101 | 101/101 | 0.979 ([0.90, 1.10]) | **PASS** `[L: RB-04]` | **0/101 (fails)**; lag 101/101 |
| width-2 sine, output learning rate ÷4096, branch-point start, 120 seeds | 120/120 | 120/120 | 1.016 ([0.90, 1.10]) | **PASS** `[L: RB-05]` | **0/120 (fails)**; lag 120/120 |

- **Say.** The forecasters are optimizer-specific. The SGD forecaster (the one that passes at a = 1.58), applied to
  the a = 1.77 Adam runs, misses: median ratio 1.252, 67 of 98 within the lag tolerance, and 21 of 98 runs with no
  forecast at all (descriptive) `[L: RB-07]`. The a = 1.77 pass uses a separate Adam forecaster (§3.2a).
- **Say.** At f = 0.90 the lag forecasts still pass in all three settings, but the crossing-step forecasts fail. A
  forecast made earlier has to extrapolate further, and the step error grows faster than the lag error. The
  post hoc horizon analysis (Figure 4) shows how this error grows with f; label it post hoc.
- **Say.** The registered tests at a = 1.65 and the registered Adam ordering test at a = 1.85 read each run at or after
  its crossing, so they are not forecasts. Their per-run accuracy claims are withdrawn. Their causal re-scores are
  reported as above, failures included.
- **Say (correction, not a withdrawal): the ordering under SGD at a = 1.65.** The registered ordering criterion
  (Spearman rank correlation of predicted and observed lag ≥ 0.5) `[Reg: track_a_registration.md §6, L3]` also holds
  when each path-conditioned prediction is replaced by the strictly causal forecast at f = 0.90 (post hoc): Spearman
  0.995, n = 63 `[L: RB-08]`. The submitted 0.995 was path-conditioned. Under Adam the same causal rank correlation is
  0.218, n = 66 `[L: RB-08]`, which fails the criterion.

### 3.2a What differs between the two causal passes and the two failing causal re-scores (facts only)

All four use the width-one sine network in free training from a uniform(−1, 1) initialisation, with a budget of 32,000
steps, and the same cutoff rule: the first step at which the output scale reaches f = 0.90 times the switch scale of
the branch occupied at the causal rule point `[Reg: phase1c_registration.md §4]` `[Reg: trackA_registration.md §4.1]`.
They differ in the following ways.

- **a = 1.58, SGD (pass).** Learning rate 0.3; 100 fresh seeds; registered as a causal forecast. The forecaster is
  Phase 1C's: a quadratic extrapolation of log s over the last 5% of the visible path (at least 50 steps). Its
  f = 0.90 and its tolerances (crossing 10 steps, lag 15 steps, band ±0.20) were chosen after the Phase 1A estimation
  seeds of the same setting had been seen (their median ratio was 1.172 at f = 0.90 and 1.094 at f = 0.95) `[L: RB-01]`
  `[Reg: phase1c_registration.md §2, §8]`.
- **a = 1.77, Adam (pass).** Learning rate 0.01; 100 fresh seeds; registered as a causal forecast. The forecaster is
  different and Adam-specific: a linear extrapolation of s over the last (up to) 20 visible steps, with Adam's
  second-moment estimate read up to the cutoff and carried forward along the forecast path to the forecast switch. Both rules were chosen after an exploration on 120 separate
  seeds at a = 1.85 (not the seeds of the a = 1.85 test). Its tolerances (crossing 10, lag 10, band ±0.10) came from a
  40-seed pilot at a = 1.77 by a rule fixed before the pilot; f = 0.90 was fixed at registration
  `[Reg: trackA_registration.md §1, §2, §4.2, §7]`.
- **a = 1.65, SGD and Adam (re-score fails).** Learning rates 0.3 (SGD) and 0.01 (Adam); 80 seeds per optimiser.
  The test was registered before submission for a path-conditioned prediction: a trajectory-integrated lag that reads
  each run's measured output-scale path `[Reg: track_a_registration.md §4, §5]`. It was not designed for a causal
  forecast. The re-score is post hoc. It uses Phase 1C's forecaster unchanged, with the tolerances and band set for
  a = 1.58 under SGD; under Adam it freezes the preconditioner at the rule point, as that test's registration did.
- **a = 1.85, Adam (re-score fails).** Learning rate 0.01; 80 seeds. The test was registered to read Adam's
  preconditioner at the first step the output scale reaches the switch scale, a rule chosen post hoc on the a = 1.65
  runs `[Reg: track2a_registration.md §2, §3, §4]`. That step lies after any causal cutoff, so the test was not designed
  for a causal forecast. The re-score is post hoc. It uses Phase 1C's forecaster unchanged, with the a = 1.58 SGD
  tolerances and band, and the preconditioner one step before the cutoff.
- **The forecaster under Adam (descriptive).** Applied to the a = 1.77 runs at that test's tolerances, Phase 1C's
  extrapolation gives no forecast in 21 of 98 runs; the crossing is within tolerance in 77/98 and the lag in 67/98, and
  the median ratio is 1.252 `[L: RB-07]`. This variant was fixed at registration as descriptive (no verdict).

### 3.3 The predicted-lag boundary

- **Say.** In a registered forced-ramp test (width-1 sine, SGD, a = 1.30 and 1.50), the lag law held within 25% in all
  6 cells with κχ ≤ 0.1 (cell medians 0.990–1.209) `[L: "2B C1 PASS"]` `[L: "2B C1 cells invalid"]`. All 4 cells with
  κχ ≥ 0.3 were invalid, so that criterion is **unresolved** `[L: "2B C2 cells invalid"]`: the runs destabilised
  (η·λ_max > 1) instead of lagging. The law is supported up to κχ ≈ 0.1; where it stops holding is **not located**.
- **Say.** In standard training, |κχ| ≤ 0.1 at the crossing for 73% of width-1 Adam runs (n 408), 89% of width-1 SGD
  runs (n 207), and 100% of GELU (n 114) and band-task (n 705) runs, but **only 12% of width-2 runs (n 75)**
  `[L: RB-10]`.

### 3.4 Per-run under Adam

- **Say.** Under Adam the per-run forecast is good only when Adam's preconditioner is read close enough to the cutoff.
  The registered causal test at a = 1.77 does this and passes, with a thin margin on the median ratio. Read at the
  earlier rule point, as in the a = 1.65 re-score, it fails on crossing and lag. That comparison is post hoc.

### Do not say

- "Individual predictions are accurate under SGD, where every run is within 10%" (abstract, Table 2) as a forecast
  claim. It is withdrawn.
- "Median per-run ratios of 1.07 under Adam and 1.06 under SGD" as evidence of prediction. These read the run through
  the predicted step.
- The submitted SGD rank correlation 0.995 as a forecast result. Cite the causal 0.995 (n = 63), labelled post hoc.
- Any f = 0.95 pass without its f = 0.90 crossing result beside it.
- "The lag law is proved for Adam" or "for the tested runs."
- "κχ ≲ 0.1 is the validity boundary." The upper cells were unresolved, so it is a bound on where it holds, not a
  located boundary.
- "The trajectory-integrated prediction brings every arm within 5%" as a forecast. That agreement is path-conditioned
  (see the 36-arm note when it is ready).

---

## Block 4: scope, by regime

One row per setting, grouped by training regime. Regime (a): the runs grow slowly relative to relaxation, either
because the output learning rate is slowed or because they start warm or at a branch point. Regime (b): standard
training.

### (a) Slow output growth, warm start, or branch-point start: the six registered, strictly causal passes

| Setting | What passed (n) | Caveat (state it with the result) |
|---|---|---|
| width-1 GELU, SGD | crossing and lag forecasts, 74/79 each; median ratio 0.995 `[L: RB-03]` | warm start at half the population switch scale; f = 0.95 chosen after the pilot; crossing fails at f = 0.90 (9/79) |
| width-2 sine, output LR ÷1024 | 101/101 and 101/101; median ratio 0.979 `[L: RB-04]` | only 101 of 200 seeds reached the tested branch; crossing fails at f = 0.90 (0/101) |
| width-2 sine, output LR ÷4096, branch-point start | 120/120 and 120/120; median ratio 1.016 `[L: RB-05]` | a mechanism test from a branch point; crossing fails at f = 0.90 (0/120) |
| tanh width-4, simplicity-bias benchmark, scale-only rule, 22 rates | the crossing lands at the point where the occupied solution disappears (the fold), inside [s_F, 1.25·s_F] at 22/22 rates; delay exponent 0.698 `[L: RB-11]` | one fixed dataset; deterministic given the rate (the 22 rates are the replicates); the exponent's interval is about 5.4 times too narrow (post hoc); f = 0.95 chosen after the exploration |
| same benchmark, output learning rate ÷16, 80 seeds | more slab use and better shifted-test accuracy at matched scale (lower end +0.118) and at matched loss (+0.031) `[L: RB-12]` | not a forecast; costs 12.35 times the steps; worse at matched steps (−0.240) |
| width-4 sine, asymmetric windows, branch-point starts | crossing and lag forecasts: 110/112 and 107/112 (median ratio 1.0014); 119/119 and 117/119 (1.0297) `[L: RB-13]` | random starts never reached a genuine four-unit solution (0/710); the crossing tolerance (125 steps) exceeds the median predicted lag (87 steps); f = 0.95 |

### (b) Standard training

| Setting | Result (n) |
|---|---|
| width-2 sine, standard training | only 12% of crossings (n 75) are in the lag law's regime `[L: RB-10]`; crossings come a median 2.79 times later than the switch, and training tracks no fixed-scale minimiser (post hoc) `[L: "W2F cross/switch median"]` |
| standard ReLU MLP, MNIST-CIFAR dominoes, output learning rate ÷16 (registered null) | no gain in feature use at matched loss, as predicted: all three lower ends below 0 (−0.0015, −0.0051, −0.0099); n 40 `[L: RB-14]`. The prediction was informed by an exploration that had found the same null |
| same, LayerNorm before the readout (explored, not registered) | slowing the output layer costs ×1.00–1.09 in steps at the three loss levels, against a ×2.0 bar; 8 seeds `[L: RB-17]` |
| tanh width-4, per-seed samples (registered) | **fails** its rule for which seeds should not cross: 23/30 = 0.767 < 0.80 `[L: RB-15]`; per-seed delay exponents have SD 0.031 across 40 seeds, 7/40 lie below 2/3, and they show no trend toward 2/3 (post hoc) `[L: RB-16]` |
| tanh width-4, standard training | runs leave the linear-dominant solution before its fold and track no fixed-scale solution (post hoc) `[L: "1A s_fold"]` |
| grokking in diagonal linear networks (derivation) | the account gives no sharp prediction of the grokking time |

- **Say.** Every pass is in regime (a). In regime (b) the account does not predict feature use or crossing time, and
  branch selection outside width-one sine is not predicted.
- **Do not say** "outside this setting, registered predictions fail or remain unresolved" (submitted abstract), or
  "agreement on a tracked branch outside width-one sine is post hoc" (submitted §6). Both are now false: there are six
  registered causal passes in regime (a).
- **Do not say** that any regime-(a) pass shows standard training follows the minimiser.

---

## Glossary (six terms; each defined without reference to a later term)

| Term | Definition |
|---|---|
| output scale | The magnitude s = \|w₂\| of the output weight, which multiplies the hidden unit's response. |
| placed | A hidden unit is placed when its response is higher on every positive input than on every negative input, so that some output bias classifies all inputs correctly. |
| fixed-scale minimiser | For a fixed output scale, the hidden parameters and output bias that minimise the training loss. |
| switch scale | The output scale at which the fixed-scale minimiser changes from not placed to placed. |
| lag | The relative amount by which the output scale at which training becomes placed exceeds the switch scale. |
| growth-to-relaxation ratio | The rate at which the output scale grows, divided by the rate at which the hidden parameters relax toward the fixed-scale minimiser. |
