# Adversarial audit before external review (2026-08-24)

Read-only audit: every number below recomputed from raw artifacts
(`results/*.csv`), never from summary documents. No result was modified,
no sweep launched, nothing fixed — findings first. The Task F CIFAR build
was running throughout; checks requiring new computation are marked
**deferred** rather than run.

Recomputation scripts were run inline; the key ones are reproducible from
the commands recorded in the session and from `src/width_effect_report.py`-style
helpers (Fisher via `math.comb`, CP via `src/threshold_report.py`).

---

## 1. Discrepancies found

### Claim-invalidating

**None found.** No recomputation overturned the substance of any current
ledger claim. The closest calls are P7 and T19, below — both are ledger-
or wording-level, with the underlying artifacts sound.

### Claim-qualifying

1. **P7's ledger wording inverts its own source finding.** Ledger:
   "Between-class margin < 1 ULP does not predict the 15 impure rows."
   Recomputed: all 15 impure rows have margin < 1 ULP (15/15), against a
   base rate of 40/3,667 among pure rows — a necessary condition with
   zero exceptions and enormous association, exactly as
   `between_class_margin.md` correctly states ("necessary … not
   sufficient; over-predicts, flagging 55 where 15 are impure"). The
   ledger compression says the opposite of the document it cites. A
   reviewer recomputing this would find the ledger claim false as
   worded.

2. **T19/T3 annealing interval does not reproduce under any definition.**
   Claim: every trace loses separation at a\* ∈ [1.1, 1.8], median
   1.275. Recomputed from `search_anneal.csv` (12 traces): the median
   1.275 is reproduced only by the midpoint-of-first-loss definition,
   whose values span **[1.075, 2.275]** — outside the claimed interval
   at both ends. First-failure values span [1.05, 2.25]. One trace
   (depth 5, seed 10) is **re-entrant**: it fails at 2.25–1.75, regains
   separation at 1.70–1.40, then fails for good — unreported, and it
   breaks "loses at a\*" for any single a\*. "Never below the analytic
   threshold a = 1" survives (min 1.05). The qualitative claim stands;
   the stated interval and the unreported re-entrance do not.

3. **T24 (winding) contains three wording/statistic mismatches.**
   - "Rates fall monotonically with |lk| at every width for every
     separating activation": false per-activation (GELU width 4:
     0% at q=3 → 15% at q=4; GELU width 5: 10% → 15%; sin width 5:
     95% → 100%; sin width 6: 70% → 75%). True only when the two
     activations are **pooled** per width. The wording implies the
     per-activation statistic; the data support the pooled one.
   - "Dense attrition grows with q (22% → 46%)": actual sequence
     22%, 28%, 46%, **42%** — non-monotone; the quoted endpoint is
     q = 3, not q = 4.
   - "tanh tracking one width behind" sin: true at q = 1 (5 vs 4),
     two behind at q = 2 (6 vs 4), and tanh never reaches a majority at
     any width ≤ 6 for q ≥ 3.
   - Knife-edge, worth a resampling note per the standing rule: the
     "majority width 6 at q = 4" rests on sin width-5 being exactly
     10/20 = 50%; one flipped run moves the claim.

4. **T1's restarts stratum is 400, not 360.** `search_restarts.csv`
   holds 400 monotonic width-3 rows (200 tanh + 200 sin a=0.95), all
   0 separations. The distinct-run total is therefore **5,580**, not
   5,540. The claim gets *stronger*; the stated n is wrong. Curiously
   this also rescues T2: at n = 5,540 the exact bound is 0.05406% —
   "below 0.054%" is false by rounding — while at n = 5,580 it is
   0.0537%, and the claim as worded becomes true.

5. **The ledger's own corrections block is stale.** It declares the
   "artifact-derived total" to be 5,570 — which is the pre-dedup,
   pre-winding sum (240+1,280+1,440+1,890+360+360). T1's row uses the
   post-dedup, winding-inclusive accounting. Two mutually inconsistent
   "authoritative totals" now coexist in the authority file, and both
   differ from the recomputed 5,580.

6. **T12's distillation evidence is thinner than stated.** (a) The
   three layer-1 student rows are **bit-identical across all three
   student seeds** (eval MSE 0.188561, worst-point distance 1.881540,
   errors 1023, to every recorded digit) — the seed almost certainly
   never entered that code path, making the layer-1 result effectively
   n = 1. Layers 2–3 vary normally. (b) "Full-depth student → 88
   errors" quotes the best of {88, 89, 1000}; the third seed sat at
   chance and is unmentioned — the project's own extreme-with-
   distribution rule applied to no other claim would allow this.
   Direction unaffected (1023 = chance either way); the stated n and
   the minimum-quoting are wrong.

7. **T31/2c: the estimator-validation harness was never committed.**
   Commit 66672be adds `estimator_validation.csv` and
   `estimator_anisotropy.csv` with **no generating source file**; no
   module in `src/` references them. The audit question "did the
   validation test the estimator actually used, or a reimplementation?"
   is unanswerable from the repository. The validated conclusions may
   be fine; they are not reproducible as committed. **Deferred:**
   reconstruct the harness and re-run.

8. **T25's headline 1.4M-point evaluation has no committed generator.**
   `src/offset_witness.py` has no `main`; the "0 errors on 1,400,000
   fresh points (four samples)" figure's sample seeds live only in
   prose. The pinned test verifies 10,000 points at seed 930001 — the
   exhibit stands, at 10k-scale, deterministically; the 1.4M figure is
   currently take-our-word-for-it. Same reproducibility class as (7).

9. **T29's scaled-down arm is not correction-robust.** Recomputed
   two-sided Fisher: up-vs-std 0.0096, down-vs-std 0.0461, up-vs-down
   2.9e-6 (the recorded 0.0048/0.023/1.5e-6 are the one-sided values —
   labeled as such in `gelu_scale_results.md` and directionally
   registered, but the ledger row omits "one-sided"). Under a Bonferroni
   ×3 within the experiment, up-vs-std (0.029) and up-vs-down survive;
   **down-vs-std (0.138) does not**. The "two-sided manipulation"
   headline rests on the down arm at nominal significance only.

10. **The dense/eval sampler is not volume-uniform over the solid
    tori.** `linked_tori` draws θ uniform, cross-section area-uniform —
    but the torus volume element carries a (R + r·cosφ) factor, so the
    tube's inner equator is oversampled ~1.5× relative to the outer
    (r/R = 0.2 ⇒ density ratio 1.2/0.8). Coverage is complete and
    bounded below at ~0.67× uniform, so 100k-point zero-error checks
    remain strong evidence of regional separation; but any claim that
    reads "dense" as "uniform" overstates slightly, and per-point error
    *rates* on dense samples are measured under this tilt. Applies
    equally to winding-link samplers built the same way.

11. **T28's "ε₅₀ flat ≈ 0.1–0.3" doesn't match a natural definition.**
    Recomputed recovery curves put the 50% crossing between 0.3 and 1.0
    for a = 1.10–3.0 (rates at ε = 0.3: 0.5–0.85) and between 0.1 and
    0.3 only at a = 1.09. "Flat" (no trend in a) survives — the
    substantive negative stands; the quoted range and the undefined
    estimator ("ε₅₀" never pinned to a formula) do not.

### Cosmetic

- T30 ledger: "57% at α = −0.05" — actual 119/200 = 59.5% at −0.05;
  57.8% is the pooled α<0 rate (the source doc states both correctly).
- T30/`fold1d_results.md`: "basin-to-solution ratios 400–10,000×"
  excludes GELU's 16,194× (box 5).
- T27 intervention (v2) reuses seeds 0–39 inside the standard arm's
  0–199 (a paired design in effect, unstated; with 17/40 vs 0/400 the
  conclusion is indifferent). Contrast: gelu_scale arms use disjoint
  seed ranges.
- `interleaved_predictions.md` had one word edited ("demonstrates" →
  "establishes") inside its results commit — banned-word cleanup, in a
  registration file, in a results commit.
- Ledger T1 n is stale w.r.t. new width_effect data (another 420
  monotonic width-3 runs at 0, depths 3 and 6, seeds 100–199/100–139).

## 2. Verified claims (recomputed = stated unless noted)

| Claim | Recomputed | Stated | Match |
|---|---|---|---|
| T1 zero (all strata) | 0 separations in 5,580 distinct | 0 in 5,540 | zero ✓, n off (finding 4) |
| T1 dedup | 720 dup rows; 270 monotonic-w3; 1,890→1,620 | same | ✓ exact |
| T2 single-cell bound | 3.68% | <3.7% | ✓ |
| T3 swap / CMA | 24 rows, min 52 errors; 0/120 mono; control 12/20 d3 (5/20 d5) | same | ✓ |
| T4 | 3 CMA train-0 (eval 19/16/18); SGD min 2; 8 runs < 9 (restarts) | same | ✓ exact |
| T5 intervals | A: 0/40 at 1.08, 1/40 at 1.09 → (1.08,1.09]; B: 0/40 at −0.20, 1/40 at −0.25 → (−0.20,−0.25]; n=720 | same | ✓ exact |
| T7 | 35 GELU dense survivors | 35 | ✓ (witness 2M: deferred, test pins 10k... see finding 8 — that is T25; T7's witness is Part-2a's, pinned by its own test) |
| T8 | 56/163 dense failures; GELU 46/81 | same | ✓ exact |
| T9 | 34 raw → 28 distinct; 15/18 configs; 5 dense; 14/18 zero-dense | same | ✓ exact |
| T13 | 8/8 toward other component; clustering 4/8 | 8/8, "some" | ✓ |
| T14 | w4 dense: ReLU 1/5, leaky 3/10 | same | ✓ |
| T21 | A_paper 1/30; strong configs 1–5/30 each, all ≥1 | same | ✓ |
| T25 searches | 476 = 400+24+12+40, all fail; best 3 errors | same | ✓ exact |
| T26 | n=1,350; cliff at 0.5 (w4 0/30 both; w6 gelu 13/30, tanh 6/30); w3 dense total 11 | same | ✓ |
| T27 A_req | (978,1057)→~1,020; (243,263)→253; (121,131)→126; reach med 52.7/max 152.8 at 1.02; 44 med at 1.09, 333 max at 1.05 | same | ✓ exact |
| T27 intervention | v2 17/40; v1 0/40 preserved; standard 0/400 | same | ✓ |
| T27 barriers | 1,065.2 / 125.1 / 10.2 / 9.1 / 3.3 / 1.7 / 1.0; constructed 1.10 = 43.3 (4.8× found) | 1,065→10→1 compression | ✓ |
| T29 counts | 6/16/34 of 200; attrition 33% vs 11%; CI 16/200 = [4.6, 12.7] | same | ✓ (p-labels: finding 9) |
| T30 | mono 0/2,000; onset 0@1.30, 2/15/50/81; B 113–119/200; GELU 93/200; geometry table exact incl. empty-in-box at 1.05/1.10 and basin jump 0→0.0037→0.052→0.502 | same | ✓ |
| P1 | bf16 tanh 56.88% ± 4.92 vs fp32 0.35% ± 0.35 (5 seeds) | same | ✓ exact |
| P5 | float32 full-precision collision pairs: 0 in all 18 rows | same | ✓ |
| P6 | post-hoc superset 126/126; interleaved containment fails 89/126 | holds / fails | ✓ |
| Estimator tests | 32/32 pass (unlink/Hopf±/torus-2,4/chain/intersecting + traces) | — | ✓ |
| Determinism tests | assert `torch.equal` on state dicts across independent runs — bit-level, not tautological | — | ✓ |

Registration integrity (2g): all 14 prediction files were committed
before their results commits. `gelu_scale_prediction.md` explicitly
discloses its standard-arm-first ordering. Post-hoc edits: the known
header fix in `corrugation_prediction.md` (outcomes appended below
intact predictions — sanctioned), the one-word cleanup above, and the
pre-run dated amendment in `width_prediction.md`. Every registered
prediction has a reported outcome, including the failures (P-basin,
P-calibration, 2a-match, P3-band, P-W1-at-8, bottleneck test A).

Gate selection (2a): every headline count (T1, T5, T24, T26, T29, T30,
width_effect) is computed over **all** runs, not gate-survivors —
verified by matching row counts. Gate pass-rate asymmetry is real and
large (threshold w3: monotonic 0.8% vs non-monotonic 14.8%) but no
audited claim conditions on `passed`. The exception is the precision
strand (P1–P3): those means are over accepted runs, the gate excludes
failures asymmetrically by construction, and the ledger already lists
that as P1's known threat. Not quantifiable from `saturation.csv` alone
(failed runs absent): **deferred**.

Dense consistency (2b): 50,000/class = 100k points and a 0-error
criterion in all eight dense-checking modules; crc32-derived seeds
disjoint from training seeds by construction (900k–970k bases).
Sampler uniformity: finding 10.

## 3. Code issues, with affected claims

- **`linking.py: defined = closest > 0.0`** — the estimator itself only
  rejects *exact* coincidence; the 0.02 artifact threshold lives in the
  callers. `linking_trace.py` and `parametrization_linking.py` gate
  correctly (native-space distance, pre-projection). **`winding.py:
  validate` emits `rounded` with no artifact-regime check** — benign
  today (designed cores with recorded clearances ~0.6), but it is a
  bypass path; a future caller with crushed curves would get a value
  where the trace modules would refuse one. Affects: T20/T24 validation
  numbers (currently safe), any future reuse (unsafe).
- **`localization.py` layer-1 distillation seed** (finding 6a): three
  recorded seeds, one effective run. Affects T12's stated n.
- **`data.py: linked_tori`** volume non-uniformity (finding 10).
  Affects the meaning of every dense rate; not the zeros.
- **`corrugation.py`** Reading-B amplitude ignored — known, documented
  (T23), pinned by test. Affects T9/T18 counts — already corrected.
- **`fold1d_geometry.py: basin_mask`** trains all 6,561 grid inits in
  one batched Adam — batching couples nothing across rows (per-element
  ops only), checked: gradients are row-independent. Clean.
- **`gelu_scale.py`** arms differ only in `pattern` and seed base;
  data seeds matched across arms (paired data, independent inits) —
  clean, though the Fisher tests treat paired-data arms as independent
  samples (conservative in expectation here; note only).
- No fallback-returns-plausible-value paths found in the audited
  modules: the failure modes raise (`PipelineFailure`,
  `RecoveryDivergence`, artifact-lock collisions) rather than default.

## 4. Untested exposures (what a hostile reviewer will ask)

1. **The 2D/3D-nested generality gap** (T30): 3 solves in 2,200 runs.
   Falsifier not attempted: an order-of-magnitude larger budget at those
   exact tasks. Genuinely out of reach on this hardware; say
   "demonstrated in 1D and 3D-linked, untested between" verbatim.
2. **The bridge to real settings is currently a failed positive
   control** (bottleneck reversal; Task F in progress). The convergence-
   regime question — does tanh's MNIST advantage survive training to
   plateau — is cheap and simply not done (it was outside the
   registered protocol). If Task F's CIFAR arm restores the control,
   the width prediction gets its test; if not, the negative map is the
   result.
3. **Family B's link-setting offset mechanism is an analogy** (T28):
   A_req is undefined for B by construction; nothing measures B's
   required scale in the link setting. A targeted construction for
   shear-degenerate folds would decide it; nontrivial design work,
   not just compute.
4. **T31's link-setting extrapolation is argued, not measured** — and
   now also not reproducible (finding 7). Cheap once the harness is
   recommitted.
5. **The monotonic zero shares Adam/Kaiming ancestry** across all
   strata except the 360-run protocol stratum (ledger already flags
   this). A different optimizer family (L-BFGS, SGD+momentum at scale)
   is a cheap-ish unrun falsifier for the *reachability* reading —
   CMA-ES partially covers it.
6. **Dense sampling tilt** (finding 10): a volume-uniform or
   boundary-weighted resample of a few dense survivors would confirm
   the zeros are not sampler-shaped. Cheap; deferred (compute).
7. **T12 rests on a single teacher network** (ledger notes it); three
   students of one teacher, one of them effectively unreplicated
   (finding 6).
8. **Intrinsic-dimension method dependence** in the Task E/F bridge:
   TwoNN vs MLE disagree by ~1.7× (registered, spread-tested) — but
   both are k-NN-family; a genuinely different estimator (persistent
   homology, GeoMLE) is unrun.
9. **reproduce.py --verify not run in this audit** (competes with the
   live build) — the last full verification predates the Task B/E/F
   artifacts. **Deferred, should be run before submission.**

## 5. Honest assessment

The quantitative core of this project survives independent recomputation
to the digit: every exact-zero count, every dense-verified rate, the
bisection intervals, the amplification crossings, the intervention
counts, the fold1d geometry, and the precision-strand numbers all
reproduce from raw artifacts, and the registration trail is genuinely
ordered (predictions before results, failures reported). What does not
survive cleanly is a layer of *summary compression on top of sound
artifacts*: one ledger row states the opposite of its source (P7), one
interval-and-median claim fits no definition of its own statistic (T19),
one row's "every activation" is true only pooled (T24), two headline
totals in the authority file disagree with each other and with the
artifacts (5,540/5,570 vs 5,580), and two headline exhibits (T25's 1.4M
evaluation, T31's validation) rest on artifacts whose generating code
was never committed. Nothing found here overturns a finding; several
things found here would cost credibility if a reviewer found them first.
The fix list is short, mechanical, and almost entirely in documents
rather than code — the two real code findings (layer-1 distill seed,
winding validate bypass) affect stated n and future safety, not present
conclusions.

---

## Resolutions (2026-08-24, post-audit, user-directed)

- **Finding 4/5 (counts):** fixed in `CLAIMS.md` (ea61c45) — T1 restarts
  400, total 5,580, T2 bound restated, corrections block reconciled.
- **Finding 8 (T25 generator):** reconstructed as
  `src/offset_witness.py::main` (4ef6309), seeds 931001–931004:
  **0 errors on 1,400,000 points re-verified**, margins 0.54–0.85
  (`offset_witness_dense.csv`).
- **Finding 7 (T31 harness):** reconstructed as
  `src/estimator_validation.py`. The noise convention had to be
  recovered by reconciliation: whole-vector RMS-relative noise gives
  systematically lower recovery (found ε=0.3: 56% vs 87%); **per-
  coordinate |θi|-relative noise reproduces the committed table** —
  found object exact to within 1.7% (its θ is bit-reproduced, so this
  is a true validation of harness + convention), constructed object
  within 6–10 points at ε ≥ 0.1 (its θ is re-derived from design
  constants; the original tensor was never persisted). All 2c
  conclusions reproduce. The RMS-convention run is kept as
  `estimator_*_rms_sensitivity.csv`: a ~30-point convention spread at
  ε=0.3, which is a standing caveat on radius-style numbers and further
  support for 2b's volume-over-radius conclusion.
- Documentary fixes (findings 1–3, 6, 9–11 wording) remain **deferred**
  per user direction. CIFAR conclusions held pending replication.

---

## Hardening pass (2026-08-25): execution record

All 11 qualifying + 5 cosmetic findings executed, plus one new
claim-invalidating finding discovered during the theorem work.

| finding | resolution | commit |
|---|---|---|
| 4/5 bound + stale totals | `monotonic_zero.md` + `src/zero_decomposition.py`: full stratum decomposition regenerable from artifacts; **0 in 5,580**, one-sided exact bound **0.0537%** | a895cb8 |
| 1 P7 inversion | ledger restated (necessary-not-sufficient, 15/15 vs base 40/3,667); repo-wide grep found no other instance | d8960cf |
| 2 T19 interval | restated with explicit definition; **2 re-entrant traces** found and reported (depth-5 seeds 10, 16) | d8960cf |
| 3 T24 mismatches | pooled-vs-per-activation scope fixed; attrition sequence given in full; knife-edge cell flagged | d8960cf |
| 9 T29 sidedness | one-sided labelled; **Bonferroni ×3 stated: down-arm fails (0.138)**, headline rests on the up arm | d8960cf |
| 6 T12 n | investigated as a code defect and **is not one** — seeds propagate (init max|Δ| 0.90), the three runs converge to a common optimum (finals agree 1.5e−8); extended to 5 seeds: **{88, 89, 1000, 1000, 1000}**, chance is the majority outcome | d8960cf |
| live bypass | artifact gate applied at **every** linking emitter; `tests/test_artifact_gate.py` adds behavioural + static-scan protection (`cancellation.py` exempt by design, documented) | 09ac6f0 |
| 10 sampler | density derived exactly (R/(R+ρcosφ); ±25%, ratio 1.5, floor 0.833); **84 survivors, the 4 interval-defining runs, and both witnesses at 1M points re-verified volume-uniform — no claim moves** | 956701c |
| 11 ε₅₀ | withdrawn as an undefined statistic; "flat in a" retained | d8960cf |
| 7/8 generators | both reconstructed, calling production code by import path (`from .fold1d import …`, `train_offset_witness`); **1.4M re-verified at full size, 0 errors**; estimator table reproduced after recovering the per-coordinate noise convention (RMS run kept as a labelled sensitivity annex) | eb8fe53, 4ef6309 |
| cosmetics | T30 rate, ratio range, seed-overlap note, registration one-word edit — all recorded | d8960cf, 05123d6 |
| **NEW: claim-invalidating** | **T30 box-emptiness withdrawn** — in-box solutions exist at a = 1.02 with \|w₂\| = 1; "required \|w₂\| ≈ 58" was the 41⁴ grid's own resolution requirement | 4eacf25 |
| follow-on | grid-derived measures audited and replaced with an analytic instrument; 2a's smooth-growth claim **survives on better evidence** (true measure positive and strictly monotone at every a > 1); component counts and basin/solution ratios withdrawn | 846280a |

Coverage additions: overclaim sweep (12 edits, 05123d6), reverse
coverage T32–T36 (879eb42), theorem T37 (8a1a7e7),
`notes/instrument_artifacts.md` on the two instrument-limitation errors.

231 tests pass. Deferred by scope: per-family volume-uniform samplers
for deformed links; `reproduce.py --verify` (competing with the live
CIFAR convergence run).
