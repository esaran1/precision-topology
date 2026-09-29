# Block 2: certificate auditability (export every certificate, check each with an independent checker)

**For the rebuttal revision** (the submission's certificates are unchanged).

- **Checker**: `src/verify_certificates.py`, python-flint (Arb) 0.9.0 at 80 bits. It imports nothing from `src/`;
  every decision is an Arb comparison. Tests: `tests/test_verify_certificates.py` (11 tests at first; 33 as of
  2026-09-28, including constructed cases it must reject).
- **Export**: `src/cert_export.py`. It reruns each certifying search with a recording hook that is off by default
  (results asserted identical to the committed artifacts). Data files go to `results/certificates/`, which is not
  committed; their SHA-256 hashes are in `certificates_manifest.csv`, each with its regeneration command. The total
  so far is 10 MB.

## Status by certificate family

| family | certificates | checker verdict | time / memory | record |
|---|---|---|---|---|
| finite-a glob brackets (a = 1.30–1.60, lo and hi ends) | 12 | **all pass** (every leaf claim verified, exact tiling, lemma W, hashes) | 554–658 s each at one worker, ≈ 75 MB | `verify_certificates_finite.log` |
| Ĝ(a) enclosures (a = 1.10–3.0) | 11 | **structure passes; the float endpoints miss by ≤ 1.1e−15** (see below) | 1–90 s check; export ≤ 0.6 GB | `verify_certificates_ghat.log` |
| Ĝ(1.05) | 1 | as for a ≥ 1.10: structure passes (18.0M leaves, exact tiling, every leaf below hi); the endpoints miss by ≤ 4e−16 | export 119 s, 1.5 GB; check 394–397 s. **The first check peaked at 3.05 GB, over the 3 GB rule.** I had not extrapolated it: the set-based tiling check held 18M tuples. The vectorised tiling check (same verdicts on constructed and real cases) brought it to 2.85 GB | `verify_certificates_ghat.log` |
| Ĝ(1.02) | 1 | as for the others: structure passes (110.7M leaves, exact tiling by the per-level check); the endpoints miss by ≤ 1.4e−16 | streamed export 747 s, 2.08 GB (under the rule); check 2,307 s. **The check peaked at 3.74 GB, over the 3 GB rule**: I had not extrapolated the per-level tiling check's copies (`setdiff1d` and per-child temporaries on a 36M-key level). It is fixed (in-place marking, preallocated children; tests pass). Re-measured on this certificate: the tiling check alone peaks at 2.54 GB (50 s) and returns an exact tiling | `verify_certificates_ghat.log` |
| limit-switch status, A = 0.68125 ("minus") and A = 0.6875 ("plus"): both ends of the search-certified bracket A* ∈ [0.68125, 0.6875] | 2 | **both pass**: every losing-region leaf lies outside its region or has a rigorous lower bound of L0* above U (72,075 and 69,102 leaves); hashes, exact data x-symmetry, window symmetry, the monotone-logit step of the localisation, exact tiling of both regions, U's point in its region, and U below the localisation bounds B(24) = 0.38797 and B_full = 0.47739 (recomputed by exact PAVA) | 3,046 s at one worker and 2,114 s at two; ≈ 80 MB | `certificate_checks/limit_A_lo.json`, `limit_A_hi.json` |
| localisation B(24), B_full | (inside the two limit checks) | **pass** (recomputed rigorously in each limit check) | | as above |
| K = sup G₀ and its domain lemma (math note §8) | 1 claim, 2 targets | **pass** (Track 5 item 1, 2026-09-25): a fresh Arb branch and bound over [0, 8] × [−12, 12] gives sup G₀ ≤ the published hi and ≤ the tight hi 0.5794559217; the lower end is attained at the published argmax; the domain lemma is checked in exact rational arithmetic | 23 s and 82 s | `certificate_checks/K_base.json`, `K_base_target0.5794559217.json` |
| Krawczyk boxes for c₁ | 2 boxes | **pass** (Track 5, 2026-09-25): the switch system has a unique zero in the 1e−9 box, A* ∈ [0.6854452375756532, 0.6854452375756537], A′(0)/A* ∈ [0.66215478, 0.66215495], the active set is unique; the K vertex is unique | — | `certificate_checks/c1_krawczyk.json` |
| PD boxes (Link 4): U × [0.66, 0.71] and the solve 0.05 box × [1.05875, 1.06] | 500 + 100 boxes | **pass** (Track 5, 2026-09-25): Sylvester's criterion on the Arb Hessian enclosure with b* bracketed; λ_min ≥ 0.0473 and ≥ 0.0209 | 6.7 s and 1.7 s | `certificate_checks/pd_glob_neighbourhood.json`, `pd_solve_neighbourhood.json` |
| ring (Link 3): 0.05 ≤ \|θ − θ_c\|_∞ ≤ 0.15, 40 annulus sub-intervals and the solve bracket | 21,600 + 540 boxes | **pass** (Track 5, 2026-09-25): 0 is excluded from ∂_p or ∂_q on every box | 199 s and 5.8 s | `certificate_checks/ring_glob_annulus.json`, `ring_solve_annulus.json` |
| outer exclusion (Link 2): 40 annulus sub-intervals of [0.66, 0.71] and the solve bracket [1.05875, 1.06] | 41 | **all pass** (Track 4, 2026-09-28; see below): certified margin ≥ 1.006e−4 on every annulus sub-interval and ≥ 1.649e−4 on the solve bracket | 39–41 s each (1,577 s for the 40), one process, ≤ 108 MB | `certificate_checks/outer_glob_annulus.json`, `outer_solve.json` |
| limit solve bracket: both ends, A = 1.05875 ("fails") and A = 1.06 ("solves") | 2 | **both pass** (Track 4; see below) | 7.4 s including the re-run ring and PD for the bracket | `certificate_checks/solve_limit_ends.json` |
| finite-a solve brackets (a = 1.30–1.60, lo and hi ends) | 12 | **all 12 pass**: every end's sign equals the published one ("fails" at every lower end, "solves" at every upper end), decided on every kept leaf (44–52 per end); the winning region's 15,432–23,292 exported leaves are verified; the losing region is shown above U by a fresh branch and bound (3,270–3,972 cells) (Track 4; see below) | export 7.0–14.5 s (11 ends; the a = 1.45 hi certificate is the timing sample's); check 359–519 s each at one worker; ≤ 484 MB | `certificate_checks/solve_finite_checks.json`, `solve_a*.json` |

## Ĝ enclosure certificates: what passed and what did not

- **Claim.** Ĝ(a) ∈ [lo, hi].
  - The upper end comes from a quadtree over [0, a] × [0, 2π] (w₁, b₁). Every leaf satisfies G(centre) + (1 + a)(2.8·hw + 2·hb) ≤ hi.
  - The lower end is a value attained at a recorded point.
  - The paper's R uses Ĝ_cert, the zoom search's attained value (`ghat_certified_all.csv`).
- **Passed at all 11 a**:
  - the files match the committed hashes;
  - the leaves tile the domain exactly (42k–4.1M leaves);
  - the domain contains the analytic reduction to w₁ ∈ (0, a/1.4];
  - Ĝ_cert ≤ hi.
- **Not passed (strict checks, not relaxed)**:
  - the published hi is exceeded by the rigorous bound on 1–2 top leaves at a = 1.30, 1.35, 1.60 and 3.0;
  - the search's attained lo is not rigorously attained at 6 a;
  - Ĝ_cert is not rigorously attained at 8 a.
  - Every discrepancy is at most **1.1e−15** absolute, and the discrepancies fall on both sides.
- **Cause.** The searches evaluate G in float64 and use the result as the bound. A rigorous re-evaluation differs in
  the last one or two ulps. The top leaf of the search attains hi with zero slack, and the float 2.8 in the
  Lipschitz step lies below the real 2.8.
- **What is proved.** The checker reports the rigorous enclosure at each a (`rigorous_lower`, `rigorous_upper` in
  the log). The published values differ from it by ≤ 1.1e−15, i.e. ≤ 6e−14 relative. No printed digit of Ĝ or of
  any R changes.
- **Decision needed (author).** How to state the certified values:
  - **(a)** the rigorous enclosures, with Ĝ_cert replaced by the rigorous lower bound; or
  - **(b)** the published values rounded outward to a stated number of significant digits, with the checker
    verifying those.

  The finite-a certificates were not affected: their comparisons have margins far above rounding.

## Author's decision (2026-09-24): option (a), implemented

- **The published Ĝ values are replaced by the checker's rigorous enclosures** (`ghat_rigorous.csv`, from
  `certificate_checks/ghat_a*.json`; producer `src/ghat_rigorous.py`).
  - Ĝ_cert is now the Arb lower bound of G at the same witness as the old float value, rounded down.
  - Ĝ_hi is the Arb bound over every leaf, rounded up.
  - Every code path that computes R reads `ghat_rigorous.ghat_R` (17 modules switched).
- **No printed digit changes.** The largest relative change is δ = 8.4e−14 (at a = 1.02). All 376 printed-number checks
  in the ledger keep their printed digits with their values scaled by 1 ± δ (`ghat_digit_stability.csv`; WP-7).
- **The strict checks on the old float endpoints keep reporting as they are.**
- **The branch and bound's own attained point** gives a larger proven lower bound at nine a (by up to 8.4e−5
  relative). It is reported in WP-7 and not adopted, because it is a different number, not a rounding correction.
- **Confirmed (2026-09-24).** R keeps the rigorous value at Ĝ_cert's witness. The enclosure of G* = Ĝ(a), with the best
  certified lower bound at each a, is reported separately (WP-7 (ii)). The largest relative difference from Ĝ_cert is
  8.36e−5, at a = 1.10, where it is 1.48e−6 absolute. The largest absolute difference is 7.09e−6, at a = 1.60, where it
  is 3.16e−5 relative.

## Limit-switch checks (2026-09-25): what was checked, and three checker fixes found on the way

- **What the status claim needs** (author's decision): the winning region's rigorous upper bound U, and every
  losing-region leaf outside its region or with L0* > U. The losing leaves are checked against U. The winning
  region's leaves are tiled (coverage) but their own search bounds are not re-verified, since they do not enter the
  claim.
- **Three checker defects, found by timing a sample before the full run, fixed and tested before the results above.**
  1. **python-flint's `arb ** 3` returns NaN on a ball containing 0, and the limit activation h used it.** A NaN
     could be skipped by `min`/`max` in the range function, which could give a too-narrow range and a false
     "outside the region" pass. h now uses products, and every range function returns the whole line if any
     candidate is non-finite. **An A_lo run made before this fix passed; that result is void and was rerun.**
     f_a (the finite-a and Ĝ checks) never used powers, so those checks are unaffected.
  2. **The bias bracket's float search** was limited to ±80. It now covers the monotone range, and the sign of F is
     computed in tail form (exact integer count of the saturated units; log-domain comparison on the balanced
     plateau), so the bracket stays decidable when every logit is ~10⁴–10⁵. Arb still decides every bracket.
  3. **A direct (zeroth-order) cell lower bound** is taken alongside the mean-value bound. It uses the per-point loss
     at its favourable logit endpoint, given the certified bias bracket. Limit checks may subdivide to depth 10; the
     results above never needed more than depth 3.
- **Probe of larger A** (the author's fallback for the upper end): with the fixed checker, the upper end A = 0.6875
  itself verified all 300 sampled leaves (projected 0.9 CPU-hours), so the primary bracket is verified directly and no
  wider bracket is needed.

## Track 4 (2026-09-28): outer exclusion and solve brackets, checked independently

**What "independent" means here.** Every check below is in `src/verify_certificates.py`, which imports nothing from
`src/`. It does not replay the producers' searches (`certificates_v2.outer_exclusion`, `math_note_v2_checks.solve_limit`
and `solve_finite`); it re-derives each claim by its own method, in Arb at 80 bits. Inputs:
- the 800-point data, from `limit_A_lo.npz` after its SHA-256 matches the manifest (exact x-symmetry re-checked);
- the published parameters, read as data: the centres θ_c (`limit_switch.csv` argmin; for the solve bracket, the
  centre of the `mn2_solve_limit.csv` enclosure at A = 1.06), the 40-interval A grid, the solve bracket, the published
  signs;
- for the finite-a ends, the exported certificates (hash-checked).

The driver is `src/track4_certs.py`; `certificate_checks/track4_summary.json` collects the results.

**Timing sample first.**
- The outer check on the first annulus sub-interval [0.66, 0.66125] took 39 s (2,060 cell evaluations).
- The limit solve ends took 7.4 s.
- One finite-a end (a = 1.45, s = 4.025, "hi") took 451 s to check. This is its kept rerun; the first run's record was
  superseded by the per-leaf-sign fix below. The certificate itself is the sample run's export, whose time is not
  recorded; the other 11 exports took 7.0–14.5 s each.
- Extrapolated, the total was 1.95 h of single-process CPU (41 outer intervals, 2 limit ends, 12 finite ends).
  That is within the ≈ 2 h budget, so everything was run. The actual total was 1.93 h.

### (a) Outer exclusion (Link 2 of the four-link chain, math note §3(c)): **checked**, 41 of 41 pass

- **Claim checked.** For every A in the interval and every θ ∈ K(24), p ≥ 0, outside the closed 0.15 box around θ_c:
  L0*(θ; A) > L0*(θ_c; A) + 1e−4. The half p < 0 follows by the exact x-symmetry of the data.
- **Method** (`outer_exclusion_arb`). This is a fresh branch and bound, not a replay of the producer's float search:
  - dyadic cells from a unit grid on [0, 24] × [−52, 52], each split in four until certified;
  - a cell is dropped only if it lies in the closed box (exact rational test) or is disjoint from K(24) (Arb test);
  - A is carried as a **ball** in the objective. The producer instead bounds the loss at the midpoint A_c and adds a
    convexity A-term;
  - the logit is rewritten as A·(h(σ) − c) + b′ with a per-cell constant c. This is the same model, since b is profiled
    over ℝ. It keeps the A-ball narrow where h(σ) is large and nearly constant;
  - each cell's bound is the checker's own rigorous lower bound (the mean-value and direct bounds of the status
    checks). It holds for every θ in the cell and every A in the ball;
  - branch side: max(U(A_lo), U(A_hi)), with U an Arb upper bound at θ_c. This rests on L0*(θ_c; ·) being convex in A:
    the loss is jointly convex in (A, b), and minimising over b keeps convexity. This is stated, not machine-checked.
- **Results.**
  - Annulus, 40 sub-intervals: all pass, with no fail and no unresolved cell. The smallest certified margin is
    1.006e−4, on [0.68625, 0.6875]. It is at least the 1.0024e−4 that Theorem G (§16) takes as its outer margin, so the
    Theorem G input is independently confirmed as a lower bound. Per sub-interval, our margins and the producer's
    differ (both are lower bounds), and each is ≥ 1e−4.
  - Solve bracket: passes, with margin ≥ 1.649e−4 (the producer's is 1.476e−4, printed as ≥ 1.48e−4 in the math note).
  - Both runs also re-check Link 1: the branch upper bound (≤ 0.35805 and ≤ 0.28117) is below B(24) = 0.38797, which
    is recomputed by exact PAVA in Arb.
  - Cost: the annulus took 82,686 cell evaluations, depth ≤ 6, 1,577 s; the solve bracket 2,161 evaluations, 41 s.
    Peak RSS was ≤ 108 MB.
- **Tests.** Constructed cases:
  - pass: a region wholly outside the box;
  - fail: a centre shifted by 0.3, so the true minimiser lies outside the box; a witness point is found;
  - unresolved: a true but too-thin margin within a depth cap, and an evaluation cap;
  - the shifted-ball cell bound lies below the float profiled loss at 75 sampled (θ, A);
  - the exactness of the domain geometry.

### (b) Solve brackets

**Limit, A_solve ∈ (1.05875, 1.06]: checked** (`limit_solve_end`, `track4_solve_limit`).
- At each end A, Krawczyk's test proves a unique zero of the (p, q, b) gradient in a box of radius 1e−9. That box's
  (p, q) lies inside the 0.05 box around the solve centre.
- By the chain, that zero is the global minimiser (unique up to p → −p). All four links are checked independently for
  every A in the bracket:
  - localisation (B(24) recomputed);
  - outer exclusion ((a) above);
  - ring (re-run here with the centre recorded: 540 boxes, |gradient component| ≥ 8.2e−3);
  - PD on the 0.05 box (re-run: 100 boxes, b* bracketed).
- The solve margin, enclosed over the Krawczyk box:
  - at A = 1.05875: [−5.00373e−4, −5.00367e−4], "fails";
  - at A = 1.06: [1.67144e−4, 1.67150e−4], "solves".
  - Both signs equal the published ones, and both enclosures lie inside the published ones.
- **Not checked independently:**
  - the transversality d(margin)/dA ∈ [0.532, 0.536] and λ_min ≥ 0.1369 at the ends (the PD re-run checks PD only);
  - that the sign changes exactly once inside the bracket. The chain holds for every A in the bracket, so a sign change
    exists by continuity, but its uniqueness rests on the producer's transversality.
- **Tests:**
  - pass at both ends;
  - fail: a Krawczyk box that misses the zero, and a zero outside a displaced 0.05 box;
  - undecided: the margin over a box too wide to decide it.

**Finite a = 1.30–1.60, both ends: checked, 12 of 12 pass** (`check_solve`, `region_excluded_arb`; export `cert_export.solve`).
- **Export.** For each end (a, s) of `mn2_solve_finite.csv`:
  - the status branch and bound at tolerance 1e−7;
  - then the winning region again at 1e−11 (its argmin enclosure, as the producer's `solve_finite`);
  - the leaves are recorded, and their SHA-256 is appended to the manifest. The 12 certificates hold 41.4 MB (2.3–5.3 MB per npz).
- **Check.**
  - Winning region: its exported leaves (hashes, lemma W, exact tiling, every leaf's claim in Arb, every discarded
    leaf above U).
  - Losing region: a fresh Arb branch and bound shows L* > U on all of it. Its exported leaves are not used.
  - Hence the global minimiser lies in the winning region's kept leaves. On **every** kept leaf the solve margin's sign
    is decided in Arb, and it must equal the published sign.
  - The kept set has two mirror clusters (w₁ ↦ −w₁), so a bounding box of them decides nothing; the first timing run
    (unresolved, rerun) showed this.
- **Results.** All 12 ends pass.
  - The signs equal the published ones: "fails" at every lower end (s = 7.1, 5.7125, 4.7375, 4.0125, 3.475, 2.7 for
    a = 1.30–1.60) and "solves" at every upper end. The sign is decided on each of the 44–52 kept leaves per end.
  - Winning region: 15,432–23,292 leaves, 33,394–37,588 Arb evaluations, subdivision depth ≤ 3.
  - Losing region: the fresh branch and bound passes at every end (3,270–3,972 cells, depth ≤ 4, 48–65 s).
- **Two changes found on the way:**
  1. **The export, as first designed, recorded both regions at 1e−11.** At a = 1.45, s = 4.025, the losing region
     alone gave millions of leaves (to depth > 30), beyond the checker's 63-bit tiling keys. Checking them leaf by leaf
     would have taken far beyond the budget. That run was discarded, so these sizes are from its log, not from an
     artifact.
     - The losing region only has to lie above U. The checker now shows that by its own branch and bound
       (53 s at a = 1.45 hi).
     - So the export searches it at 1e−7. That file was deleted and its manifest rows removed before the rerun.
     - The status certificates' export is unchanged: the new key is written only for solve certificates.
  2. **The export's `regenerate` field named `finite` instead of `solve`.** Fixed before any solve certificate was
     kept.
- **Not checked independently:**
  - the published margin enclosures as numbers (e.g. [4.1e−6, 3.1e−4] at a = 1.45 hi); only their signs are checked;
  - that the sign changes once inside each bracket;
  - the exported leaves' cell edges, which are computed in floats from the integer indices, as in all earlier
    finite-a checks. The comparisons have margins far above that rounding. For the sign, each kept leaf is widened by
    1e−12.
- **Tests:**
  - the fresh branch and bound: pass; fail (a target above the losing region's infimum, with a witness); unresolved (a
    target 1e−6 below the infimum within a depth cap); the other region excluded below U and not above it;
  - `check_solve`'s own steps on the exported a = 1.45 hi certificate (its leaf checks stubbed): pass; fail (a flipped
    published sign; the losing region not excluded); undecided (a kept leaf coarsened to side 0.05).

### What remains unchecked by the independent checker (limit chain and thresholds)

- The ring and PD checks for the annulus (Track 5) were run with the centre as a parameter that their JSON files do
  not record. For the solve bracket they were re-run here with the centre recorded.
- Theorem G's finite-ε parts (§16) are not checked: the ε-inflated Krawczyk boxes, the 110 A-sub-interval branch boxes
  and the localisation at finite ε. Only its outer-margin input is confirmed, in (a).
- The limit and finite-a transversality statements are not checked (see (b)).
- The convexity-in-A lemma and the localisation lemmas are used as proved in the math note, not machine-checked.
