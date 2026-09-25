# Block 2: certificate auditability (export every certificate, check each with an independent checker)

**For the rebuttal revision** (the submission's certificates are unchanged).

- **Checker**: `src/verify_certificates.py`, python-flint (Arb) 0.9.0 at 80 bits. It imports nothing from `src/`;
  every decision is an Arb comparison. Tests: `tests/test_verify_certificates.py` (11 tests, including constructed
  cases it must reject).
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
| solve brackets (finite and limit), outer exclusion and ring, PD boxes, K = sup G₀ (its domain lemma is written; math note §8), Krawczyk boxes for c₁ | — | pending | | |

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
