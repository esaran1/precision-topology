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
| Ĝ(1.02) | — | **not exported: stopped under the 3 GB rule.** It has 147M search cells, 6× a = 1.05, which peaked at 1.5 GB, so ≈ 9 GB is extrapolated. It needs the recording to stream leaves to disk, or a tighter certifier | | |
| solve brackets, limit switch, outer exclusion and ring, PD boxes, K, Krawczyk boxes, localisation | — | to do | | |

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
