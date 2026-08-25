# Overclaim sweep (hardening Part 3, 2026-08-25)

Full-text scan of every results document, the ledger, FINDINGS.md and
notes/ for: forbidden verbs (shows/demonstrates/establishes/confirms/
proves/validates), causal language, all-cells generality, numbers
without n, and single-run claims stated as more.

## Fixes applied (12 edits, commit below)

| document | count | nature |
|---|---:|---|
| results/witness.md | 3 | "Established:" header → "Measured and held:"; two "establish" → "support" |
| results/gap_results.md | 2 | "demonstrated by manipulation" → "shown by two-sided manipulation"; "confirmed by intervention" → "borne out" |
| results/offset_results.md | 2 | "proves representability" → "realizes"; "established independently" → "found" |
| results/gelu_scale_results.md | 1 | "confirms the low side" → "bears out" |
| results/parametrization_sensitivity.md | 1 | "establishes that" → existence-case phrasing |
| results/interleaved_quantization.md | 1 | "established" → "recorded" |
| results/linking_width3.md | 1 | "retracing confirmed" → "agreed with" |
| CLAIMS.md (T31) | 1 | "is validated" → "checks out against exact ground truth" |

## Deliberately not edited, with reasons

- **Registration files (`*_prediction.md`)**: their language was
  registered before results; retroactive edits would damage the
  registration trail worth more than the wording. (Their two
  "confirms"-family usages are pre-outcome conditionals.)
- **Withdrawn/superseded documents** (`linking_projected.md`):
  historical text stays under its correction blocks.
- **"Validation" as a procedure name** (estimator validation suite,
  validation gate) and **"verified" for mechanical exact-integer
  checks** (lk = −q against designed geometry): these denote checks,
  not claim inflation.
- **T10's "confirmed existence proof"**: phrasing was directed
  verbatim by the project owner (2026-08-23 instruction).
- **notes/**: internal reading notes, not claims documents.

## Causal-language scan

All "because/due to/driven by" instances are deductive
(geometry/arithmetic: rotation invertibility, grid overshoot, gate
attrition) except two mechanism statements inside the weight-scale
account's domain, which rests on the two-sided manipulation (T29/T27)
and is the one place mechanism language is licensed. One instance —
`fold1d_results.md` "empty within both boxes because required
|w₂| ≈ 58 exceeds the box" — is **flagged as possibly wrong**, pending
the Part 5 theorem work (the emptiness may be grid-resolution rather
than true emptiness); it will be corrected there if the construction
check says so, not silently here.

## Generality and n scans

- "at every width for every activation" (T24): fixed in the 1c package.
- Single-run claims: T12 corrected in the 1d package; the witness and
  offset-witness exhibits are stated as single-object exhibits, which
  is what they are.
- Remaining all-quantifiers ("all 11 dense-verified", "every stratum",
  "163/163") were each checked against cell-complete recomputation in
  `AUDIT.md` Part 1 and stand.
