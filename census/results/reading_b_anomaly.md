# The Reading-B dense-zero: resolved, with a discovered implementation flaw

The anomaly: Reading B had zero dense-verified separations in all nine of
its configurations, including its mildest — which did not fit the
corrugation-strength account, since Reading B leaves the core untouched and
baseline geometry has dense survivors. Investigation of the sampled
distributions found the mechanism, and on the way found an implementation
flaw that contaminates the corrugation sweep's counts.

## Finding 1: Reading B has no mild arm — its amplitude parameter is ignored

`corrugation.sample` under Reading B modulates the sampled radius as
`radius · (1 + 0.5·sin(f·θ))`, clipped to the tube. **The amplitude
parameter never enters**; the modulation depth is hardcoded at 0.5. Every
Reading-B configuration carries the same strong 50% radial modulation, and
the four zero-noise frequency-100 configurations — `B_a0.05`, `B_a0.15`,
`B_a0.5`, `B_paper` — differ in nothing but their names.

Verified in the recorded sweep: their per-run eval accuracies are
**bit-identical across all four labels** (same seeds → same data → same
deterministic training). Four configuration labels, one condition, run
four times. So "Reading B at its mildest" never existed: the premise of
the anomaly — that mild-B should resemble baseline — was false. The
amplitude axis under Reading B was vacuous by construction.

## Finding 2: the clip concentrates ~20% of points exactly on the tube surface

Comparing sampled distributions (50,000 per class, same seed):

| | flat | Reading B (any config) |
|---|---:|---:|
| points exactly on the tube surface | 0.0000 | **0.1988** |
| points within 2% of the surface | 0.0395 | 0.2112 |
| min between-class distance @100k | 0.6013 | **0.7000** |

The clip `min(r·(1+0.5 sin), tube)` creates an atom of probability on the
extremal surface. Two consequences:

- **Proximity is ruled out as the mechanism**: Reading B's minimum
  between-class distance is *larger* than flat's (0.70 vs 0.60) — the
  classes are farther apart, not closer.
- **Dense verification becomes a surface probe.** A 100,000-point dense
  sample contains ~20,000 points exactly on the surface, against
  essentially zero for flat sampling. A decision boundary correct on the
  solid interior but marginally wrong on surface patches passes flat's
  dense check and fails B's. Direct check on the failed run `B_paper` d5
  s6: **60 of its 72 dense errors lie exactly on the tube surface** (83%,
  against a 20% base rate).

## Resolution

The Reading-B dense-zero is **not an anomaly against the
corrugation-strength account, because Reading B never had a mild arm to
test that account with**, and its dense failures are surface-margin
failures under a sampling scheme that probes the extremal surface with
finite probability per point — a measurement-regime difference from
baseline, unrelated to folding. After deduplication Reading B has **5
distinct separations, 0 dense survivors** (1 is a flip case passing 0/0 on
extra samples); n = 5 with a mechanism attached, not an open anomaly.

## The contamination, deduplicated

The three duplicate labels add 720 bit-identical rows to the corrugation
sweep (of 5,040). Corrected counts, distinct condition-runs only:

| Quantity | labeled | distinct |
|---|---:|---:|
| corrugation sweep rows | 5,040 | 4,320 |
| corrugated monotonic width-3 runs (all zero separations) | 1,890 | **1,620** |
| **project-wide monotonic width-3 total (T1)** | 5,570 | **5,300** |
| power bound on the monotonic rate | 0.054% | **0.057%** |
| GELU width-3 corrugated separations | 34 | **28** |
| GELU width-3 runs ≤5 errors (corrugated) | 60 | **48** |
| corrugated configurations | 21 | **18** |
| configurations with GELU w3 separations | 18 | **15** |
| width-4 separation rates | — | unchanged to ~0.5% (duplicates were unbiased copies) |

The monotonic zero is unaffected in kind — duplicates of zero are zero —
but the run count and the bound now use distinct runs only. `CLAIMS.md`
T1/T2 updated; status notes added to `corrugation_results.md` and
`corrugation_dense_rates.md`. The permutation tests and medians in the
corrugation writeup treated duplicated rows as independent; with 90 of 630
GELU width-3 rows duplicated the medians are essentially unchanged, and
the tail-count corrections are in the table above.

The generator keeps its recorded behaviour — changing it would orphan the
recorded artifacts — with the flaw now documented in the module and pinned
by a test (`test_corrugation_reading_b.py`). Any future Reading-B sweep
should thread amplitude through the modulation depth and regenerate from
scratch.
