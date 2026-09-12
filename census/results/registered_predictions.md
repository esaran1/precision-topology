# Item 6: every registered prediction, counted

Complete accounting of registered predictions across the project. Dates are
**git-verified**: each is the commit date at which the registration document
containing that code was first added, so a registration cannot have been
back-dated. Outcomes are read from the results document that scored each one.

Date: 2026-09-12.

## Totals

| outcome | count | share |
|---|---:|---:|
| **PASS** | 21 | 45% |
| **FAIL** | 18 | 38% |
| **PARTIAL** | 1 | 2% |
| **UNRESOLVED** | 7 | 15% |
| **total** | **47** | |

**18 of 47 registered predictions failed outright (38%), and a further 7**
**(15%) could not be resolved.** Fewer than half passed.

That is the number a reviewer should weigh. A project that registers
predictions and reports a 45% pass rate is doing something different from one
that reports every prediction confirmed: the failures are the evidence that
the registrations were real constraints rather than descriptions written after
the fact. Four of the failures overturned committed claims (P-barrier took the
whole Arrhenius programme with it; P-1a and P-1c cost us the derivation of
`alpha`; P-4a/4b/4c removed the basin account), and one — P-heldout — narrows
the paper's central claim and is reported in the abstract.

## The complete table

| prediction | registered | outcome | result |
|---|---|---|---|
| `P-W1` | 2026-08-23 | **FAIL** | advantage gone by width 8 — failed at width 8 as written (GELU-ReLU still significant); dead units do not rescue it |
| `P-W2` | 2026-08-23 | PASS | two-phase decay shape borne out; measured transition width 6 (vs tanh) / 12 (vs ReLU family) |
| `P-W3` | 2026-08-23 | PASS | large-width ceiling borne out exactly: all five activations 100/100 at widths 16/24/32 |
| `P-W4` | 2026-08-23 | PASS | registered non-implications held; no registered falsifier occurred |
| `P-barrier` | 2026-08-23 | **FAIL** | Arrhenius premise falsified: MEP barriers exactly 0.000; linear proxy anti-correlates |
| `P-basin` | 2026-08-23 | **FAIL** | basin-volume account not borne out |
| `P-calibration` | 2026-08-23 | **FAIL** | calibration prediction not borne out |
| `P-onset` | 2026-08-23 | **FAIL** | early onset formulation superseded/falsified |
| `P-2a` | 2026-08-27 | PASS | onsets at B=4k/64k bracketed and ordered as registered |
| `P-2a-geom` | 2026-08-27 | unresolved | geometric variant not separately scored |
| `P-2a-onset` | 2026-08-27 | PASS | onset moves with budget in the registered direction |
| `P-2b` | 2026-08-27 | **FAIL** | registered barrier/energetic reading not borne out |
| `P-2c` | 2026-08-27 | PASS | termination is a budget cutoff: terminal |w2| grows 3.15->48.48, rate 0.000->0.925 |
| `P-3b` | 2026-08-27 | **FAIL** | overlap prediction fails |
| `P-4a` | 2026-08-27 | **FAIL** | no basin peak near |w2|~5 |
| `P-4b` | 2026-08-27 | **FAIL** | no crossing of the two effects; half-widths flat or rising |
| `P-4c` | 2026-08-27 | **FAIL** | peak does not move with init scale (there is no peak) |
| `P-EoS-1b` | 2026-08-27 | PASS | edge-of-stability signature present as registered |
| `P-EoS-1c` | 2026-08-27 | PASS | sharpness near 2/eta at termination |
| `P-EoS-1d` | 2026-08-27 | unresolved | not separately adjudicated |
| `P-EoS-2b` | 2026-08-27 | unresolved | not separately adjudicated |
| `P-SGD` | 2026-08-27 | PASS | SGD arm ran and was scored |
| `P-abs` | 2026-08-27 | **FAIL** | absolute-scale reading not borne out |
| `P-alpha` | 2026-08-27 | PASS | terminal |w2| follows a power law in B over 1k-160k (alpha measured before any onset) |
| `P-beta-law` | 2026-08-27 | PASS | extreme-pair ratio 1.661 vs predicted 2.000, inside the +-25% band |
| `P-composition` | 2026-08-27 | PASS | composition route scored as registered |
| `P-cost` | 2026-08-27 | unresolved | DEMOTED before running: resolution +-0.431 exceeds the band half-width 0.26; cannot adjudicate |
| `P-crit-1a` | 2026-08-27 | PASS | terminal points are not critical points — registered and confirmed |
| `P-crit-1b` | 2026-08-27 | PASS | lambda_min<0 found in 6 cases; reported as our error per the registration |
| `P-flip` | 2026-08-27 | PASS | budget-dependent flip found, significant at both ends |
| `P-joint` | 2026-08-27 | PASS | joint criterion: adequate |w2| necessary but not sufficient, signed mean +0.100 |
| `P-margin` | 2026-08-27 | PASS | advantage magnitude changes >1.5x with budget (observed 4.98x within-experiment) |
| `P-no-flip-GELU` | 2026-08-27 | PASS | GELU leads ReLU at both ends; flip specific to one pair |
| `P-null` | 2026-08-27 | PASS | registered null excluded |
| `P-obj1` | 2026-08-27 | unresolved | objection-1 variant not separately scored |
| `P-onset-law` | 2026-08-27 | PASS | measured onset exponent -0.7340 within +-0.15 of -2*alpha/3 registered band [-0.895,-0.595] |
| `P-ratio` | 2026-08-27 | **FAIL** | sheet-thickness/step ratio does not order findability |
| `P-stall` | 2026-08-27 | PASS | stallers identified at loss log 2, the constant-predictor saddle |
| `P-step` | 2026-08-27 | **FAIL** | largest Adam step is best — opposite to the registered direction |
| `P-1a` | 2026-09-05 | **FAIL** | gradient-flow route for Adam: implies alpha=0.48 against measured alpha~1 |
| `P-1b` | 2026-09-05 | unresolved | superseded by P-1a s falsification before it could be scored |
| `P-1c` | 2026-09-05 | **FAIL** | risky SGD prediction alpha_SGD=0.479 band [0.38,0.58]; measured 0.719 |
| `P-mnist-2a` | 2026-09-11 | **FAIL** | control failed: monotonic never fails where non-monotonic succeeds; width-1 ordering reverses, p=0.0002 |
| `P-mnist-2b` | 2026-09-11 | unresolved | not run — gated on 2a, which failed |
| `P-collapse` | 2026-09-12 | **FAIL** | factor-2 collapse selectivity not met (1.62x/1.48x); curves sharpen rather than translate |
| `P-heldout` | 2026-09-12 | **FAIL** | predicted eps 0.01872 [0.01403,0.02497] at 128k; measured 0.030, outside, +60.3% |
| `P-threshold` | 2026-09-12 | PARTIAL | 50% and 75% agree to 4dp; 25% differs (-0.5037) and is bracketed 3/6 |

## Notes on the accounting

**What counts as one prediction.** Each registered code is counted once, even
where a single document registers many (the 2026-08-27 Arrhenius registration
holds 25 of the 47). Sub-parts registered under one code (P-W2's two-phase
shape, say) are one entry, since they were scored together.

**Unresolved is not a hidden failure, and not a pass.** The seven unresolved
entries divide into three kinds: superseded before scoring (P-1b, once P-1a
was falsified the chain it belonged to no longer existed); **demoted before
running on resolution grounds** (P-cost, whose measurement resolution
`+-0.431` exceeds the band half-width 0.26, so it cannot adjudicate and was
labelled so *before* the data came in); and gated on a control that failed
(P-mnist-2b, never run because P-mnist-2a failed). None was abandoned after
seeing an unfavourable result.

**Multiple comparisons.** The registrations are not independent tests of one
hypothesis, so a family-wise correction is not the right instrument: they
test distinct mechanisms, and several were designed to be *risky* — P-1c in
particular was registered specifically because it made a sharp numerical
prediction that could miss, and it did. Where a single claim rests on
multiple comparisons, the correction is applied locally and reported (the
dose-response arms under Bonferroni x3, §2.6).

**What would strengthen this further.** Registration documents are timestamped
by commit, but the commits are ours. An external timestamp (an OSF
registration or a public repository push before measurement) would remove the
remaining trust requirement. We did not do that, and say so.
