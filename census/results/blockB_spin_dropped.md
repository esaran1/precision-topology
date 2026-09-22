# `R_spin` is dropped: the conditional landscape has a single continuous switch

The `a = 1.60` refinement (`blockE_window_refinement.md`) showed that window's
hysteresis gap was the 0.05 grid. The audit below extends that to every `a`.

Artifact: `results/blockB_fine_windows.csv`. Same frozen Block B procedure
(sha256 `9f1b10741d8bf48c`), grid step **0.01** instead of 0.05. Per the freeze
rule this change is justified by **resolution alone**; the registration itself
called for it, and the result's direction (windows shrink) is the opposite of what
agreement-seeking would produce, since a wider window would have supported the
spinodal reading.

---

## `R_glob = R_spin` at every `a`, and every window is one grid step

| `a` | `R_fold` | `R_glob` | `R_spin` | width | grid steps | 0.05-grid `R_spin` | 0.05 width |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.35 | 0.21514 | 0.21568 | 0.21568 | 0.00053 | **1.0** | 0.22689 | 0.01335 |
| 1.40 | 0.21670 | 0.21734 | 0.21734 | 0.00064 | **1.0** | 0.22506 | 0.01608 |
| 1.45 | 0.21861 | 0.21936 | 0.21936 | 0.00076 | **1.0** | 0.22693 | 0.01891 |
| 1.50 | 0.21984 | 0.22071 | 0.22071 | 0.00087 | **1.0** | 0.23991 | 0.02181 |
| 1.60 | 0.22291 | 0.22402 | 0.22402 | 0.00111 | **1.0** | 0.25077 | 0.02786 |

`a = 1.30` already had `R_glob = R_spin = 0.2145` at the coarse grid, which is what
triggered the original no-hysteresis falsifier.

`R_glob` and `R_spin` agree to **1e-15** — they differ only in floating-point
representation of the same `|w2|`. The `G <= 0` branch followed upward is lost at
**exactly** the `|w2|` where the global minimiser first has `G > 0`, and the
`G > 0` branch followed downward is lost one grid step below. That is resolution,
not bistability, at all six activation values.

**Independent confirmation from the scaling limit**: Block B run directly on
`h(sigma)` gives `R_glob^inf = R_spin^inf = 0.19991` at `W = 0.690`
(`scaling_limit_results.md`). The coincidence holds in the limit problem too, with
no `f_a` in it.

## Consequences — `R_spin` is removed from every claim

**1. The drift comparison keeps only `R_glob`.** Recomputed on the refined values:

| | 1.30 -> 1.60 | correlation with measured |
|---|---:|---:|
| measured crossing | **+10.0%** | — |
| `R_glob` (refined) | **+4.5%** | **+0.9965** |

The refinement *improves* the correlation (+0.980 -> **+0.9965**) while lowering
the predicted drift (+6.5% -> +4.5%). The `+16.9%` figure previously quoted for
`R_spin` is withdrawn: it was the coarse grid's artifact, and its `a = 1.60`
endpoint moves from 0.25077 to 0.22402.

**2. D-1 is vacuous.** `blockF_lag_prediction.md` § D-1 compares
`M-spin: R_spin(a) + c*rate(a)` against `M-glob: R_glob(a) + c*rate(a)`. With
`R_spin = R_glob` at every `a`, **the two models are identical** and the comparison
has nothing to distinguish. D-1's registered expectation — "neither will be
distinguished" — is satisfied, but for a stronger reason than the one registered:
not that the data lack leverage, but that **the two hypotheses coincide**.
Recorded as vacuous rather than scored as a pass.

**3. The "crossings track the upper spinodal" test is withdrawn entirely.** It was
already recorded as non-discriminating (`blockB_results_final.md`); it is now
recorded as **meaningless**, because there is no upper spinodal distinct from the
global switch. The prospective/post-hoc distinction on that test (3 of 3) no longer
refers to anything and is withdrawn with it.

**4. The bifurcation-with-hysteresis framing stays dropped**, as it has been since
the `a = 1.30` falsifier. This audit closes the possibility that it survived at
larger `a`.

## The statement that replaces it

> The conditional landscape has a **single continuous switch** at `R_glob(a)`.
> Following the `G <= 0` branch upward and the `G > 0` branch downward, the two
> meet within one grid step of 0.01 in `|w2|` at all six `a`, and the gap rises
> smoothly through zero. Measured crossings sit **above** `R_glob` by a factor of
> **1.087 to 1.144** (mean **1.115**, sd 0.020), one-signed at every `a`, which is
> relaxation lag: runs track the branch (Block C: 1.09 -> 0.008) while `|w2|`
> keeps moving the target.

`R_fold` is retained only as the downward-continuation check that establishes the
one-step coincidence; it is not a separate switch point either.
