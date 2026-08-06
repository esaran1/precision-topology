# Layerwise linking at width 3

**Everything in this file is at width 3, where `m + n + 1 = 1 + 1 + 1 = 3 = d`
holds and the estimate is the Theorem 4.7 invariant in the layer's own space.
No projection is applied anywhere in this document.** Results at widths above 3
are reported separately and under an explicitly stated projection convention;
the two must never share a table.

Method: ordered samples of the two Hopf-link core circles (512 points each) are
propagated through the trained network separately from the training data, which
consists of sampled solid tori. The linking number is defined on the cores, not
on the tubes. Networks are reconstructed from recorded width-sweep seeds and
refused unless train and evaluation accuracy reproduce exactly.

At the input the estimator returns −1.000025, rounding to **−1**, with minimum
inter-curve distance 1.000000.

## Reportability is a result, not missing data

A linking value is quoted only when the curves are disjoint **and** the minimum
inter-curve distance exceeds 0.02, the artifact threshold calibrated in
`linking_validation.md`. Below that the Gauss integrand is ill-conditioned and
any number it returns is an artifact, exactly as Ren and Lim's Table 8 reports
when ReLU collapses minimum distance to zero.

Rates over all hidden layers at width 3 (n = 140 layer-observations per
activation, seeds 0-4, depths 3, 5, 8, 12):

| Activation | Monotonic | Undefined (curves meet) | In artifact regime | **Reportable** |
|---|---|---:|---:|---:|
| ReLU | yes | **61.4%** | 100.0% | **0.0%** |
| tanh | yes | 12.1% | 96.4% | 3.6% |
| leaky-ReLU | yes | 10.0% | 97.9% | 2.1% |
| GELU | **no** | **0.0%** | 75.0% | **25.0%** |

Fraction **not** reportable, by layer:

| Layer | GELU | leaky-ReLU | ReLU | tanh |
|---:|---:|---:|---:|---:|
| 1 | 65.0 | 95.0 | **100.0** | 85.0 |
| 2 | 60.0 | 95.0 | **100.0** | 90.0 |
| 3 | 85.0 | 95.0 | **100.0** | 100.0 |
| 4 | 80.0 | 100.0 | **100.0** | 100.0 |
| 5 | 80.0 | 100.0 | **100.0** | 100.0 |
| 6 | 90.0 | 100.0 | **100.0** | 100.0 |

**ReLU is never reportable at any layer**, and in 61.4% of its layer
observations the two curves have literally met. This is the behaviour Table 8
describes, measured here across 20 runs per activation rather than one best
seed.

Runs in which the two core curves ever truly intersect (minimum distance
exactly 0):

| Activation | Runs with an intersection |
|---|---:|
| ReLU | **15 / 20 (75%)** |
| tanh | 3 / 20 (15%) |
| leaky-ReLU | 2 / 20 (10%) |
| GELU | **0 / 20 (0%)** |

Minimum distance at layer 1:

| Activation | Mean | Median | Min |
|---|---:|---:|---:|
| GELU | 0.03202 | 0.01367 | 0.00073 |
| tanh | 0.01697 | 0.00896 | 0.00104 |
| leaky-ReLU | 0.00893 | 0.00826 | 0.00010 |
| ReLU | 0.00207 | 0.00072 | **0.00000** |

## The six GELU runs that reach 1.0000 at width 3

These are the only runs in the entire 2,880-run sweep where separation succeeds
at the width the obstruction is stated for, so they are reported individually.
All six are exact `d = 3` measurements with no projection.

**depth 3, seed 10**

| Layer | link raw | rounded | min distance |
|---:|---:|---:|---:|
| 0 | −1.000025 | **−1** | 1.000000 |
| 1 | 0.000010 | **0** | 0.146992 |
| 2 | −0.000002 | 0 | 0.400156 |
| 3 | −0.000000 | 0 | 2.400370 |

**depth 3, seed 17**

| Layer | link raw | rounded | min distance |
|---:|---:|---:|---:|
| 0 | −1.000025 | **−1** | 1.000000 |
| 1 | −0.000069 | **0** | 0.145893 |
| 2 | 0.000001 | 0 | 0.665495 |
| 3 | −0.000000 | 0 | 2.282335 |

**depth 5, seed 0**

| Layer | link raw | rounded | min distance |
|---:|---:|---:|---:|
| 0 | −1.000025 | **−1** | 1.000000 |
| 1 | −0.000015 | **0** | 0.117229 |
| 2 | −0.000028 | 0 | 0.132508 |
| 3 | −0.000002 | 0 | 0.340270 |
| 4 | 0.000000 | 0 | 1.099553 |
| 5 | −0.000000 | 0 | 4.735525 |

**depth 5, seed 1**

| Layer | link raw | rounded | min distance |
|---:|---:|---:|---:|
| 0 | −1.000025 | **−1** | 1.000000 |
| 1 | 0.000036 | **0** | 0.156938 |
| 2 | −0.000008 | 0 | 0.263501 |
| 3 | 0.000000 | 0 | 0.436558 |
| 4 | −0.000003 | 0 | 1.383654 |
| 5 | −0.000001 | 0 | 4.275936 |

**depth 8, seed 14**

| Layer | link raw | rounded | min distance |
|---:|---:|---:|---:|
| 0 | −1.000025 | **−1** | 1.000000 |
| 1 | −0.000012 | **0** | 0.098043 |
| 2 | 0.000009 | 0 | 0.179825 |
| 3 | −0.000003 | 0 | 0.095311 |
| 4 | −0.000002 | 0 | 0.227945 |
| 5 | 0.000000 | 0 | 0.551936 |
| 6 | 0.000000 | 0 | 0.366043 |
| 7 | −0.000000 | 0 | 0.910530 |
| 8 | 0.000000 | 0 | 3.484730 |

**depth 12, seed 2**

| Layer | link raw | rounded | min distance |
|---:|---:|---:|---:|
| 0 | −1.000025 | **−1** | 1.000000 |
| 1 | −0.000002 | **0** | 0.142232 |
| 2 | 0.000005 | 0 | 0.096501 |
| 3 | −0.000000 | 0 | 0.284869 |
| 4 | 0.000000 | 0 | 0.058937 |
| 5 | 0.000000 | 0 | 0.104616 |
| 6 | 0.000000 | 0 | 0.279258 |
| 7 | −0.000000 | 0 | 0.542426 |
| 8 | −0.000000 | 0 | 1.248290 |
| 9 | −0.000000 | 0 | 1.510020 |
| 10 | 0.000000 | 0 | 1.661193 |
| 11 | 0.000000 | 0 | 2.099866 |
| 12 | −0.000000 | 0 | 5.525163 |

### What the six traces share

1. **Linking changes at layer 1 in all six**, from −1 to 0, at every depth from
   3 to 12.
2. **Minimum distance is strictly positive at every layer of every run.** The
   curves never meet. Linking goes to zero while the components stay disjoint.
3. **Minimum distance grows toward the output**, ending at 2.40, 2.28, 4.74,
   4.28, 3.48, and 5.53 against an input value of 1.00, having first dipped to
   roughly 0.10-0.16 at layer 1.
4. Residuals are of order 1e−5 or smaller, far inside the noise floor
   calibrated at 1.65e−04 for 200-point curves.

This is the signature the paper attributes to genuine unlinking, as opposed to
the ReLU case where linking appears to reach zero only because the curves have
been driven into each other. Ren and Lim report the same qualitative shape for
GELU and ReLU+skip in Table 8: link reaching 0 with minimum distance dipping and
then rising steadily.

The distinction matters because it separates two ways of reaching link 0. One
destroys the geometry, so the invariant stops being defined; the other performs
a fold, so the invariant changes while remaining defined. Only the latter is
consistent with perfect classification, and only GELU produces it here.

## Reportable linking values at width 3

Across all reportable hidden-layer observations:

| Activation | link = −1 | link = 0 | link = +1 |
|---|---:|---:|---:|
| GELU | 4 | **27** | 4 |
| tanh | 2 | 0 | 3 |
| leaky-ReLU | 1 | 2 | 0 |

GELU is the only activation with a substantial count at link 0 while the
curves remain disjoint. The monotonic activations are almost never reportable
at all, and when they are, they are usually still at ±1. The ±1 values for GELU
occur at layers where the fold has not yet happened or where a later layer has
re-linked a still-disjoint pair.

## Caveats

- **This is not a test of Theorem 4.7.** The theorem forbids a width-3
  monotonic network from linearly separating the two *continuous* curves. We
  measure a finite sample of the cores propagated through networks trained on
  thickened tubes. The results are in the direction the theorem describes but
  do not establish it.
- **Optimisation is confounded with expressivity.** The monotonic runs that
  drive the curves into intersection have failed to find a separating map; the
  theorem says none exists, but a failed search and an impossible search look
  identical from outside.
- **The six GELU successes are 6 of 80.** GELU reaches 1.0000 at width 3 in
  only 7.5% of runs. The mechanism is available to it, not reliable for it.
- The artifact threshold of 0.02 is a calibrated convention, not a sharp
  boundary; degradation on approach is gradual.

## Status

Width 3 complete. Widths above 3 pending and will be reported separately, under
the projection convention, never in the same table as these numbers.
