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

## The central result: disjointness is what makes these six runs mean something

Linking number reaching 0 is not by itself evidence of anything. There are two
ways to get there, and they mean opposite things.

**Destroy the geometry.** Drive the two curves into each other. Once they
intersect, the linking number is undefined, and any value the Gauss integral
returns is a numerical artifact. Nothing has been unlinked; the configuration
has simply been ruined, and no downstream layer can recover a separation from
it because an intersection propagates.

**Perform a fold.** Move the curves past each other while they remain disjoint.
Linking changes from −1 to 0 and stays defined throughout. This is the operation
that permits linear separation, and it is what a non-monotonic activation makes
available.

The two are distinguished by exactly one quantity: **minimum inter-curve
distance, which must stay strictly positive**. In all six GELU runs it does, at
every layer, dipping to 0.098-0.157 where the fold occurs at layer 1 and then
**growing to 2.28, 2.40, 3.48, 4.28, 4.74, and 5.53** at the output, against an
input value of 1.000. The curves separate further and further after the fold.
That trajectory — dip, then sustained growth — is the fold signature, and it is
the reason these six runs are informative rather than merely lucky.

The monotonic activations do not produce it. The contrast is direct:

| | GELU (non-monotonic) | ReLU (monotonic) |
|---|---|---|
| Runs where curves intersect outright | **0 / 20 (0%)** | **15 / 20 (75%)** |
| Hidden layers with a reportable value | **25.0%** | **0.0%** |
| Hidden layers where curves have met | **0.0%** | **61.4%** |
| Minimum distance at layer 1 (mean) | 0.03202 | 0.00207 |
| Minimum distance at layer 1 (min) | 0.00073 | **0.00000** |

The important reading is not that the monotonic activations fail to unlink. It
is stronger than that: **they fail to maintain disjointness while attempting
it.** ReLU never yields a reportable linking value at any layer of any run,
because by the time its linking number would have changed, the two curves are
already touching. Three quarters of its runs end with an outright intersection.
tanh and leaky-ReLU sit between the two extremes, intersecting in 15% and 10%
of runs and reportable at 3.6% and 2.1% of layers, but they are far closer to
ReLU than to GELU.

This is the distinction Ren and Lim draw in their Table 8 interpretation, where
ReLU's fractional link values at layers 0 and 1 are starred as artifacts of the
integral becoming ill-conditioned as minimum distance approaches zero, rather
than treated as partial unlinking. Our measurement reproduces that behaviour
across many runs.

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

### The change happens at layer 1 regardless of available depth

All six runs change linking at **layer 1**, whether the network has 3, 5, 8, or
12 layers to work with. The topological work is done immediately rather than
distributed across the available depth, and the remaining layers are spent
increasing separation: minimum distance climbs from its smallest value to the
output in every run, by factors of 15.6x, 16.3x, 27.3x, 36.6x, 40.4x, and
93.8x.

This is recorded as an observation, not an explanation. **No mechanism is
offered**, and it rests on six runs, all from a single activation at a single
width. Whether the immediacy is a property of the fold operation, of the
optimiser, of this initialisation, or of the particular geometry here is not
determined by this measurement. It would be worth checking against a larger
sample and against non-monotonic activations other than GELU before any weight
is put on it.

### Relation to Table 8

This measurement is stronger than Table 8 in one specific respect: **it reports
20 runs per activation with a calibrated noise floor and an explicit
undefined-rate, where Table 8 reports a single best seed.**

That is a difference in purpose, not a deficiency in the original. Table 8 is
presented as a mechanistic illustration — its role in the paper is to show what
the impossibility theorem looks like layer by layer in a concrete network, and
one clearly annotated trace does that job. It also does the most important
thing correctly: it stars the fractional ReLU values as artifacts and states
that linking number is only defined for disjoint curves, which is the exact
point our artifact threshold operationalises.

What the additional runs add is the ability to say how often each behaviour
occurs rather than that it can occur. The 75% intersection rate for ReLU, the
0% for GELU, the 0.0% versus 25.0% reportability, and the seed-level dispersion
underlying them are quantities a single trace cannot supply. The calibrated
noise floor (1.65e−04 at 200 points, with the estimator surviving jitter to 20%
of curve radius) additionally lets a value be called converged rather than
merely plausible, which matters when the claim is that a number equals an
integer.

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

---

> **Status note (2026-08-22).** Two later findings bear on traces reported
> here. (1) Dense verification (`dense_check.md`): separation labels are
> sample-level; on the baseline link nearly all width-3 GELU separations
> survive dense checking, so the traces in this file are essentially
> unaffected. (2) Trace methodology (`localization_results.md`): the sign of
> the linking number legitimately flips under orientation-reversing affine
> layers (negative determinant), so |lk| is the layer-invariant quantity;
> and 512-point traces can under-resolve deep, strongly distorting networks
> — high-resolution retracing confirmed the GELU baseline traces used here
> are stable under resolution doubling.
