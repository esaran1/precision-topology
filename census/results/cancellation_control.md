# Cancellation control: projection can hide nonzero linking

**Outcome: cancellation is real, reproducible, and frequent. The uniform-zero
conclusion in `linking_projected.md` does not hold and is withdrawn.**

> **The single most important number here: the Hopf link with `|lk| = 1` — the
> exact configuration our original fidelity control used — returns a clean
> integer 0 in 17–23% of random projections once the embedding is distorted.**
> That control reported 150 of 150 recoveries because it embedded by *rigid
> rotation*, and rigid maps preserve precisely the structure that cancellation
> destroys. The control was not merely weak; it was constitutionally unable to
> detect the failure mode it was meant to rule out.

## What was tested

The author's concern is that projecting to `R^3` creates self-intersections, so
the projected image can be several joined rings, and positive and negative
crossings can cancel to give linking number 0 while nontrivial linking is
present. Our existing fidelity control cannot detect this: it used a Hopf link
with `|lk| = 1` embedded by a **rigid rotation**, where there is nothing to
cancel and the geometry is undistorted.

Two things were varied here: winding number, following the author's suggestion
to increase winding rather than duplicate Hopf links, and — decisively — whether
the embedding is rigid or distorted.

Configurations, with true linking verified in `R^3` by the validated estimator
before any projection:

| Configuration | True linking | Residual | Min distance |
|---|---:|---:|---:|
| torus link, winding 1 | −1 | 5.74e−07 | 1.6000 |
| torus link, winding 2 | −2 | 2.26e−06 | 1.5278 |
| torus link, winding 3 | −3 | 6.38e−06 | 1.1165 |
| torus link, winding 4 | −4 | 1.40e−05 | 0.8741 |

## Result 1: rigid embeddings never cancel

Embedded into `R^5` and `R^8` by a Haar-random rotation, then projected back
through 100 random 3-frames each:

| Configuration | True | Fraction zero | Fraction correct | Distinct values |
|---|---:|---:|---:|---|
| winding 2, `R^5` | −2 | **0.000** | 1.000 | −2, 2 |
| winding 2, `R^8` | −2 | **0.000** | 1.000 | −2, 2 |
| winding 3, `R^5` | −3 | **0.000** | 0.990 | −8, −3, 3 |
| winding 3, `R^8` | −3 | **0.000** | 1.000 | −3, 3 |
| winding 4, `R^5` | −4 | **0.000** | 0.980 | −6, −4, 3, 4 |
| winding 4, `R^8` | −4 | **0.000** | 0.990 | −4, 4, 5 |

**Zero cancellation in 600 projections.** Sign flips occur, as expected from
orientation reversal, but the magnitude survives. This reproduces our original
control and explains why it passed 150 of 150.

## Result 2: distorted embeddings cancel readily

The rigid case is not what a network does. Replacing the rotation with a random
non-orthogonal linear map followed by `tanh` — what a layer actually computes —
changes the picture completely:

| Configuration | True | Fraction zero | Fraction correct | Distinct values |
|---|---:|---:|---:|---|
| winding 2, `R^5` | −2 | **0.100** | 0.770 | −2 … 2 |
| winding 2, `R^8` | −2 | **0.280** | 0.340 | −2 … 2 |
| winding 3, `R^5` | −3 | **0.120** | 0.510 | −3 … 3 |
| winding 3, `R^8` | −3 | **0.200** | 0.300 | −3 … 3 |
| winding 4, `R^5` | −4 | **0.070** | 0.370 | −4 … 6 |
| winding 4, `R^8` | −4 | **0.160** | 0.270 | −4 … 5 |

**Between 7% and 28% of projections return exactly 0 while the true linking
number is −2, −3, or −4.** The estimator is not merely noisy: it returns a clean
integer zero on a configuration that is definitely linked.

Fraction correct falls as low as 0.27, so under distortion the majority of
projections give the wrong magnitude, not just the wrong sign.

## Result 3: the driver is distortion, not winding

The `|lk| = 1` Hopf link — our original control configuration — also cancels
once the embedding is distorted:

| Embedding | Configuration | Fraction zero |
|---|---|---:|
| Rigid rotation, `R^5`/`R^8` | Hopf, `|lk| = 1` | **0.000** (20/20 correct) |
| tanh-distorted, `R^5` | Hopf, `|lk| = 1` | **0.170–0.190** |
| tanh-distorted, `R^8` | Hopf, `|lk| = 1` | **0.230** |

So higher winding is not required to produce cancellation. **The variable that
matters is whether the embedding distorts the geometry**, and our fidelity
control exercised only rigid rotations. That is precisely why it reported 150 of
150 recoveries and gave false reassurance.

## Consequence: the uniform-zero conclusion is withdrawn

`linking_projected.md` concluded that the uniform projected zero across 535
reportable final-layer observations was a real null rather than a blind measure,
resting on three arguments. Two of them no longer hold:

1. **"The control recovers linking whenever it is present."** It does so only
   for rigid embeddings. Under the distortion a trained layer actually applies,
   it fails 7–28% of the time on configurations that are genuinely linked, and
   17–23% of the time even on a Hopf link.
2. **"A destructive projection would fail inconsistently."** This was the
   strongest argument and it is now the weakest. Cancellation does not require
   inconsistency across cells: if the representation at a given layer sits in a
   regime where most projections cancel, every observation in that cell returns
   0 uniformly. Uniformity is consistent with systematic cancellation, not only
   with absence.

The third argument is **theoretical, not empirical**, and must not be presented
as remaining support for the null:

3. **Two circles in `R^4` and above are always unlinked.** This is a statement
   about the ambient dimension. It is not evidence about our measurement, and it
   would be true whatever our estimator returned. Treating it as support for the
   null would be circular: it is a prior expectation, not a result.

**The accurate position is that the projected analysis cannot currently
distinguish absence from cancellation.** The theoretical expectation happens to
point the same way, which is reassuring but is not corroboration — a measurement
that cannot discriminate between two hypotheses provides no evidence for either,
regardless of which one theory favours.

The projected zeros therefore cannot be used as evidence in either direction.
The width-3 results in `linking_width3.md` are unaffected: they are exact,
unprojected, and in the dimension where the invariant is defined.

## A general methodological point

This is not specific to our setup. **Any study that projects high-dimensional
representations to `R^3` to compute a linking number is exposed to the same
failure**, and the natural control does not catch it.

The pattern is:

1. The invariant is only defined in `R^3` for two curves, so a projection is
   needed to measure anything at all.
2. Projection creates self-intersections; the image can be several joined rings
   whose signed crossings cancel.
3. The obvious control — embed a known link, project it back, check the value is
   recovered — passes cleanly if the embedding is a **rotation**, because rigid
   maps preserve exactly the structure that cancellation destroys.
4. The control therefore certifies an estimator that will return 0 on linked
   configurations once the representation is genuinely deformed, which is the
   only case that arises in practice.

A control built on rigid rotations tests whether a link can be *recovered*. It
never tests whether a link can be *hidden*, and those are different questions.
The minimal fix is to sample the **distribution** over many random projections
rather than checking one, and to include a distorted embedding rather than an
isometric one. Reporting the fraction of projections returning 0, alongside the
distribution of values, makes the failure visible where a single number hides
it.

Ren and Lim already label their own PCA-3D CIFAR-10 analysis as projection- and
sampling-dependent and treat it as correlational. The measurements here give a
quantitative sense of how large that dependence can be: up to 28% of random
projections returning 0 on a configuration with `|lk| = 4`.

## What this does not show

- It does not show that our width>3 representations *are* linked. The
  theoretical expectation points to absence, and nothing here contradicts it.
  What is withdrawn is the claim that the *measurement* supports that
  conclusion.
- The cancelling projections are not estimator failures in the numerical sense.
  Each returns a clean integer with a small residual; the integer is simply the
  linking number of the projected image, which differs from that of the
  original.
