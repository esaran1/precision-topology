# Part 4: generator diff against the published appendix

**Provenance: raised in external expert review** -- a mismatch between the
appendix description and our released generator, with details to follow.
Posture per the brief: **assume ours is wrong until shown otherwise.**

## What I could verify, and what I could not

**Could not**: the repository does not hold the appendix's torus construction
text (major/minor radii, centre offsets, sampling law). `notes/` records
Appendix G.2's architecture statement and G.8's linking method, but not the
geometry specification. **I therefore cannot perform the requested diff
against the published text**, and I am not going to reconstruct the appendix
from memory and diff against that -- it would manufacture agreement or
disagreement with equal ease. **The diff is deferred pending the reviewer's
details or the appendix text.**

**Could verify**: what our generator actually produces, measured from samples
rather than read from code, and whether it is internally consistent.

## Our generator, measured from 200,000 samples

| property | measured | intended |
|---|---|---|
| class 0 core | unit circle, z = 0 plane, centre (0,0,0) | same |
| class 1 core | unit circle, y = 0 plane, centre (1,0,0) | same |
| major radius R | 1.0 | 1.0 |
| tube radius r | 0.2000 | 0.2 |
| core separation | 1.0 = R | R |
| disjointness 2r < R | 0.400 < 1.0, holds | holds |
| min cross-class distance | 0.6020 | R - 2r = 0.600 |
| linking number of cores | **-1.000002** | +-1 |

The construction is a standard Hopf link -- two unit circles in orthogonal
planes whose centres are separated by R, each passing through the other's
disc -- and the Gauss integral on the cores returns -1 to 2e-6. Sampling is
uniform in (theta, cross-sectional area).

## Discrepancies found, reported whether or not they favour us

1. **Class supports are 3-D solids, not the theorem's 1-D curves.** Theorem
   3.7 concerns disjoint closed oriented *curves*; we sample solid tubes
   around linked cores. **This was already recorded** in
   `notes/icml_paper_notes.md` (item 2) and is a genuine mismatch between what
   the theorem constrains and what the experiments train on. Every empirical
   result here is about thickened samples; the theorem is not directly tested
   by them, which the caveats in `width_sweep.md` already state.

2. **The sampler is not volume-uniform** (found in our own audit, 2026-08-25):
   density per unit volume is R/(R + rho*cos(phi)), so the inner equator is
   oversampled ~1.5x relative to the outer. Coverage is complete (density
   bounded below at 0.833x uniform), and 84 dense survivors plus both
   witnesses were re-verified under an exactly volume-uniform sampler with no
   claim moving (`uniform_resample.md`).

3. **No discrepancy found between the generator and its own documented
   intent**: every measured quantity matches the docstring and the parameters
   to within sampling error.

## Status

The specific mismatch the reviewer identified **cannot be checked without
their details**, and this document is the placeholder that says so rather
than claiming a clean bill of health. What is verified is that our generator
produces a genuine Hopf link with the stated parameters; what is known to
differ from the theorem's setting is the solid-versus-curve distinction,
which predates this review and is already documented.
