# §2 The link setting: where the phenomenon appears

*Draft in publication prose. Companion to `paper/results_draft.md` (§3–§6),
which this section precedes. Verification of every number against raw
artifacts is in `paper/link_cifar_verification.md`.*

---

## 2.1 Setting

We train small feedforward networks to separate two Hopf-linked solid tori in
`R^3`. Each class is a tube of radius `rho` sampled around one of two circular
cores lying in orthogonal planes, positioned so that each core passes through
the other's disc; the Gauss integral on the cores returns `-1` to within
`2e-6`. Networks are width 3 throughout this section — the representation
space is `R^3` at every layer, so width never provides an escape — and we vary
depth, activation, initialization, training protocol, and the geometry of the
link itself.

The task is **regional separation**: a network separates if its decision
region correctly classifies the two solid tori, not merely the training
sample. This distinction is not pedantic, and §2.4 reports what it cost us to
learn.

**Relationship to the impossibility theorem, stated precisely.** Ren & Lim's
Theorem 3.7 proves that a width-3 feedforward network with affine layers and
monotonic activations cannot render two disjoint closed *curves* with nonzero
linking number linearly separable. Our classes are **3-D solid tori, not 1-D
closed curves**. This is a genuine mismatch between the object the theorem
constrains and the object our networks train on, and we state it here rather
than in a footnote: **the experiments in this section do not test Theorem
3.7.** A solid torus is homotopy equivalent to its core circle, so the
obstruction is morally the same, but "morally the same" is not a proof, and we
claim no proof. What follows is an empirical statement about what training
finds on thickened samples. Its force does not depend on the theorem, and the
theorem's correctness does not depend on it.

## 2.2 The monotonic zero

Across every monotonic-activation width-3 configuration we have trained —
**5,580 distinct runs — not one produced a separating network.**

| stratum | runs | separations | what varies |
|---|---:|---:|---|
| width sweep | 240 | 0 | fixed tanh / ReLU / leaky-ReLU |
| threshold sweep | 1,280 | 0 | family A `a <= 1`, family B `alpha >= 0` |
| parametrization sweep | 1,440 | 0 | 12 link parametrizations |
| corrugation sweep | 1,620 | 0 | corrugated links, readings A and B |
| protocol sweep | 360 | 0 | the original authors' training protocol |
| search restarts | 400 | 0 | 200 tanh + 200 `sin` at `a = 0.95` |
| winding sweep | 240 | 0 | winding links, `q = 1..4` |
| **total** | **5,580** | **0** | |

The pool spans four distinct link geometries — the Hopf link, corrugated Hopf
links under two independent readings of the corrugation parameters, and
winding links at `|lk| = 2, 3, 4` — together with twelve parametrizations of
the Hopf link (varying major and tube radii, aspect, offset, obliquity and
rotation), two training protocols, and four independent search methods.

With `k = 0` events in `n = 5,580` trials, the one-sided exact 95% upper bound
on the per-run separation rate is

> `p <= 1 - 0.05^(1/5580) = 5.37e-4`, i.e. **below 0.0537%**.

For any single 80-run cell the same calculation gives only 3.68%, which is why
the pooled statement carries the weight: no individual cell excludes a rate of
a few percent, and the pool excludes a rate of one in two thousand.

**Three scope conditions travel with this number.** First, the pool is a
heterogeneous mixture of link geometries, depths and protocols, so the bound
applies to the *mixture*, not to any one condition. Second, every stratum but
the protocol sweep shares Adam/Kaiming ancestry, so this is a statement about
a family of standard training setups rather than about all possible
optimizers. Third, evaluation uses 2,000 held-out points per run; because
regional separation implies sample-level separation, a sample-level zero is
the **conservative** direction — a run counted as failing cannot secretly have
separated the region.

The denominator has moved during this project (3,330 to 4,610 to 4,970 to
5,570 to 5,540 to 5,580) as strata were added, de-duplicated and recounted;
every version has been a zero, and only the denominator changed. The current
figure is regenerated from raw artifacts by `src/zero_decomposition.py`.

## 2.3 Four search methods, and what CMA-ES showed

Gradient descent finding nothing is weak evidence if only gradient descent was
tried. Three non-gradient searches were run, each reported separately and
never pooled into the 5,580:

- **swap-descent** (24 runs x 2 configurations): no separation.
- **simulated annealing** (12 traces): every trace loses separation before
  reaching `a = 1`. First-failure midpoints span `a* in [1.075, 2.275]`
  (median 1.275); two traces are re-entrant, failing, recovering and failing
  again, with final losses at 1.375 and 1.125. No trace fails below the
  analytic threshold `a = 1`.
- **CMA-ES** (120 restarts): no separation, against a positive control that
  succeeded 12 times in 20 at depth 3 — so the optimizer was capable of
  finding separations when they existed.

The CMA-ES result is the most informative, because of *how* it failed.
Monotonic networks under CMA-ES reach **zero training error on the 2,000-point
sample** (3 restarts), and separate SGD runs come within **2 eval errors**.
The barrier sits at exactly zero eval errors and is approached arbitrarily
closely from above without ever being reached.

The reading is that **monotonic networks can shatter the sample and cannot
separate the region.** Given enough capacity to memorize 2,000 points, a
monotonic width-3 network does so; asked to get the region right, it fails at
every scale of effort we applied. An earlier version of this work argued for a
hard "floor" at 9–15 errors; that argument is **retracted** as an artifact of
searching only three fixed activations with SGD, and the correct statement is
the one above.

## 2.4 Dense verification, and a correction

Our first pass scored separation on the 2,000-point evaluation sample. That
was wrong, and it inflated our counts.

Re-checking every claimed width-3 separation at 100,000 points, **56 of 163
failed** — including 46 of 81 GELU separations. **Roughly a third of what we
had called separations were sample-level only**: networks whose decision
region cuts through one of the tori in a place the evaluation sample happened
not to visit.

We report this as a correction to our own method rather than as a methods
detail, because it changes what the surviving numbers mean. Every separation
count in this section is dense-verified at 100,000 points; the strongest
witness is verified at 2,000,000 points at 0 errors with margin 0.28. Dense
sampling remains sampling, and survivors include margins as small as 0.017, so
we describe these as **correctly-classifying regions** rather than as
solutions or minima, and we do not claim any of them is exactly separating.

This correction does not touch §2.2. The monotonic zero is a zero at the
sample level, and regional separation implies sample-level separation, so
denser verification can only confirm it.

## 2.5 The advantage is width-specific, and not monotonicity all the way up

If the obstruction is topological, it should vanish once the representation
space is large enough to unlink, and the activation should stop mattering.
Broadly this is what we find, but the boundary is not clean, and reporting it
cleanly would misstate it.

At **width 3** the effect is categorical: monotonic activations separate in
**0 of 300** fresh-seed runs at depth 3 and **0 of 120** at depth 6, while
GELU separates 2 of 100 and `sin(1.5)` 9 of 100, dense-verified.

Above width 3 the picture resolves into two distinct effects:

- The **monotonicity-specific** advantage — GELU against tanh, the comparison
  that isolates monotonicity while holding smoothness fixed — is Fisher
  non-significant **from width 6**.
- The advantage of GELU over the **ReLU family** persists to **width 8**. This
  is not a monotonicity effect: the same deficit appears in tanh-versus-ReLU
  comparisons, where both activations are monotonic (**98 vs 84 separations at
  width 8, p = 8e-4**). It is a ReLU-family optimization deficit.

Every pairwise comparison is non-significant from width 12, and all cells
reach 100/100 at widths 16–32.

**A registered prediction failed here.** P-W1, registered before the sweep,
predicted that the activation advantage would be gone by width 8. **It failed
at width 8 as written**, because the GELU-over-ReLU gap was still significant
there. Dead units do not rescue it: leaky-ReLU, which cannot die, still fails
7 of 100 at width 8. The prediction's companions fared better — P-W2's decay
shape and P-W3's ceiling were borne out — and a depth-6 slice shows the same
shape with no detected interaction (n = 40). We report P-W1 as failed rather
than reinterpreting it around the ReLU-family diagnosis we found afterwards.

The consequence for the paper's framing: **the monotonicity-specific
phenomenon is a narrow-width phenomenon.** Above width 6 what remains is an
optimization difference between activation families that has nothing to do
with folding, and by width 12 nothing remains at all.

## 2.6 A dose-response on weight scale

The width-3 GELU separation rate responds monotonically to initialization
scale, in both directions, with everything else held fixed:

| initialization | dense-verified rate | n |
|---|---:|---:|
| 0.3x standard | 3.0% | 200 |
| standard | 8.0% | 200 |
| found-separator pattern | 17.0% | 200 |

Across the extremes this is **p = 1.5e-6** (Fisher exact, one-sided,
directions registered in advance; two-sided 2.9e-6). The individual contrasts
are weaker: up-versus-standard `p = 0.0048` and down-versus-standard
`p = 0.023` one-sided, and under Bonferroni correction for three comparisons
the **down arm does not survive**: 3 x 0.023 = 0.069 on the one-sided test,
above 0.05. (Our ledger quotes 0.138 here, which is 3 x the *two-sided* 0.046
— a two-sided correction applied to a one-sided p; either way the arm fails
correction, and we state the one-sided figure because the direction was
registered in advance.) The dose-response headline
therefore rests on the scaled-up arm, with the down arm directionally
consistent at nominal significance only.

A registered expectation was exceeded in an informative way. We had predicted
(R2) that endpoint weight scale would not distinguish separators from
failures, and it does not — endpoint spectral products sit at percentiles
0.34–0.86, carrying no signal. Yet initialization scale moves the rate by a
factor of nearly six. The operative variable is therefore **the scale the
trajectory traverses early**, set by initialization, not the scale it ends at.
Scaled initialization also raises dense attrition (33% versus 11%), which is
why these rates are quoted dense-verified.

## 2.7 What this section establishes, and what it does not

Within a mixture of standard training setups on thickened Hopf and winding
links at width 3, monotonic activations produce correctly-classifying regions
at a rate below 0.0537%, while non-monotonic ones do so at single-digit
percentages that respond by a factor of six to initialization scale alone.
Monotonic networks are not merely slower here; they shatter the sample and
still fail the region, under four independent search methods.

What the section does not establish is *why*, and it cannot. Nothing in this
setting is computable in closed form: the basin structure is inaccessible, the
required parameter norm is unmeasurable, and the impossibility that would
explain the zero is a theorem about curves rather than about the solids we
train on. Every candidate explanation — capacity, conditioning, barrier
height, initialization scale — remains live, and the setting provides no
instrument sharp enough to exclude any of them.

That is the motivation for the remainder of the paper. §3 constructs the
smallest task carrying the same obstruction: one input, one hidden unit, four
parameters, where impossibility is provable in a line for the actual object
under study, the set of correctly-classifying parameters is computable
exactly, and each standard explanation can be tested by direct measurement
rather than inferred.
