# Independent re-reading of Ren and Lim, reconciled against the prior notes

Source: Junyu Ren and Lek-Heng Lim, "Low-dimensional topology of deep neural
networks", arXiv:2606.31856v1, submitted 30 June 2026. ICML 2026 poster 60602.

Method. The paper was re-read from the arXiv PDF independently, before the
prior notes (`icml_paper_notes.md`) were consulted in detail, so the two
readings are separable. Full text was extracted with `pypdf` and searched
directly rather than through a summarizer; the first summarizer pass returned a
theorem numbering that does not match the paper (it invented "Theorem 3.1
activation function bounds" and "Theorem 5.1") and was discarded. All quotations
below are from the extracted text.

The prior notes are **retained unchanged**. This file records the diff.

## Headline: the two readings agree

I found no substantive disagreement with `icml_paper_notes.md`. Every claim in
it that I could check against the text is corroborated, including several
specific numbers. The items below are verifications, then refinements.

## Priority item: `d + c`

**Verified, and this is the strongest result of the re-read.** A regex search
over the complete 93,779-character extracted text returns **zero** occurrences
of `d + c`, `d+c`, or any spacing variant, and zero occurrences of a bound of
the form `c < 5`. There is no quantity `c` in this paper.

What the paper actually states, verbatim:

> "Width `d+ 1` is therefore tight: width `d` is impossible by the lower bound
> below, while width `d+ 1` suffices." (Appendix D)

> "As a corollary, universal approximation with such activations requires width
> at least `d+ 1`." (Introduction)

The threshold is exact and additive-by-one, not additive-by-`c`. Theorem D.1
gives sufficiency at width `d+1` via Hanin–Sellke universal approximation;
Theorem E.1 gives the matching lower bound `w_min >= d + 1` for any continuous
coordinate-wise monotonic activation.

The prior notes' claim that widths 3–8 are "motivated by the reviewer feedback,
not documented by this paper" is correct. I confirm it independently.

The nearest empirical material is Appendix G.7, Table 7, which I can now quote
exactly (`R^7`, `S^3 ⊔ S^3`, `k=10` copies, depth 5, **15 ReLU seeds per
width**):

| Width | 7 | 8 | 10 | 14 | 21 | 28 | 35 | 49 |
|---|---|---|---|---|---|---|---|---|
| Multiplier of `d` | 1× | 1.1× | 1.4× | 2× | 3× | 4× | 5× | 7× |
| Mean (%) | 87.9 | 86.7 | 91.2 | 95.4 | 98.8 | 99.4 | 99.6 | 99.7 |
| Max (%) | 93.1 | 93.9 | 94.8 | 98.0 | 99.2 | 99.7 | 100.0 | 99.9 |

Note the mean *decreases* from width 7 to width 8 (87.9 → 86.7). The paper calls
these "small finite-seed nonmonotonicities". Accuracy saturates near 99–100%
around width `≈5d`, not at `d + small constant`. The related `R^5` experiment
reaches 98.5% mean at width 20, which is `4d`.

**Consequence for the width sweep.** If an additive `d+c` regime with `c < 5`
were the design premise, this paper does not supply it. Its own empirical curve
is multiplicative — the obstruction is relieved gradually over `1×` to `5×d`,
not at `d+1` to `d+5`. For our Hopf link `d = 3`, the theoretically obstructed
width is 3 and width 4 is already sufficient in principle; the paper's empirical
analogue would suggest looking out to roughly `5d = 15`, not 8. A sweep of
widths 3–8 is defensible as an empirical transition study but should be labeled
as externally motivated, and its upper end is not where this paper's own data
puts saturation.

## Verified specifics

Theorem numbering in the prior notes matches the paper exactly: 3.2 (linking
number), 3.5 (rank-deficient intersection), 3.6, 3.7 (link separation
impossibility), 4.1, 4.3, 4.4, 4.5, 4.6, 4.7 (general impossibility), 5.2
(ResNet), 5.3 (two-token attention), A.5 (autoencoder), C.1–C.8, D.1, E.1,
H.1–H.2. The summarizer's numbering was wrong; the prior notes' was right.

**Theorem 4.7**, verbatim hypotheses: `M^m, N^n ⊂ R^d` disjoint closed oriented
submanifolds, complementary dimension `m+n+1 = d`, `link(M,N) ≠ 0`, `F: R^d →
R^d` any width-`d` feedforward network with affine transformations and
coordinate-wise monotonic activations. Conclusion: `F(M)` and `F(N)` are not
linearly separable and perfect classification is impossible. The paper adds:

> "The result applies equally to smooth activations (sigmoid, tanh) and
> piecewise-linear activations (ReLU): the crucial property is monotonicity, not
> differentiability."

This confirms the prior notes' point 3 — monotonicity, not boundedness, is the
operative property — in the paper's own words.

**Corollary A.5** confirms the bottleneck reading: "The bottleneck dimension `d`
alone determines the topological constraint; the input dimension `n` is
irrelevant."

**Theorem 5.2** confirms the ResNet identity `|x| = x + 2 ReLU(-x)` and that the
escape "relies crucially on discrete, non-infinitesimal residuals", so Neural
ODE flows do not escape.

**Table 8** (Appendix G.8) confirms the prior notes: width-3 depth-5, **best
seed, 200 points per class**, no seed-level uncertainty. ReLU `d_min` collapses
to 0.00 by L1 and the fractional link values 0.50 and 0.18 are starred as
artifacts, with the paper stating "linking number is only defined for disjoint
curves". GELU and ReLU+skip reach link 0 with `d_min` growing to 1.41 and 3.73.

**Appendix H** confirms the point-cloud pipeline: PCA to `R^3`, ε-filtered k-NN
graphs per class, fundamental cycle basis, Gauss integral over `O(β_X · β_Y)`
basis pairs, ε "typically a percentile of nearest-neighbor distances".

## Refinements the re-read adds

1. **The width-expansion curve is multiplicative, and non-monotone at its
   start.** The prior notes said Table 7 "does not establish an additive
   constant below five", which is right but understates it. The 87.9 → 86.7 dip
   from width 7 to 8 means the paper's own data shows accuracy *falling* at
   `d+1` relative to `d`, attributed to finite-seed noise across 15 seeds. Any
   width sweep of ours should expect non-monotonicity at the narrow end and
   should not treat a dip as a finding.

2. **G.7 seed count is 15 per width, and is stated.** The prior notes did not
   record this. It is a useful calibration point: our census used 5 seeds, and
   the paper needed 15 to make claims at this granularity while still observing
   nonmonotonicity.

3. **The paper never uses the terms "solid torus", "tube", or "sampled".** It
   uses "thicken" (7 occurrences). The prior notes' point 2 — that our sampled
   solid tori are not the theorem's 1-D closed curves — stands, and the paper's
   own experiments likewise train on thickened samples while its theory and its
   Table 8 estimator concern the underlying curves. This is a genuine gap in
   both works, not something the paper resolves for us.

4. **Table 1 is explicitly scoped.** Its caption reads "Topological expressivity
   of width-`d` architectures, scored on the linking/folding transformations
   studied here. ✗ = cannot perform under our hypotheses; ✓ = can perform via
   the construction we give." This is stronger support for the prior notes'
   framing that the ordering is not a general expressivity ordering: the paper
   scopes it in the caption itself.

## Disagreements with the prior notes

None substantive. Two presentational notes:

- The prior notes describe Table 7 as observing "roughly 99–100% accuracy near
  width `5d`". Exact: 99.6% mean / 100.0% max at width 35 = `5d`. The
  characterization is accurate.
- The prior notes say the paper "does not report a seed-level uncertainty or a
  calibrated numerical noise floor for Table 8". Confirmed: Table 8 is a single
  best seed. G.7/Table 7 does report 15 seeds but gives only mean and max, no
  standard deviation, so seed-level dispersion is unavailable there too.

## Items still undeterminable from the paper

Unchanged from the prior notes, and re-confirmed:

- No `d + c` definition, no `c`, no bound on `c`.
- No reproducible discretization detail for the Table 8 Gauss integral, and no
  uncertainty estimate across seeds or discretizations.
- No method for assigning a classical linking number to two 1-D curves embedded
  in hidden width > 3 without a projection convention. The paper's own detector
  projects to `R^3` via PCA and it labels the CIFAR-10 result
  projection-dependent and correlational.
- No floating-point or quantization analysis anywhere in the paper. It does not
  speak to our precision census directly.
