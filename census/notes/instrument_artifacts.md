# Instrument limitations read as properties of the object

A short methodological note, written because this project has now made
this class of error **seven times** — twice in unrelated strands, a
third time in a distinct form within the same instrument, and a fifth in
the write-up rather than the measurement — and caught each only through
an independent route. It is transferable to anyone
measuring a solution set by scanning, or a topological invariant by
projection.

The five instances: a projection that returns the right answer for the
wrong reason (1), a scan whose resolution floor was read as a property
of the object (2a), the same scan's *parameter-dependent* bias
contaminating a comparison across that parameter (2b), a probe whose own
construction dominated the quantity it was measuring (3), a verifier
that checked the document against itself and so could not see that every
copy of a number was the same wrong number (4), and a hand-transcribed value
in a committed artifact, which inherits no check that operates on artifacts
(5).

## The pattern

> A measurement instrument has a resolution limit. The limit is a
> property of the instrument. When the instrument returns "nothing
> here", that can mean *nothing is there* or *what is there is below
> resolution* — and if the resolution limit varies with the same
> parameter you are sweeping, the instrument's response curve will look
> exactly like a real effect in that parameter.

The second clause is what makes this hard to catch by inspection: the
artifact is not noise, it is a smooth, monotone, plausible-looking
trend, aligned with the hypothesis under test.

## Instance 1: projection cancellation (T16, withdrawn 2026-08)

Projected linking number was measured on width > 3 representations by
projecting to R³ and applying the Gauss estimator. Result: uniform zero,
read as "the representation has unlinked the components".

The control that overturned it: distorted Hopf embeddings — known to be
linked — **also project to a clean integer 0 in 17–23% of random
projections**. The zero was partly a property of projection, not of the
representation. The original fidelity control used only rigid rotations,
which preserve the invariant and so could not detect the failure mode.

## Instance 2a: solution-set gridding — hard resolution floor (T30 clause, withdrawn 2026-08-25)

The 1D solution set was measured by scanning a 41-point grid per axis.
Result: empty box at a ≤ 1.10, and a required |w₂| ≈ 58 inferred from
"where solutions start appearing" — read as *solutions migrate to large
norm near the threshold*, which was then promoted to "the box-dependence
is the mechanism made visible".

What was actually true: the admissible b₂ interval has width
|w₂|·gap(a), and gap(a) → 0 as a → 1⁺. A grid node lands inside only
when |w₂| > step/gap(a). **The "required |w₂|" was the scan's own
resolution requirement**, and because gap(a) grows with a, the
instrument's detectability improved with a — producing a smooth,
monotone, entirely artifactual "growth of the solution set".

The correction: solutions exist inside [−5, 5]⁴ at a = 1.02 with
|w₂| = 1, and the true measure (computed analytically, no grid) is
positive and monotone at every a > 1, undercounted by the grid by
10–19× with the undercount factor itself varying with a.

## Instance 2b: the same grid, a *parameter-dependent bias* (2026-08-25)

Distinct from 2a and worth naming separately. After the box-emptiness
error was corrected, the surviving grid numbers were re-derived
analytically. The grid did not merely miss solutions below its
resolution — where it *did* register solutions it **undercounted the
measure by 10–19×, with the undercount factor itself varying in the
swept parameter** (≈19× at a = 1.35 falling to ≈10× at a = 3.0), because
detectability improves as the admissible interval widens.

This is a different failure mode from a floor:

- A **hard floor** produces zeros — conspicuous, and it invites the
  question "is that a real zero?" (which is how 2a was eventually
  caught).
- A **parameter-dependent bias** produces plausible non-zero numbers
  whose *ratios across the swept parameter are wrong*. It is invisible
  to sanity checks, and it survives exactly the comparisons one most
  wants to make.

The reason it matters here: the claim at stake was a **comparison
across a** ("solution measure grows smoothly while basin volume jumps").
A *constant* multiplicative bias would have cancelled in that
comparison and done no harm. This one did not cancel — it inflated the
apparent growth, because the instrument was getting better at seeing
exactly as the quantity grew. The claim survived re-derivation, but on
different numbers than the ones originally offered for it.

## Instance 3: the barrier collapse — the probe dominated the measurement (2026-08-27)

The most expensive of the four: it sent a full brief down a wrong path.

`gap_results.md` reported linear-interpolation barrier heights of
**1,065 → 125 → 10 → 1** as `a` crossed the findability onset. Three
orders of magnitude, tracking findability. That motivated an Arrhenius
account of findability — P(find) ~ exp(−ΔE/T) — with a full
experimental program built on it.

Measured properly in the 1D task, the effect does not exist:

- Every **minimum-energy-path** barrier is exactly **0.000**, at every
  `a`, for both endpoint types. Solutions are downhill-connected to
  typical initializations even where the solve rate is 0/200.
- The **linear** proxy *anti*-correlates with findability once the
  endpoint is held fixed: ΔE rises 0.020 → 0.103 as `a` goes 1.02 → 3.0
  while the rate rises 0% → 70%.

What the original 1,065 measured: those barriers ran to *constructed*
endpoints at **amplification 600**. A path from a unit-scale
initialization to a weight vector of norm ~600 passes through
intermediate weights at half that scale, and the loss there is enormous.
**The barrier height was mostly the endpoint's own weight scale** — a
property of the probe we built, not of the landscape we were probing.
`gap_results.md` had even flagged this in passing ("for constructed
endpoints the barrier height partly *is* the amplification"), and the
caveat was not carried forward into the claim that used it.

The distinguishing feature of this instance: unlike 2a/2b, the
instrument had no resolution limit and no bias in the usual sense. It
measured a real quantity correctly. The error was that the quantity was
dominated by a property of the object we inserted into the measurement,
so it varied with `a` for a reason unrelated to the hypothesis. **A
correct measurement of the wrong quantity.**

The control that would have caught it immediately, and now does: hold
the endpoint fixed while sweeping the parameter. Constructed
|w₂| = 1 endpoints at every `a` show the barrier is flat-to-rising,
killing the account in one table.

## Instance 4: the consistency check that could not fail (2026-09-11)

The first four instances are about instruments. This one is about the
*verification*, and it is the cheapest error on the list to make.

A headline quantity — the through-origin slope of onset exponent against
1/beta — was quoted in eleven documents, including the results draft and
a figure caption. A cross-sentence verification pass was run over the
draft specifically to catch disagreements: it tabulated every quantity
appearing more than once and confirmed that no quantity appeared with two
different values. It reported zero inconsistencies, and it was right.

The number was nonetheless wrong. The fit had been written with a
hardcoded four-tuple that merged two families sharing `beta = 1.5` into a
single point and kept one family's measured exponent while silently
discarding the other's. Every later document had copied the published
figure. Recomputing from `beta_law_points.csv` gives **1.0984**, not
1.1240 — and the headline agreement with the independently measured
`alpha` weakens from 0.6% to **1.7%**.

The failure mode is specific and worth naming:

> **Internal consistency is not correctness.** A document can be
> perfectly self-consistent and disagree with every artifact it
> describes. A consistency check compares text to text; its pass
> condition is that copies agree, and copies of a wrong number agree
> perfectly. The more thoroughly a number has propagated, the more
> confidently such a check will clear it.

What caught it was a different pass entirely — recomputing each derived
quantity **from the raw artifact at full precision** and comparing
against what the text claimed, with no reference to whether the text
agreed with itself. That pass also found a second, smaller case in the
same sweep: an exponent that reproduced only when refitted from two
decimal-place *display* values rather than from the stored data
(1.5123 vs the artifact's 1.5109).

Note what the wrong number survived: a figure regeneration script with
inline provenance, a ledger verifier recomputing eighteen headline
numbers, a bootstrap of the full pipeline, and an adversarial
interpretive audit. None of those touched this fit, because it was
written inline in a results document rather than in a checked script.
**A quantity computed inside prose is outside every verifier you own.**

## Instance 5: a hand-transcribed value in a committed artifact (2026-09-13)

Every verifier in this project checks **numbers against artifacts**. That
structure has an exact blind spot, and this instance is it.

`results/onset_curves.csv` held three wrong rate values at `B = 8,000`:
0.875 / 0.850 / 0.475 at `a` = 1.20 / 1.18 / 1.16, against 0.775 / 0.600 /
0.200 in the run's own log and in a from-source regeneration. **The log and
the regeneration agree exactly; the artifact was wrong.**

The cause is mechanical. `src/onset_resume.py` was written to resume a sweep
killed mid-run by a stray `pkill`. The cells that had already completed were
**not recomputed** — they were transcribed by hand from the log into a
literal list, and three were mistyped. The script then wrote those literals
into the artifact in the same format as measured values, where nothing
distinguishes them.

> **A hand-transcribed value inherits no check that operates on artifacts.**
> Every verifier we own — the ledger verifier, the figure audit, the
> quotation checker, the full-precision recomputation of §14 — compares a
> claim against an artifact. All of them passed, correctly, because the claim
> *did* match the artifact. The artifact did not match the code that was
> supposed to have produced it, and no check we had was pointed in that
> direction.

**What caught it**: regenerating the sweep from current source and diffing
against the committed artifact. This is the only detector in the project that
compares **artifact against code** rather than claim against artifact, and it
was run once, at the end, because the brief asked for it.

**`onset_resume.py` is the only script in the repository that hardcodes prior
measurements** (checked across all of `src/`). It is now annotated to say so.
That is the transferable rule: any script that embeds previously-measured
values rather than recomputing them is outside every artifact-based check, and
should be marked as such at the point where the literals appear.

**Two things this instance also settled, both favourable and neither
anticipated:**

1. **The bracketing rule absorbed the error.** The onset at `B = 8,000` is
   1.18 under both sets of values, because 0.475 and 0.200 are both below the
   50% threshold and 1.20 is above it either way. A rule that reads only
   *which side of a threshold* a cell falls on is insensitive to exactly this
   class of corruption — an unintended argument for the rule.
2. **Both derived quantities moved toward the rest of the evidence.** The
   collapse minimising `theta` went 0.7250 → **0.7500** against an
   onset-derived 0.7340, and the 25% threshold exponent went −0.5037 at 3 of 6
   bracketed → **−0.6581 at 4 of 6**, close to the 50% level's −0.7367. **The
   outlier that made threshold-independence look weak was the transcription
   error.** A caveat in the paper became supporting evidence.

## Instance 6: an orientation sign flip that would have produced a *clean, wrong* answer (2026-09-21)

The others produced numbers that were wrong. This one would have produced the
**opposite conclusion**, cleanly, with no tell.

The Phase 1 decomposition splits every failure into *placement* (`G <= 0`) or
*bias* (`b2` outside the admissible interval). For `w2 > 0` that interval is
`(-w2 m_O, -w2 M_I)`. For `w2 < 0` the roles of the windows mirror, and
dividing the defining inequality by a negative `w2` **flips the order of the
endpoints**: the interval is `(-w2 M_O, -w2 mi_I)`. The first implementation
kept the `w2 > 0` ordering, giving `lo > hi` — **an empty interval, always**.

Every one of the roughly half of runs with `w2 < 0` would have been classified
a **bias failure**. That is not noise: it is exactly the registered competing
hypothesis P1-c ("bias failures dominate, and the paper reframes around the
output bias"). The bug would have *manufactured* the alternative finding, in a
form indistinguishable from a real result.

**What caught it**: a validity condition registered *before* the data — that
the placement/bias classification must agree with `solves()` **run for run**,
because the decomposition is an algebraic identity rather than a model. It
fired on the first test case, a known solver: predicted not-solved, `solves()`
True. A single disagreement was enough, because an identity admits none.

**The generalisable rule**: *when an analysis is an identity, register the
identity as a test.* Most checks compare a measurement against an expectation
and must tolerate slack. An identity tolerates none, which makes it the
sharpest possible detector — and it costs one assertion.

**Audit of every other orientation-dependent site**, done immediately after,
since the same flip could sit elsewhere:

| site | verdict |
|---|---|
| `fold1d_theorem` margin and bound | **safe** — margin is read off *signed logits*, never reconstructed from a sign-branched interval; 0 of 66 violations, min slack 1.0394 |
| per-placement bound `\|w2\| >= 2m/G(w1,b1)` | **safe** — never previously computed; computed now with orientation handled: **0 violations of 66**, 22 of them `w2 < 0`, median slack 1.2488 |
| `exclusion_table` (Table 1) | **safe** — every row has `w2 > 0`, so the branch is never exercised |
| `r_variable.oriented_gap` | **safe** — takes `max` over both orientations rather than branching; agrees with the sign-branch on 66 of 66 solvers |
| `R = \|w2\| G*(a)/2` in the pooled data | **safe** — uses the *uniform* `G*`, orientation-free by the symmetry argument |

**Nothing changed.** The flip was confined to the new code, which is where the
identity test was pointed.

## Why all of these were caught only from outside the measurement

None of these errors was found by examining the measurement more
carefully.

- Instance 1 was caught by a **positive control on a known object**
  (does the instrument return the right answer for something whose
  answer we know?).
- Instance 2a was caught by **deriving theory that predicted the
  opposite** (the lower bound says |w₂| ≥ c/D(a) with c ≈ 1.2, which at
  a = 1.02 permits |w₂| ≈ 1 — flatly contradicting "requires 58"). The
  theorem was right and the measurement was wrong. Separately, the
  overclaim sweep had flagged the same sentence hours earlier purely on
  the word **"because"**, without any knowledge of the mathematics —
  causal language marking a place where a correlation had been promoted
  to a mechanism.

- Instance 2b was caught by **replacing the instrument entirely** —
  re-deriving the same quantity analytically, which was possible only
  because three of the four dimensions integrate in closed form. Note
  that 2b was found *while cleaning up after 2a*: the first error
  prompted the re-derivation that exposed the second. Neither the
  control nor the theorem would have caught it, since the biased
  numbers were finite, plausible, and in the right direction.

- Instance 3 was caught by **re-measuring the same quantity with the
  confound removed** — the same endpoint at every parameter value, plus
  a second estimator (MEP) that does not share the linear path's
  sensitivity to endpoint norm. Both said the same thing. Note the
  cost: it was *not* caught by the passing caveat in the document that
  first reported it, which shows a flagged-but-uncarried caveat is
  worth about as much as no caveat at all.

- Instance 6 was caught by **registering an identity as a test before the data
  existed**. No external route was needed because the relation is exact: one
  disagreement falsifies the implementation. Unlike the others, this detector
  was cheap and immediate, and the error it caught was the most dangerous —
  a clean answer of the opposite sign.

- Instance 5 was caught by **regenerating the artifact from code and diffing**
  — the only check that runs in that direction. Every artifact-based verifier
  passed, and each was right to: the claim matched the artifact. The defect
  was upstream of everything they inspect.

- Instance 4 was caught by **recomputation from source**, and could not
  have been caught by the check designed for it: the consistency pass
  compared copies of the number to each other, and they agreed. The
  detector had to bypass the text entirely and go to the CSV.

Seven independent detectors, none of them "look at the scan again":
a known-answer control, a theoretical prediction, a language audit, an
independent re-derivation, a confound-removing re-measurement, and a
recomputation from source.

## Checklist this yields

1. **Never infer a threshold from where an instrument starts
   registering.** Ask what the instrument's detection limit is as a
   function of the swept parameter, and plot that curve next to the
   result. If they have the same shape, you have measured the
   instrument.
2. **Check whether the instrument's bias is constant in the swept
   parameter, not merely whether it exists.** A known constant bias is
   harmless for comparisons across that parameter — it cancels. A bias
   that *varies* with the parameter contaminates precisely the
   comparison the sweep exists to make, and unlike a resolution floor
   it leaves no zeros or other visible tell. Ask: "if my instrument's
   sensitivity improves along this axis, does my trend still hold?" —
   and answer it by re-deriving at least two points with an
   independent method.
3. **Run a positive control on an object with a known answer**, chosen
   so the known answer is *not* the one the instrument is biased
   toward — rigid rotations could not have caught instance 1.
4. **Prefer analytic reduction to scanning** wherever the structure
   permits. In instance 2 three of four dimensions were analytically
   integrable, and doing so removed both the resolution floor (2a) and
   the parameter-dependent bias (2b) at once.
5. **Treat causal language in a results document as a flag**, not a
   style issue. "Empty *because* required |w₂| exceeds the box" asserted
   a mechanism where only a correlation between two grid-derived
   quantities existed.
6. **Ask what fraction of your measurement is the probe.** When a
   quantity is measured between a fixed reference and an object you
   constructed, vary the construction while holding the parameter
   fixed, and vary the parameter while holding the construction fixed.
   If the first sweep moves the number as much as the second, you are
   measuring your probe. This is cheap and would have saved a brief.
7. **A caveat recorded in passing is not a caveat.** The endpoint-scale
   confound was noted in `gap_results.md` when the barriers were first
   reported, then not carried into the claim built on them. Either a
   caveat gates the claim or it does not exist.
8. **Verify text against artifacts, never against other text.** A
   cross-reference pass that checks whether a number is quoted
   consistently has a pass condition — "the copies agree" — that a
   propagated error satisfies exactly. Every derived quantity in a
   document needs one line tracing it to the file and the computation
   that produced it, and the trace has to be *re-run*, not read.
9. **Any number computed inline in prose is unverified.** Fits,
   averages and ratios written directly into a results document sit
   outside every script your verifiers cover. Either the computation
   lives in a checked script that emits the number, or it gets its own
   recomputation pass.
10. **Regenerate artifacts from code, not just claims from artifacts.** Every
    verifier that compares a claim to a stored number shares one blind spot:
    it cannot see an artifact that the current code would not produce. Run the
    sweep again and diff. Once, before submission, is enough to find the
    class.
11. **Mark every hardcoded measurement at the point it appears.** A literal
    copied from a log looks identical to a computed value once written to a
    CSV. If a script must embed prior results, say so in a comment beside the
    literals, because nothing downstream can tell.
12. **When an analysis is an identity, register the identity as a test.** An
    identity admits no slack, so a single run-for-run disagreement falsifies
    the implementation. This is the cheapest detector in the list and it caught
    the only error that would have produced a clean result of the opposite
    sign.
13. **Audit every sign branch the moment one is found wrong.** A flipped
    orientation is a copy-paste class of error; if it appears once, enumerate
    every site that branches on the same sign and check each, rather than
    assuming the one found was the only one.
14. **Derive theory even in an empirical project.** The single highest-
   value output of the theorem work so far was not the theorem; it was
   the empirical error the theorem exposed.
