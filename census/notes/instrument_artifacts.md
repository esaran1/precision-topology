# Instrument limitations read as properties of the object

A short methodological note, written because this project has now made
this class of error **three times** — twice in unrelated strands, and a
third time in a distinct form within the same instrument — and caught
each only through an independent route. It is transferable to anyone
measuring a solution set by scanning, or a topological invariant by
projection.

The three instances: a projection that returns the right answer for the
wrong reason (1), a scan whose resolution floor was read as a property
of the object (2a), and the same scan's *parameter-dependent* bias
contaminating a comparison across that parameter (2b).

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

Four independent detectors, none of them "look at the scan again":
a known-answer control, a theoretical prediction, a language audit, and
an independent re-derivation.

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
6. **Derive theory even in an empirical project.** The single highest-
   value output of the theorem work so far was not the theorem; it was
   the empirical error the theorem exposed.
