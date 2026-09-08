# Cross-optimizer scope: three attempts, one conclusion

Three separate tests asked whether the onset relationship extends beyond Adam.
Reported as one sequence because separately they read as three caveats;
together they are a finding.

| # | test | what it probed | outcome |
|---|---|---|---|
| 1 | **Dynamical form** (T45) | does the optimizer enter only through alpha? | **failed quantitatively**: exponent ratio 0.447 vs alpha ratio 0.643 (30% gap) on the quantity where grid resolution partly cancels |
| 2 | **Geometric ratio** (T47) | does exponent ~ 1/beta hold under SGD? alpha cancels | **could not execute**: q0.667 never solves under SGD at any eps or budget (0/4 bracketed); q4 unresolved by an Adam-derived grid |
| 3 | **Same-family comparison** (T47) | q4 at fixed beta = 1.25, both optimizers | **failed decisively**: SGD flat (+0.0056, \|exp\| < 0.0185) vs Adam -0.8305 -- at least **45x** smaller in magnitude |

## The conclusion these support jointly

**The relationship is real under Adam, and every attempt to extend it beyond
Adam has either failed or proved unmeasurable.**

That is not three caveats. It is a positive finding: **whatever governs the
geometry-to-onset relationship is not optimizer-independent.** The four-family
law (T44) holds quantitatively under Adam -- through-origin slope 1.1240
[1.008, 1.240] against an independently measured alpha 1.1173 [0.999, 1.236],
four families, betas verified analytically -- and does not survive an
optimizer change in the one family where the change could be measured.

Test 3 is the decisive one and does not rest on a resolution argument:

- the law predicts the **smallest** beta gives the **steepest** exponent;
- q4 has the smallest beta of any family tested;
- under Adam it is duly the steepest (-0.8305);
- under SGD it is **flat**, bounded at \|exponent\| < 0.0185;
- the registered prediction -alpha_SGD/beta = -0.5750 is missed by **0.58**,
  far outside anything a grid explains;
- and SGD is **not globally static**: family A moves under SGD (-0.3255), so
  the flatness is beta-specific, not optimizer-wide.

## Consequence for the paper

**The central quantitative claim is single-optimizer, and the scope belongs in
the abstract rather than in limitations.** A reader who learns it on page one
reads the rest as a scoped result; one who finds it on page seven reads it as
something we hoped they would miss.

We know the scope because we tested it, not because a reviewer will.

## What remains open, precisely

- Whether the 1/beta scaling holds under SGD for **beta >= 1.5** families:
  untested, because SGD cannot reach q0.667 (beta = 2.5) and q2/q1 were not
  run under SGD.
- Why q4 is flat under SGD while family A moves: the two differ in beta (1.25
  vs 1.5) and in activation family, and the measurement cannot separate those.
- The **lower-tail hypothesis** (SGD's terminal weights differ in distribution
  *shape*, so its 50%-threshold onset is governed by a nearly
  budget-independent lower tail) remains the candidate explanation for the
  dynamical failure. Unregistered, one look, not pursued.
