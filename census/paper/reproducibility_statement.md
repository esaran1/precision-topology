# Reproducibility Statement

All code, committed artifacts, the claims ledger and the figure script are in
the supplementary material. A single command regenerates every figure and
re-verifies the headline numbers against their artifacts:

    PYTHONPATH=. python3 paper/make_figures.py

**Verification is in two directions, not one.** Most of this project's checks
compare a **claim** against an **artifact**: a ledger verifier recomputes 18
headline numbers, the figure script recomputes the three multi-valued
quantities at render time and refuses to plot on a mismatch, and a quotation
checker re-tests every quoted passage from the primary sources against the
retrieved text. Six analyses were additionally reimplemented from their
definitions in a separate module importing only numpy, agreeing with the
production code in five of six (the sixth, an onset location, differed by one
grid step and was diagnosed as sampling rather than implementation).

Those checks share a blind spot: **they cannot see an artifact that the current
code would not produce.** Before submission we therefore regenerated the sweeps
from source at the submitted commit and diffed the output against the committed
artifacts. Priorities and outcomes:

| sweep | result |
|---|---|
| onset cells feeding the six bracketed budgets | **1 discrepancy found** (below) |
| pooled R dataset (subset: 40 of 200 seeds per cell) | bit-exact, max abs difference 4.4e-16 |
| monotonic zero, all seven strata | exact: 5,580 runs, 0 separations |
| exclusion-table geometry (`G*`, `D`, `kappa`) | exact at all eight values of `a` |

**The discrepancy, and what it cost.** Three rate cells at `B = 8,000` in
`onset_curves.csv` disagreed with regeneration. The cause was mechanical: a
sweep killed mid-run was resumed by a script that **hand-transcribed** the
already-completed cells from the run's log rather than recomputing them, and
three values were mistyped. The original log and the regeneration agree exactly;
the artifact was wrong. Every artifact-based verifier had passed, correctly,
because the claim did match the artifact.

**No reported number moved.** The onset at `B = 8,000` is 1.18 under both sets
of values, because the affected cells fall on the same side of the 50%
threshold either way — the bracketing rule is insensitive to this class of
corruption, which we did not anticipate when adopting it. Two derived
quantities changed and both moved **toward** the rest of the evidence: the
collapse-minimising exponent from 0.7250 to 0.7500 against an onset-derived
0.7340, and the 25% crossing exponent from −0.5037 (4 of 6 bracketed after
correction, 3 before) to −0.6581. A caveat in an earlier draft — that
threshold-independence held only at 50% and 75% — was itself an artifact of the
transcription error, and the corrected data supports the stronger claim now
made in §6.

The script responsible is the only one in the repository that embeds
previously-measured values rather than recomputing them; it is now annotated at
the point where the literals appear. The transferable form: **a hand-transcribed
value inherits no check that operates on artifacts.**

**Scope of what regeneration covers.** The 1D sweeps, the monotonic-zero strata
and the geometric quantities are fully regenerable on a laptop CPU. The pooled R
dataset was regenerated on a documented subset (40 of 200 seeds per cell, every
cell) rather than in full, and the CIFAR runs were not regenerated; both are
stated here rather than omitted. Random seeds are fixed and recorded, so the
subset is a genuine sample of the same computation rather than a rerun under
different conditions.

**Known non-global numerical step.** `G*(a)` is obtained by numerical
maximisation over a neighbourhood of the analytic optimum, not the whole
parameter plane. Refinement finds gaps 0.3–0.7% larger and never smaller, so the
value used is a lower bound on the true `G*`; since the bound under test is
`2m/G*`, this makes the empirical verification conservative rather than
optimistic. Re-checking all 66 solvers with `G*` inflated by 0.7% leaves 0
violations.
