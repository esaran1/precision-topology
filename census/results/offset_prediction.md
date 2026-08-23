# Registered predictions: existence versus reachability at a = 1.02 (A4)

Written before the search batch runs. Two construction-design pilots have
already run (both 0/20; recorded in `offset_results.md` when written) —
they are positive-control engineering for the existence exhibit, not
searches. Everything below is registered before execution.

## Setting

`a = 1.02` sits inside the offset region: monotonicity threshold at 1.0,
first observed separation at 1.10, zero observed at 1.05 (0/80). A1
established the theory asserts existence of a separating width-3 network
for every `a > 1` with no margin condition, and that float32 practicality
does not block it at 1.02 (window 0.4, amplification ~190×).

## The exhibit (existence)

An F.2-style construction: frozen affine placing the fold coordinate
across `f_{1.02}`'s local maximum, frozen amplification, trained monotone
tanh continuation (the Part 2a decomposition standard: the only
non-monotonicity is the `f_{1.02}` layer). Design iterations on the frozen
part are expected and legitimate — this is a constructive exhibit, not a
search; its bar is one network that separates densely (100k+ points).

**Prediction E: an exhibit exists and will be found.** Grounds: A1's
amplification arithmetic and the Part 2a precedent. Failure to construct
one after honest effort would be reported as a failed exhibit, which
would leave existence resting on the theory citation alone — weaker, and
stated as such.

## The searches (reachability), Part 2b machinery at a = 1.02

1. **Mass restarts**: 200 seeds, plain `sin_family(1.02)` width-3
   networks, depth 8 (best monotonic-side depth) and depth 5.
2. **Fine-tune from a = 3.0 separating solutions**: reconstruct ≥10
   separating `a = 3` runs, set `a = 1.02`, fine-tune 2,000 steps at two
   learning rates.
3. **Targeted annealing**: anneal `a` from 3.0 to exactly 1.02 in 0.05
   steps with 200 adaptation steps each (the protocol that located
   `a* ∈ [1.1, 1.8]`), then 2,000 further steps at 1.02.
4. **CMA-ES** at `a = 1.02`, 20 restarts, the passing positive-control
   machinery (smooth cross-entropy objective, counts for verification).

**Prediction R: all four searches fail to produce a dense-verified
separation at a = 1.02.** Grounds: the sweep's 0/80 at 1.05 and the
annealing traces all dying above 1.1. Sample-level-only separations
(train-0 or eval-0 that fail densely) may appear, as they did in Part 2b.

## Outcomes and readings, fixed in advance

- **Exhibit separates + searches fail** (predicted): the offset is
  reachability, not expressivity. The transition marks where the
  optimization problem becomes tractable, and the architecture/SGD gap
  the project turns on is now located at the threshold itself.
- **Any search finds a dense-verified separation**: the offset is *soft*
  reachability (rare but reachable), the transition location becomes a
  rate statement, and the 0/80 at 1.05 becomes a bound rather than a
  wall. Report prominently; this changes A2's interpretation.
- **Exhibit fails + searches fail**: theory says possible, exhibition and
  four searches say no — reported as an open tension with the honest
  caveat that a failed exhibit may reflect our construction skill.
- **Exhibit fails + a search succeeds**: existence settled by the search
  itself; the exhibit failure is then only about our construction.
