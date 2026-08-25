# The transition offset, resolved: reachability, not expressivity (Part A)

Predictions for A4 were registered in `offset_prediction.md`; A1's theory
reading is `threshold_theory.md`. Data: `offset_search.csv`,
`bisection.csv`, `offset_witness_weights.json`.

## The answer

**A separating width-3 network exists at `a = 1.02`, deep inside the
offset region, and four independent search families cannot find one.**
The offset between the analytic monotonicity threshold and the observed
onset of separation is a gap between possible and reachable — the same
SGD-versus-architecture gap the project turns on, now located and measured
at the threshold itself.

## A1 (recap): the theory rules out a representational margin

The possibility direction (Appendix F.2) requires only a strict local
extremum on an open interval, which both families have for every parameter
past threshold. No margin condition exists in the hypothesis, and float32
practicality does not create one near the transition (`threshold_theory.md`).

## A4: the exhibit and the searches

**Existence, by exhibition.** Following F.2's recipe exactly: a frozen
affine map places the fold coordinate across `f_{1.02}`'s local maximum
with shrink 0.06 — inside the near-quadratic neighbourhood, so the
realized fold is a near-symmetric inverted parabola of depth 5.3e−3 —
a frozen affine amplifies it 600×, and a trained monotone tanh
continuation finishes. The only non-monotonicity is `f_{1.02}` itself.
Result: **0 errors on 1,400,000 fresh points** (four samples, margins
0.53–0.91), and the linking trace shows lk −1 → 0 **exactly at the
`f_{1.02}` layer** (residual 3e−9), preserved at 0 by the continuation.
Deterministic (`src/offset_witness.py`, seed 16); pinned by tests.

Construction honesty: two design iterations failed first (recorded here) —
data placed inside the decreasing branch (a pure orientation flip, no
fold), then a large-shrink tent whose 50:1 shear SGD cannot train a
continuation on. Shape isolation on idealized folds showed the inverted
parabola is the trainable shape (best 1 error in 10 seeds) while the
sheared tent and upright parabola are not (best 54 / 71); the working
exhibit uses the small-shrink quadratic regime accordingly.

**Reachability, by search — all four fail at `a = 1.02`, as registered:**

| Search | n | dense-verified separations | best |
|---|---:|---:|---|
| mass restarts (depths 5, 8) | 400 | **0** | 3 errors (22 runs < 9) |
| fine-tune from `a = 3` separators | 24 | **0** | swap lands at 642–1000; best 10 after tuning |
| anneal 3.0 → 1.02 + 2,000 extra steps | 12 | **0** | 6 errors |
| CMA-ES (passing positive control) | 40 | **0** | 4 train-0 candidates; best eval 7 |

The searches get closer here than anywhere previously (best 3, against 26
for tanh and 2 for `sin(0.95)`) and still never touch 0. The registered
outcome "exhibit separates + searches fail" obtained.

## A2: the bisection

40 runs per value (4 depths × 10 seeds), dense verification built in:

| Family A `a` | 1.001–1.07 | 1.08 | 1.09 | 1.10 |
|---|---|---|---|---|
| eval-0 / dense | 0 / 0 (all) | 1 / 0 | 2 / **1** | 2 / **2** |

| Family B `α` | −0.005, −0.02, −0.05 | −0.11 | −0.13, −0.16, −0.20 | −0.25 |
|---|---|---|---|---|
| eval-0 / dense | 0 / 0 | 1 / 0 | 0 / 0 | 2 / **1** |

- Refined dense-verified intervals: **A: (1.08, 1.09]**; **B: (−0.20,
  −0.25]**. Per-point CIs at onset are wide (1/40 → [0.1%, 13%]); the
  interval statements are count-based at grid resolution.
- Which stopping condition occurred: **the rate fell below resolution**,
  not the interval becoming tight — below `a = 1.08` all cells are 0/40
  (ceiling 8.8%), while the exhibit realizes representability down to at
  least 1.02. Min errors at the near-threshold probes are 3–8: SGD gets
  within a handful of errors even at `a = 1.001` and never to zero.
- The sample-level-only events (A at 1.08, B at −0.11) both failed dense
  verification — near the onset, eval-0 without regional separation
  appears first, one more instance of the Part 2c pattern.

## A3: no matched unit survives

Candidate scalar summaries computed for both families, transitions
compared in each unit (conventions and truncation sensitivity in the
analysis): fold width, fold depth, multi-preimage measure, and
effective fold strength are all **disjoint by 1–2 orders of magnitude**
between the families — ruled out. Min-derivative, the last survivor,
touched at |min f′| = 0.10 on the original grid; the bisection breaks it:
**A transitions at |min f′| ∈ (0.08, 0.09], B at (0.20, 0.25]** —
disjoint. **No candidate unit makes the two families coincide.** Per the
registered reading, a universal negative here argues for reachability —
which A4 found independently. The two families simply have
different SGD-reachability onsets, and there is no single activation
scalar that predicts where separation becomes findable.

## A5: no optimization discontinuity

Across the bisection grid (`a` = 1.001 → 1.10): median final loss flat at
0.049–0.057, gradient norms fluctuating with no trend through the onset,
inactive-unit fraction exactly 0.0 everywhere, no at-chance runs.
Training health is smooth through the transition while the separation
rate turns on — the transition is not a trainability collapse. Combined
with A4, the picture is of rare basins becoming accessible, not of
gradients breaking.

## What the transition now means

The analytic threshold `a = 1` separates impossible from possible
(theory + exhibit). The observed onset near `a ≈ 1.09` separates
unfindable from findable for SGD at our protocol and budget, is
family-specific, tracks no activation scalar we tested, and sits at
smooth training health. The interval (1, 1.08] is the measured
possible-but-unreached zone — with the exhibit standing at 1.02 as
proof that the zone is genuinely possible, not merely asserted.

---

> **Status note (2026-08-24, audit follow-up).** The 1,400,000-point
> evaluation above originally ran from uncommitted session code
> (`AUDIT.md` finding 8). A committed generator now exists
> (`src/offset_witness.py::main`, sample seeds 931001–931004) and
> reproduces the result: **0 errors on 1,400,000 fresh points, minimum
> margins 0.54–0.85** (`offset_witness_dense.csv`). The claim rests on
> the committed run.
