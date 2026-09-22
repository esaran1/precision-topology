# Correction to registered prediction G-3, recorded before any Block G crossing was measured

`blockG_prediction.md` (committed `da8621f`) states:

> `G3_far_outer` is predicted **>= base** (a superset of the outer window can
> only help the supremum, since the inner window is unchanged — this one is a
> near-certainty and is stated so as to be checkable, not as a discriminating
> test).

**The direction is backwards, and provably so.** `Ghat` is a supremum over
placements of a **min-over-`O` minus max-over-`I`** quantity (in either
orientation). Enlarging `O` can only **lower** `min_O f` and **raise** `max_O f`,
so for **every** placement the gap on the larger window is `<=` the gap on the
smaller one, and taking the supremum preserves the inequality:

    O' superset of O,  I' = I    =>    Ghat(O', I) <= Ghat(O, I)

A superset of the outer window can only **hurt**. The parenthetical reasoning in
the registration ("can only help the supremum") confused enlarging the *search
space over placements*, which would help, with enlarging the *region the network
must classify correctly*, which cannot.

## Measured, consistent with the corrected direction

| window | `Ghat` at `a = 1.30` | `Ghat` at `a = 1.50` |
|---|---:|---:|
| `G2_narrow_gap` | 0.021043 | 0.042901 |
| `G3_far_outer` | **0.071530** | **0.145175** |
| `base` | 0.085846 | 0.174761 |
| `G4_shifted` | 0.143781 | 0.283729 |
| `G1_wide_gap` | 0.166415 | 0.337440 |

`G3 < base` at both `a`, as the corrected inequality requires.

**The rest of G-3 is confirmed as registered**: `G2_narrow_gap < base <
G1_wide_gap` at both `a`, which was the substantive part of the prediction and
was stated with the right reasoning (a wider class margin admits a larger
achievable gap).

**Verification of the inequality itself**, placement by placement, 2,000 random
`(w1, b1)`: 125 cases show `G3` exceeding `base`, all by at most **1.5e-7**. That
is discretisation — both windows are sampled at 4,000 outer points, so `G3`'s
grid spacing is 2.25x coarser in absolute terms. The inequality holds; an initial
check used a 1e-12 tolerance and reported 8 of 200 spurious violations.

## Status

Logged as a **registered prediction failed on a sub-claim, through an error of
reasoning rather than a surprise in the data**. `G4_shifted` remains exploratory
as registered. G-1, G-2 and G-4 are untouched by this and are still open — no
crossing has been measured on any window at the time of writing.
