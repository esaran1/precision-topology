# Registration: which switch point do measured crossings track?

**Status by `a`.** For `a = 1.30` this is **POST HOC** — the measured crossing
(0.2332) and the coarse conditional switch (`R in (0.172, 0.215)`) were both in
hand before this was written. For **every other `a` it is prospective**: no
crossing statistic for `a = 1.40, 1.45, 1.50, 1.60` has been computed or read,
and `a = 1.35`'s crossing median was read once (0.2378) and is therefore also
post hoc. Prospective set: **`a` = 1.40, 1.45, 1.50, 1.60**.

Date: 2026-09-21.

---

## The four switch points

The conditional loss is the training loss minimised over `(w1, b1, b2)` with
`|w2|` **held fixed**. As `|w2|` rises the landscape can change character in
four distinguishable ways:

| symbol | definition |
|---|---|
| `R_glob` | smallest `\|w2\|` at which the **global** conditional minimizer has `G > 0` |
| `R_spin` | **upper spinodal**: where the `G <= 0` branch, followed by continuation from small `\|w2\|`, vanishes or its smallest Hessian eigenvalue reaches 0 |
| `R_fold` | **lower spinodal**: where the `G > 0` branch, followed downward from large `\|w2\|`, vanishes |
| `R_solve` | smallest `\|w2\|` at which the conditional minimizer on the fold branch is sign-correct on the dense grid |

If `R_fold < R_glob < R_spin` the landscape has a **hysteresis window**.

## Registered prediction

**Measured crossings track `R_spin`, not `R_glob`.**

Tolerance: **median measured crossing `R` within 15% of `R_spin(a)`**, and
**closer to `R_spin` than to `R_glob`** at a majority of the four prospective
`a` values.

**Grounds, stated before the prospective data**: a run initialised at small
`|w2|` sits on the `G <= 0` branch. Gradient descent is local, so it follows
that branch as `|w2|` grows and does **not** jump to a lower minimum elsewhere
merely because one appears. It leaves only when the branch it occupies ceases
to exist. That predicts `R_measured ≈ R_spin`, and since `R_spin >= R_glob`,
it predicts the **under-estimate already observed at `a = 1.30`** (measured
0.2332 against `R_glob = 0.2037`, +12.6%).

**Also registered**: measured solve thresholds track **`R_solve(a)`**, same 15%
tolerance. The existing measured value to beat is `R50 = 0.3705`.

**Also registered**: `R_spin`, `R_glob` and `R_solve` are each roughly constant
in `a` when expressed in `R` (CV <= 0.15 across the six values), and each
corresponding `|w2|*` falls roughly as `1/Ĝ(a)`.

## What would falsify the adiabatic picture entirely

- The conditional minimizer **never switches** over the `|w2|` range trained
  runs reach.
- Switches exist but sit **more than 50% away** from measured crossings at a
  majority of prospective `a`.
- **No hysteresis window** (`R_spin ≈ R_glob` to within the grid step) *and*
  measured crossings sit well above both.

Any of these is recorded in **Read first** and the adiabatic account is dropped
rather than repaired.

## Data convention

The training loss is over 400 points resampled per seed. The landscape is
therefore computed **on a dense uniform sample of the windows as the population
loss**, and separately on **five seed datasets**, so the spread of each switch
point across seed datasets is reported rather than assumed negligible.

All Block B work is **float64**.
