# Part 4: no basin peak; SGD's |w2| is set by travel distance, not by basin size

Registered P-4a/4b/4c in `arrhenius_prediction.md`. Data:
`manifold_basin.csv` (a = 1.5, centred solutions at 11 values of |w2|;
basin by 60 perturb-retrain draws each; reach by 100 trained runs each).

## P-4a falsified: basin size does not peak near |w2| ~ 5

| \|w2\| | b1 half-width | b2 half-width | basin (eps=0.3) | basin (eps=1.0) | reach (standard init) |
|---|---|---|---|---|---|
| 1 | 0.050 | 0.039 | 0.77 | 0.18 | 0.70 |
| 3 | 0.050 | 0.116 | 0.60 | 0.22 | 0.65 |
| **5** | 0.050 | 0.194 | 0.70 | 0.28 | **0.21** |
| 8 | 0.050 | 0.310 | 0.72 | 0.42 | 0.00 |
| 20 | 0.050 | 0.776 | 0.73 | 0.35 | 0.00 |

Basin recovery is **flat in |w2|** at eps = 0.3 (0.60-0.77, no trend)
and **rises monotonically** at eps = 1.0 (0.18 -> 0.42). There is no
peak anywhere, and certainly not at 5. The registered prediction that
basin volume peaks where SGD lands is **false**: by basin size alone,
larger |w2| is weakly *better*.

The geometry explains why. The sheet's half-width in b1 and w1 is
**independent of |w2|** (0.050, 0.061 at every value — these are set by
the fold's position and traversal, which the construction holds fixed),
while the b2 half-width grows **linearly** in |w2| (0.039 -> 0.776,
exactly proportional, since the admissible b2 interval is |w2|*gap).
So the solution set gets *thicker* with |w2| in one direction and no
thinner in the others. Nothing about the solution set disfavours large
|w2|.

## What does explain |w2| ~ 5: reach

`reach_standard` — the fraction of ordinary trained runs whose final
|w2| gets at least that large — falls off a cliff exactly where SGD
stops landing: 0.70 at |w2| = 1, 0.65 at 3, **0.21 at 5, 0.01 at 6,
0.00 at 8**. SGD lands near 5 because 5 is about as far as it travels
in 2,000 steps from a unit-scale initialization, not because 5 is
special in the landscape.

This closes the loop with the step-size result
(`step_size_results.md`): raising the step to lr = 3e-2 moves the
solutions found to |w2| = 9.9-14.9, precisely the region that
`reach_standard` says is unreachable at lr = 1e-2. Travel distance,
not basin geometry, sets where solutions are found.

## P-4b: the registered two-effect crossing does not exist

The account predicted a peak where two constraints cross: b1-fragility
below, reachability above. **The lower constraint is absent** — b1
half-width is constant in |w2| (0.0498 at every value), so small-|w2|
solutions are not more fragile in b1. Only the upper constraint
(reachability) is real, and a single monotone constraint produces a
cliff, not a peak. Consistent with the data: the solve rate falls with
|w2| exactly like reach, with no low-|w2| suppression.

## P-4c falsified, with the sign reversed

Predicted: the peak moves outward with initialization scale, since the
upper constraint is about reachability. Measured (3x initialization
scale, a = 1.5, 100 runs): final |w2| median falls **3.82 -> 2.19**,
and the solve rate falls **38/100 -> 7/100**. Larger initialization
reaches *less* far in |w2| and solves *worse*.

`reach_scaled3x` in the table shows the same: at |w2| = 3 reach drops
0.65 -> 0.21 under 3x init, at |w2| = 5 it drops 0.21 -> 0.02.

This does **not** contradict the link-setting intervention (T27/T29),
which rescaled *per-layer spectral norms into a specific pattern*, not
the isotropic uniform-scale change tested here. The two manipulations
are different objects; the isotropic version is harmful in this task.

## Summary of Part 4

- Solution-set thickness is flat (b1, w1) or growing (b2) in |w2|.
- Basin recovery is flat-to-rising in |w2|. No peak.
- Reach collapses between |w2| = 5 and 8 at standard settings.
- Therefore the |w2| ~ 5 landing point is a **reachability** fact about
  the optimizer's travel distance, not a basin-geometry fact.
- Isotropic init scaling reduces both reach and solve rate.
