# Block B final: switch points across all six activation values

Registered in `blockB_prediction.md`. Procedure frozen (sha256 `9f1b10741d8bf48c`).
float64, population loss on 800 dense uniform points.

---

## Switch points, and the measured crossings

| `a` | `R_fold` | `R_glob` | `R_spin` | `R_solve` | measured cross | meas/spin | meas/glob |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.30 | 0.2037 | 0.2145 | 0.2145 | 0.3067 | **0.2330** | 1.086 | 1.086 |
| 1.35 | 0.2135 | 0.2162 | 0.2269 | 0.3070 | **0.2375** | 1.047 | 1.098 |
| 1.40 | 0.2090 | 0.2186 | 0.2251 | 0.3054 | **0.2418** | 1.074 | 1.106 |
| 1.45 | 0.2080 | 0.2194 | 0.2269 | 0.3064 | **0.2456** | 1.082 | 1.119 |
| 1.50 | 0.2181 | 0.2225 | 0.2399 | 0.3053 | **0.2489** | 1.038 | 1.119 |
| 1.60 | 0.2229 | 0.2285 | 0.2508 | 0.3065 | **0.2563** | 1.022 | 1.122 |

## Registered prediction: passes, but on THREE prospective `a`, not four —
## and the test is not discriminating

**Correction (timestamp audit).** The registration named `a` = 1.40, 1.45,
1.50, 1.60 as prospective. **`a = 1.40` is not.** Its crossing median was read
in the Block A interim at **18:41:03**, while the seeding fix that *creates*
the `R_fold`/`R_spin` separation at `a >= 1.35`, and the frozen procedure
sha `9f1b10741d8bf48c`, both landed at **18:56:15** — fifteen minutes later.
Before that fix no window existed anywhere, so the `R_spin` values at
`a >= 1.35` come from a procedure that postdates my having seen crossings at
1.30, 1.35 and 1.40.

| status | `a` |
|---|---|
| **genuinely prospective** | **1.45, 1.50, 1.60** (3 values) |
| post hoc | 1.30, 1.35, 1.40 |

On the corrected prospective set the result is **3 of 3** closer to `R_spin`
than `R_glob`, all within 15%. Overall: within 15% of `R_spin` at 6 of 6;
closer to `R_spin` at 5 of 6 (`a = 1.30` ties, the two coincide there).

**And the test does not discriminate.** `R_spin >= R_glob` by construction, and
**every** measured crossing exceeds `R_spin`. So "closer to `R_spin`" follows
automatically from the ordering and would hold equally under a pure-lag account
in which the true switch is `R_glob` and the run overshoots it. The registered
outcome is recorded as it stands, but it is **not evidence for the spinodal
over the global switch**. The discriminating version is registered in
`blockF_lag_prediction.md` (§ D-1) and tests `R_spin + lag` against
`R_glob + lag` with the lag predicted from growth rate alone.

## Drift: the predicted switch moves the way the measurement moves

| | 1.30 → 1.60 | correlation with measured |
|---|---:|---:|
| measured crossing | +10.0% | — |
| `R_glob` | +6.5% | **+0.980** |
| `R_spin` | +16.9% | **+0.947** |

Both drift **upward, same sign**. The measured drift sits between the two
predicted drifts.

## CV across `a`

| quantity | CV |
|---|---:|
| `R_solve` | **0.0020** |
| `R_glob` | 0.0208 |
| measured crossing | 0.0311 |
| `R_spin` | 0.0504 |

`R_solve` is the most constant quantity in the study — a training-free number
that varies by 0.2% across a 2.6× range of `Ĝ`.

## Hysteresis: absent at `a = 1.30`, narrow but present above

At `a = 1.30`, `R_fold = 0.2037` against `R_glob = R_spin = 0.2145` — **one
0.25 grid step**, i.e. resolution, not a window. That triggered the registered
falsifier and the bifurcation framing was dropped.

At `a >= 1.35` the fixed downward seeding does find windows, `R_spin - R_fold`
between 0.013 and 0.028. These are **a few grid steps wide** and are reported
as such; they are not treated as a primary result, and Block E's primary test
remains **reversibility** rather than hysteresis.

## The degenerate solution: never the minimiser

The `w1 = 0` constant predictor outputs a constant, so its conditional loss is
the class-prior entropy at **every** `|w2|`. The population data are balanced
(`P(y=1) = 0.5000`), so that loss is `log 2 = 0.693147` throughout.

| `|w2|` | 2.0 | 4.0 | 6.0 | 8.0 | 10.0 |
|---|---:|---:|---:|---:|---:|
| minimiser loss | 0.4614 | 0.3811 | 0.3180 | 0.2659 | 0.2226 |
| degenerate loss | 0.6931 | 0.6931 | 0.6931 | 0.6931 | 0.6931 |

**The degenerate solution is never lower**, at any `|w2|` in the scan. So
excluding it needs no special justification for the *definition* of the switch —
it was only ever a hazard for the cheap screening pass, where a bad start can
collapse `w1` and reach `log 2` faster than it can find a placement.

## The solve threshold, compared correctly

`R50 = 0.3705` is **terminal** `R` pooled over runs and includes post-solve
growth, so it is the wrong comparison. Using **`R` at the solve step**, per run:

| `a` | median `R` at solve step | n | CV |
|---|---:|---:|---:|
| 1.30 | **0.3568** | 129 | 0.077 |
| 1.40 | 0.3710 | 142 | 0.077 |
| 1.50 | 0.3838 | 159 | 0.075 |
| 1.60 | 0.3960 | 166 | 0.073 |

Against `R_solve(1.30) = 0.3067`: **+16.4%**, versus **+20.8%** if `R50` is
used. Either way the **solve gap is about twice the placement gap** (+8.7%),
which the lag account must explain — a run has further to travel past the
solve switch than past the placement switch, so it accumulates more lag.

## Registered prediction that FAILED

`R·ρ` was registered as the lower-CV quantity at the solve step, on the grounds
that the bias interval has width `|w2| G = 2Rρ`. Measured over 882 solving runs:

| quantity | median | CV |
|---|---:|---:|
| **R** | 0.3747 | **0.0830** |
| `R·ρ` | 0.2440 | 0.1933 |

**`R` governs the second stage, not `R·ρ`.** The registered reasoning was wrong
because `ρ` is already near 1 by the time runs solve (Phase 1 median 0.967), so
`R·ρ` mostly re-imports `ρ`'s variance without adding information.
