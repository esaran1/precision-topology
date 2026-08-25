# The possible-versus-findable gap: mechanism (Parts 1–4)

Predictions registered in `basin_prediction.md` and
`amplification_prediction.md`; pipeline control in `gap_part4_control.md`.
Data: `basin_recovery.csv`, `basin_distances.csv`, `basin_profiles.csv`,
`required_amplification.csv`, `amplification_window.csv`,
`scaled_init_intervention.csv`, `scaled_init_v2.csv`,
`achievable_amplification.csv`, `part3_basin.csv`.

## The one-paragraph answer

The gap is a **weight-scale barrier, shown by two-sided manipulation**: the
solutions at `a = 1.02` require ~1,000× fold amplification; standard
training reaches spectral-norm products of ~50 (median) to ~150 (max)
there and never travels to the required scale; and the required scale
crosses *into* the trained distribution exactly at the findability onset.
The decisive test: **initializing layers 1–2 at the construction's scale
pattern — random directions, standard Adam otherwise — produces 17/40
dense-verified separations at `a = 1.02`, where standard initialization
produced 0/400.** The gap is now something we manipulate, not something
we describe. Basin radius, the other candidate mechanism, failed its
registered predictions and is not the explanation.

## Preliminary: the exhibit upgraded

Pure `sin_family(1.02)` **depth-4 all-f_a networks** separate densely
(4/40 continuation seeds) — the standard architecture itself represents
the map at 1.02; the earlier mixed-activation caveat is gone.

## 1a–1b: the basin account fails its registered predictions

Recovery rate (of 20) after relative perturbation ε, recovery = Adam
2,000 steps then eval-0 + 100k dense:

| solution | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| 1.02 constructed | 2 | 0 | 0 | 0 | 0 | 0 |
| 1.05 constructed | 5 | 2 | 1 | 0 | 0 | 0 |
| 1.10 constructed | 9 | 5 | 2 | 0 | 0 | 0 |
| 1.09 found | 20 | 20 | 20 | 20 | 7 | 0 |
| 1.10 found | 20 | 20 | 20 | 19 | 12 | 1 |
| 1.25 / 2.0 / 3.0 found | 20 | 19–20 | 17–20 | 19–20 | 10–17 | 1–2 |

- **P-calibration failed**: at the same `a = 1.10`, constructed basins
  are ~2 orders of magnitude narrower than found basins. Constructed
  solutions are atypical objects; basin claims restrict to found
  solutions, per the registered fallback.
- **P-basin failed**: among found solutions, ε₅₀ is flat (≈ 0.1–0.3)
  from 1.09 to 3.0 while SGD success rates vary 2.5% → 29%. Basin radius
  does not grow with `a` and does not explain findability.
- P-onset is untestable as stated (no found solutions below the onset;
  constructed ones atypical). 1b's metric caveat materialized exactly:
  normalized init–solution distance is ≈ 1.000 ± 0.0002 for constructed
  solutions (their norm is all amplification) versus 1.04–1.26 for found
  ones — raw distance is dominated by the solution's own scale, as
  registered.

## 1c: barriers fall three orders of magnitude across the onset — borne out

Median linear-interpolation barrier (Goodfellow-style loss profiles,
init → solution, 20 inits each): **1,065 (1.02c) → 125 (1.05c) → 10.2
(1.09 found) → 9.1 (1.10 found) → 3.3 (1.25) → 1.7 (2.0) → 1.0 (3.0,
min 0: some paths barrier-free)**. Constructed solutions at 1.10 sit
behind 4–6× higher walls than found ones at the same `a`. P-barrier is
borne out — with the caveat that for constructed endpoints the barrier
height partly *is* the amplification (mid-path weights at half-scale
produce huge losses), i.e. 1c independently points at weight scale.

## 2a: required amplification — half a result, honestly

`A_req` by construction success (10 continuation seeds), Family A:
**~1,020 at 1.02 → ~253 at 1.05 → ~126 at 1.09**, and a working window
extending to 12,800 with a saturation ceiling by 20,000 (too much
amplification kills the continuation — discovered when the first
bisection, which assumed monotonicity, returned an artifact; replaced
with a window scan).

**Prediction 2a-match failed**: `A_req` is **undefined for Family B** —
the construction fails at every amplification on the grid (its fold is a
1/|α| sheared tent, and the shear, not the scale, blocks the
continuation; meanwhile plain SGD *does* find B solutions at −0.25, so
the recipe, not the family, is the limit). Required amplification is a
real, monotone quantity within Family A and is not the cross-family
matched unit. The cross-family onset difference remains unexplained at
the level of a single scalar.

## 2b: the amplification account — borne out by intervention

**2b-tail, borne out with the crossing at the onset.** Spectral-norm
products of standard trained depth-5 networks versus `A_req`:

| a | trained median | trained max | A_req |
|---|---:|---:|---:|
| 1.02 | 53 | 153 | **~1,020** — 7× beyond the max |
| 1.05 | 25 | 333 | ~253 — at the tail's edge |
| 1.09 | 44 | 281 | ~126 — inside the distribution |

**2b-intervention, the decisive result.** v1 (all three units at the
fold point — an implementation error that destroyed the pass-through
coordinates) failed 0/40 and is recorded. v2 implements the
construction's actual pattern — one fold-scale unit, two pass units,
amplification-scale second layer, random directions, everything trained
by standard Adam: **17/40 dense-verified separations at `a = 1.02`**
against 0/400 under standard initialization. The registered strongest
outcome obtained: the gap moves when the initialization scale moves.

## Part 3: the original setting

- **GELU construction failed** at every point of a 2-shrink × 5-amp grid
  (0/10 each), reported as a failed exhibit. GELU's existence at width 3
  does not need it: 35 dense-verified *found* GELU solutions already
  exist. In the standard setting the gap is a **rate gap** (exists;
  found in ~5% of dense-verified runs), not a zero gap.
- Basin geometry of found GELU (ε₅₀ ≈ 0.3, barrier 0.24) and found
  pwl(−0.25) (ε₅₀ ≈ 0.1, barrier 0.12) is indistinguishable from found
  sin solutions — and **does not rank-order SGD success rates** (GELU:
  low barrier, low rate; sin(3.0): higher barrier, high rate). The
  registered strongest-available Part 3 result — predicting GELU's
  empirical rate from basin geometry — **is not available**, stated
  plainly. Local geometry around solutions is roughly universal; what
  varies is whether SGD's trajectory ever enters, which 2b ties to
  weight-scale demand.

## Part 4 (recap): the pipeline is a recipe, not a solution finder

Monotonic activations fail at step 1 — no strict local extremum, f′ ≥ 0
everywhere, provably — and the failure boundary is exactly the analytic
threshold (`sin(0.98)` fails, `sin(1.001)` passes). Two detector bugs
were caught by the control itself (`gap_part4_control.md`).

## Scorecard of registered predictions

| Prediction | Outcome |
|---|---|
| P-basin (radius grows with a) | **failed** (flat among found) |
| P-onset (radius ≈ init distance at onset) | untestable as stated |
| P-calibration (constructed ≈ found basins) | **failed** (2 orders apart) |
| P-barrier (barriers fall to onset) | borne out (1,065 → 10 → 1) |
| 2a-match (A_req aligns families) | **failed** (undefined for B) |
| 2b-tail (A_req beyond trained scale at 1.02, crossing at onset) | borne out |
| **2b-intervention (scaled init unlocks 1.02)** | **borne out: 17/40 vs 0/400** |
| Part 3 rate-from-geometry | not available; reported as such |

Three registered failures, and they sharpen the result: the mechanism is
not basin size, not any activation scalar, not cross-family universal —
it is the required weight scale, family-specific, and causally
manipulable.
