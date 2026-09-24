# Design (for review, nothing run): what sets the residual — inherited displacement or optimiser memory (Block 4b)

**Status: design only.** Nothing in this design has been computed. The post hoc analysis in Block 4a
(`residual_posthoc.py`, from existing states) is exploratory and runs separately. **This file becomes a
registration only after review**, with its producer, its tests and its commit hash, and before any
intervention is run.

## Question

The deconfounded lag test (`lag_test2_prediction.md`, scored) showed where the residual's sensitivity to w₂'s
growth rate comes from:
- slowing w₂ from the plateau exit removes about three-quarters of the residual;
- slowing it only from 0.7·R_own removes 11–14%.

So the residual depends on the middle phase of training. There are two candidate mechanisms:

- **Inherited displacement (ID)**: during the middle phase the hidden parameters (w₁, b₁) fall behind the
  occupied branch's conditional minimiser along a slow direction, and that displacement persists to the
  crossing.
- **Optimiser memory (OM)**: Adam's moment estimates, shaped during the middle phase, bias the later approach.

## Runs and switch point (identical to the deconfounded lag test)

- **Runs**: n = 6,400 training sets; a ∈ {1.30, 1.50}; seeds 300,000–300,049. The included runs are the lag
  test's 48 per a (seeds 300,006 and 300,028 do not cross within 32,000 steps).
- **Switch point**: each run's primary-rule state, the first step with R ≥ max(0.7·R_own, the commitment
  level) after the last plateau exit. These are the saved full states `lag_test2_states/primary_a*_s*.pt`:
  parameters, the Adam state and the step count. The lag test regenerated them deterministically.
- **Continuation**: φ = 1 (standard protocol) from the intervened state to the crossing, budget 32,000 steps in
  total.

## Four arms, from the same saved state

| arm | intervention at the switch point |
|---|---|
| control | none |
| teleport | move (w₁, b₁) onto the occupied branch's conditional minimiser at the current scale |w₂|; set b₂ to its profiled optimum there; **keep** the Adam state |
| reset | reset Adam's moments only (exp_avg = exp_avg_sq = 0, step = 0); parameters unchanged |
| teleport + reset | both |

- **Occupied branch**: canonical orientation w₂ > 0; the branch is sign(w₁) in that orientation, as in the
  lag test's diagnostic.
- **The branch's conditional minimiser**: `mirror_branches.half_min(|w₂|, a, x, y, branch, step = 0.05,
  n_refine = 16)` on the run's own 6,400 points. This is the same routine as the lag test's diagnostic.
- **Mapping back**: the minimiser is mapped back to the run's orientation. b₁ is taken as the 2π-representative
  nearest the current b₁. |w₂| is unchanged.

## Outcome

- **Per run and arm**: the residual r = R_cross/R_own − 1, with R_own the run's own global threshold. This is the
  lag test's own definition.
- **Per a and arm**: residual(arm) = the median over included runs.
- **Paired contrasts**: control − arm, over the same runs. Bootstrap 95% intervals of the difference of
  medians, resampling runs (10,000 resamples, seed 0). The claim is about runs within one setting, so the run
  is the unit.

## Registered predictions (scored separately at each a)

"Arm X removes at least half" means both:
- residual(X) ≤ ½·residual(control);
- the interval of residual(control) − residual(X) lies above 0.

"Arm X does not" means residual(X) > ½·residual(control).

| outcome | teleport | reset | verdict |
|---|---|---|---|
| **ID** (inherited displacement) | removes ≥ half | does not | ID supported |
| **OM** (optimiser memory) | does not | removes ≥ half | OM supported |
| **Competing** (neither) | does not | does not | neither supported; mechanism unresolved |
| both remove ≥ half | removes | removes | reported as "both"; neither hypothesis alone supported |

- teleport + reset is **reported** beside the verdict, with no criterion. It shows whether the effects add.
- **Also reported** per arm, with no criterion:
  - the crossing branch;
  - the mirror share;
  - the distance to the branch minimiser at the crossing (the lag test's diagnostic);
  - the median steps from the switch to the crossing.

## Validity and stop conditions (each exercised on constructed pass and fail cases before registration)

| check | stop / exclusion | constructed fail case |
|---|---|---|
| **control reproduces the original run bit for bit**: crossing step and \|w₂\| at crossing, exact round-trip parsing, every included run | stop if any differs | a state perturbed by one ulp in w₁ |
| **teleport does not itself change placement**: G(w₁, b₁) of the teleported hidden configuration, exact extrema, must stay ≤ 0 (as before the teleport) | per run, **exclude and count** violators; stop if more than 10% of runs at an a are excluded | a planted target with G > 0 |
| teleport lands on the minimiser | the profiled gradient norm at the teleported (w₁, b₁) ≤ 1e−6 and b₂ equals its profiled optimum to 1e−10 | a target moved off the minimiser |
| reset zeroes the moments | exp_avg = exp_avg_sq = 0 and step = 0 exactly | a partial reset (exp_avg only) |
| crossings | at least 40 crossings per arm and a; a cell with fewer is reported as insufficient, not extended | — |
| scorer | a constructed ID pattern scores ID; an OM pattern scores OM; flat residuals score competing; both-halving scores "both" | the four constructed patterns |

## Compute (measured pilots, not estimates)

- **Branch minimiser per run** (`half_min` on 6,400 points at the switch scale): about 25–30 s each, measured
  in Block 4a. This is 96 runs, about 45 minutes on one worker. Block 4a computes the same quantity at the same
  states, so its minimisers can be reused after an exact-equality check.
- **Continuations**: under 1 s each at φ = 1 (the lag test's measurement). 4 arms × 96 runs is a few minutes.
- **Total**: under 1 CPU-hour, under 0.5 GB per worker.

## What would change the paper

- **ID supported**: the residual is explained as displacement inherited from the middle phase. The writer
  inputs would say so, with its size.
- **OM supported**: the residual is an optimiser-state effect, specific to Adam's moment dynamics.
- **Neither or both**: the mechanism remains open, and the wording stays as it is ("Its mechanism remains
  open").

## Approval — 2026-09-24 (author), with one added check

- **Added validity check (stop)**: after teleporting, the state must be on the same mirror branch the run
  occupied at the switch point, in canonical orientation (w₂ > 0, branch = sign(w₁)). **Stop if any run changes
  branch.** Constructed fail case: a target on the other mirror branch.
- **Order**: runs after Block 4a (post hoc) is reported.
