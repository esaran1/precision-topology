# Test 2B registration: the validity boundary of the lag law in κχ (redesigned), 2026-09-29

**Status: written before any registered run.** No registered seed has been trained and no crossing, gap or placement of
any 2B training trajectory has been evaluated.
- This text is committed first.
- It becomes the registration when the frozen inputs of §9 are committed with the SHA-256 manifest
  `results/track2b/registration.sha256`, which includes this file's hash (the registration commit).
- That commit is timestamped with OpenTimestamps (`results/track2b/registration_stamp.txt` and its `.ots` proof).

## 1. Design reference

- Approved design: `results/designs/2B_boundary_design.md` (committed in `c3840fe`). The author approved it as designed
  on 2026-09-29, with no changes. Nothing in it is changed here.
- Where the page is silent, the convention of the first boundary test (`results/ramp_boundary_registration.md`) is used.
  Every such choice is listed in §7. None of them changes the meaning of a criterion.
- Code: `src/track2b.py`, which reuses the first boundary test's machinery (`src/ramp.py`, `src/ramp_boundary.py`).
  Tests: `tests/test_track2b.py`, which cover every decision rule on constructed PASS, FAIL and UNRESOLVED cases, the
  η_cell rule and the realised-trajectory check. The training loop reproduces `ramp.batch_ramp` bit for bit at η = 0.3.

## 2. Why a redesign

- The first boundary test (registered 2026-09-26) ended with B1 and B2 UNRESOLVED: only 3 of 10 cells met validity.
- Diagnosed post hoc on its runs: the fast-ramp runs did not lag; they destabilised. At a = 1.30 the non-crossers ended
  far from placement, with growing |w₁| and several windings. SGD at a fixed lr 0.3 was used while the Hessian grows
  with s.
- 2B therefore sets the learning rate per cell a priori, so that η·λ_max ≤ 1 on the population branch along the whole
  ramp. It indexes cells by the predicted lag κχ, not χ. Descriptively, the first test's error grew with κχ.

## 3. Setting and seeds

- Width 1, f_a(t) = t + a sin t, at a = 1.30 (natural winding k = −1) and a = 1.50 (k = 0).
- **Seeds 2,020,000–2,020,039** (40 seeds). Each seed has its own 400-point sample `fold1d.make_data(200, seed)`.
  - Checked 2026-09-29: none of these integers appears, with or without thousands separators, in any .py, .csv, .json,
    .jsonl, .md, .log, .txt, .tex, .yaml, .toml or .sh file of the repository (3,708 files scanned).
  - No `range(lo, hi)` literal in any .py file overlaps them.
- The same 40 seeds are used in all 12 cells (§7, choice 6).
- Full-batch SGD (torch.optim.SGD, lr η_cell, no momentum), float64. The hidden coordinates θ = (w₁, b₁, b₂) train;
  w₂ = +s(t) is set every step.
- **Warm start:** every seed starts at the population branch point θ\*(s₀) on the natural winding (`ramp.branch_init`),
  with s₀ = 0.5·s\*. s\* is the population switch (the certified glob-bracket midpoint, `ramp._s_star_pop`):
  4.95625 at a = 1.30 and 2.53125 at a = 1.50.
- **Held phase:** s = s₀ for steps 1–4,000. Then s = s₀e^{γ(t−4000)}, up to the first step with s ≥ s_end,
  s_end = s\*·(1 + 6κχ).

## 4. The rules, exactly

**η_cell (a priori; `track2b.branch`, `design_rows`):**
- The population branch (800-point population objective) is continued by Newton from θ\*(s₀) on a geometric grid of
  ratio 1.0025, with every cell's s_end added as a grid point, up to 3.4·s\*.
- η_cell = min(0.3, 1/max λ_max(H_pop(s))) over the grid points in [s₀, s_end].
- **Continuation validated at every point:**
  - gradient residual < 1e-10;
  - H positive definite;
  - continuation step < 0.01 in (w₁, b₁, b₂/s);
  - the branch passes through the committed switch point of `lag_law/kappa.csv` to within 1e-6, up to the winding copy;
  - λ_max is nondecreasing in s, so the maximum over the continuous interval is the value at the grid point s_end.

**γ (a priori):**
- γ = (κχ_target/κ)·η_cell·λ_min(H_pop). H_pop is the committed population Hessian at the switch (`lag_law/kappa.csv`).
  κ is the committed SGD κ for the natural winding (`lag_law/kappa_by_winding.csv`): 7.6196 at a = 1.30, 4.0744 at
  a = 1.50.
- So χ = γ/(η_cell λ_min) = κχ_target/κ =: χ_target exactly.

**Frozen per seed (before any run; `ramp_boundary._switch_job`, unchanged from the first test):**
- **s_branch:** the forced branch's own-sample switch. The branch is continued by Newton on the seed's sample from
  θ\*(s₀) on the natural winding, in 1% steps, until its gap turns positive, then bisected to 1e-6 relative.
- **λ_min_own:** the smallest eigenvalue of the own-sample Hessian at the forced branch point at s_branch·(1 − 1e-6).
- χ_own = γ/(η_cell·λ_min_own). It is used for validity only.

**Training (`track2b.train`):**
- Each cell's 40 runs are trained in one vectorised batch, to the last ramp step.
- **No gap or placement is evaluated.**
- At every step t, λ_max of the own-sample Hessian is recorded. It is taken at the point where that step's gradient is
  evaluated, (θ_{t−1}, s_t), in closed form.
- Each run's path (θ after every step, and the λ_max series) is saved and its SHA-256 recorded. The predictions table
  (`predictions.csv`) and its SHA-256 are committed before `observe` runs.

**Observation (`track2b.observe`, after the predictions commit; asserts every hash):**
- **Crossing:** the first step t ≥ 1 with G(θ_t) > 0 on the dense windows of `ramp.batch_ramp` (4,001 inner and
  2 × 2,000 outer points, w₂ > 0 orientation). This is detection at every step.
  - If t ≤ 4,000, the run was placed in the held phase: it is not a crossing, and the run counts as not crossed.
  - Otherwise s_cross = s_t, the scale used by that step.
- **Observed lag:** obs r = s_cross/s_branch − 1.
- **Predicted lag:** pred r = κ_k·χ_target. κ_k is the committed SGD κ (population switch, P = I) for the run's winding
  at its crossing, k = round((b₁ − b₁\*)/2π) (`ramp.predict_run`'s rule, with η_cell in χ). κ_k for k = −5…5 is frozen
  in `kappa_k.csv`. At the natural winding, pred r = κχ_target exactly.
- **Per run:** ratio = obs r/pred r.
- **Scored run:** a crossing run with a finite frozen s_branch (all 80 seed-a pairs have one; §9).
- **Cell statistic:** the median ratio over the cell's scored runs.
- **Realised stability, per run:** max of η_cell·λ_max over steps 1 through the crossing step (crossing runs), or
  through the last ramp step (all other runs). This includes the held phase. A non-finite λ_max on that range counts as
  a violation.

## 5. Registered criteria (`track2b.score_cells`)

The band is B = [0.75, 1.25], inclusive. Numbers are compared as computed in float64, with no rounding.

| criterion | cells | PASS if |
|---|---|---|
| **C1** | κχ ∈ {0.02, 0.05, 0.1} at both a (6 cells) | the median ratio is in B in every one of them |
| **C2** | κχ ∈ {0.3, 0.4} at both a (4 cells) | the median ratio is outside B in every one of them |
| **κχ\*** (descriptive, per a) | all 6 cells of that a | the smallest target whose median ratio is outside B |

**Validity, per cell (all three required):**
- V1: at least 30 of the 40 runs cross (scored runs).
- V2: the median χ_own over the scored runs is within 30% of χ_target: |median/χ_target − 1| ≤ 0.30.
- V3: η_cell·λ_max ≤ 1 holds along the realised trajectory of **every** run of the cell (§4).

**Unresolved:**
- A criterion is **UNRESOLVED if any of its cells is invalid**, whatever the valid cells show. UNRESOLVED takes
  precedence over FAIL.
- Otherwise the criterion is PASS or FAIL as in the table.
- The κχ = 0.2 cells enter only κχ\*.
- κχ\* is taken over the cells with at least one scored run, regardless of validity. Each cell's validity is reported
  beside it.

## 6. Competing predictions and falsifiers

| outcome | reading |
|---|---|
| C1 PASS, C2 PASS | The lag law holds within ±25% up to κχ = 0.1 and fails by κχ = 0.3: the boundary lies between 0.1 and 0.3 (κχ\* locates it). |
| C1 FAIL | The law degrades already at κχ ≤ 0.1. The first test hinted at this at a = 1.30 (valid cell χ = 0.03, κχ ≈ 0.23: ratio 1.67). |
| C2 FAIL | At least one κχ ≥ 0.3 cell stays within ±25%: the law holds further than expected, possibly to κχ = 0.4. |
| C1 or C2 UNRESOLVED | Some cell had fewer than 30 crossings, χ_own off target, or a realised η·λ_max > 1. The test says nothing about that criterion. |

- **Falsifier of "the boundary is between 0.1 and 0.3":** a FAIL of C1 or of C2.
- A failure stays a failure. Post hoc readings may be placed beside it, labelled POST HOC.

## 7. Details the page left open (the choices made, fixed now)

Each follows the first boundary test's convention where it had one.

1. **s\* in s₀ and s_end is the population switch** (certified glob-bracket midpoint, `ramp._s_star_pop`), as in the
   first test's s₀. The page uses one symbol s\* for both.
2. **λ_min in γ** is that of the committed population Hessian at the switch (first test).
3. **κ** is the committed SGD κ for the natural winding (first test). The per-run prediction uses κ_k for the winding at
   crossing (first test).
4. **λ_max over [s₀, s_end]:**
   - taken on a 0.25% geometric grid with s_end as a grid point;
   - the validated monotonicity makes the grid maximum the continuous maximum;
   - the continuation is on the natural winding copy. The Hessian is identical on every copy.
5. **Held phase:** the page's "held for 4,000 steps" is the first test's warm-up. The runs train at η_cell with s = s₀.
6. **The same 40 seeds in every cell** (first test: one seed set across its cells). This makes the cells paired in
   seeds.
7. **Realised stability:**
   - It uses the Hessian of the run's own training loss (the loss SGD descends) at the point where each step's gradient
     is evaluated.
   - It covers every step from the first held step through the crossing step, or through the ramp end if the run does
     not cross.
   - It must hold in every run of the cell. η·λ_max = 1 exactly holds.
   - Steps after a run's crossing are not part of its realised trajectory. The lag is read at the crossing.
8. **Crossing detection:** `ramp.batch_ramp`'s rule, at every step. A placement during the held phase is not a crossing
   (first test).
9. **V1 and V2 use the scored runs** (crossing runs with a finite frozen switch); V2's median is over them (first test).
10. **UNRESOLVED:** any invalid cell makes its criterion UNRESOLVED, and this takes precedence over FAIL (first test).
11. **κχ\*:**
    - per a;
    - "leaving the band" means a median ratio outside B;
    - it is taken over cells with at least one scored run, regardless of validity (first test);
    - "first" means the smallest target.
12. **Training without evaluating crossings; predictions and path hashes committed before observing** (Test 2A's
    convention, as the brief requires).
13. **Feasibility gate:** added before registration, not a criterion. The design would be infeasible if in some cell
    more than 10 seeds had their predicted crossing, s_branch·(1 + κχ), at or beyond s_end. Then V1 could not be met
    even if the law held exactly, and 2B would have stopped here. The gate passed (§8).

## 8. A-priori risks, stated now (from the frozen inputs only; `feasibility.json`)

- **The a = 1.50, κχ = 0.02 cell is close to its crossing threshold.**
  - Its ramp ends at 1.12·s\* = 2.835.
  - For 8 of the 40 seeds, s_branch·1.02 ≥ 2.835. The seeds' s_branch spans 0.874–1.214·s\*.
  - So at most 32 runs can cross, against the 30 required. Two more non-crossers would make C1 UNRESOLVED.
  - The median in that cell is over the 32 or fewer seeds with the smaller switches.
  - At a = 1.30, κχ = 0.02, 4 seeds are ruled out, so at most 36 can cross.
  - At twice the predicted lag, the counts are 9 (1.50, 0.02), 7 (1.30, 0.02) and 3 (1.50, 0.05).
- **η_cell = 0.3 (the cap) in 9 of the 12 cells.** Along the population branch, λ_max stays at or below 3.88 at
  a = 1.30 and 2.62 at a = 1.50.
  - η_cell < 0.3 only at a = 1.30, κχ = 0.2, 0.3 and 0.4, where it is 0.2766, 0.2630 and 0.2578.
  - So the learning-rate change from the first test is small. The larger change is the shorter ramp end: s_end is
    s\*(1 + 6κχ), against the first test's s\*·max(3, 1 + 4κχ).
  - Whether the fast cells now stay on the branch is not known in advance.
- **V3 has no margin where η_cell = 1/λ_max** (a = 1.30, κχ ≥ 0.2): η·λ_max(H_pop) = 1 at s_end.
  - A run that has not crossed by s_end is checked there on its own sample's Hessian.
  - At a = 1.30, κχ = 0.1 (a C1 cell), η·λ_max(H_pop) reaches 0.970 at s_end.
- **V2 cannot bind:** χ_own/χ_target lies in 0.951–1.081 (a = 1.30) and 0.933–1.075 (a = 1.50) over all 40 seeds.

## 9. Frozen values (computed before any training; in the registration commit)

**Population branch** (`population_branch.csv`, `branch_validation.json`, `branch.log`): 774 grid points per a, from
0.5·s\* to 3.4·s\*.

| a | max residual | min λ_min | max step | switch-point match | λ_max nondecreasing | validated |
|---|---|---|---|---|---|---|
| 1.30 | 9.9e-16 | 0.0732 | 9.3e-4 | 4.1e-14 | yes | yes |
| 1.50 | 4.9e-16 | 0.0702 | 1.4e-3 | 2.2e-13 | yes | yes |

**Design** (`design.csv`): λ_min(H_pop at the switch) is 0.150673 at a = 1.30 and 0.149318 at a = 1.50.

| a | κχ | χ_target | s_end | max λ_max | η_cell | γ | ramp steps | total steps |
|---|---|---|---|---|---|---|---|---|
| 1.30 | 0.02 | 0.002625 | 5.5510 | 2.7174 | 0.300000 | 1.186455e-04 | 6798 | 10798 |
| 1.30 | 0.05 | 0.006562 | 6.4431 | 2.9373 | 0.300000 | 2.966139e-04 | 3222 | 7222 |
| 1.30 | 0.1 | 0.013124 | 7.9300 | 3.2329 | 0.300000 | 5.932277e-04 | 1961 | 5961 |
| 1.30 | 0.2 | 0.026248 | 10.9038 | 3.6151 | 0.276620 | 1.093991e-03 | 1355 | 5355 |
| 1.30 | 0.3 | 0.039372 | 13.8775 | 3.8028 | 0.262966 | 1.559986e-03 | 1105 | 5105 |
| 1.30 | 0.4 | 0.052496 | 16.8513 | 3.8794 | 0.257772 | 2.038897e-03 | 941 | 4941 |
| 1.50 | 0.02 | 0.004909 | 2.8350 | 1.8203 | 0.300000 | 2.198879e-04 | 3668 | 7668 |
| 1.50 | 0.05 | 0.012272 | 3.2906 | 1.9750 | 0.300000 | 5.497197e-04 | 1739 | 5739 |
| 1.50 | 0.1 | 0.024544 | 4.0500 | 2.1819 | 0.300000 | 1.099439e-03 | 1058 | 5058 |
| 1.50 | 0.2 | 0.049087 | 5.5688 | 2.4474 | 0.300000 | 2.198879e-03 | 674 | 4674 |
| 1.50 | 0.3 | 0.073631 | 7.0875 | 2.5755 | 0.300000 | 3.298318e-03 | 523 | 4523 |
| 1.50 | 0.4 | 0.098174 | 8.6063 | 2.6244 | 0.300000 | 4.397757e-03 | 436 | 4436 |

**κ_k** (`kappa_k.csv`; SGD, population switch):

| a | k = −2 | k = −1 | k = 0 | k = +1 | k = +2 |
|---|---|---|---|---|---|
| 1.30 | 22.68 | 7.620 | −7.439 | −22.50 | −37.56 |
| 1.50 | 19.84 | 11.96 | 4.074 | −3.809 | −11.69 |

**Per seed** (`frozen_switches.csv`, `freeze.log`):
- All 80 seed-a pairs have a forced-branch switch, with no note.
- s_branch: 4.1354–5.6619 (median 4.9336) at a = 1.30, and 2.2125–3.0724 (median 2.5846) at a = 1.50.
- λ_min_own: 0.1393–0.1584 at a = 1.30, and 0.1389–0.1600 at a = 1.50.

**Manifest:** `results/track2b/registration.sha256` holds the SHA-256 of:
- `src/track2b.py`, `src/ramp.py`, `src/ramp_boundary.py`, `src/lag_law.py`, `src/fold1d.py`,
  `src/width2_conditional.py`, `tests/test_track2b.py`;
- this file and the design page;
- the committed inputs `lag_law/kappa.csv`, `lag_law/kappa_by_winding.csv` and `cond_certified_brackets.csv`;
- the frozen files `population_branch.csv`, `branch_validation.json`, `design.csv`, `kappa_k.csv`,
  `frozen_switches.csv` and `feasibility.json`.

`train` and `observe` assert every one of these hashes before running.
