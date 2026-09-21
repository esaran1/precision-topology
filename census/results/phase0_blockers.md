# Phase 0: blockers

Read-only audit against artifacts and source. No measurement. 2026-09-21.

---

## 0a. Pooled counts — same population, AdamW added

| population | file(s) | n | below `R=0.30` | above `R=0.50` |
|---|---|---:|---|---|
| **current (abstract)** | `r_pooled.csv` + `r_adamw.csv` | **3,150** | **0 of 2,285** | **460 of 461** |
| *superseded* | `r_pooled.csv` alone | *2,910* | *0 of 2,160* | *402 of 403* |

**AdamW was added to the pool** — the 240 AdamW runs are the whole difference
(2,910 + 240 = 3,150). These are **not different populations**: the earlier
figures are the same dataset before the third optimiser existed, and are stale.

**Caveat for any future script**: AdamW lives in a **separate file**.
`r_pooled.csv` still holds 2,910 Adam+SGD rows; the 3,150 union is formed at
render time. Anything quoting a pooled figure must read both.

### The one non-solving run above `R = 0.50`

| | |
|---|---|
| cell | `a = 3.0`, budget **2,000**, **Adam**, seed **13** |
| source | `fold1d_sweep.csv` (**float32**) |
| terminal | `w1 = -1.452066`, `b1 = -1.944588`, `w2 = 1.099398`, `b2 = 2.331415` |
| `R` | **0.576580** |
| 200-point sample | **0 errors** |
| dense 4,001-point | **11 outer violations**, `x` in [1.2000, 1.2040], worst logit **-0.011** |

A sample-level-only apparent solution: correct on every sampled point, violating
the region on a sliver at the outer window's edge. Same failure mode the dense
criterion caught 56 times in 163 in the link setting.

## 0b. Configurations, from source

**Task** (`src/fold1d.py`): `INNER_MAX = 0.8`, `OUTER_MIN, OUTER_MAX = 1.2, 2.0`.

**Data** (`make_data`): `numpy default_rng(seed)`; 200 points per class;
inner `U(-0.8, 0.8)`; outer magnitude `U(1.2, 2.0)` with sign `±1` uniform;
cast to **float32** at construction. **Full batch — all 400 points every step,
no minibatching.**

**Initialisation**: `torch.empty(4).uniform_(-1.0, 1.0)`, i.e. `U(-1,1)^4`, all
four parameters. `budget_law.run_full` multiplies by a `scale` argument,
default 1.0.

| optimiser | lr | other | where |
|---|---|---|---|
| Adam | **1e-2** | defaults | `fold1d.LR`, `budget_law`, `onset_law`, `family_onsets` |
| SGD | **0.3** | no momentum | `sgd_onsets.LR` — comment records it as the only SGD lr that solves at all |
| SGD (α comparison) | **0.3** | no momentum | `alpha_composition.main` |
| AdamW | **1e-2** | **weight_decay 0.01** | third-optimiser sweep |

**Seeds and budgets**

| sweep | seeds | budgets |
|---|---|---|
| onset sweeps | 40 (`SEEDS = 40`) | 2k, 4k, 8k, 32k, 64k, 128k |
| `budget_law` α | 30 | 1k, 2k, 4k, 8k, 16k, 40k, 80k, 160k |
| `alpha_composition` | 30 | 1k, 2k, 4k, 8k, 16k, 40k |
| `family_onsets` | 40 | 2k, 8k, 32k, 128k |
| `fold1d_sweep` | 200 | 2,000 (`STEPS`) |
| AdamW | 30 | 1k, 2k, 4k, **5k, 6k**, 8k, 16k, 40k |

**Link-setting runs** (`src/train.py`, `src/census.py`): **Adam, lr 1e-2,
max_steps 2,000**, width 3. This is the optimiser and budget for the pooled
link runs, the search families, the depth comparison and the constructed-vs-found
comparison.

**Precision**: `fold1d_sweep` and the sweep drivers run **float32**; the
theorem verification and Figure 1's recovered parameters run **float64**.

**Corrected 2026-09-21**: an earlier version of this note said the two "differ
materially", citing `a = 1.5` seed 0 as solving in float64 and failing in
float32 (169 errors). **That was an RNG artifact, not an arithmetic one.**
`torch.empty(4, dtype=d).uniform_()` consumes the random stream differently per
dtype, so drawing the initialisation separately in each precision yields
**different networks**, not the same network computed two ways. Given the
**same** initialisation, float32 reproduces the float64 solver exactly: 0
errors, `solves()` True, parameters agreeing to ~1e-5. Across 180 paired runs
at `a = 1.40/1.45/1.50`, **0 flips**. **float32 and float64 agree run-for-run
given a shared initialisation.**

## 0c. Family B — **the geometric quantity is not well defined**

**α values used**: `-1.0, -0.5, -0.25, -0.1, -0.05` (non-monotone half;
threshold is `α = 0`).

**How `Ĝ` was computed**: a grid search over `w1 ∈ (0.02, 4.0]`,
`b1 ∈ [-4, 4]`, taking the max over both orientations. **There is no
normalisation of `(w1, b1)`.**

**This is fatal for the quantity as defined.** `f(t) = max(t, αt)` is positively
homogeneous, so `G(c·w1, c·b1) = c·G(w1, b1)` and the unrestricted supremum is
**infinite**. Measured directly, `Ĝ` scales exactly with the search box:

| box `w1 ≤` | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| `Ĝ(α = -0.5)` | 0.2000 | 0.4000 | 0.8000 | **1.6000** | 3.2000 |

The committed value **1.600** is the box edge at `w1 ≤ 8`, not a supremum. The
committed values are exactly `3.2|α|` for `|α| ≤ 0.5`, i.e. an **implicit
normalisation at a fixed `w1` scale that is nowhere stated**.

**Consequence**: family B's `R` is not comparable with the other families', and
the "family B breaks the product structure" finding (T54) rests on an
arbitrary box. **Phase 6 must recompute under an explicit scale-invariant
normalisation before that conclusion stands.** Flagged now rather than at
Phase 6 because it changes what T54 can claim.

## 0d. Reproducibility statement — confirmed as stated

Headline numbers recomputed from artifacts after the transcription fix:

| quantity | value now | headline |
|---|---|---|
| onset exponent | **−0.7340** (R² 0.9900, 6 bracketed) | −0.7340 |
| `alpha` | **1.1172** | 1.1172 |
| slope vs `1/β` | **1.0984** | 1.0984 |

**No headline number changed.** The two derived quantities that moved are as
you state: collapse `θ` **0.7250 → 0.7500**, and the 25% onset exponent
**−0.5037 (3 of 6 bracketed) → −0.6581 (4 of 6)**. Both moved *toward* the rest
of the evidence. Commit `8ff92ad`.

## 0e. Logging — **Phase 1 cannot run on stored data**

**No artifact stores `(w1, b1, w2, b2)`** for any sweep population. Coverage:

| artifact | rows | parameter columns |
|---|---:|---|
| `fold1d_sweep.csv` | 4,800 | `w1_abs`, `w2_abs` — **magnitudes only** |
| `fold1d_refine.csv` | 800 | `w1_abs`, `w2_abs` |
| `r_pooled.csv` | 2,910 | `w2` only |
| `r_adamw.csv` | 240 | `w2` only |
| `r_family_b.csv` | 1,000 | `w1_abs`, `w2_abs` |
| `box_counterexample.csv` | 5 | all four — but constructed, not trained |

**`b1` and `b2` are absent everywhere.** Both are required: placement needs
`b1`, and the bias condition is a statement about `b2`. Signs are also required
for orientation, and `w1_abs`/`w2_abs` discard them.

**No intermediate checkpoints exist** for the 1D sweeps.

**Rerun cost, measured**: **217 ms per run** at 2,000 steps in float64.

| rerun | runs | cost |
|---|---:|---|
| Phase 1 population (`sweep` + `refine`, `a > 1`) | 2,400 | **~9 min** |
| Phase 2b grid (9 `a` × 5 budgets × 40 seeds) | 1,800 | **~12 min** on 4 cores |

Cheap, as the brief anticipated. Phase 1 therefore reruns with full logging
rather than analysing stored magnitudes. The rerun records both precisions, and
they **agree run-for-run given a shared initialisation** (see the corrected
precision note above), so Phase 1 describes the same population as the
headlines.
