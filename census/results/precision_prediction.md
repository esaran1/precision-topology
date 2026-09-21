# RETRACTED 2026-09-21 — the effect it registers against does not exist

> **This registration is void.** It was written against a 'precision effect' that
> was an artifact of my own measurement. `torch.empty(4, dtype=d).uniform_(-1,1)`
> consumes the RNG stream differently for float32 and float64, so the two arms
> **started from completely different initialisations** and were never the same
> seed. The comparison measured ordinary sampling variation between two
> independent draws, not arithmetic.
>
> **The tell was in the data before I interpreted it**: a seed-set overlap ratio
> of 1.0 against the independence expectation *is* what independent samples look
> like. I read it as decorrelation caused by the perturbation.
>
> **Controlled test, initialisation held fixed and only arithmetic varied:**
>
> | `a` | shared init | per-dtype init |
> |---|---|---|
> | 1.45 | **60/60 agree (100%)** | 46/60 (77%) |
> | 1.50 | **60/60 agree (100%)** | 36/60 (60%) |
>
> Plus 180 paired runs at a = 1.40/1.45/1.50 with shared init: **0 flips,
> relative final-parameter distance 0.0000**.
>
> **float32 and float64 agree run-for-run.** There is no divergence, no
> knife-edge, and no reason to treat the transition band as irreproducible.
> H-divergence and H-knife-edge are both moot. `src/phase1_relog.py` is fixed to
> draw the initialisation once and cast.

---

# Registration: is the precision effect a terminal knife-edge or trajectory divergence?

**Written before any trajectory distance or divergence step was computed.** The
only quantities in hand are the per-cell solve counts and the seed-set overlap
already reported.

Date: 2026-09-21.

---

## What is already measured

Same seeds, same code, float32 against float64:

| `a` | solved 32 | solved 64 | overlap | E[overlap] if independent | ratio |
|---|---:|---:|---:|---:|---:|
| ≤ 1.30 | 0 | 0 | 0 | 0 | — |
| 1.35 | 2 | 2 | **0** | 0.0 | 0.00 |
| 1.40 | 15 | 17 | **0** | 1.3 | **0.00** |
| 1.45 | 50 | 46 | 15 | 11.5 | **1.30** |
| 1.50 | 81 | 85 | 35 | 34.4 | **1.02** |
| 2.00 | 143 | 146 | 105 | 104.4 | **1.01** |

**Aggregate rates are stable** (50 vs 46, 81 vs 85, 143 vs 146) while **seed
identity carries almost no outcome information**: the overlap ratio sits at 1.0
where the counts are large enough to measure it, and at 1.40 the two solving
sets are *disjoint*.

## The two hypotheses

**H-knife-edge.** Trajectories track each other closely throughout. The final
parameters differ only at rounding scale, but sit on opposite sides of a
narrow admissible bias interval. *Predicts*: small float32-to-float64 distance
between final parameters, comparable for flipped and agreeing runs; divergence
step late, near the end of training.

**H-divergence.** The optimisation is chaotic in the transition band: a
rounding perturbation is amplified until the two runs reach genuinely different
parameters. *Predicts*: large final-parameter distance, **much larger for
flipped runs than agreeing ones**; divergence step early, far from the end.

## Registered prediction

**H-divergence**, and specifically:

1. Median relative final-parameter distance among **flipped** runs is
   **> 10x** that among **agreeing** runs in the same cell.
2. The first step at which the two trajectories differ by more than 1%
   relative is **before 50% of the budget** for the median flipped run.
3. The distance distribution for flipped runs is **not** concentrated at
   rounding scale (median relative distance > 0.1).

Registered because the overlap ratio is already at the independence value: if
trajectories merely straddled a threshold at the end, seeds would retain
structure and the ratio would exceed 1. It does not. The knife-edge reading was
my first description of this effect and I am registering against it.

**If H-knife-edge holds instead**, the effect is a measurement-boundary artifact
and the pooled float32 numbers describe the same underlying object. **If
H-divergence holds**, individual run outcomes in the transition band are not
reproducible across arithmetic, only rates are, and every paper claim in that
band must be a rate claim.

## Consequences fixed in advance

- **Phase 3** becomes a **rate comparison**, not paired-seed contrasts, with a
  **null-perturbation arm** (relative 1e-6 on `(w2, b2)`, or a precision switch)
  to establish the per-seed flip rate as the noise floor. All Phase 3 arms run
  in **one precision**.
- **Phase 2b** carries a registered risk: if trajectories diverge this readily,
  early-checkpoint `R` may predict final outcome poorly among unsolved runs
  **regardless of mechanism**. Headline metrics are **log loss and calibration**
  at the cell level, not run-level AUC alone.
- The paper states transition-band claims as **rate claims**.
