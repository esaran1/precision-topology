# P-step: registered direction falsified; a larger effect runs the other way

Registered in `arrhenius_prediction.md`: *smaller* steps raise findability
near the onset (thin sheet overshot by a large step). Data:
`step_size_sweep.csv` (2 optimizers x 4 values of a x 5 rates x 200 seeds).

## Result: findability rises with the LARGEST step, non-monotonically

Adam, dense-verified solve rate (n = 200):

| lr | a=1.25 | a=1.35 | a=1.45 | a=1.50 |
|---|---|---|---|---|
| 3e-2 | **0.415** | **0.850** | **0.880** | **0.880** |
| 1e-2 (baseline) | 0.000 | 0.010 | 0.250 | 0.405 |
| 3e-3 | 0.000 | 0.025 | 0.400 | 0.585 |
| 1e-3 | 0.000 | 0.080 | 0.625 | 0.735 |
| 3e-4 | 0.000 | 0.000 | 0.000 | 0.045 |

**P-step is falsified as registered.** The largest step is best everywhere,
and at a = 1.25 — where the baseline rate is 0/200 and the committed
onset is (1.30, 1.35] — lr = 3e-2 gives **83/200**. The onset moves below
1.25 at large step size.

There is a weaker secondary effect in the registered direction: within
lr <= 1e-2 the rate does rise as lr falls (0.250 -> 0.400 -> 0.625 at
a = 1.45), before collapsing at 3e-4. So the curve is non-monotonic with
a maximum at the largest step and a second local rise at small steps.

## Controls

- **Budget is not the explanation.** The sweep scaled steps as 1/lr; a
  control at *fixed* 2000 steps for every lr reproduces the effect more
  cleanly (a=1.25: 43/100 at 3e-2, 0/100 at every smaller lr;
  a=1.45: 87, 22, 0, 0, 0).
- **The solutions are genuine.** At a = 1.25, lr = 3e-2, 43/100 pass an
  8,001-point exact-region check (the sweep used 4,001).
- **Realized vs nominal step.** Adam compresses a 100x nominal range to
  ~4x realized (0.00027 -> 0.0012) and is non-monotonic per coordinate,
  so nominal lr is not the operative variable. Plotted against
  `realized_mean` the Adam curve is still non-monotonic: the 3e-3 and
  1e-2 cells have nearly identical realized steps (0.00100, 0.00103) and
  very different rates (0.400 vs 0.250 at a=1.45).

## Plain SGD: the clean step test finds almost nothing

| lr | 1.25 | 1.35 | 1.45 | 1.50 |
|---|---|---|---|---|
| 1.0 | 0 | 0 | 0.000 | 0.000 |
| 0.3 | 0 | 0 | 0.005 | **0.185** |
| 0.1 | 0 | 0 | 0.000 | 0.000 |
| 0.03 | 0 | 0 | 0.000 | 0.000 |
| 0.01 | 0 | 0 | 0.000 | 0.000 |

SGD solves at all only at lr = 0.3, a >= 1.45, and never below the
committed onset. It shows no monotone step-size trend in either
direction. So the effect is **not a generic step-size effect** — it is
specific to Adam.

## Why: large steps reach a different part of the manifold

At a = 1.25, lr = 3e-2, the solutions found have **|w2| = 9.9-14.9
(median 12.4)**. Standard-lr solutions sit at |w2| ~ 5, and the
analytic construction at |w2| = 1. Large-step Adam is not threading a
thin sheet more accurately; it is **travelling further and landing in a
higher-|w2| region of the solution manifold** that small steps never
reach within budget.

This is consistent with the cross-section picture only in the loose
sense that capture depends on where the trajectory goes, not on energy.
The specific mechanism registered — small steps avoid overshooting a
thin sheet — is **not** what happens.

## Status of the successor hypothesis

- Energetic account: falsified (`barrier_results.md`).
- P-step as registered (smaller steps better): **falsified**, sign
  reversed at the dominant effect.
- P-ratio: consistent but weakly diagnostic (thickness rises
  monotonically in a, so any monotone function orders the rates
  equally well), and quantitatively wrong (onset at ratio 20-30, not
  order 1).
- What survives: findability is dynamical and depends on **which region
  of the manifold the optimizer reaches**, which is what Part 4
  measures directly.

## Consequence for the committed onset

The onset (1.30, 1.35] in `fold1d_results.md` / T30 is **optimizer- and
step-size-specific**, not a property of the task. At lr = 3e-2 it lies
below 1.25. This does not affect the monotonic zero (provable) or the
gap's existence, but the onset's *location* must be quoted with its
optimizer configuration. Ledger updated.
