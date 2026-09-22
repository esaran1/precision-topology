# Saturation, not trapping: the jump/cold stall is an optimisation artifact

Scored against `results/blockE_stall_prediction.md` (registered `ce4c7df`, before
any gradient norm, loss excess or extended-budget run was inspected). Artifacts:
`blockE_stall.csv`, `blockE_stall_loss.csv`, `src/blockE_stall.py`.

**Registered expectation was TRAPPING. The data say SATURATION. The registered
expectation is wrong.**

---

## 1. Gradient norms — the saturation signature

Median over 40 seeds, `a = 1.30`:

| step | `control` | `jump` | `cold_high` | jump / control |
|---|---:|---:|---:|---:|
| 100 | 3.35e-02 | 3.26e-01 | 3.40e-01 | 9.7x |
| 500 | 1.57e-02 | **1.13e-05** | **4.97e-06** | **0.00072** |
| 2,000 | 9.55e-03 | 1.13e-05 | 4.97e-06 | 0.0012 |
| 8,000 | 2.95e-03 | 3.36e-05 | 1.24e-05 | 0.011 |
| 11,999 | 1.22e-03 | 3.01e-04 | 2.96e-05 | 0.25 |

At step 500 the jumped runs' gradients are **0.07% of control's** — three orders
of magnitude down, and far below S-A's "under 10%" criterion. The large initial
logit saturates the sigmoid almost immediately (the step-100 spike is the
transient before saturation sets in).

## 2. Loss against the conditional minimum — the trapping signature

For each stalled run, `best_conditional` at that run's own final `|w2|`
(frozen Block B procedure, 24 restarts):

| arm | n | median `loss_run - loss_min` |
|---|---:|---:|
| `jump` | 12 | **+0.341834** |
| `cold_high` | 12 | **+0.367765** |

Both are positive and **68-74x** S-B's 0.005 criterion. The stalled runs are
nowhere near the best achievable configuration at their own output scale.

## 3. Extended budget — the tiebreaker, and it favours saturation

| arm | placed at 12,000 | placed at **60,000** |
|---|---:|---:|
| `jump` | 2/15 | **7/15 (46.7%)** |
| `cold_high` | 1/15 | **3/15 (20.0%)** |

S-A required **more than 25%** at the longer budget for `jump`: **46.7%, passes**.
S-B required the 60,000-step rate to be **within 10 points** of the 12,000-step
rate: `jump` moves **13.3 -> 46.7**, a 33-point rise, **fails**.

## Verdict: S-A satisfied, S-B not — SATURATION

Two of three measurements point to saturation (collapsed gradients, and a 5x
budget substantially recovering placement); one points to trapping (loss far above
the conditional minimum). The extended budget is decisive: **a trapped run cannot
be rescued by more steps, and these are.** The large loss excess is consistent
with saturation once the geometry is examined.

## What the stalled runs are actually at

Gradient and Hessian at the 12,000-step endpoint, `jump`, three seeds:

| seed | `w1` | loss | `|grad|` | Hessian eigenvalues |
|---|---:|---:|---:|---|
| 0 | -0.004934 | 0.692546 | 9.1e-04 | -5.1e-06, +2.3e-05, +32.9, +50.2 |
| 1 | -0.002916 | 0.693020 | 1.3e-05 | -8.4e-06, +3.7e-09, +25.7, +30.2 |
| 2 | -0.001617 | 0.693092 | 1.0e-05 | -1.1e-07, +1.7e-07, +30.2, +42.3 |

`w1 ~ -0.003`, loss ~ `log 2 = 0.693147`: the runs sit in the **near-constant
predictor** region — the same degenerate configuration Block B had to exclude by
`is_degenerate` to keep its switch points meaningful.

It is **not** a stationary point. `w1 = 0` exactly would be stationary only if the
two classes had matching first moments, and on the sampled data they do not
(`E[x | inner] = +0.063`, `E[x | outer] = +0.148`), so the gradient at `w1 = 0` is
**0.18-0.36**, not zero. What the runs find is an **extremely ill-conditioned flat
valley**: two Hessian eigenvalues at `1e-6` or below against two at `25-50`, a
condition number above `10^7`, with the small eigenvalues marginally negative
(a shallow saddle, not a minimum).

**This reconciles the mixed signature.** Gradients are tiny because the valley is
flat (saturation-like); the loss is 0.34 above the achievable minimum because the
valley is the wrong place to be (trapping-like); and a 5x budget helps because the
valley is a *slow region*, not a basin with a barrier.

## Consequence: the annealing framing is NOT adopted

Per the standing instruction and the registration's own reporting condition, the
annealing interpretation of output scale enters `paper_claims_delta.md` only if
this test returns **trapping**. It returns **saturation**. The framing is
therefore **not adopted**, regardless of what the growth-rate dose-response shows.

What is reported instead, which the data do support: **pre-supplying `|w2|` at
initialisation drives the run into an ill-conditioned near-constant-predictor
region from which recovery is slow** — 2/15 at 12,000 steps against 7/15 at
60,000, versus control's 37/40 at 12,000. This is a statement about optimisation
conditioning, not about annealing or basin structure, and the paper should say so.

The Block E conclusion is unchanged and does not depend on this: `R > R_glob` is
necessary for placement to persist and not sufficient for it to be found. What
this test settles is **why** the jump arm fails — saturation and conditioning, not
a wrong basin.
