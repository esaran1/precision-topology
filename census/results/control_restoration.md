# Task F 1b verdict: three of four restoration criteria pass; the plateau criterion fails

Criterion fixed in advance in `control_criterion.md`. Data:
`control_search_mnist.csv` (map), `control_replication.csv` (MNIST
follow-up), `control_search_cifar.csv` (pilot map),
`cifar_replication.csv` (n = 10 fresh seeds 3–12, this verdict's basis).
Permutation tests: 100,000 resamples, seed 0, two-sided.

## The map (1a), in one table

| setting | GELU vs ReLU | verdict |
|---|---|---|
| MNIST MLP depth 4 | ReLU better | no advantage |
| MNIST MLP depth 8, n=10 fresh | −13 errors, p = 0.070 | failed replication |
| MNIST MLP depth 12, n=10 fresh | −3 errors, p = 0.98 | pilot gap was seed noise |
| CIFAR CNN depth 4, n=10 fresh | **−116 errors, p = 0.006** | replicated |
| CIFAR CNN depth 8, n=10 fresh | **−132 errors, p = 0.00012** | replicated |
| CIFAR CNN depth 8 + warmup (pilot, n=3) | −11 errors | advantage erased |

The folk advantage is real but narrow: absent in shallow MLPs, dead on
MNIST at any depth we tried, present in CIFAR CNNs at both depths
tested — and, in the pilot, erased by a 500-step warmup, an
initialization-trajectory interaction consistent with the weight-scale
account and predicted by no smoothness story (n = 3, not replicated;
recorded as a lead, not a claim).

## The four criteria (`control_criterion.md`), scored

1. **Effect** — pass. GELU's mean test errors below ReLU's at both
   depths (2,050 vs 2,182; 2,498 vs 2,614).
2. **Statistics at n ≥ 10** — pass. p = 0.00012 (depth 8), p = 0.006
   (depth 4), fresh seeds, pilots excluded.
3. **Replication in ≥ 2 nearby configurations** — pass. Both depths,
   same sign, both independently significant.
4. **Training-loss plateau (< 1% final-epoch relative improvement)** —
   **fail, everywhere.** Median final-epoch improvement: 9–20% by
   cell; minimum 2.1%. 2 of 45 runs carry the plateau flag and one of
   those has *rising* loss (improvement −23%), i.e. instability, not
   convergence. Every cell is still mid-descent at 12 epochs.

**Verdict under the registered criterion: the control is NOT formally
restored.** Three of four conditions pass, strongly; the fourth was
written precisely to prevent declaring an activation ranking from
mid-descent snapshots — the MNIST bottleneck reversal (tanh ahead at 3
epochs) is the cautionary case in this very project.

## Why this failure mode is weaker here than in the 2a reversal

In the bottleneck experiment tanh led on *training* error, so the test
ranking plausibly reflected optimization speed alone. Here GELU leads
ReLU on **training error too, in the same direction as test, at both
depths** (146 vs 300 at depth 4; 224 vs 604 at depth 8) — there is no
crossover pending that a plateau would obviously reveal, and the gap
*widened* from depth 4 to depth 8 while ReLU's train deficit tripled.
The mid-descent caveat is real but the specific artifact that burned us
in 2a (leader in test trailing in train) is not present.

## 1c kill switch: not fired, at replication scale

tanh is the *worst* of the three wherever the GELU advantage exists
(depth 8: 2,514 vs GELU's 2,050, p = 0.0004; train error 905 vs 224).
The advantage tracks non-monotonicity, not smoothness. The smoothness
explanation of GELU-over-ReLU is contradicted in this setting — a
smooth monotonic activation does strictly worse than ReLU here.

## What would decide it, and the budget position

The missing piece is one run: the depth-8 cell trained to plateau
(~30+ epochs, ~80 min/run, ~20 runs ≈ 1–2 days of CPU) checking that
the GELU−ReLU gap survives convergence. Part 1's hard budget (one to
two days) is now spent: MNIST arm + failed replication + CIFAR pilot +
this replication ≈ two days of wall time. Per the brief's Part 3, the
budget is not extended by default; the decision to fund the plateau
run (or accept the scope sentence with "replicated mid-descent, not
confirmed at convergence") is the user's.

No conclusions beyond this table are drawn; Part 2 (the width
prediction in this setting) is gated on that decision.
