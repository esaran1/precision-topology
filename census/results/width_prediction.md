# Task E Part 1a: registered prediction — width dependence of the activation advantage

Registered 2026-08-23, before any Task E run. Setting: linked tori at
`d = 3`, the census pipeline (`src/width_sweep.py` lineage), full-batch
Adam, 2,000 steps.

## Prior data this registration must declare

This is not a blank slate, and pretending otherwise would make the test
worthless. `results/width_sweep.md` (2,880 runs, widths 3–15, depths
3–12, seeds 0–19, committed before this brief existed) already shows:

- the binary perfect-rate advantage of GELU over the monotonic
  activations is large at width 3 (6/80 vs 0/240), present at width 4,
  and gone by width ~6–8, where every activation saturates;
- from width 8 to 15 GELU holds no advantage at all (it is nominally
  *below* tanh and leaky-ReLU there);
- ReLU's late catch-up is dying-unit resolution, not expressivity.

So the coarse shape of the Part 1 prediction is **partly postdiction**
with respect to sample-level perfect rates at widths 3–15. What follows
separates what is genuinely predictive in the new experiment from what
is not.

## What is genuinely out-of-sample in 1b

1. **New widths 16, 24, 32** — no prior data above 15.
2. **Fresh seeds 100–199** at every width (prior sweep used 0–19), so
   every cell of the new sweep is out-of-sample even where widths overlap.
3. **Dense-verified rates** (0 errors on 100,000 fresh points) as the
   primary outcome — the prior sweep was sample-level; dense checks
   exist only for a small width-3/4 subset.
4. **A family A representative, sin_family a = 1.5** — never run above
   width 4 (threshold sweep width-3 rate: 16/80).
5. **Convergence speed** (steps to training criterion) — never measured
   as a function of width.
6. **Fixed depth 3** — a single-depth slice at n = 100/cell, versus the
   prior 20/cell pooled over four depths.

## The registered prediction

**P-W1 (decay).** The non-monotonic-vs-monotonic advantage in
dense-verified separation rate decays monotonically with width and is
statistically indistinguishable from zero (Fisher exact, two-sided,
α = 0.05, n = 100 per cell) at every width ≥ 8. This holds for both
GELU and sin(1.5) against each of tanh, leaky-ReLU, and ReLU-with-live-
units (ReLU cells are additionally reported with dead-at-init runs
flagged, since dying ReLU is a separate, already-documented mechanism).

**P-W2 (decay shape).** The account implies a two-phase shape, not a
parametric curve, and we register the shape rather than inventing a
functional form:

- *Width 3 — categorical.* The obstruction is topological at width = d:
  monotonic dense-verified rate is exactly 0 (this extends the
  project's 0-in-5,540 body of evidence with 300 more monotonic runs);
  non-monotonic rates are positive.
- *Widths 4–6 — findability remnant.* At width d + 1 the obstruction is
  gone (an embedding into R⁴ can unlink the tori; monotonic layers
  suffice thereafter), so any remaining advantage is a findability
  gap, which should shrink as width adds redundant units.
- *Width ≥ 8 (~2.5d) — zero.* No advantage on any measure of
  separation.

The transition from categorical to zero should be **monotone in width**
for each activation pair; we do not register its rate.

**P-W3 (large-width ceiling).** At widths 16, 24, 32 every activation's
dense-verified rate is high and mutually indistinguishable (pairwise
Fisher n.s.), including ReLU at depth 3 (its chance-collapse rate at
depth 3 was already ≤ 5% by width 8 in the prior sweep).

**P-W4 (what the account does *not* predict — the 1c discriminator).**
The fold account is a claim about separation, not about optimization
speed. We register no prediction that convergence-speed differences
vanish at large width. The registered interpretation rule, stated
before looking: **if at width ≥ 16 GELU retains a steps-to-criterion
advantage while holding no dense-verified separation advantage, that
speed advantage is an optimization effect and is not evidence for the
fold mechanism** — and will be reported as such, separately, not folded
into the headline.

## What would falsify the account's width claim

- A dense-verified separation advantage for GELU or sin(1.5) over any
  monotonic activation that is Fisher-significant at any width ≥ 12.
- An advantage that **grows** with width anywhere above 4.
- Non-monotone decay: an advantage that vanishes and then reappears at
  a larger width.
- (Bug-alarm, not falsifier: any monotonic dense-verified separation at
  width 3 — stop everything and find the bug, per standing rule.)

## Measures (fixed now)

- **Dense-verified separation**: 0 errors on the 2,000-point held-out
  eval *and* 0 errors on 100,000 fresh points (seed derived by crc32 as
  in `src/dense_check.py`). Reported with Clopper–Pearson 95% intervals;
  exact-zero cells get the rule-of-three bound.
- **Minimum errors**: distribution of eval errors per cell (counts, not
  means), so near-misses at width 3 remain visible.
- **Convergence speed**: first step at which training accuracy reaches
  1.0, checked every 50 steps; censored at 2,000 and reported as
  censored, never imputed.

Grid: widths 3, 4, 5, 6, 8, 12, 16, 24, 32 × {tanh, relu, leaky_relu,
gelu, sin_family(1.5)} × seeds 100–199, depth 3, lr 1e-2, 2,000 steps,
linked tori tube radius 0.2, 1,000/class train and eval. Budget note:
4,500 runs, CPU; the width-32 cells are the cost driver and the reason
depth is fixed at 3.
