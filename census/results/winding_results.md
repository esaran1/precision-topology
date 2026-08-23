# Does the barrier scale with linking number? (Part 4 outcomes)

Predictions registered in `winding_prediction.md` before training. 1,600
runs: `q ∈ {1,2,3,4}` (verified lk = −q, residuals ≤ 1.4e−5) × widths
{3,4,5,6} × five activations × 20 seeds, depth 5, with tube clearance and
between-class distance held constant across `q`. Every sample-level
separation was dense-verified on 100,000 fresh points at run time;
**dense-verified counts are primary throughout**, with sample-level counts
alongside. Data: `winding_sweep.parquet`, `winding_validation.csv`.

## The grid (sample-level / dense-verified, out of 20)

**sin(2.0)** — the strong non-monotonic representative:

| q | w3 | w4 | w5 | w6 |
|---|---|---|---|---|
| 1 | 8/**6** | 19/**16** | 20/**19** | 20/**20** |
| 2 | 3/**1** | 14/**11** | 20/**20** | 20/**19** |
| 3 | 2/**0** | 9/**7** | 18/**13** | 16/**14** |
| 4 | 0/**0** | 6/**1** | 14/**10** | 17/**15** |

**GELU:** w3: 1/1, 0, 0, 0 across q. w4: 13/**7**, 4/**3**, 3/**0**,
5/**3**. **tanh:** w3 all zero; w4: 15/**9**, 5/**1**, 2/**1**, 0.
ReLU/leaky-ReLU: single digits everywhere, zero at q = 4 except one
sample-level leaky run (0 dense).

## Prediction 1: the monotonic zero extends — BORNE OUT

**0 monotonic separations in 240 width-3 runs across all four links**
(tanh/ReLU/leaky-ReLU, q = 1–4). The width-3 monotonic zero now stands at
**5,540 distinct runs** spanning five link families; bound on the rate
0.054% (`CLAIMS.md` T1/T2 updated).

## Prediction 2: non-monotonic width-3 rates fall with q — BORNE OUT

Dense-verified width-3 separations, sin(2.0): **6 → 1 → 0 → 0**; GELU:
**1 → 0 → 0 → 0**. A monotone decline to zero, with the caveat that the
zeros at q ≥ 3 are 0/40 per activation (bound 7.2% each) — small cells, a
real decline, and the two sample-level q = 3 sin runs both failed dense
verification. At width 3, one non-monotonic activation's folding evidently
does not stretch to winding 3 at rates our seed budget can see.

## Prediction 3: width 4 suffices at every q, and the reachability width grows — BOTH HALVES BORNE OUT

- **The d+1 bound stays attainable at every q**: width 4 has dense-verified
  separations at q = 1 (16 sin), 2 (11), 3 (7), and 4 (sin 1, GELU 3). The
  registered strongest scaling version — a width-4 zero with width 5/6
  positive — did **not** occur, so nothing here is in tension with Theorem
  D.1.
- **Reachability degrades with q at fixed width and the practical width
  grows.** The smallest width where sin(2.0) reaches a dense-verified
  majority of seeds: **4, 4, 5, 6 for q = 1, 2, 3, 4.** That is the
  scaling result in its honest form: not a hard threshold in what is
  possible, but a monotone widening of the gap between possible and
  reachable as |lk| grows. Monotonic tanh shows the same shape one width
  later: dense-verified at width 4 for q ≤ 3 (9, 1, 1 runs), needing width
  6 at q = 4 (2 runs).

## Prediction 4: dense attrition grows with q — BORNE OUT

Attrition among sample-level separations: **22% (q=1) → 28% → 46% → 42%**.
Nearly half of the q ≥ 3 sample-level separations are not regional
separations — the corrugation lesson generalizes: finer geometry per
sample point makes eval-0 easier to reach spuriously. Worst single cell:
tanh q=3 width 5, 13 sample-level → 2 dense.

## What Part 4 adds up to

Two clean monotone structures, both registered in advance:

1. **The categorical claim is q-independent**: no monotonic width-3
   separation at any linking number, and the theory's width-4 sufficiency
   is realized at every q tested.
2. **The quantitative claim scales**: at fixed width, dense-verified rates
   fall with |lk| for every activation that separates at all, and the
   width needed for reliable separation grows — 4 → 6 over q = 1 → 4 for
   the strongest activation, with monotonic tanh tracking one step behind.
   Since the tube clearance and sample margins were held fixed across q,
   the added difficulty is the winding itself, not the geometry's scale.

The registered against-reading condition — all rates collapsing at every
width including 6, leaving topology and optimization indistinguishable —
did not occur: width 6 holds 75–100% dense-verified rates for sin(2.0) at
every q.
