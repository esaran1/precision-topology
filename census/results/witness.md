# A semi-analytic width-3 witness (Part 2a)

A width-3 network that separates the linked tori, in which the entire
non-monotonic contribution is **one hand-designed fold layer**, placed at a
geometrically meaningful plane, with everything after it purely monotone.
Weights: `witness_weights.json`. Construction and training:
`src/witness.py`, deterministic from seed 2.

## The construction

    layer 1 (fixed, by hand):  (x, y, z) -> (|x - 1|, y, z)
    layers 2-3 (trained):      width-3 tanh
    head (trained):            linear

The fold plane `x = 1` contains component B's core circle (centred at
(1,0,0) in the xz-plane) and touches component A's core at its single
extreme point. A lies in `x ≤ 1` except for a 0.2-thick tube cap, so the
fold maps A almost affinely, while B — which straddles the plane
symmetrically — is folded 2-to-1 onto an arc. In folded coordinates
`(u, y, z)` with `u = |x−1|`: A's core becomes the circle `(u−1)² + y² = 1`
in the plane `z = 0`, and B's core becomes the half-circle arc
`u² + z² = 1, u ≥ 0, y = 0`, each point covered twice. The fold converts
*circle through circle* into *arc through disk*, which is no longer a link.

The fold layer is exactly expressible in the project's activation zoo as a
`pwl_family(−1)` (absolute value) layer — weights I, bias (−1, 4, 4), the
+4 offsets keeping coordinates two and three positive on the data where
`|·|` is the identity (pinned by a test).

## Verification

- **Sample criterion:** 0 errors on the standard 2,000-point eval set.
- **Dense verification: 0 errors on 2,000,000 fresh points** — five
  independent 200,000-point samples and one 1,000,000-point sample, six
  data seeds, worst logit margin over all samples **0.284** (per-sample
  minima 0.28–0.56). This is regional separation as far as sampling can
  establish it, not sample shattering.
- **Linking trace** (all stages are width 3; no projection anywhere):

  | stage | lk | residual | min distance |
  |---|---:|---:|---:|
  | input | −1 | 1.6e−06 | 1.000 |
  | after fold | **0** | 2.1e−15 | 1.000 |
  | after tanh 1 | 0 | 7.3e−13 | 0.124 |
  | after tanh 2 | 0 | 8.6e−14 | 1.146 |

  The hand fold alone takes the linking number from −1 to exactly 0; the
  monotone continuation never changes it again. (The folded B core is
  traversed twice; the Gauss integral over the parametrized loops remains
  defined since the curves stay disjoint.)

## Controls: the fold is the operative ingredient, and its placement matters

10 training seeds each, monotone tanh continuation of depth 2 throughout,
frozen first layer varied (initial experiment, offsets (·, y+4, z+4)):

| Frozen first layer | separations | best errors |
|---|---:|---:|
| fold at `x = 1` | **4/10** | **0** |
| fold at `z = 0` | 0/10 | 57 |
| same affine, no fold | 0/10 | 65 |

The no-fold control shows a frozen affine layer plus monotone continuation
cannot do it (consistent with 4,970 SGD runs). The `z = 0` fold shows not
any fold works: folding along z leaves the folded-B arc with both
endpoints on A's plane — one inside the hole, one outside, a staple
through the ring — while the `x = 1` fold leaves the arc's crease
endpoints off A's plane on opposite sides, crossing A's disk once. The
geometric account of *why* one staple resolves and the other does not is
an observation, not a theorem; what the controls establish is only that
placement matters.

In the final witness parametrization (no +4 offsets) the selection sweep
separated 2/30 seeds (seeds 2 and 14). Success *rate* varies with the
frozen map's constants; existence is what the witness claims.

## The cautionary specimen

Selection seed 14 also reached 0 errors on the 2,000-point eval set — and
failed dense verification with 2 errors at 50,000 points (min margin
−1.30). **Within a 30-seed selection, one of the two eval-perfect networks
was not regionally separating.** This is the Part 2c distinction made
concrete, and it is why every separating network in this project now gets
dense verification before being called a separation.

## A rounding attempt, failed

Rounding the continuation weights to one decimal broke the witness (42
errors on 200,000 points). The witness is therefore semi-analytic: the fold
is analytic and the continuation is trained, recorded exactly, and
reproducible deterministically. A fully hand-derived continuation was not
found; the trained one's structure (near-opposite head rows, two similar
rows in the last hidden layer) hints at a cleaner form but did not survive
naive simplification.

## What the witness establishes, and what it does not

Established:

- Separating width-3 networks exist whose non-monotonicity budget is
  exactly **one non-injective fold at one plane** — the existence claim no
  longer rests on SGD finding GELU solutions.
- The fold is where the linking dies, by direct measurement, with the rest
  of the network monotone — consistent with the fold-layer-1 observation
  in all 34 traced GELU runs, and now by construction rather than
  correlation.

Not established:

- Nothing here says a monotonic width-3 network *cannot* separate — that
  remains the (attacked, surviving) zero of Part 2b.
- The construction is for this link and parametrization; the fold plane
  was chosen using knowledge of the geometry. Corrugated or rotated links
  would need their own fold plane, and nothing here shows one always
  exists.

> **Count pointer (2026-08-22).** "4,970 SGD runs" reflects the total at
> the time of writing; the audited total is 5,570 (`CLAIMS.md`, T1).
