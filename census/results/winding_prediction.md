# Registered predictions: does the barrier scale with linking number? (Part 4)

Written after geometry validation and **before any training**. The links: a
plain ring (A) and a second ring winding `q` times around it (B), for
`q = 1, 2, 3, 4`, verified at linking number −q exactly (residuals ≤
1.4e−5), with tube clearance (0.150) and minimum between-class sample
distance (≈0.16) held constant across `q` so that, as far as the sampled
geometry goes, only the topology scales. Validation numbers are recorded by
the sweep before training and any invalid configuration aborts.

## Grid

`q ∈ {1,2,3,4}` × widths {3,4,5,6} × activations {tanh, ReLU, leaky-ReLU,
GELU, sin(2.0)} × 20 seeds, depth 5, our protocol. `sin(2.0)` is included
as a strong non-monotonic representative because GELU's dip (−0.129) is
near the practical onset and its separation rates are low; a scaling
question needs a non-monotonic activation with headroom. **Every separating
run is dense-verified immediately (100,000 fresh points)**, per Part 2c;
dense-verified rates are the primary outcome and sample-level rates are
reported alongside.

## Predictions

**P1 — the monotonic zero extends to every `q` at width 3.** 0 separations
for tanh/ReLU/leaky-ReLU at width 3 for all `q`. Any monotonic separation
overturns the central claim (and would be doubly surprising at `q > 1`).

**P2 — non-monotonic width-3 rates fall with `q`.** One fold undoes a Hopf
link; a `q`-winding link plausibly needs more geometric work than one fold
supplies. Expected shape: `sin(2.0)` separates at `q = 1` at roughly its
baseline-link rate, with the rate falling — possibly to zero within our
seed budget — by `q = 3` or `4`. GELU may already be at zero for `q ≥ 2`.

**P3 — width 4 suffices at every `q` (the theory's bound), but
reachability degrades with `q`.** Theorem D.1 says width d+1 = 4 is enough
for any link. The interesting outcome is the gap between possible and
reachable: we predict the width-4 separation rate falls with `q`, and the
width at which separation is *commonly reached* (say, majority of seeds
for the best activation) grows with `q`. If instead rates at width 4 are
flat in `q`, that is a clean vindication of the width bound with no
reachability gap; if width 4 produces zero separations at some `q` while
width 5 or 6 produces them, that is a reachability threshold growing with
linking number — the scaling-law outcome.

**P4 — dense verification will again cut the sample-level counts**, more
at higher `q` (finer geometry per sample point, the corrugation lesson).
This is why dense-verified rates are primary.

## What would count against which reading

- Monotonic separation anywhere: overturns the monotonic zero, full stop.
- Width-3 non-monotonic rates *flat or rising* in `q`: the folding account
  is wrong about what a fold costs — a single fold would evidently undo
  any winding, which would itself be a finding about the geometry of
  folds.
- Width-4 rates at zero for high `q` with width 5/6 nonzero: a
  width-reachability threshold that grows with `|lk|` — the strongest
  version of the scaling claim, and in tension with reading the d+1 bound
  as practically attainable.
- All rates collapsing at high `q` at every width including 6: the task
  got uniformly harder in a way widths don't buy back; then the sweep
  cannot distinguish topology cost from optimization cost, and says so.
