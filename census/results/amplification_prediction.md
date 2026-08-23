# Registered predictions: the amplification account (2a–2b)

Written before 2a and 2b run.

## The account under test

A3 found no activation scalar that aligns the two families' findability
onsets. The construction suggests why: the operative quantity may be the
**amplification required to make the fold usable** — a joint property of
activation shape and reachable weight scale, invisible to any pure
activation scalar. At `a = 1.02` the fold is 5.3e−3 deep and the
construction needed ~600×.

## 2a: required amplification

Definition, fixed in advance: `A_req(activation, parameter)` is the
smallest amplification factor at which the F.2 construction (generalized
recipe of `src/fold_construction.py` / `src/basin.py`, fixed shrink
policy, 10 continuation seeds) produces at least one dense-verified
separation, located by bisection on a log grid between 1 and 20,000.

**Prediction 2a-match:** the two families' findability onsets (A ≈ 1.085,
B ≈ −0.22) have `A_req` of the same order, while every activation scalar
differed by 1–2 orders — making required amplification the matched unit
A3 could not find.

**Named failure:** if `A_req` at the two onsets differs by an order of
magnitude, the account fails and the gap is family-specific for reasons
still unidentified. A second failure mode is specific to Family B: its
fold is a sheared tent (slope ratio 1/|α|), and the continuation may be
untrainable at any amplification for small |α| — in that case `A_req` is
undefined below some |α| and the account must say so rather than fit a
number.

## 2b: achievable amplification and the intervention

**Prediction 2b-tail:** the product of layer-wise weight scales that
standard training reaches (measured as the distribution of per-layer
weight norms across trained runs at each `a`) sits far below `A_req` at
`a = 1.02`, and the two distributions approach each other near the onset.

**Prediction 2b-intervention (the decisive one):** initializing training
with the first layers scaled toward the required amplification — or
equivalently training from an initialization whose fold-coordinate
pathway already carries `A_req`-scale weights — makes separation appear
at `a = 1.02` under otherwise standard SGD, where 400 standard restarts
found nothing. Concretely: rerun 40 seeds at `a = 1.02` with the
first-layer/second-layer weights initialized at the construction's scale
pattern but random directions; any dense-verified separation confirms
the account. Zero separations in 40 runs would be evidence against it —
reported unsoftened, as with every registered failure so far.
