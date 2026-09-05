# §7 empirical content: architecture comparisons are budget-dependent

Data: `budget_flip.csv` (depth-8 CIFAR-10 CNN, fresh seeds 100-107, n = 8 per
cell), converged values from `cifar_convergence.csv` (T38, n = 10).
Permutation tests: 100,000 resamples, seed 0, two-sided. All predictions
registered in `arrhenius_prediction.md` before the cells landed.

## Main result: the magnitude of a reported advantage varies 5.65x with budget

GELU over ReLU, the comparison whose *sign* is stable:

| budget | test error rate | GELU | ReLU | **GELU advantage** | p |
|---|---|---|---|---|---|
| 2 epochs | 35.1% | 3,173.1 | 3,923.5 | **750.4** | 0.00019 |
| 5 epochs | 24.7% | 2,167.0 | 2,585.4 | **418.4** | 0.00023 |
| converged | ~21% | 2,047.0 | 2,179.7 | **132.7** | 0.00003 |

**5.65x** between the largest and smallest advantage. Significant at **every**
budget, monotone in budget, no sign change, and the 5-epoch and converged
cells sit at 21-25% error -- regimes people publish from.

This is the cleanest form of the methodological claim. A practitioner reports
an advantage and its size at whatever budget they ran; the size of that
advantage is a property of the budget as much as of the architectures.
Nothing here depends on a degenerate regime: both ends are respectable.

**Registered as P-margin (>1.5x change) before measurement; observed 5.65x.**

## Extreme case: at short budgets the sign reverses

tanh versus ReLU:

| budget | error rate | tanh | ReLU | ordering | p |
|---|---|---|---|---|---|
| 2 epochs | 35.1% | 3,424.5 | 3,923.5 | **tanh ahead by 499** | **0.0004** |
| 5 epochs | 24.7% | 2,653.9 | 2,585.4 | ReLU ahead by 68 | 0.187 (n.s.) |
| converged | ~23% | 2,591.8 | 2,179.7 | **ReLU ahead by 412** | **0.0004** |

Both ends individually significant, as the pre-registered flip criteria
require. This was the pair P-flip named in advance, for the reason given in
advance: tanh underfits at convergence (train error 744 vs ~180 per 10k, T38).

**Scope, stated as registered before the data landed:** the crossing lies
between 2 and 5 epochs, i.e. **above ~25% test error**. A flip whose crossing
sits only in the very-early regime supports the narrow claim that
early-training rankings are unstable -- closer to known than to new. It is
reported here as the extreme case of the margin-shift, **not** as the
headline. By 5 epochs the pair is already statistically tied.

**GELU versus ReLU never flips** (p = 0.00019 and 0.00023 at 2 and 5 epochs,
same direction), so the reversal is pair-specific rather than general
instability of the setup.

## Weight norms grow with budget, as the account requires

Mean L2 weight norm:

| budget | GELU | ReLU | tanh |
|---|---|---|---|
| 2 epochs | 29.0 | 28.0 | 28.1 |
| 5 epochs | 41.8 | 38.0 | 40.2 |
| 12 epochs | 60.3 | (pending) | (pending) |

Norms are near-identical across activations at 2 epochs (28.0-29.0), so the
early ranking is not itself a travel-distance effect; what changes with
budget is how far training has travelled. This is the same quantity that
governs the controlled result.

## What this licenses, and what it does not

**Licensed**: at least one standard architecture comparison, on a standard
dataset, has an advantage whose magnitude varies 5.65x with training budget
in the publishable regime, and one pair whose sign reverses at short budget.
Comparisons conducted at a fixed budget measure reachability at that budget.

**Not licensed**: that all architecture comparisons are budget-dependent, or
that any specific published comparison is wrong. One architecture family, one
dataset, one optimizer, no augmentation, single learning rate.

**The controlled result remains the quantitative core** (§6): the onset moves
1.60 -> 1.03 across a 64x budget range with exponent -0.7340 (R^2 = 0.990,
6/6 bracketed cells) -- and, per the family-B double dissociation, does so
**only where the required scale diverges** (`family_b_budget.md`).
