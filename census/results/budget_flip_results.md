# Part 1: a budget-dependent ranking flip on CIFAR-10 (interim, 33/120 runs)

Registered in `arrhenius_prediction.md` (P-flip, P-no-flip-GELU, P-margin)
and the flip-reporting/crossing-regime criteria, all fixed before the tanh
cells landed. Data: `budget_flip.csv`. Depth-8 CIFAR CNN, fresh seeds
100-107, n = 8 per cell. Permutation tests: 100,000 resamples, seed 0.

## The flip: registered, found, significant at both ends

| | tanh | ReLU | ordering | p |
|---|---|---|---|---|
| **2 epochs** (this sweep) | **3,425 ± 150** | 3,924 ± 180 | **tanh ahead by 499** | **0.00044** |
| **converged** (T38, 30-40 ep) | 2,592 ± 36 | 2,180 ± 45 | **ReLU ahead by 412** | **0.00039** |

Same task, architecture, optimizer and data; only the budget differs. Both
ends individually significant, as the criteria require. This is the pair
P-flip named in advance, for the stated reason (tanh underfits at
convergence: train error 744 vs ~180 per 10k, T38).

**P-no-flip-GELU also borne out**: GELU leads ReLU at 2 epochs
(3,173 vs 3,924, p = 0.00023) and at convergence (2,047 vs 2,180), so the
flip is specific to one pair rather than general instability of the setup.

## Where the crossing sits -- the vulnerability, reported as registered

| budget | mean test error | regime |
|---|---|---|
| 2 epochs | 31.7-39.2% | very early; a third of examples wrong |
| 5 epochs | GELU 21.7% (others pending) | near-converged quality (GELU converged: 20.5%) |

The crossing is **bounded between 2 and 30 epochs** by present data. The
5- and 12-epoch cells locate it, and they were still running at the
three-hour reporting deadline.

**Registered interpretation rule, restated so it cannot be chosen after the
fact:** a flip whose crossing exists only in the ~35%-error regime supports
the narrow claim that early-training rankings are unstable -- closer to known
than to new -- and must NOT be led with. It carries the paper only if the
crossing sits at 5 epochs or later (roughly <50% error, i.e. a regime people
publish from).

**Status: undetermined.** The flip exists and is significant; whether it
clears the regime bar depends on cells not yet in.

## Weight norms move with budget, as the account requires

GELU L2 norm: **29.0 at 2 epochs -> 41.8 at 5 epochs**. The account is that
reachability is set by how far the optimizer travels, so a ranking that
changes with budget should be accompanied by norm growth. It is. (Norms at
2 epochs are near-identical across activations, 28.0-29.0, so the 2-epoch
ordering is not itself a travel-distance effect.)

## Interim quantification

By what factor does the reported advantage change? tanh-vs-ReLU moves from
**-499 errors (tanh better)** to **+412 errors (ReLU better)** -- a swing of
911 errors, 9.1 percentage points, with the sign reversing. The GELU-vs-ReLU
gap moves from -750 to -133, a factor of **5.6x** shrinkage without changing
sign (P-margin's >1.5x threshold cleared).

## Not yet in

5-epoch tanh/ReLU, all of 12/30/60 epochs. 33 of 120 runs complete. The
sweep continues; this document is the three-hour snapshot and will be
superseded.
