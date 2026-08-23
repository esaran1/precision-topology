# Part 4: negative control on the construction pipeline

The F.2 pipeline (`src/fold_construction.py`) is structured so that its
precondition — a strict local extremum of the activation, located as a
sign change of f′ with usable slope on both branches — is an explicit
first step that either succeeds or raises a named failure. The attempt
on monotonic activations *is* the precondition check, and it fails at
step 1 in every case, for a provable rather than numerical reason:

| Activation | Step-1 outcome | Why, analytically |
|---|---|---|
| tanh | **fails** | f′ = sech² > 0 everywhere; no sign change exists |
| ReLU | **fails** | f′ ∈ {0, 1}; never negative (the flat branch is not a *strict* extremum, and its fold would be rank collapse, not a 2-to-1 fold) |
| leaky-ReLU | **fails** | f′ ∈ {0.01, 1} |
| `sin(0.98)` — just below threshold | **fails** | f′ = 1 + 0.98 cos x ≥ 0.02 > 0 everywhere — the same structural failure as tanh, exactly as required; nothing numerical about it |
| `pwl(+0.05)` | **fails** | f′ ∈ {0.05, 1} |
| `sin(1.001)` | fold at t\* = π ± arccos(1/a) | genuine 0.089-wide fold |
| `sin(1.02)` | fold at −3.3400 (max) | the exhibit's fold |
| `pwl(−0.05), (−0.22)` | fold at 0 (min) | the kink |
| GELU | fold at **−0.7518** (min) | true local minimum |
| SiLU (not engineered for) | fold at −1.2785 (min) | found unprompted |

The pipeline is not a general-purpose solution finder pointed at
non-monotonic activations: it cannot start on a monotonic activation,
and the boundary of "cannot start" is exactly the analytic monotonicity
threshold — `sin(0.98)` fails and `sin(1.001)` passes.

Two detector bugs were found and fixed by the control itself, both worth
recording: (1) GELU's far tail (x ≈ −7.6) produces sign flicker in f′ at
magnitudes ~1e−13 — underflow noise, not a fold — now excluded by a
slope floor requiring both branches to carry ≥1e−6 slope; (2) the first
version of that floor rejected `sin(1.001)`'s genuine 0.089-wide fold
because a fixed-width test window overshot the narrow decreasing branch
— the sign test is now local and the slope test respects branch
boundaries. A detector that rejects a genuine fold or accepts underflow
noise would have quietly corrupted Parts 1–3, which is what this control
was for.
