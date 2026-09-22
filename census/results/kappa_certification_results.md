# The certified κ interval: T50's width survives, its endpoints move up

Scored against `results/kappa_certification_prediction.md` (registered `1ed3788`,
before any certified bound was computed). Artifacts: `kappa_certified.csv`,
`kappa_boundary.csv`, `kappa_K_certified.csv`, `src/kappa_certify.py`.

**Naming**: this is the κ certification, **not** Block H — Block H is the
growth-rate dose-response, registered `3c1c0f4` and reported in
`blockH_rate_results.md`.

---

## The certificate

`Ĝ(a)` is a **supremum** over placements, so a finite grid can only
**under-estimate** it. With `L = (1+a)·max(1, max|x|)` the Lipschitz constant of
`G` in `(w₁, b₁)` — from `|f_a'| = |1 + a cos t| ≤ 1+a` — a grid of spacing `h`
gives

    Ĝ_grid  ≤  Ĝ  ≤  Ĝ_grid + L·h/2

the same device as T57's `solves()` certificate, applied to a supremum instead of
a sign. `h` is refined by zooming on the running argmax until `L·h/2 < 0.001·Ĝ`.

## C-1: the certificate closes — PASS

| `a` | `L` | `h` | `Ĝ` lower | `Ĝ` upper | rel. width | `κ` lower | `κ` upper |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.02 | 4.040 | 2.6e-07 | 0.0016267 | 0.0016273 | **0.032%** | 0.307747 | 0.307844 |
| 1.05 | 4.100 | 1.3e-06 | 0.0063601 | 0.0063627 | 0.041% | 0.308396 | 0.308523 |
| 1.10 | 4.200 | 6.4e-06 | 0.0176733 | 0.0176867 | 0.076% | 0.309429 | 0.309664 |
| 1.25 | 4.500 | 6.4e-06 | 0.0665031 | 0.0665175 | 0.022% | 0.312224 | 0.312292 |
| 1.30 | 4.600 | 3.2e-05 | 0.0860982 | 0.0861718 | 0.086% | 0.313079 | 0.313346 |
| 1.35 | 4.700 | 3.2e-05 | 0.1069138 | 0.1069890 | 0.070% | 0.313910 | 0.314131 |
| 1.40 | 4.800 | 3.2e-05 | 0.1287721 | 0.1288489 | 0.060% | 0.314688 | 0.314876 |
| 1.45 | 4.900 | 3.2e-05 | 0.1515399 | 0.1516183 | 0.052% | 0.315424 | 0.315587 |
| 1.50 | 5.000 | 3.2e-05 | 0.1751248 | 0.1752048 | 0.046% | 0.316149 | 0.316294 |
| 1.60 | 5.200 | 3.2e-05 | 0.2243666 | 0.2244498 | 0.037% | 0.317499 | 0.317616 |

Every width is **0.02% to 0.09%**, against a registered 0.2%. Reached with
`h ≥ 2.6e-07`, above the 1e-6 floor at which float64 window sums would dominate.

## C-2: FAILS on containment, passes on magnitude — and T50's substance survives

| | interval | width |
|---|---|---:|
| **certified**, `a ≤ 1.60` | **[0.307747, 0.317616]** | **3.207%** |
| T50 as reported | [0.305440, 0.315430] | 3.271% |

Registered: the certified interval **contains** T50's and endpoints move **< 1%**.

- **Magnitude criterion PASSES**: lower endpoint moves **+0.755%**, upper
  **+0.693%**, both under 1%.
- **Containment criterion FAILS**: T50's lower endpoint 0.30544 sits **below** the
  certified minimum 0.307747, so the certified interval does not contain it.

**Both endpoints moved up, which is the only direction possible** — the
registration said as much in advance, and a downward move would have indicated a
bug. The grid-600 search T50 used under-estimated κ at **every** `a`:

| `a` | 1.02 | 1.10 | 1.30 | 1.40 | 1.60 |
|---|---:|---:|---:|---:|---:|
| grid-600 `κ` | 0.305640 | 0.307349 | 0.311933 | 0.314275 | 0.315432 |
| certified lower | 0.307747 | 0.309429 | 0.313079 | 0.314688 | 0.317499 |
| under-estimate | **0.685%** | 0.672% | 0.366% | 0.131% | **0.651%** |

**What this changes and what it does not.** T50's headline is that κ is *constant
in `a` to 3.2%*, and that is what the theorem chain consumes — the **width**, not
the endpoints. Certified width **3.207%** against reported **3.271%**: the claim
survives, very slightly tightened. What must be corrected is the **interval's
location**: it is `[0.3077, 0.3176]`, not `[0.3054, 0.3154]`.

This is a correction to T50, not a refutation. It is also a vindication of doing
the certification: the under-estimate is real, systematic, and of the same order
(0.1–0.7%) as the agreements the paper elsewhere reports as tight — including
T58's κ₀ match, re-examined below.

## C-3: the compactness reduction is sound — PASS

On the boundary shell `|w₁| = 3` or `|b₁ − π| = 6`:

| `a` | boundary max `G` | interior max | verdict |
|---|---:|---:|---|
| 1.02 | −0.0000000 | 0.0016267 | **strictly below** |
| 1.30 | −0.0000000 | 0.0860982 | **strictly below** |
| 1.60 | −0.0000000 | 0.2243666 | **strictly below** |

The boundary gap is non-positive at every `a` tested — no placement on the shell
separates the classes at all — so the domain cut cannot have excluded the optimum.

## C-4: `K` and `κ₀` certified — PASS

On `h(σ) = −σ + σ³/6`, with `|h'| = |−1 + σ²/2|` bounding the Lipschitz constant:

    K   ∈ [0.579454977, 0.579950977]   rel. width 0.086%
    κ₀  ∈ [0.3073024,   0.3075655]

Registered: width below 0.1% (**0.086%, passes**) and containing the reported
value. **`K = 0.579454926` and `κ₀ = 0.307302` both sit at the certified lower
endpoint**, to 7 and 6 digits respectively — the nested-refinement search had
already found the supremum to that precision.

## Consequence for T58's κ₀ agreement

T58 reported `κ₀ = 0.307302` against measured `κ(1.02) = 0.307483`, a **0.061%**
agreement. Under certification the measured value becomes
`κ(1.02) ∈ [0.307747, 0.307844]` and `κ₀ ∈ [0.3073024, 0.3075655]`.

**The two intervals are disjoint**, by about **0.06%**: certified `κ(1.02)`
exceeds certified `κ₀` by 0.145% at the closest approach. The 0.061% figure was a
coincidence of two under-estimates of different size.

This does **not** overturn S-1, whose registered tolerance was **3%** — the
agreement is 0.145% certified, still 20× inside tolerance. But the paper should
quote **0.15% certified** rather than 0.061% sampled, and should not present the
match as exact. The honest statement: `κ₀` predicts `κ(1.02)` to about **0.15%**,
with both quantities now interval-valued rather than point estimates, and the
small residual is the finite-ε correction the scaling limit predicts.

## Net

| prediction | verdict |
|---|---|
| **C-1** certificate width < 0.2% | **PASS**, 0.02–0.09% |
| **C-2** certified interval contains T50's, endpoints move < 1% | **magnitude PASSES (0.69–0.76%), containment FAILS** — T50's location corrected to [0.3077, 0.3176]; its 3.2% **width** survives at 3.207% |
| **C-3** boundary below interior | **PASS**, boundary gap ≤ 0 at every `a` |
| **C-4** `K` certified, width < 0.1% | **PASS**, 0.086%, reported value at the lower endpoint |

κ is no longer a sampled estimate. The drift T50 reports is now bounded away from
being a grid artifact — which after Block G's G4 episode, where a fixed grid
mis-measured `Ĝ` by 18.8%, was a live concern rather than a formality.
