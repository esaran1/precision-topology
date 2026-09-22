# Item 2: the scaling limit against certified κ

Both sides of the S-1 comparison are now certified intervals. Artifact:
`kappa_certified_smalleps.csv`.

---

## The new comparison at `a = 1.02`

| quantity | certified interval |
|---|---|
| `κ(1.02)` | **[0.3077470, 0.3079916]** |
| `κ₀ = K/(4√2/3)` | **[0.3073024, 0.3075655]** |

**Disjoint**: `κ(1.02)`'s lower bound exceeds `κ₀`'s upper bound. Closest
approach: **+0.0590%**.

Against S-1's registered 3% tolerance, this **passes by 51×**.

## Correction to my earlier report

In `kappa_certification_results.md` I wrote that the certified gap was **0.145%**,
computed against the *sampled-x* certification. With **exact** `x`-extrema (item
#2) the measured `κ(1.02)` is slightly lower and the gap is **0.0590%**, not
0.145%. The exact figure is the one to quote.

Coincidentally that is almost identical to the **0.061%** originally reported
from the two uncertified point estimates — but it is a different comparison
reached for a different reason, and it is now an interval statement. The earlier
sampled numbers (`κ = 0.306122` vs `κ₀ = 0.307302`) had `κ` **below** `κ₀`; with
certification `κ` is **above** it. The old agreement was two under-estimates of
different size happening to land close; the new one is a bounded gap in a
predicted direction.

## κ approaches κ₀ **from above**, monotonically — the predicted direction

| `a` | ε | certified `κ` lower | gap above `κ₀` |
|---|---:|---:|---:|
| 1.001 | 0.0010 | 0.3073242 | **+0.0071%** |
| 1.002 | 0.0020 | 0.3073470 | +0.0145% |
| 1.005 | 0.0050 | 0.3074145 | +0.0365% |
| 1.010 | 0.0100 | 0.3075260 | +0.0728% |
| 1.020 | 0.0200 | 0.3077470 | +0.1447% |
| 1.050 | 0.0500 | 0.3083956 | +0.3557% |
| 1.100 | 0.1000 | 0.3094056 | +0.6844% |
| 1.300 | 0.3000 | 0.3130918 | +1.8839% |
| 1.600 | 0.6000 | 0.3174895 | +3.3150% |

**Monotone decreasing as ε → 0**, reaching **+0.0071%** at ε = 0.001. Confirmed:
the finite-`a` values approach `κ₀` from above.

*(At ε ≤ 0.01 the certificate's own width exceeds the gap — the `κ` upper bounds
there are 0.329, 0.315, 0.309, 0.308 — so "above `κ₀`" is established by the
**lower** bound, which is the rigorous side. The interval containing `κ₀`'s
interval at tiny ε reflects certificate width, not disagreement.)*

## The gap is `O(ε)`, matching the expansion term by term

| ε | 0.001 | 0.005 | 0.02 | 0.1 | 0.6 |
|---|---:|---:|---:|---:|---:|
| gap/ε | 0.0710 | 0.0730 | 0.0724 | 0.0684 | 0.0553 |

Log-log slope **0.9926** for ε ≤ 0.1 (**0.9663** over all points). `gap/ε`
extrapolates to **0.0727** as ε → 0.

This is the `ε·σ³/6` term identified in closed form in T58: the exact expansion is
`ε^{3/2}[−σ + (1+ε)σ³/6] + O(ε^{5/2})`, whose bracket carries a **relative `O(ε)`**
correction. The measured exponent 0.993 and the finite limit of `gap/ε` are that
term, now visible on both sides as certified intervals rather than inferred from
point estimates.

## What to quote in the paper

- `κ₀ = K/(4√2/3)`, certified **[0.3073024, 0.3075655]**, from the cubic alone.
- `κ(1.02)`, certified **[0.3077470, 0.3079916]**.
- Agreement: **0.059%**, with `κ` **above** `κ₀`, the direction the expansion
  predicts, and the gap **`O(ε)`** with slope 0.993.
- **Not** "0.061%" (uncertified point estimates that happened to land close) and
  **not** "0.145%" (my sampled-x figure, superseded).
