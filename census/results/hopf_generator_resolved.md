# Item 2: the corrugated-Hopf parametrisation, resolved against their code

Unblocked by the repository reference. Cloned
`github.com/7pocheR/low_dimensional_topology` and read the generator directly.

**Their code implements NEITHER of our two readings — and not the appendix's own
description either.**

---

## What their code actually does

`exp_1_table2_relu_gelu_hopf/run_table2_trial.py`, class `ThickenedHopfLink`
(the file comments it as "verbatim from RELU_VS_GELU.py"); identical code in
`exp_2_table3_plain_relu_vs_resnet_hopf/run_table3_trial.py`:

```python
def generate_curve1(self, t_values):
    x = np.cos(t_values) + 0.3 * np.sin(100 * t_values)
    y = np.sin(t_values) + 0.3 * np.sin(100 * t_values)
    z = 0.5 * np.cos(100 * t_values)
    return np.column_stack([x, y, z])

def generate_curve2(self, s_values):
    x = 1 + np.cos(s_values) + 0.3 * np.sin(100 * s_values)
    y = 0.5 * np.cos(100 * s_values)
    z = np.sin(s_values) + 0.3 * np.sin(100 * s_values)
    return np.column_stack([x, y, z])
```

`generate_data` then adds `np.random.normal(0, noise_std)` with
`noise_std = 0.01`.

## Measured from 200,000 samples

| property | measured | implication |
|---|---|---|
| distance from the unit-circle core | **min 0.4243, max 0.5000** | a tube of radius `r` would fill `[0, r]`; this fills a shell. **Every point is determined by `t` alone** — there is no free radial parameter. |
| any `ε ~ U(0, 0.15)` draw | **none in the code** | the only randomness is **isotropic Gaussian** `σ = 0.01`, not a uniform radial offset |
| displacement ⟂ tangent | `\|cos angle\|` mean **0.383**, max **0.9999** | **not** normal-aligned |
| displacement pattern | `x` and `y` both get `+0.3 sin(100t)`; `z` gets `0.5 cos(100t)` | a **fixed ambient pattern**, not a frame-dependent normal |

## Verdict on the three candidate readings

| reading | what it says | does their code do it? |
|---|---|---|
| **Appendix G.1 as written** | sample `γ(t) + ε·n(t)`, `n` a unit normal, `ε ~ U(0, 0.15)`, plus `0.3 sin(100t)` | **NO** — no normal, no `ε`, no `0.15` |
| **our Reading A** | displace the **core**, then thicken | **NO** — there is no thickening step at all |
| **our Reading B** | modulate the **sampled radius** | **NO** — same reason |
| **what the code does** | add `0.3 sin(100t)` **directly to ambient x, y** and set `z = 0.5 cos(100t)`, then add Gaussian noise `σ = 0.01` | — |

**The `0.15` radial thickening described in Appendix G.1 is absent from their
released generator.** Their published data are **1-D curves plus isotropic
Gaussian noise**, not thickened tubes. Both of our readings assumed a tube,
because the appendix describes one.

## A further structural finding: their curve self-intersects

The oscillated curve is **not embedded**. Proven analytically, not just measured.

**The argument.** For `curve1`,

    x(t) − y(t) = cos t − sin t

**independent of the oscillation**, since `+0.3 sin(100t)` is added to *both*
`x` and `y`. So any self-intersection needs `cos t₁ − sin t₁ = cos t₂ − sin t₂`,
i.e. `cos(t₁ + π/4) = cos(t₂ + π/4)`, giving the one-parameter family

    t₂ = −t₁ − π/2   (mod 2π)

Searching that family and refining 60×:

| | |
|---|---|
| minimum self-distance | **7.8e−15** |
| at | `t₁ = 2.5171476033`, `t₂ = 2.1952412700` |
| parameter separation | **0.3219** — not a near-diagonal artifact |

**7.8e−15 is float64 epsilon on coordinates of order 1.** This is a genuine
self-intersection.

Consistent with the numerical refinement, which **fails to converge to a positive
reach**: minimum self-distance **6.07e−04 → 2.42e−05 → 1.50e−05** at 20,000 /
60,000 / 180,000 sample points. An embedded curve has positive reach and the
measured minimum would converge; here it keeps shrinking.

For scale: the between-component separation is **0.1275**, and the smooth
un-oscillated pair has separation **1.0000** — so the oscillation reduces
between-class separation ~8× *and* destroys embeddedness of each component.

**Why this matters.** A self-intersecting curve is not a knot, and the pair is not
a link in the usual sense — the linking number of the *cores* is still defined,
but each component fails to be an embedded `S¹`. The sampled dataset is
unaffected as a **classification problem** (the two classes stay 0.1275 apart, so
it is well-posed), but the topological description of the data does not hold as
stated.

Reported neutrally as a property of the released generator. It is not a claim
that their empirical results are wrong — those are about a classification task
that remains well-posed.

## What this settles for §2

1. **Our replication used a different object from theirs, in a way neither
   reading would have fixed.** Reading A and Reading B were both attempts to
   interpret a *thickening* that their code does not perform.
2. **The §2 parametrisation question is answered**: state what their generator
   does, in their own code's terms, and note that it differs from their
   Appendix G.1 description — the appendix specifies a `U(0, 0.15)` normal offset
   that the released code does not implement.
3. **`corrugation_results.md`'s Reading A / Reading B comparison remains valid as
   a study of two ways to corrugate a tube**, but neither is "their"
   parametrisation and the note should say so.
4. This is a **discrepancy between their text and their code**, reported as such.
   It is not a claim that either is wrong, and the paper should state it
   neutrally: the released generator is what produced their published numbers.
