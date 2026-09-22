# Private note: the Ren–Lim released Hopf generator

**For Junyu. Not for the paper.** Nothing here appears in our submission; this is
a question about their released code, to be raised with them directly if at all.

Date: 2026-09-22. Source: `github.com/7pocheR/low_dimensional_topology`, cloned
and read directly.

---

## 1. The code

`exp_1_table2_relu_gelu_hopf/run_table2_trial.py`, class `ThickenedHopfLink`,
commented in-file as *"verbatim from RELU_VS_GELU.py"*. Byte-identical code
appears in `exp_2_table3_plain_relu_vs_resnet_hopf/run_table3_trial.py`.

```python
class ThickenedHopfLink:
    """Thickened Hopf link dataset (verbatim from RELU_VS_GELU.py)."""

    def __init__(self, n_points_per_curve=2000):
        self.n_points_per_curve = n_points_per_curve

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

    def generate_data(self, noise_std=0.01):
        t_values = np.linspace(0, 2 * np.pi, self.n_points_per_curve)
        s_values = np.linspace(0, 2 * np.pi, self.n_points_per_curve)
        curve1 = self.generate_curve1(t_values)
        curve2 = self.generate_curve2(s_values)
        curve1 += np.random.normal(0, noise_std, curve1.shape)
        curve2 += np.random.normal(0, noise_std, curve2.shape)
        ...
```

## 2. It does not match Appendix G.1

Appendix G.1 describes sampling `γ(t) + ε·n(t)` with `n(t)` a unit normal,
`ε ~ U(0, r)`, `r = 0.15`, plus `0.3 sin(100t)` oscillations.

Measured over 200,000 samples of `generate_curve1`:

| property | measured | consequence |
|---|---|---|
| distance from the unit-circle core | **min 0.4243, max 0.5000** | a tube of radius `r` fills `[0, r]`; this fills a shell, and **every point is determined by `t` alone** — there is no free radial parameter |
| any `ε ~ U(0, 0.15)` draw | **absent from the code** | the only randomness is isotropic Gaussian, `noise_std = 0.01` |
| displacement ⟂ tangent | `\|cos angle\|` mean **0.383**, max 0.9999 | **not** normal-aligned |
| displacement pattern | `+0.3 sin(100t)` added to **both** `x` and `y`; `z = 0.5 cos(100t)` | a fixed ambient pattern, not a frame-dependent normal |

**The `U(0, 0.15)` normal thickening is not implemented.** The released data are
1-D curves plus isotropic Gaussian noise.

## 3. `curve1` self-intersects — proof

Because `0.3 sin(100t)` is added to **both** `x` and `y`,

    x(t) − y(t) = cos t − sin t

is **independent of the oscillation**. A self-intersection needs
`x(t₁) = x(t₂)` and `y(t₁) = y(t₂)`, hence
`cos(t₁ + π/4) = cos(t₂ + π/4)`, hence

    t₂ = −t₁ − π/2   (mod 2π)

Restricting to that one-parameter family and minimising `‖γ(t₁) − γ(t₂)‖`
(2,000,001-point scan, then 60 bisection refinements):

| | |
|---|---|
| minimum self-distance | **7.8 × 10⁻¹⁵** |
| at | `t₁ = 2.5171476033`, `t₂ = 2.1952412700` |
| parameter separation | **0.3219** — not a near-diagonal artifact |

7.8e−15 is float64 epsilon on coordinates of order 1. The curve is **not
embedded**.

## 4. Numerical confirmation

An embedded curve has positive reach, so a refined sampling converges to a
positive minimum. This does not:

| sample points | 20,000 | 60,000 | 180,000 |
|---|---:|---:|---:|
| min self-distance (`\|Δt\|` beyond 50/150/450 steps) | 6.07e−04 | 2.42e−05 | **1.50e−05** |

Still shrinking, consistent with a true crossing.

## 5. Classification is unaffected

| quantity | value |
|---|---:|
| **between-class minimum distance** | **0.1275** |
| min self-distance within a component | ~0 (see above) |
| smooth, un-oscillated pair, same parametrisation | **1.0000** |

**The two classes remain 0.1275 apart**, so the classification task is well-posed
and their reported empirical results are not called into question by any of this.
The oscillation reduces between-class separation ~8× relative to the smooth pair
and destroys embeddedness of each component, but the *learning problem* is intact.

## 6. What we think this means, and what we are not claiming

- We are **not** claiming their results are wrong. The task they trained on is
  well-posed and their numbers stand.
- We **are** noting that the released generator differs from the appendix
  description, and that the objects it produces are not embedded curves — so the
  topological framing ("thickened Hopf link") does not describe the released data
  as stated.
- Our own replication assumed a *tube*, because the appendix describes one. Both
  readings we tried (displace the core; modulate the sampled radius) were attempts
  to interpret a thickening their code never performs.

**Suggested handling**: raise with the authors before any of it is written down
publicly. It may be that `RELU_VS_GELU.py` is an older script superseded by one
matching the appendix, in which case the question resolves without controversy.
