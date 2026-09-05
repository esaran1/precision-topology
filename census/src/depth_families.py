"""2a: activation families whose fold depth vanishes at different rates.

Family A (committed): f(x) = x + a*sin(x).  At a = 1+eps the critical points
satisfy f'(x) = 1 + a*cos(x) = 0, i.e. cos(x) = -1/a.  Writing x = pi -+ delta
with cos(pi -+ delta) = -cos(delta) ~ -(1 - delta^2/2), the condition gives
delta ~ sqrt(2*eps), and integrating f' across the fold gives depth
D ~ (8/3)*eps^{3/2}.  So beta_A = 3/2.

General construction.  Let f(x) = x - a*h(x) with h'(x) having a zero of
order (k-1) at x = 0, i.e. h'(x) = 1 - |x|^{k-1}-like.  Then near a = 1:
   f'(x) = 1 - a*h'(x)
If h'(x) = 1 - c*|x|^q near 0, then f'(x) = (1-a) + a*c*|x|^q, which vanishes
at |x| = ((a-1)/(a*c))^{1/q} ~ eps^{1/q}, and
   D = 2 * int_0^{x*} |f'| dx ~ eps^{1 + 1/q}
so **beta = 1 + 1/q**.

  q = 2  ->  beta = 3/2   (family A's local form: h' = cos, 1 - x^2/2)
  q = 1  ->  beta = 2     (kink: h' = 1 - |x|)
  q = 4  ->  beta = 5/4   (flatter: h' = 1 - x^4)
  q = 2/3 -> beta = 5/2   (sharper than a kink: h' = 1 - |x|^{2/3})

Implemented via f'(x) = (1-a) + a*|x|^q on |x| <= 1, matched smoothly outside,
integrated in closed form so f itself is exact.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def make_family(q: float):
    """Return (f, D_exact) for the family with f'(x) = (1-a) + a*|x|^q near 0.

    f(x) = (1-a)*x + a*sign(x)*|x|^{q+1}/(q+1)   for |x| <= 1
    f(x) = f(+-1) + (1-a+a)*(x -+ 1) = f(+-1) + (x -+ 1)  outside (slope 1)
    beta = 1 + 1/q.
    """

    def f(x, a):
        ax = torch.abs(x)
        inner = (1.0 - a) * x + a * torch.sign(x) * ax.pow(q + 1) / (q + 1)
        edge = (1.0 - a) + a / (q + 1)          # f(1)
        outer = torch.sign(x) * edge + (x - torch.sign(x))
        return torch.where(ax <= 1.0, inner, outer)

    def depth_exact(a):
        """Fold depth = f(local max) - f(local min), computed exactly."""
        if a <= 1.0:
            return 0.0
        xs = ((a - 1.0) / a) ** (1.0 / q)        # critical point
        if xs > 1.0:
            return float("nan")
        # f(-xs) - f(+xs) for the descending fold
        val = (1.0 - a) * xs + a * xs ** (q + 1) / (q + 1)
        return float(-2.0 * val)                 # = 2*(a-1)*xs - 2a*xs^{q+1}/(q+1)

    return f, depth_exact


FAMILIES = {"q1_beta2": 1.0, "q2_beta1.5": 2.0, "q4_beta1.25": 4.0,
            "q0.667_beta2.5": 2.0 / 3.0}


def verify(q: float, name: str) -> dict:
    """Numerically confirm beta = 1 + 1/q by fitting D(eps) ~ eps^beta."""
    _, depth = make_family(q)
    eps = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    D = np.array([depth(1.0 + e) for e in eps])
    good = np.isfinite(D) & (D > 0)
    beta_fit = np.polyfit(np.log(eps[good]), np.log(D[good]), 1)[0]
    beta_theory = 1.0 + 1.0 / q
    return {"family": name, "q": q, "beta_theory": beta_theory,
            "beta_measured": beta_fit,
            "error": abs(beta_fit - beta_theory),
            "D_at_eps_0.01": depth(1.01)}


if __name__ == "__main__":
    print("beta verification: analytic 1 + 1/q vs numerical fit of D(eps)\n")
    rows = []
    for name, q in sorted(FAMILIES.items(), key=lambda kv: kv[1]):
        r = verify(q, name)
        rows.append(r)
        print(f"  {name:16s} q={q:6.3f}  beta_theory={r['beta_theory']:.4f}  "
              f"beta_measured={r['beta_measured']:.4f}  err={r['error']:.2e}  "
              f"D(eps=.01)={r['D_at_eps_0.01']:.3e}")
    import pandas as pd
    pd.DataFrame(rows).to_csv("results/depth_families_verification.csv", index=False)
    print("\nfamily A (sin) reference: beta = 3/2, matched by q = 2")
