"""R = |w2| * G*(a) / 2 : the margin capacity of a trained run.

R is the largest logit margin a run's terminal weight scale could support
under optimal placement of (w1, b1).  The theorem |w2| >= 2m/G* rearranges
to R >= m, so R is the quantity the bound constrains.

Two things this module exists to keep straight, both of which caused errors
during the analysis:

1. `m` in the theorem is each solution's OWN achieved logit margin, not a
   fixed constant.  Assuming m = 1 inflates the requirement ~1.5x and
   produces apparent contradictions.
2. G is orientation-dependent.  The published form is the w2 > 0 gap; runs
   with w2 < 0 realize the mirrored arrangement.  Score each run in its own
   orientation or roughly half of them look like bound violations.
"""

from __future__ import annotations

import numpy as np
import torch

from .fold1d import INNER_MAX, OUTER_MIN, OUTER_MAX

INNER = torch.linspace(-INNER_MAX, INNER_MAX, 401, dtype=torch.float64)
_POS = torch.linspace(OUTER_MIN, OUTER_MAX, 201, dtype=torch.float64)
OUTER = torch.cat([_POS, -_POS])


def oriented_gap(f, w1: float, b1: float) -> float:
    """Achieved class gap, scored in whichever orientation is realizable."""

    with torch.no_grad():
        inner = f(w1 * INNER + b1)
        outer = f(w1 * OUTER + b1)
    plus = float(outer.min() - inner.max())
    minus = float(inner.min() - outer.max())
    return max(plus, minus)


def margin_capacity(w2_abs: float, gstar: float) -> float:
    """R = |w2| * G* / 2."""

    return float(w2_abs) * float(gstar) / 2.0


def separation_auc(values: np.ndarray, solved: np.ndarray) -> float:
    """P(solved run ranks above unsolved run); 1.0 is perfect separation."""

    pos = np.asarray(values)[np.asarray(solved, dtype=bool)]
    neg = np.asarray(values)[~np.asarray(solved, dtype=bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
