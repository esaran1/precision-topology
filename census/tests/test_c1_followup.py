"""Track 3 c1 follow-up: the scoring rule and validity check on constructed pass / fail / invalid cases (written and
run before any added certificate is computed), the registered design, and the frozen inputs."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import c1_followup as cf

ROOT = Path(__file__).resolve().parents[1]
C1 = (0.2852299861201777, 0.2852302997512185)
RINF = (0.19859263779939, 0.19859265097239762)
ORIG_EPS = (0.01, 0.02, 0.03, 0.04)


def _law(eps, c1, c2=-0.13, c3=-0.016):
    r = 0.5 * sum(RINF)
    e = np.asarray(eps, float)
    return r * (1 + c1 * e + c2 * e ** 2 + c3 * e ** 3)


def _intervals(eps, c1, rel_w, pos=0.5):
    R = _law(eps, c1)
    return R * (1 - rel_w * pos), R * (1 + rel_w * (1 - pos))


def _score(eps, lo, hi):
    f = cf.feasible(eps, lo, hi, *RINF)
    return f, cf.verdict(f[0], f[1], *C1)


def test_verdict_rule_cases():
    assert cf.verdict(0.26, 0.30, *C1) == "PASS"
    assert cf.verdict(0.2852301, 0.2852301, *C1) == "PASS"
    assert cf.verdict(0.29, 0.33, *C1) == "FAIL"                         # above
    assert cf.verdict(0.20, 0.28, *C1) == "FAIL"                         # below
    assert cf.verdict(np.nan, np.nan, *C1) == "FAIL (no two-term law fits)"
    assert cf.verdict(0.22, 0.33, *C1) == "INVALID (feasible width > 0.1)"  # contains c1 but too wide: not a pass
    assert cf.verdict(0.30, 0.41, *C1) == "INVALID (feasible width > 0.1)"
    assert cf.verdict(0.25, 0.35, *C1) == "PASS"                         # width exactly 0.1 is valid


def test_constructed_pass():
    eps = list(ORIG_EPS) + [a - 1 for a in cf.A_ADD]
    lo, hi = _intervals(eps, 0.28523, 6.25e-4, pos=0.4)
    (f_lo, f_hi), v = _score(eps, lo, hi)
    assert v == "PASS" and f_lo <= 0.28523 <= f_hi and f_hi - f_lo < 0.05


def test_constructed_fail_wrong_c1():
    eps = list(ORIG_EPS) + [a - 1 for a in cf.A_ADD]
    lo, hi = _intervals(eps, 0.36, 6.25e-4)
    (f_lo, f_hi), v = _score(eps, lo, hi)
    assert v.startswith("FAIL") and (not np.isfinite(f_lo) or f_lo > C1[1])


def test_constructed_fail_no_law():
    eps = list(ORIG_EPS) + [a - 1 for a in cf.A_ADD]
    lo, hi = _intervals(eps, 0.28523, 6.25e-4)
    lo, hi = lo.copy(), hi.copy()
    lo[-1] *= 1.01; hi[-1] *= 1.01                                         # one bracket off any two-term law
    (f_lo, f_hi), v = _score(eps, lo, hi)
    assert v == "FAIL (no two-term law fits)" and not np.isfinite(f_lo)


def test_constructed_invalid_wide():
    eps = list(ORIG_EPS) + [a - 1 for a in cf.A_ADD]
    lo, hi = _intervals(eps, 0.28523, 1e-2)
    (f_lo, f_hi), v = _score(eps, lo, hi)
    assert f_hi - f_lo > 0.1 and v == "INVALID (feasible width > 0.1)"


def test_original_four_alone_reproduce_registered_inconclusive_width():
    t = pd.read_csv(ROOT / "results/first_order_finite.csv", float_precision="round_trip")
    f = cf.feasible(t.eps.values, t.R_lo.values, t.R_hi.values, *RINF)
    assert abs((f[1] - f[0]) - 0.10498428399595014) < 1e-9
    assert cf.verdict(f[0], f[1], *C1) == "INVALID (feasible width > 0.1)"


def test_registered_design_and_timing_a_excluded():
    assert cf.A_ADD == (1.08, 1.09, 1.10, 1.11, 1.12)
    assert 1.055 not in cf.A_ADD
    assert cf.VALID_WIDTH == 0.1


FROZEN_SHA = {
    "results/first_order_finite.csv": "ff68d364027fbdd1aee5a6f67edf7c76d1a18ac602e6edaf4cf2b10ef5b9c5bb",
    "results/first_order_c1.csv": "e7fb2f5a1373b6182848c8e1b36ad68dbd72dcaa2a83447a7da45c6b154c2397",
    "results/c1_followup_model.json": "f8bb88aa1e5101cf5111b6870457a540069e1f3c919d1a774a64d578d6ee0926",
    "results/c1_followup_timing_a1.0550.json": "5bf17f20d0cbeff242f2cebc1feb2a02e573e8496b19785555ab531930849f60",
}


@pytest.mark.parametrize("path", sorted(FROZEN_SHA))
def test_frozen_inputs(path):
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == FROZEN_SHA[path]
    assert path in cf.FROZEN
