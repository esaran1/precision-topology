"""§16 (local to global): the remainder lemma, the link arithmetic and the gap enclosure on constructed cases, and the
committed certificate's conclusions.  The Krawczyk sweep itself (7 minutes) is not re-run here."""

import json
import math
from pathlib import Path

import numpy as np
from mpmath import iv, mp

from src import local_global as lg

ROOT = Path(__file__).resolve().parents[1]


def _phi(sig, eps):
    u = math.sqrt(eps) * sig
    return (u - (1 + eps) * math.sin(u)) / eps ** 1.5


def test_lemma_G1_remainder_bounds_hold():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        eps = 10 ** rng.uniform(-6, -1)
        sig = rng.uniform(-30, 30)
        h = -sig + sig ** 3 / 6
        assert abs(_phi(sig, eps) - h) <= eps * lg.m0(abs(sig)) * (1 + 1e-6) + 1e-9 * abs(h)


def test_link_arithmetic_passes_at_eps0_and_fails_beyond():
    c = lg.limit_inputs()
    ok = lg.link_conditions(1.4e-12, c)
    assert all(v["slack"] > 0 for v in ok.values())
    bad = lg.link_conditions(3e-12, c)                       # constructed fail: past the outer-link limit 1.79e-12
    assert bad["outer"]["slack"] < 0


def test_gap_enclosure_signs_on_constructed_points():
    ph = lg.Inflated(1e-12)
    pt = lambda v: iv.mpf([mp.mpf(v) - 1e-9, mp.mpf(v) + 1e-9])
    lo, hi = lg.gap_bounds(pt(0.0), pt(0.0), ph)             # constant unit: G = 0 exactly
    assert lo <= 0 <= hi
    lo, hi = lg.gap_bounds(pt(1.6056), pt(1.2042), ph)       # near K's vertex: G0 ~ 0.5795 > 0
    assert lo > 0.5 and hi < 0.6
    lo, hi = lg.gap_bounds(pt(3.0), pt(0.0), ph)             # a steep ramp through 0: unplaced
    assert hi < 0


def test_committed_certificate():
    d = json.loads((ROOT / "results/local_global.json").read_text())
    assert d["theorem_G_holds"] and d["EPS0"] == 1.4e-12
    g = d["gap"]
    assert g["covered"] and g["all_krawczyk_in_U"] and g["unique_sign_change"]
    assert d["switch_box"]["ok"] and d["switch_box"]["active_pair_ok"]
