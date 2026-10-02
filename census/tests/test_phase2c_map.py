"""Phase 2C (src/phase2c_map.py; DESCRIPTIVE): constructed cases for the summary functions, the fold-rate fit, the
pooling and the rendering; the committed map regenerates from the committed sources."""

import json
import math

import numpy as np
import pytest

from src import phase2c_map as M


def test_summarize_constructed_quantiles_and_fractions():
    v = [0.01, 0.02, 0.05, 0.1, 0.2, float("nan"), float("inf")]
    s = M.summarize(v)
    assert s["n"] == 5
    assert (s["min"], s["max"], s["median"]) == (0.01, 0.2, 0.05)
    assert s["q10"] == pytest.approx(0.014)              # linear interpolation between 0.01 and 0.02
    assert s["q90"] == pytest.approx(0.16)
    assert s["n_abs_le_0.1"] == 4 and s["frac_abs_le_0.1"] == pytest.approx(0.8)    # 0.1 itself counts (≤)
    assert s["n_abs_le_0.02"] == 2 and s["frac_abs_le_0.02"] == pytest.approx(0.4)
    assert s["n_negative"] == 0 and not M.signed_differs(s)
    assert s["abs_median"] == s["median"]


def test_summarize_signed_and_abs_differ_with_negatives():
    s = M.summarize([-0.5, -0.01, 0.03, 0.2])
    assert s["n_negative"] == 2 and M.signed_differs(s)
    assert s["min"] == -0.5 and s["abs_min"] == 0.01 and s["abs_max"] == 0.5
    assert s["n_abs_le_0.1"] == 2 and s["n_abs_le_0.02"] == 1            # fractions use |·|
    assert s["median"] == pytest.approx(0.01) and s["abs_median"] == pytest.approx(0.115)


def test_summarize_empty():
    s = M.summarize([float("nan")])
    assert s["n"] == 0 and s["median"] is None and s["frac_abs_le_0.1"] is None and s["n_abs_le_0.1"] == 0
    assert not M.signed_differs(s)
    row = M._row("x", "0 / 0", s)
    assert row.count("–") >= 7


def test_fold_rate_recovers_a_constructed_square_root_fold():
    """λ_min(P^{1/2}HP^{1/2})² = 4 m (s_F − s) exactly (one hidden unit: 4 training coordinates, P blocks = 1):
    |m_c| = m, Λ_F = √(m s_F), ε_F = (ṡ/s_F)/(lr Λ_F)."""
    m, s_f, sdot = 0.003, 4.0, 0.02
    ss = np.linspace(s_f - 0.05, s_f - 1e-4, 40)
    Hs = [np.diag([math.sqrt(4 * m * (s_f - s)), 10.0, 10.0, 10.0]) for s in ss]
    Lam, eps, mc, lam = M.fold_rate(Hs, ss, s_f, [1.0, 1.0, 1.0, 1.0], 1, sdot)
    assert mc == pytest.approx(m, rel=1e-9)
    assert Lam == pytest.approx(math.sqrt(m * s_f), rel=1e-9)
    assert eps == pytest.approx((sdot / s_f) / (0.01 * math.sqrt(m * s_f)), rel=1e-9)
    # a uniform P scale c multiplies λ by c, so |m_c| by c² and Λ_F by c (ε_F by 1/c)
    Lam2, eps2, mc2, _ = M.fold_rate(Hs, ss, s_f, [4.0, 4.0, 4.0, 4.0], 1, sdot)
    assert mc2 == pytest.approx(16 * m, rel=1e-9) and eps2 == pytest.approx(eps / 4, rel=1e-9)


def test_committed_eps_reproduces_the_population_fold_constant():
    E = json.loads((M.OUT / "sb_eps_F.json").read_text())
    p = E["population"]
    assert p["eps_F_recomputed_median_P"] == pytest.approx(p["eps_F_committed"], rel=1e-12)
    assert p["Lambda_F_recomputed_median_P"] == pytest.approx(p["Lambda_F_committed"], rel=1e-12)
    assert p["lambda_path_max_rel_diff"] < 1e-12 and p["P_median_max_rel_diff"] < 1e-12
    assert E["n_runs"] == 40 and all(r["eps_F"] > 0 for r in E["runs"])


def test_committed_map_regenerates_and_pools_consistently():
    J = json.loads((M.OUT / "kappa_chi_map.json").read_text())
    assert J == json.loads(json.dumps(M.finalize(M.build())))
    assert M.MD.read_text() == M.render_md(J)
    for k, p in J["pooled"].items():
        st = k.split(" | ")[0]
        n = sum(c["summary"]["n"] for c in J["cells"] if c["setting"] == st and c["cell"] in p["cells"])
        assert p["summary"]["n"] == n
        assert not any("not pooled" in c for c in p["cells"])
    assert "simplicity bias | Adam" not in J["pooled"]                   # κ undefined: no pooled κχ
    assert all(c["quantity"] != "kappa_chi" for c in J["cells"] if c["setting"] == "simplicity bias")
    assert J["label"].startswith("DESCRIPTIVE")
