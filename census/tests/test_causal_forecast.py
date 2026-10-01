"""Phase 1A: the causal forecaster (src/causal_forecast.py) and its pilot driver (src/phase1a_pilot.py).

(a) a deliberately leaky forecaster is caught (guard; NaN/garbage poisoning catches a guard bypass);
(b) the real forecasters pass under the guards;
(c) their output is identical when every row at or after the cutoff is NaN, or garbage;
(d) the same for every auxiliary input (Adam moments: rows ≥ t_c; hidden state: every row but the one allowed);
constructed pass / fail / unresolved cases for the extrapolation and the integration.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import causal_forecast as C
from src import linear_response as LR

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "results" / "phase1a" / "fixtures"


# ================================================================================================ the guard
def test_guard_blocks_rows_at_or_after_cutoff():
    a = np.arange(20.0).reshape(10, 2)
    g = C.guard(a, 6, name="x")
    assert len(g) == 6 and g.shape == (6, 2)
    assert np.array_equal(g[:], a[:6]) and np.array_equal(g[-1], a[5]) and g[2, 1] == a[2, 1]
    assert np.array_equal(np.asarray(g), a[:6])
    for bad in (lambda: g[6], lambda: g[9], lambda: g[0:7], lambda: g[[1, 6]], lambda: g[6, 0], lambda: g[-7],
                lambda: g[np.r_[np.zeros(6, bool), True, False, False, False]]):
        with pytest.raises(C.CausalityViolation):
            bad()
    assert g.max_index_read == 5 and len(g.violations) == 7


def test_guard_allowed_rows_and_single_row():
    a = np.arange(30.0).reshape(10, 3)
    g = C.guard(a, 8, rows={0})
    assert np.array_equal(g[0], a[0])
    with pytest.raises(C.CausalityViolation):
        g[1]
    h = C.guard(a, 8, max_rows=1)
    assert np.array_equal(h[4], a[4]) and np.array_equal(h[4], a[4])      # the same row twice is one row
    with pytest.raises(C.CausalityViolation):
        h[5]
    with pytest.raises(C.CausalityViolation):
        h[8]                                                              # never at/after the cutoff
    with pytest.raises(C.CausalityViolation):
        C.guard(a, 8, rows={0})[:]                                        # the whole hidden path is not readable


def test_guard_is_read_only_copy():
    a = np.arange(5.0)
    g = C.guard(a, 3)
    a[0] = 99.0
    assert g[0] == 0.0
    v = g[:]
    v[0] = 7.0
    assert g[0] == 0.0


def test_cutoff_is_a_stopping_time():
    s = np.array([1.0, 1.2, 1.5, 1.9, 2.4, 3.0])
    t = C.cutoff_step(s, 1.8)
    assert t == 3
    s2 = s.copy()
    s2[t + 1:] = np.nan
    assert C.cutoff_step(s2, 1.8) == t                                    # rows after t_c do not move t_c
    assert C.cutoff_step(s, 10.0) is None


# ================================================================================================ synthetic problem
ETA, HH = 0.05, 0.5                       # 1/(ηh) = 40 steps of lag in the steady state
S0, G_RATE, S_SW = 1.0, 2e-3, 2.0


def _syn_path(n, rate_after=None, t_change=None):
    t = np.arange(n, dtype=float)
    y = math.log(S0) + G_RATE * t
    if rate_after is not None:
        y = np.where(t > t_change, math.log(S0) + G_RATE * t_change + rate_after * (t - t_change), y)
    return np.exp(y)


def _syn_forecast(out, t_c, cfg, budget=4000, leak=None):
    """A forecaster on the synthetic branch θ*(s) = (log s, 0, 0), H = h·I, gap(θ) = θ₀ − log s_sw, δ₀ = the steady
    state −(g/(ηh), 0, 0) at release.  leak: 'guarded' reads out[t_c]; 'bypass' reads the private data (a guard cannot
    stop that; the poisoning test does)."""
    s = np.abs(C._prefix(out, t_c))
    if leak == "guarded":
        s = s * (1 + 0 * float(out[t_c]))
    if leak == "bypass":
        s = s * 0 + np.asarray(out._GuardedArray__data)[:t_c] * (1 + 1e-3 * np.nan_to_num(out._GuardedArray__data[t_c]))
    s_hat, st, info = C.extrapolate_scale(s, cfg, S_SW, budget - t_c + 1)
    if st != "ok":
        return {"status": st}
    s_ext = np.concatenate([s, s_hat])
    r = C.lag_forecast_1d(s_ext, 0, lambda v: np.array([math.log(v), 0.0, 0.0]), lambda v: HH * np.eye(3),
                          lambda v: 0.1 <= v <= 100.0, lambda th: th[0] - math.log(S_SW),
                          np.array([-G_RATE / (ETA * HH), 0.0, 0.0]), ETA, S_SW)
    return C._finish(r, s_ext, t_c, info, (out,))


def _expected_cross(rate=G_RATE):
    """Analytic: δ stays at −g/(ηh); the crossing is the first t with log s_t − g/(ηh) > log s_sw."""
    return int(math.floor((math.log(S_SW) + rate / (ETA * HH) - math.log(S0)) / rate)) + 1


@pytest.mark.parametrize("family", C.FAMILIES)
def test_exact_exponential_growth_is_extrapolated_exactly(family):
    s = _syn_path(3000)
    cfg = C.ForecastConfig(f=0.9, family=family)
    t_c = C.cutoff_step(s, cfg.f * S_SW)
    s_hat, st, info = C.extrapolate_scale(s[:t_c], cfg, S_SW, 3000 - t_c)
    assert st == "ok" and abs(info["growth_rate_at_cutoff"] / G_RATE - 1) < 1e-8
    np.testing.assert_allclose(s_hat, s[t_c:t_c + len(s_hat)], rtol=1e-9)


@pytest.mark.parametrize("family", C.FAMILIES)
def test_constructed_pass_exact_forecast_and_known_lag(family):
    s = _syn_path(4001)
    cfg = C.ForecastConfig(f=0.8, family=family)
    t_c = C.cutoff_step(s, cfg.f * S_SW)
    fc = _syn_forecast(C.guard(s, t_c, name="s"), t_c, cfg)
    t_exp = _expected_cross()
    assert fc["status"] == "ok" and fc["t_fc"] == t_exp
    assert fc["t_sw"] == C.cutoff_step(s, S_SW, 0)
    assert fc["lag_steps_fc"] in (40, 41)                                 # 1/(ηh) = 40 steps, up to the grid
    assert fc["max_index_read"]["s"] == t_c - 1 and t_c < t_exp
    # the reference on the actual path gives the same step (the path IS exponential)
    ref = _syn_forecast(C.guard(s, len(s) - 1, name="s"), len(s) - 1, cfg)
    assert ref["t_fc"] == t_exp


def test_constructed_fail_growth_change_after_cutoff_is_an_extrapolation_error():
    """The actual path's growth rate doubles at step 250 (after the cutoff at step 203): the forecast (lin) stays on the
    old rate's analytic crossing; the actual crossing (reference on the actual path) is much earlier."""
    t_change = 250
    s = _syn_path(4001, rate_after=2 * G_RATE, t_change=t_change)
    cfg = C.ForecastConfig(f=0.75, family="lin")
    t_c = C.cutoff_step(s, cfg.f * S_SW)
    assert t_c < t_change
    fc = _syn_forecast(C.guard(s, t_c, name="s"), t_c, cfg)
    ref = _syn_forecast(C.guard(s, 4000, name="s"), 4000, C.ForecastConfig(f=1.0, family="lin"))
    assert fc["t_fc"] == _expected_cross()
    assert ref["t_fc"] <= fc["t_fc"] - 40                                  # the error is caught by the comparison


def test_constructed_unresolved_cases():
    falling = 1.5 - 1e-3 * np.arange(600.0)
    fc = _syn_forecast(C.guard(falling, 300), 300, C.ForecastConfig(f=0.7, family="lin"))
    assert fc["status"].startswith("no forecast: extrapolated path not growing")
    s = _syn_path(4001)
    t_c = C.cutoff_step(s, 0.9 * S_SW)
    fc = _syn_forecast(C.guard(s, t_c), t_c, C.ForecastConfig(f=0.9, family="lin"), budget=t_c + 20)
    assert fc["t_fc"] is None and fc["status"].startswith("no forecast")


def test_leaky_forecaster_is_caught_by_the_guard():
    s = _syn_path(4001)
    cfg = C.ForecastConfig(f=0.8)
    t_c = C.cutoff_step(s, cfg.f * S_SW)
    g = C.guard(s, t_c, name="s")
    with pytest.raises(C.CausalityViolation):
        _syn_forecast(g, t_c, cfg, leak="guarded")
    assert g.violations


def _poison(a, t_c, how):
    b = np.array(a, float, copy=True)
    if how == "nan":
        b[t_c:] = np.nan
    else:
        b[t_c:] = np.random.default_rng(123).uniform(-50, 50, size=b[t_c:].shape)
    return b


def _same(f1, f2):
    keys = set(f1) | set(f2)
    for k in keys:
        a, b = f1.get(k), f2.get(k)
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            continue
        if a != b:
            return False
    return True


@pytest.mark.parametrize("how", ["nan", "garbage"])
def test_poison_catches_a_guard_bypass_and_passes_the_real_one(how):
    s = _syn_path(4001)
    cfg = C.ForecastConfig(f=0.8)
    t_c = C.cutoff_step(s, cfg.f * S_SW)
    clean = _syn_forecast(C.guard(s, t_c, name="s"), t_c, cfg)
    pois = _syn_forecast(C.guard(_poison(s, t_c, how), t_c, name="s"), t_c, cfg)
    assert _same(clean, pois)
    leak_clean = _syn_forecast(C.guard(s, t_c, name="s"), t_c, cfg, leak="bypass")
    leak_pois = _syn_forecast(C.guard(_poison(s, t_c, how), t_c, name="s"), t_c, cfg, leak="bypass")
    assert not _same(leak_clean, leak_pois)                               # the bypass is detected


def test_extrapolate_v_exact_constant_shares():
    t = np.arange(3000.0)
    V = np.column_stack([0.1 * np.exp(1e-3 * t), 0.9 * np.exp(1e-3 * t)])
    cfg = C.ForecastConfig(f=0.9, family="rate")
    s = V.sum(axis=1)
    t_c = C.cutoff_step(s, 0.9 * 10.0)
    Vh, st, _ = C.extrapolate_v(V[:t_c], cfg, 10.0, 3000 - t_c)
    assert st == "ok"
    np.testing.assert_allclose(Vh, V[t_c:t_c + len(Vh)], rtol=1e-9)
    V2 = V.copy()
    V2[t_c - 3, 0] *= -1
    assert C.extrapolate_v(V2[:t_c], cfg, 10.0, 100)[1].startswith("no forecast: an output weight changes sign")


def test_causal_rule_step_reads_before_cutoff_and_matches_track_a():
    from src.track_a import rule_step
    s = np.array([0.3, 0.6, 0.4, 0.55, 0.7, 0.9, 1.1, 1.3, 1.6, 2.1])
    s_fr = 2.0
    t_c = C.cutoff_step(s, 0.8 * s_fr)                                    # 8
    assert C.causal_rule_step(s[:t_c], s_fr) == 6 == rule_step(s, s_fr)       # last below 1.0 at step 5
    assert C.causal_rule_step(s[:3], s_fr) is None


# ================================================================================================ real adapters
def _fixture(name):
    p = FIX / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"fixture {p.name} not present")
    d = np.load(p, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    return d, meta


def _w1_run(opt, n_steps):
    from src.phase1a_pilot import w1_frozen, w1_train
    seed = 9_310_002                                                      # a dev seed
    fr = w1_frozen(seed)
    W, M, V = w1_train(seed, opt=opt, budget=n_steps)
    return seed, fr["s_frozen"], W, M, V


def _w1_inputs(seed, s_fr, W, M, V, opt, t_c, Wo=None, Hh=None, Mo=None, Vo=None, guarded=True):
    from src.phase1a_pilot import _w1_sample
    x, y = _w1_sample(seed)
    gd = (lambda a, **kw: C.guard(a, t_c, **kw)) if guarded else (lambda a, **kw: a)
    Wo = W if Wo is None else Wo
    Hh = W[:, [0, 1, 3]] if Hh is None else Hh
    return C.W1Inputs(a=1.58, x=x, y=y, s_frozen=s_fr, opt=opt, lr=0.3 if opt == "sgd" else 1e-2, t_c=t_c,
                      budget=32_000, out=gd(Wo[:, 2], name="w2"), hid=gd(Hh, max_rows=1, name="hidden"),
                      M=None if M is None else gd(M if Mo is None else Mo, name="M"),
                      Vhat=None if V is None else gd(V if Vo is None else Vo, name="Vhat"))


@pytest.fixture(scope="module")
def w1_sgd_run():
    return _w1_run("sgd", 2500)


@pytest.fixture(scope="module")
def w1_adam_run():
    return _w1_run("adam", 2500)


def test_w1_sgd_real_forecaster_passes_guards_and_poisoning(w1_sgd_run):
    seed, s_fr, W, M, V = w1_sgd_run
    cfg = C.ForecastConfig(f=0.9)
    s = np.abs(W[:, 2])
    t_c = C.cutoff_step(s, cfg.f * s_fr)
    assert t_c is not None and t_c < len(W) - 1
    clean = C.run_forecast("w1", _w1_inputs(seed, s_fr, W, M, V, "sgd", t_c), cfg)
    assert clean["max_index_read"]["w2"] == t_c - 1 and clean["max_index_read"]["hidden"] == clean["t_rule"] < t_c
    for how in ("nan", "garbage"):
        Wp = _poison(W, t_c, how)
        out = C.run_forecast("w1", _w1_inputs(seed, s_fr, W, M, V, "sgd", t_c, Wo=Wp, Hh=Wp[:, [0, 1, 3]]), cfg)
        assert _same(clean, out)
        # unguarded plain arrays, poisoned: identical as well (nothing reads past the cutoff by any route)
        out2 = C.ADAPTERS["w1"](_w1_inputs(seed, s_fr, W, M, V, "sgd", t_c, Wo=Wp, Hh=Wp[:, [0, 1, 3]],
                                           guarded=False), cfg)
        assert _same({k: v for k, v in clean.items() if k != "max_index_read"},
                     {k: v for k, v in out2.items() if k != "max_index_read"})
    # (d) the hidden state: every row but the rule point NaN
    Hn = np.full_like(W[:, [0, 1, 3]], np.nan)
    Hn[clean["t_rule"]] = W[clean["t_rule"], [0, 1, 3]]
    assert _same(clean, C.run_forecast("w1", _w1_inputs(seed, s_fr, W, M, V, "sgd", t_c, Hh=Hn), cfg))


def test_w1_nested_cutoff_is_a_stopping_time_and_matches_the_forecaster(w1_sgd_run):
    from src.phase1a_pilot import _w1_sample
    seed, s_fr, W, M, V = w1_sgd_run
    x, y = _w1_sample(seed)
    s = np.abs(W[:, 2])
    t_c, tR, s_run = C.w1_cutoff_on_s_run(s, lambda k: W[k, 2], 0.9, s_fr, 1.58, x, y, lambda k: W[k, [0, 1, 3]])
    assert t_c is not None and s[t_c] >= 0.9 * s_run and tR == C.causal_rule_step(s[:t_c], s_fr) and tR < t_c
    Wp = _poison(W, t_c + 1, "nan")
    assert C.w1_cutoff_on_s_run(np.abs(Wp[:, 2]), lambda k: Wp[k, 2], 0.9, s_fr, 1.58, x, y,
                                lambda k: Wp[k, [0, 1, 3]]) == (t_c, tR, s_run)
    fc = C.run_forecast("w1", _w1_inputs(seed, s_fr, W, M, V, "sgd", t_c), C.ForecastConfig(f=0.9))
    assert fc["t_rule"] == tR and fc["s_run"] == s_run and fc["max_index_read"]["w2"] == t_c - 1


def test_w1_leaky_adapter_is_caught(w1_sgd_run, monkeypatch):
    seed, s_fr, W, M, V = w1_sgd_run
    cfg = C.ForecastConfig(f=0.9)
    t_c = C.cutoff_step(np.abs(W[:, 2]), cfg.f * s_fr)
    real = C.forecast_w1

    def leaky_out(inp, cfg):
        inp.out[inp.t_c]                                                  # peeks at the cutoff step
        return real(inp, cfg)

    def leaky_hidden(inp, cfg):
        inp.hid[inp.t_c - 1]                                              # a second hidden row (not the rule point)
        return real(inp, cfg)
    for leaky in (leaky_out, leaky_hidden):
        monkeypatch.setitem(C.ADAPTERS, "w1", leaky)
        with pytest.raises(C.CausalityViolation):
            C.run_forecast("w1", _w1_inputs(seed, s_fr, W, M, V, "sgd", t_c), cfg)


@pytest.mark.parametrize("adam_P", ["rule_point", "cutoff"])
def test_w1_adam_moments_only_before_cutoff(w1_adam_run, adam_P):
    seed, s_fr, W, M, V = w1_adam_run
    cfg = C.ForecastConfig(f=0.9, adam_P=adam_P)
    t_c = C.cutoff_step(np.abs(W[:, 2]), cfg.f * s_fr)
    if t_c is None:
        pytest.skip("the short Adam run does not reach the cutoff")
    clean = C.run_forecast("w1", _w1_inputs(seed, s_fr, W, M, V, "adam", t_c), cfg)
    assert clean["max_index_read"]["M"] == clean["t_rule"] < t_c
    assert clean["max_index_read"]["Vhat"] == (clean["t_rule"] if adam_P == "rule_point" else t_c - 1)
    for how in ("nan", "garbage"):
        out = C.run_forecast("w1", _w1_inputs(seed, s_fr, W, M, V, "adam", t_c, Mo=_poison(M, t_c, how),
                                              Vo=_poison(V, t_c, how)), cfg)
        assert _same(clean, out)
    # moments NaN everywhere except the rows the rule names
    keep = {clean["t_rule"], clean.get("t_P", clean["t_rule"])}
    Mn, Vn = np.full_like(M, np.nan), np.full_like(V, np.nan)
    for r in keep:
        Mn[r], Vn[r] = M[r], V[r]
    assert _same(clean, C.run_forecast("w1", _w1_inputs(seed, s_fr, W, M, V, "adam", t_c, Mo=Mn, Vo=Vn), cfg))
    Ml = M.copy()
    with pytest.raises(C.CausalityViolation):
        g = C.guard(Ml, t_c, name="M")
        g[t_c]


def _real_fixture_inputs(name, cfg, how=None, guarded=True):
    from src import phase1a_pilot as PP
    d, meta = _fixture(name)
    row = meta["row"]
    setting = row["setting"]
    out, hid_rows, hid = d["out"], d["hid_rows"], d["hid"]
    s = np.abs(out) if out.ndim == 1 else np.abs(out).sum(axis=1)
    t_c = C.cutoff_step(s, cfg.f * row["s_ref"])
    if how is not None:
        out = _poison(out, t_c, how)
    ctx = PP.context(setting, row)
    kind = {"w1_sgd": "w1", "gelu_random": "gelu"}.get(setting, "w2a")
    return kind, PP.make_inputs(setting, row, out, hid_rows, hid, t_c, ctx, guarded=guarded), t_c, meta


@pytest.mark.parametrize("name", ["gelu_random", "w2a_T"])
def test_real_adapters_gelu_w2a_pass_guards_and_poisoning(name):
    cfg = C.ForecastConfig(f=0.95)
    kind, inp, t_c, meta = _real_fixture_inputs(name, cfg)
    clean = C.run_forecast(kind, inp, cfg)
    assert clean["max_index_read"][inp.out.name] == t_c - 1 and clean["max_index_read"]["hidden"] == 0
    assert clean["status"] == "ok" and clean["t_fc"] is not None
    assert clean["t_fc"] == meta["expected"]["t_fc"]                     # the committed pilot forecast reproduces
    for how in ("nan", "garbage"):
        kind, inp2, _, _ = _real_fixture_inputs(name, cfg, how=how)
        assert _same(clean, C.run_forecast(kind, inp2, cfg))
    real = C.ADAPTERS[kind]

    def leaky(inp, cfg):
        inp.hid[1]                                                        # a hidden state after release
        return real(inp, cfg)
    kind, inp3, _, _ = _real_fixture_inputs(name, cfg)
    C.ADAPTERS[kind] = leaky
    try:
        with pytest.raises(C.CausalityViolation):
            C.run_forecast(kind, inp3, cfg)
    finally:
        C.ADAPTERS[kind] = real


# ================================================================================================ the pilot driver
def test_pilot_seeds_fresh_and_disjoint():
    from src import phase1a_pilot as PP
    allseeds = [s for d in PP.SEEDS.values() for ss in d.values() for s in ss]
    assert len(allseeds) == len(set(allseeds))
    assert all(v == [] for v in PP.registered_overlap().values())


def test_committed_summary_regenerates():
    from src import phase1a_pilot as PP
    p = ROOT / "results" / "phase1a" / "summary.json"
    if not p.exists():
        pytest.skip("summary not yet written")
    assert json.loads(p.read_text()) == json.loads(json.dumps(PP._jsonable(PP.build_summary())))
    assert (ROOT / "results" / "phase1a" / "error_estimate.md").read_text() == PP.render_md(json.loads(p.read_text()))
