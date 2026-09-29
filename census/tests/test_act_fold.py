import numpy as np

from src import act_fold as af

C = 2.0


# Constructed saddle-node: L(z; s) = z0³/3 − (C − s)·z0 + ½(z1 − 0.3 z0)² + ½ z2².
# ∇L = 0  ⇔  z0² = C − s, z1 = 0.3 z0, z2 = 0: the stable branch z0 = +√(C − s) ends in a fold at s = C (z0 = 0).
def sn_F(X):
    z0, z1, z2, s = X
    return np.array([z0 ** 2 - (C - s) - 0.3 * (z1 - 0.3 * z0), z1 - 0.3 * z0, z2])


def sn_J(X):
    z0, z1, z2, s = X
    return np.array([[2 * z0 + 0.09, -0.3, 0.0, 1.0], [-0.3, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def sn_start(s0=0.0):
    z0 = np.sqrt(C - s0)
    return np.array([z0, 0.3 * z0, 0.0, s0])


# Fold-free branch: L = ½(z0 − s)² + ½(z1 − 0.3 z0)² + ½ z2²  (z0 = s for every s).
def lin_F(X):
    z0, z1, z2, s = X
    return np.array([(z0 - s) - 0.3 * (z1 - 0.3 * z0), z1 - 0.3 * z0, z2])


def lin_J(X):
    return np.array([[1.09, -0.3, 0.0, -1.0], [-0.3, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def test_fold_detected_on_saddle_node():
    ev, path = af.continue_branch(sn_F, sn_J, sn_start(), +1, s_stop=5.0)
    assert ev["end"] == "fold"
    assert abs(ev["fold"]["s"] - C) < 1e-8
    assert abs(ev["fold"]["z"][0]) < 1e-4 and abs(ev["fold"]["lam"]) < 1e-4
    assert ev["min_lam_stable"] > 0                               # λ_min > 0 on the stable segment
    assert all(p["s"] <= C + 1e-9 for p in path)


def test_no_fold_on_monotone_branch():
    ev, path = af.continue_branch(lin_F, lin_J, np.array([0.0, 0.0, 0.0, 0.0]), +1, s_stop=5.0)
    assert ev["fold"] is None and ev["end"] == "s_stop"
    assert path[-1]["s"] >= 5.0


def test_switch_detection_before_and_without_fold():
    # G = z0 − 1: on the saddle-node branch z0 = √(C − s) the gap changes sign at s = C − 1 (placed → unplaced)
    ev, _ = af.continue_branch(sn_F, sn_J, sn_start(), +1, s_stop=5.0, gapf=lambda X: X[0] - 1.0)
    assert abs(ev["switch"]["s"] - (C - 1)) < 1e-8 and not ev["switch"]["to_placed"]
    assert ev["fold"]["s"] > ev["switch"]["s"]
    # G = z0 − 2 on the monotone branch: switch at s = 2 to placed, no fold
    ev, _ = af.continue_branch(lin_F, lin_J, np.zeros(4), +1, s_stop=5.0, gapf=lambda X: X[0] - 2.0)
    assert abs(ev["switch"]["s"] - 2.0) < 1e-8 and ev["switch"]["to_placed"] and ev["fold"] is None
    # a gap that never changes sign: no switch
    ev, _ = af.continue_branch(lin_F, lin_J, np.zeros(4), +1, s_stop=5.0, gapf=lambda X: -1.0 - X[0] ** 2)
    assert ev["switch"] is None


def test_step_halving_agrees():
    a, _ = af.continue_branch(sn_F, sn_J, sn_start(), +1, s_stop=5.0)
    b, _ = af.continue_branch(sn_F, sn_J, sn_start(), +1, s_stop=5.0, h0=af.H0 / 2, hmax=af.HMAX / 2)
    assert abs(a["fold"]["s"] - b["fold"]["s"]) < 1e-8


def test_downward_continuation():
    ev, path = af.continue_branch(sn_F, sn_J, sn_start(1.5), -1, s_stop=0.0, gapf=lambda X: X[0] - 1.2)
    assert ev["fold"] is None and ev["end"] == "s_stop"
    assert abs(ev["switch"]["s"] - (C - 1.44)) < 1e-8 and ev["switch"]["to_placed"]


def test_independent_fold_check_pass_and_fail():
    def lg(z, s):
        L = z[0] ** 3 / 3 - (C - s) * z[0] + 0.5 * (z[1] - 0.3 * z[0]) ** 2 + 0.5 * z[2] ** 2
        g = np.array([z[0] ** 2 - (C - s) - 0.3 * (z[1] - 0.3 * z[0]), z[1] - 0.3 * z[0], z[2]])
        return L, g
    zf = np.zeros(3)
    zb = np.array([np.sqrt(C * 1e-3), 0.3 * np.sqrt(C * 1e-3), 0.0])
    r = af.independent_fold_check(lg, zf, C, zb)
    assert r["ok"] and r["n_min_above_near_fold"] == 0 and r["n_min_below_near_branch"] > 0
    # a wrongly placed "fold" (s = 1, where the minimum exists on both sides) is rejected
    z1 = np.array([1.0, 0.3, 0.0])
    zb1 = np.array([np.sqrt(1 + 1e-3), 0.3 * np.sqrt(1 + 1e-3), 0.0])
    assert not af.independent_fold_check(lg, z1, 1.0, zb1)["ok"]


def test_memory_watchdog_parse_and_rule():
    p = "Pageouts: 12\n\nSystem-wide memory free percentage: 35%\n"
    s = "vm.swapusage: total = 11264.00M  used = 10335.06M  free = 928.94M  (encrypted)"
    f, w = af.parse_memory(p, s)
    assert f == 35 and abs(w - 928.94) < 1e-9
    assert af.memory_ok(f, w)
    assert not af.memory_ok(24, 928.94) and not af.memory_ok(35, 499.0)
    assert not af.memory_ok(*af.parse_memory("garbage", "garbage"))


def test_pre_state():
    traj = np.zeros((11, 4)); traj[:, 2] = np.linspace(1, 11, 11)
    k, th = af.pre_state(traj, 10, 11.0, 0.95)                    # last step < 10 with s <= 10.45
    assert k == 9 and th[2] == 10.0
    assert af.pre_state(traj, 10, 0.5, 0.95) is None
