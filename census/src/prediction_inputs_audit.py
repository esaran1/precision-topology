"""AUDIT (author-requested 2026-09-30): exactly which information the registered trajectory-integrated (r_traj) and
closed-form (r_cf) predictions use, in W2-A (primary target), GELU-T, Test 2A and Track A (a = 1.65).

Nothing registered is changed: every registered module is imported and called unchanged; no frozen file, prediction,
observation or verdict is written.  Outputs: results/prediction_inputs_audit.json (every count reported in
results/prediction_inputs_audit.md) and the replay rows results/prediction_inputs_audit/replay_*.jsonl.

Three parts.
  1. CODE_REFS: every input of every prediction, with file:line of the code that reads it and its class; `code_refs_check`
     confirms that each cited line still contains the cited token.
  2. Counts from the committed CSVs (observed_runs*.csv; scored runs only unless stated): where the prediction's last
     read step lies relative to the observed crossing t_obs, the switch t_sw and the ṡ window, the follow check.
  3. Replays from the saved paths (untracked; each path's SHA-256 is asserted against its committed row before use):
       MASKED  the registered predict function called unchanged on the run's path with every hidden coordinate after
               release (W2-A, GELU-T; except at the follow-check step t_0.8, whose hidden state the registered follow
               check reads -- a scored-set input, reported separately) / at every step other than the rule point
               (Track A, 2A) set to NaN (Adam moments: NaN except the rows the registration names).  Prediction fields that reproduce the committed values
               use no hidden-parameter state after release / the rule point (condition i) and no gap of it (ii).
       CAUSAL  (POST HOC diagnostic; it replaces no registered verdict) as MASKED, and the output weights (v for W2-A,
               w₂ otherwise) at every step t ≥ t_obs replaced by the linear extrapolation from step t_obs − 1 with the
               mean increment over the last 100 steps (t_obs − 101 → t_obs − 1).  The prediction then uses nothing
               read at or after the observed crossing.  (Choosing the cut uses t_obs: it is a sensitivity diagnostic,
               not a prediction rule.)
       TRUNCATED  as MASKED, and in addition every row (hidden and output) after the claimed last-read step L set to
               NaN (L_traj for r_traj, L_cf for r_cf; see TRUNC_FIELDS).  Reproduction shows the prediction reads
               nothing after L, so the counts of steps read at or after the crossing are not undercounts.
     Track A / 2A save only θ; Adam's moments are rebuilt by replaying the registered Adam run with a LOCAL torch
     Generator (no global RNG state) and asserting that the replayed θ equals the saved path bit for bit.

    python -m src.prediction_inputs_audit replay w2a|gelu|track_a|track2a   # resumable (paths needed)
    python -m src.prediction_inputs_audit truncated w2a|gelu|track_a|track2a   # resumable (paths needed)
    python -m src.prediction_inputs_audit summarize                        # -> results/prediction_inputs_audit.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "prediction_inputs_audit.json"
OUT_DIR = RESULTS / "prediction_inputs_audit"
REPLAY = {"w2a": OUT_DIR / "replay_w2a.jsonl", "gelu": OUT_DIR / "replay_gelu.jsonl",
          "track_a": OUT_DIR / "replay_track_a.jsonl", "track2a": OUT_DIR / "replay_track2a.jsonl"}
TRUNC = {"w2a": OUT_DIR / "truncated_w2a.jsonl", "gelu": OUT_DIR / "truncated_gelu.jsonl",
         "track_a": OUT_DIR / "truncated_track_a.jsonl", "track2a": OUT_DIR / "truncated_track2a.jsonl"}
AUDIT_MD = RESULTS / "prediction_inputs_audit.md"
SLOPE_WIN = 100
REPRO_RTOL, REPRO_ATOL = 1e-13, 1e-16   # the committed CSVs keep 16 decimal places (truncation error < 1e-16)
RSS_LIMIT = 3 * 1024 ** 3

# ------------------------------------------------------------------------------------------ 1. inputs, with file:line
FROZEN, RELEASE, OUTPUT, HIDDEN_AFTER, OPT_AFTER, GAP_TRAIN, GAP_OTHER, SCORING, DESCR = (
    "frozen before training", "release / rule-point state", "output-weight path (s_t or v_t)",
    "hidden-parameter state after release / rule point", "optimiser state after the rule point",
    "gap of a training state", "gap of a non-training point (predicted state or branch point)",
    "scored-set membership (not a prediction input)", "descriptive column (not used by any prediction or verdict)")

# (test, prediction, input, class, file, line, token that the line must contain)
CODE_REFS = (
    # ---------------- W2-A
    ("W2-A", "r_traj, r_cf", "occupied copy and winding k (fixed at release)", RELEASE, "src/width2_asym.py", 1292,
     "copy, k = classify(z_rel"),
    ("W2-A", "r_traj, r_cf", "frozen copy point z_c(s0), s_switch (D), kappa0 + winding coef, lam_min_switch",
     FROZEN, "src/width2_asym.py", 1361, "kap = kappa_winding"),
    ("W2-A", "r_cf", "lam_min at the copy's frozen switch", FROZEN, "src/width2_asym.py", 1362, "lam_sw = cf["),
    ("W2-A", "r_traj, r_cf, t_sw", "the run's output-weight path V = P[:, VI], s = |v|_1", OUTPUT,
     "src/width2_asym.py", 1359, "V = P[:, VI]"),
    ("W2-A", "r_traj, t_sw (T, T')", "own branch z*(v_t): warm-started Newton at each v_t on the seed's own sample",
     OUTPUT, "src/width2_asym.py", 738, "pred = self.Z[-1]"),
    ("W2-A", "r_traj", "release state z_release = P[0][ZI] (delta_0)", RELEASE, "src/width2_asym.py", 1390,
     "t_hit, st, mx, rad = r4_recursion"),
    ("W2-A", "r_traj", "delta_0 = z_release - z*(v_0)", RELEASE, "src/width2_asym.py", 783, "d = np.asarray(z_release"),
    ("W2-A", "r_traj", "recursion delta_{t+1} = (I - eta H_t) delta_t - (z*(v_{t+1}) - z*(v_t)): v_0..v_t_hit", OUTPUT,
     "src/width2_asym.py", 798, "d = d - eta"),
    ("W2-A", "r_traj", "hit test: gap of the PREDICTED state z*(v_t) + delta_t at v_t", GAP_OTHER,
     "src/width2_asym.py", 792, "if t >= 1 and placed_fn"),
    ("W2-A", "r_traj", "s_traj = s[t_hit] (the run's own s at the predicted step)", OUTPUT, "src/width2_asym.py", 1396,
     "out.update(t_traj=int(t_hit)"),
    ("W2-A", "t_sw, s_switch (T, T')", "own-path switch: first t with z*(v_t) placed; v_0..v_t_sw", OUTPUT,
     "src/width2_asym.py", 1370, "t_sw, s_sw = own_path_switch"),
    ("W2-A", "t_sw, s_switch (T, T')", "gap of branch points z*(v_t) (not training states)", GAP_OTHER,
     "src/width2_asym.py", 760, "if placed_fn(br.Z[t], br.V[t])"),
    ("W2-A", "t_sw, s_switch (T, T')", "bisection on the segment v_{t_sw-1} -> v_{t_sw}", GAP_OTHER,
     "src/width2_asym.py", 767, "if okm and gap_fn"),
    ("W2-A", "t_sw (D)", "t_sw = first s_t >= frozen s_switch", OUTPUT, "src/width2_asym.py", 1373,
     "t_sw, s_sw = first_ge"),
    ("W2-A", "r_cf", "s-dot = (s[t_sw] - s[t_sw - w]) / w, w = min(100, t_sw)", OUTPUT, "src/width2_asym.py", 469,
     "sdot = (s[t_sw]"),
    ("W2-A", "r_cf", "closed form kappa_k * chi at t_sw", OUTPUT, "src/width2_asym.py", 1379,
     "r_cf, chi, sdot = closed_form"),
    ("W2-A", "V7 statistic", "chi_t on 0..t_sw (s path and own-branch lam_min)", OUTPUT, "src/width2_asym.py", 1381,
     "cp = chi_path"),
    ("W2-A", "follow check (scored set)", "hidden state z at t_0.8 = P[t08][ZI]", SCORING, "src/width2_asym.py", 1324,
     "zr, vr = P[t08]"),
    ("W2-A", "gate (D, T'); T sensitivity", "gap of every pre-release hold state (v fixed at v_0)", GAP_TRAIN,
     "src/width2_asym.py", 1261, "p, u = placed_state(z, v0)"),
    ("W2-A", "descriptive", "s_min_after_release = min over the whole path", DESCR, "src/width2_asym.py", 1365,
     "s_min_after_release="),
    ("W2-A", "coupling", "joint update of (z, v) by the full gradient", OUTPUT, "src/width2_asym.py", 1248,
     "q = q - lr * grad"),
    # ---------------- GELU-T
    ("GELU-T", "r_traj, r_cf", "copy at release", RELEASE, "src/gelu_transfer.py", 814, "copy = classify_release(z_rel"),
    ("GELU-T", "r_traj, r_cf", "frozen s_switch, kappa, lam_min_switch of the copy", FROZEN, "src/gelu_transfer.py",
     848, "s_sw, kap, lam_sw = cf"),
    ("GELU-T", "r_traj, r_cf", "s = |w2| path", OUTPUT, "src/gelu_transfer.py", 846, "s = np.abs(Wp[:, 2])"),
    ("GELU-T", "r_cf", "t_sw = first s_t >= frozen s_switch", OUTPUT, "src/gelu_transfer.py", 849,
     "t_sw = t_switch(s, s_sw)"),
    ("GELU-T", "r_cf", "s-dot window s[t_sw - w .. t_sw]", OUTPUT, "src/gelu_transfer.py", 185, "sdot = (s[t_sw]"),
    ("GELU-T", "r_traj", "static own-sample branch grid from the frozen copy point (no path input)", FROZEN,
     "src/gelu_transfer.py", 854, "B = GBranch(s0"),
    ("GELU-T", "r_traj", "release state z_rel = Wp[0, [0, 1, 3]]", RELEASE, "src/gelu_transfer.py", 864,
     "z_rel = Wp[0, [0, 1, 3]]"),
    ("GELU-T", "r_traj", "segment end T_end = first s_t outside the branch grid (loop bound only)", OUTPUT,
     "src/gelu_transfer.py", 866, "T_end = len(s) if"),
    ("GELU-T", "r_traj", "theta*(s_t), H(s_t) along the run's s path", OUTPUT, "src/gelu_transfer.py", 868,
     "th_seg = np.array("),
    ("GELU-T", "r_traj", "hit test: gap of the PREDICTED state theta*(s_t) + delta_t", GAP_OTHER,
     "src/gelu_transfer.py", 873, "gapf = lambda"),
    ("GELU-T", "r_traj", "linear-response recursion (stops at the first hit)", OUTPUT, "src/linear_response.py", 332,
     "if gapf(t + 1, d) > 0"),
    ("GELU-T", "r_traj", "s_traj = s[t_hit]", OUTPUT, "src/gelu_transfer.py", 882, "out.update(s_traj=float(s_seg"),
    ("GELU-T", "follow check (scored set)", "hidden state at t_0.8 = Wp[t08, [0, 1, 3]]", SCORING,
     "src/gelu_transfer.py", 829, "st = float(s[t08]); z = Wp"),
    ("GELU-T", "gate (branch); random sensitivity", "gap of every pre-release hold state", GAP_TRAIN,
     "src/gelu_transfer.py", 768, "st = state(th.detach()"),
    ("GELU-T", "descriptive", "w2_min_after_release = min over the whole path", DESCR, "src/gelu_transfer.py", 847,
     "w2_min_after_release"),
    ("GELU-T", "coupling", "one torch SGD optimiser over (w1, b1, w2, b2)", OUTPUT, "src/gelu_transfer.py", 783,
     "opt = torch.optim.SGD"),
    # ---------------- Track A (a = 1.65), and 2A (Track A's code, P at t_sw)
    ("Track A / 2A", "r_traj, r_cf", "landscape kappa inputs H_pop, theta*', grad G at s*_pop", FROZEN,
     "src/track_a.py", 294, "d, th_pop, H_pop"),
    ("Track A / 2A", "rule point", "t_R reads s_1 .. s_t_top (t_top = first s_t >= s_frozen)", OUTPUT,
     "src/track_a.py", 102, "top = np.nonzero(s_path[1:] >= s_frozen)"),
    ("Track A / 2A", "r_traj, r_cf", "rule-point state theta_{t_R} -> Newton -> occupied branch, s*_run", RELEASE,
     "src/track_a.py", 306, "th_b, res, Hn = LR.newton(TH[tR]"),
    ("Track A / 2A", "r_traj, r_cf", "own-sample branch grid from the rule-point Newton point", RELEASE,
     "src/track_a.py", 309, "B = LR.Branch(float(s[tR])"),
    ("Track A", "r_traj, r_cf", "P_R = 1/(sqrt(v-hat_{t_R}) + eps) (Adam)", RELEASE, "src/track_a.py", 315,
     "P_R = (1.0 / (np.sqrt(V[tR"),
    ("Test 2A", "r_traj, r_cf", "v-hat at t_sw put in row t_R: P = P(t_sw) (Adam 2nd moment of the hidden coords)",
     OPT_AFTER, "src/track2a.py", 122, "Vm[t_rule] = V[t_sw]"),
    ("Test 2A", "r_traj, r_cf", "t_sw = first s_t >= s*_run", OUTPUT, "src/track2a.py", 218, "t_sw = t_switch(s, base"),
    ("Track A / 2A", "r_cf", "t_sw = first s_t >= s*_run; s-dot window", OUTPUT, "src/track_a.py", 322,
     "win = min(100, t_sw)"),
    ("Track A / 2A", "r_traj", "delta_0 = theta_{t_R} - theta*(s_{t_R})", RELEASE, "src/track_a.py", 332,
     "d0 = TH[tR]"),
    ("Track A / 2A", "r_traj", "m_0 = Adam first moment at t_R", RELEASE, "src/track_a.py", 337,
     "t_hit, path = LR.simulate"),
    ("Track A / 2A", "r_traj", "s path from t_R to the hit (theta*(s_t), H(s_t))", OUTPUT, "src/track_a.py", 326,
     "s_seg = s[tR:]"),
    ("Track A / 2A", "r_traj", "hit test: gap of the PREDICTED state", GAP_OTHER, "src/linear_response.py", 332,
     "if gapf(t + 1, d) > 0"),
    ("Track A / 2A", "r_traj", "s_traj = s[t_R + t_hit]", OUTPUT, "src/track_a.py", 346,
     "out.update(s_traj=float(s_seg"),
    ("Track A / 2A", "coupling", "one torch optimiser over (w1, b1, w2, b2)", OUTPUT, "src/track_a.py", 286,
     "W[t] = th.detach()"),
    # ---------------- further lines cited in results/prediction_inputs_audit.md (each checked like the above)
    ("W2-A", "all", "predict_one (registered)", OUTPUT, "src/width2_asym.py", 1350, "def predict_one(arm, seed, P"),
    ("W2-A", "r_traj, r_cf", "Newton at v0 from the release state", RELEASE, "src/width2_asym.py", 1288,
     "zn, gn, lamn, okn = newton(z_rel, v0"),
    ("W2-A", "r_traj", "the seed's own training sample", FROZEN, "src/width2_asym.py", 1283, "x, y = own_sample(seed)"),
    ("W2-A", "r_traj, r_cf", "eta of the arm", FROZEN, "src/width2_asym.py", 1358, "eta = ETA[arm]"),
    ("W2-A", "r_traj, r_cf", "s = |v|_1", OUTPUT, "src/width2_asym.py", 1360, "s = np.abs(V).sum(axis=1)"),
    ("W2-A", "r_traj, t_sw (T, T')", "own branch anchored at the frozen copy point, shifted to k at v_0", FROZEN,
     "src/width2_asym.py", 1367, "br = OwnBranch(V, shift("),
    ("W2-A", "r_traj, t_sw (T, T')", "Newton at v_k (warm start)", OUTPUT, "src/width2_asym.py", 739,
     "zn, _, _, ok = newton(pred, self.V[k]"),
    ("W2-A", "t_sw (T, T')", "loop over t = 1 .. t_sw", OUTPUT, "src/width2_asym.py", 757, "for t in range(max(1, t_from)"),
    ("W2-A", "t_sw (T, T')", "bisection starts at v_{t_sw - 1}", GAP_OTHER, "src/width2_asym.py", 761,
     "lo, hi, zl = 0.0, 1.0, br.Z[t - 1]"),
    ("W2-A", "t_sw (T, T')", "returns at t_sw", OUTPUT, "src/width2_asym.py", 772, "return int(t), float(np.abs(vr)"),
    ("W2-A", "r_traj", "recursion returns at the first hit", OUTPUT, "src/width2_asym.py", 793, 'return t, "ok", mx, rad'),
    ("W2-A", "V7 statistic", "chi_window_max over t < t_sw", OUTPUT, "src/width2_asym.py", 1383,
     'out["chi_window_max"] = chi_window_max('),
    ("W2-A", "rules", "observation rule: enclosure lower end > 0 (placed_state)", SCORING, "src/width2_asym.py", 284,
     "def placed_state(z, v)"),
    ("W2-A", "rules", "prediction and branch-point rule: enclosure midpoint > 0 (branch_placed)", GAP_OTHER,
     "src/width2_asym.py", 293, "def branch_placed(z, v)"),
    ("W2-A", "observation", "every-step detection on the actual state", SCORING, "src/width2_asym.py", 1580,
     "p, u = placed_state(P[t][ZI], P[t][VI])"),
    ("W2-A", "observation", "r_obs = s_obs / s_switch - 1", SCORING, "src/width2_asym.py", 1644,
     'd["r_obs"] = col("s_obs") / col("s_switch") - 1'),
    ("GELU-T", "all", "predict_one (registered)", OUTPUT, "src/gelu_transfer.py", 838, "def predict_one(arm, seed, Wp"),
    ("GELU-T", "r_cf", "w = min(100, t_sw)", OUTPUT, "src/gelu_transfer.py", 184, "w = min(SDOT_WIN, int(t_sw))"),
    ("GELU-T", "r_cf", "closed form at t_sw", OUTPUT, "src/gelu_transfer.py", 850, "r_cf, chi, sdot = closed_form(s, t_sw"),
    ("GELU-T", "V7 statistic", "chi_window_max over t < t_sw", OUTPUT, "src/gelu_transfer.py", 858,
     'out["chi_window_max"] = chi_window_max('),
    ("GELU-T", "r_traj", "segment test: s_t inside the branch grid", OUTPUT, "src/gelu_transfer.py", 865,
     "inside = (s >= B.lo) & (s <= B.hi)"),
    ("GELU-T", "r_traj", "H(s_t) along the s path", OUTPUT, "src/gelu_transfer.py", 869, "Hs = [B.hess(v) for v in s_seg]"),
    ("GELU-T", "r_traj", "stability radius over recursion indices <= t_hit", OUTPUT, "src/gelu_transfer.py", 876,
     "rad = max(LR.step_map_radius(Hs[i]"),
    ("GELU-T", "coupling", "every training step saved", OUTPUT, "src/gelu_transfer.py", 791, "Wp[t] = th.detach()"),
    ("GELU-T", "observation", "every-step detection: phase2b_ordering.state placement", SCORING,
     "src/gelu_transfer.py", 1035, 'if state(torch.tensor(Wp[t]'),
    ("GELU-T", "observation", "r_obs = s_obs / s_switch - 1", SCORING, "src/gelu_transfer.py", 1070,
     'd["r_obs"] = d.s_obs / d.s_switch - 1'),
    ("Track A / 2A", "all", "predict_one (registered)", OUTPUT, "src/track_a.py", 293, "def predict_one(opt_name, seed, W"),
    ("Track A / 2A", "rule point", "rule_step definition", OUTPUT, "src/track_a.py", 96, "def rule_step(s_path, s_frozen"),
    ("Track A / 2A", "rule point", "t_R = rule_step(s, s_frozen)", OUTPUT, "src/track_a.py", 302, "tR = rule_step(s, s_frozen)"),
    ("Track A / 2A", "r_traj, r_cf", "hidden coordinates and Adam m (only row t_R is used)", RELEASE, "src/track_a.py",
     300, "TH = W[:, [0, 1, 3]] * flip; MM = M[:, [0, 1, 3]] * flip"),
    ("Track A / 2A", "r_traj, r_cf", "s*_run = the rule-point branch's switch (no path input)", RELEASE,
     "src/track_a.py", 310, "s_run = B.switch(s_frozen)"),
    ("Track A / 2A", "r_cf", "t_sw = first s_t >= s*_run", OUTPUT, "src/track_a.py", 318, "t_sw = LR._first_ge(s, s_run)"),
    ("Track A / 2A", "r_cf", "closed form with P_R and the landscape kappa", FROZEN, "src/track_a.py", 323,
     "r_cf, kap, chi = closed_form_r(H_pop"),
    ("Track A / 2A", "r_traj", "H(s_t) along the s path", OUTPUT, "src/track_a.py", 331, "Hs = [B.hess(v) for v in s_seg]"),
    ("Track A / 2A", "r_traj", "hit test: exact gap of the PREDICTED state", GAP_OTHER, "src/track_a.py", 336,
     "gapf = lambda i, dd: LR.gap_exact("),
    ("Track A / 2A", "r_traj", "P frozen for the recursion", RELEASE, "src/track_a.py", 334, "PP = np.tile(P_R"),
    ("Track A / 2A", "r_traj", "Adam bias correction from the step count K (not a state)", FROZEN, "src/track_a.py", 335,
     "bc1 = (1 - LR.BETA1 ** K[tR"),
    ("Track A / 2A", "coupling", "one step on the full parameter vector", OUTPUT, "src/track_a.py", 282,
     "for t in range(1, BUDGET + 1)"),
    ("Test 2A", "r_traj, r_cf", "v_at_switch: P frozen at t_sw", OPT_AFTER, "src/track2a.py", 119,
     "def v_at_switch(V, t_rule, t_sw)"),
    ("Test 2A", "r_traj, r_cf", "Track A's predict_one with v-hat(t_sw) in row t_R", OPT_AFTER, "src/track2a.py", 225,
     "v_at_switch(V, base[\"t_rule\"], t_sw)"),
    ("all", "r_traj", "one recursion step uses H at step t", OUTPUT, "src/linear_response.py", 317, "g = H_of(t) @ d"),
    ("all", "r_traj", "simulate returns at the first hit", OUTPUT, "src/linear_response.py", 333, "return t + 1, path"),
)


def code_refs_check(refs=CODE_REFS, root=ROOT):
    """Cited lines that no longer contain their token (a list of strings; empty if all hold)."""
    bad, cache = [], {}
    for test, pred, what, cls, f, line, tok in refs:
        if f not in cache:
            cache[f] = (Path(root) / f).read_text().splitlines()
        lines = cache[f]
        if not (1 <= line <= len(lines)) or tok not in lines[line - 1]:
            bad.append(f"{f}:{line} does not contain {tok!r} ({test}: {what})")
    return bad


# ------------------------------------------------------------------------------------------ counting (pure functions)
def _arr(v):
    return np.asarray(pd.to_numeric(pd.Series(v), errors="coerce"), float)


def _med(v):
    v = np.asarray(v, float)
    return float(np.median(v)) if len(v) else None


def _mn(v):
    v = _arr(v)
    return float(v.min()) if len(v) else None


def _mx(v):
    v = _arr(v)
    return float(v.max()) if len(v) else None


def read_beyond(t_last, t_obs):
    """Where the last step READ by a prediction lies relative to the observed crossing step t_obs (per run).
    d = t_last − t_obs; 'at or after' is d ≥ 0 (the crossing step's own output weights are read); 'strictly after'
    is d > 0, and d is then the number of steps read after the crossing step."""
    t_last, t_obs = _arr(t_last), _arr(t_obs)
    ok = np.isfinite(t_last) & np.isfinite(t_obs)
    d = (t_last - t_obs)[ok]
    after = d[d > 0]
    return {"n": int(ok.sum()), "n_at_or_after": int((d >= 0).sum()), "n_at": int((d == 0).sum()),
            "n_strictly_after": int((d > 0).sum()), "n_before": int((d < 0).sum()),
            "max_steps_after": float(after.max()) if len(after) else 0.0,
            "median_steps_after_among_after": _med(after),
            "d_min": float(d.min()) if len(d) else None, "d_median": _med(d),
            "d_max": float(d.max()) if len(d) else None}


def sdot_window(t_sw, t_obs, win=100):
    """The closed form's ṡ window [t_sw − w, t_sw], w = min(100, t_sw), against t_obs (per run): windows ending at or
    after the crossing (t_sw ≥ t_obs), lying entirely at or after it (t_sw − w ≥ t_obs), and straddling it."""
    t_sw, t_obs = _arr(t_sw), _arr(t_obs)
    ok = np.isfinite(t_sw) & np.isfinite(t_obs)
    ts, to = t_sw[ok], t_obs[ok]
    start = ts - np.minimum(win, ts)
    end_after = ts >= to
    start_after = start >= to
    return {"n": int(ok.sum()), "n_end_at_or_after": int(end_after.sum()),
            "n_entirely_at_or_after": int(start_after.sum()),
            "n_straddling": int((end_after & ~start_after).sum()),
            "max_steps_tsw_after": float((ts - to)[end_after].max()) if end_after.any() else 0.0}



def first_ge(s, level, start=0):
    idx = np.nonzero(np.asarray(s, float)[start:] >= level)[0]
    return None if len(idx) == 0 else int(idx[0] + start)


def extrapolate_linear(X, t_cut, w=SLOPE_WIN):
    """Rows 0..t_cut unchanged; row t_cut + k = X[t_cut] + k·(X[t_cut] − X[t_cut − w])/w for k ≥ 1 (1-D or 2-D)."""
    X = np.array(X, float, copy=True)
    t_cut = int(t_cut)
    if t_cut - w < 0:
        raise ValueError("not enough steps before the cut")
    slope = (X[t_cut] - X[t_cut - w]) / w
    k = np.arange(1, len(X) - t_cut, dtype=float)
    X[t_cut + 1:] = X[t_cut] + (k[:, None] * slope if X.ndim == 2 else k * slope)
    return X


def rel_equal(a, b, rtol=REPRO_RTOL, atol=REPRO_ATOL):
    """Equal as committed (floats to rtol relative or atol absolute: the CSVs keep 16 decimal places; None/NaN only
    with None/NaN)."""
    fa = a is None or (isinstance(a, float) and not math.isfinite(a))
    fb = b is None or (isinstance(b, float) and not math.isfinite(b))
    if fa or fb:
        return bool(fa and fb)
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    a, b = float(a), float(b)
    return bool(a == b or abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def last_read(*cols):
    """Per run, the last step read by a prediction with several step-indexed inputs (e.g. t_traj and t_sw): the
    elementwise max, NaN entries ignored (NaN only where every column is NaN)."""
    A_ = np.vstack([_arr(c) for c in cols])
    with np.errstate(all="ignore"):
        return np.fmax.reduce(A_, axis=0)


def ratio_split(t_traj, t_obs, s_traj, s_obs, r_traj, r_obs):
    """L1's per-run ratio r_obs/r_traj, split by the identity.  A run with t_traj = t_obs reads the same array entry for
    s_traj and s_obs (s_traj ≡ s_obs), so its ratio is 1 up to the CSVs' 16-decimal rounding of r (the maximum
    deviation is reported).  The other runs are counted by the side of 1 their ratio falls on."""
    tt, to, st, so, rt, ro = (_arr(v) for v in (t_traj, t_obs, s_traj, s_obs, r_traj, r_obs))
    ok = np.isfinite(tt) & np.isfinite(to) & np.isfinite(rt) & np.isfinite(ro) & (rt != 0)
    same = ok & (tt == to)
    other = ok & (tt != to)
    with np.errstate(all="ignore"):
        q = ro / rt - 1
    return {"n": int(ok.sum()), "n_t_traj_equal_t_obs": int(same.sum()),
            "n_identity_s_traj_identical_s_obs": int((same & (st == so)).sum()),
            "max_abs_ratio_minus_1_among_identity": float(np.abs(q[same]).max()) if same.any() else None,
            "n_other": int(other.sum()), "n_other_ratio_below_1": int((other & (q < 0)).sum()),
            "n_other_ratio_above_1": int((other & (q > 0)).sum()),
            "n_other_ratio_exactly_1": int((other & (q == 0)).sum())}


def truncate_after(X, L):
    """A copy of X with every row after step L set to NaN: a prediction that reproduces on it reads rows 0..L only."""
    X = np.array(X, float, copy=True)
    X[int(L) + 1:] = np.nan
    return X


def md_refs(text):
    """Every `name.py:N` cited in a text, as (file basename, line), sorted and unique."""
    return sorted({(m.group(1), int(m.group(2))) for m in re.finditer(r"\b([a-z0-9_]+\.py):(\d+)", text)})


def md_refs_check(text, refs=None):
    """Cited `name.py:N` that no CODE_REFS entry covers (so no token check backs the citation)."""
    refs = CODE_REFS if refs is None else refs
    have = {(Path(f).name, int(line)) for *_, f, line, _tok in refs}
    return [f"{f}:{n}" for f, n in md_refs(text) if (f, n) not in have]


# ------------------------------------------------------------------------------------------ helpers
def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def _write(path, row):
    """Append one replay row (JSON line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(_clean(row)) + "\n")


def _done(path, key):
    if not path.exists():
        return set()
    return {tuple(r[k] for k in key) for r in map(json.loads, path.read_text().splitlines())}


def _gate(tag):
    """Memory gate (free ≥ 25%, swap free ≥ 500 MB; act_fold.check_memory); waits while it fails."""
    from .act_fold import check_memory
    while True:
        ok, f, w = check_memory()
        if ok:
            return
        print(f"memory gate WAIT {tag}: free={f}% swap_free={w}MB", flush=True)
        time.sleep(60)


def _rss():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > RSS_LIMIT:
        raise SystemExit("STOP: RSS > 3 GB")


def _nice():
    if os.nice(0) < 15:
        os.nice(15 - os.nice(0))


def _fields(o, keys):
    return {k: o.get(k) for k in keys}


# ------------------------------------------------------------------------------------------ 3. replays
W2A_KEYS = ("t_sw", "s_switch", "sdot", "chi_tsw", "r_cf", "chi_window_max", "t_traj", "s_traj", "r_traj",
            "traj_status", "branch_lost_at", "branch_steps_computed", "follows_branch", "status")
GELU_KEYS = ("t_sw", "s_switch", "sdot", "chi_tsw", "r_cf", "chi_window_max", "t_traj", "s_traj", "r_traj", "traj_T",
             "status_traj", "follows_branch", "status")
TA_KEYS = ("t_rule", "s_run", "P_w1", "P_b1", "P_b2", "sdot", "r_cf", "t_traj", "s_traj", "r_traj", "status_traj",
           "status")
T2A_KEYS = TA_KEYS + ("t_sw", "descr_r_traj_P_rule", "status_2a")
REPRO = {"w2a": ("t_sw", "s_switch", "sdot", "r_cf", "chi_window_max", "t_traj", "s_traj", "r_traj", "follows_branch"),
         "gelu": ("t_sw", "s_switch", "sdot", "r_cf", "chi_window_max", "t_traj", "s_traj", "r_traj",
                  "follows_branch"),
         "track_a": ("t_rule", "s_run", "sdot", "r_cf", "t_traj", "s_traj", "r_traj"),
         "track2a": ("t_rule", "s_run", "t_sw", "sdot", "r_cf", "t_traj", "s_traj", "r_traj")}


def _scored_w2a():
    from . import width2_asym as W
    rows = []
    for arm in W.ARMS:
        d = pd.read_csv(RESULTS / "width2_asym" / f"observed_runs_{arm}.csv")
        rows += [r for r in d[d.scored.astype(bool)].itertuples()]
    return rows


def replay_w2a():
    from . import width2_asym as W
    land, fz = W._land(), W._frozen()
    f = REPLAY["w2a"]
    done = _done(f, ("arm", "seed"))
    for r in _scored_w2a():
        if (r.arm, int(r.seed)) in done:
            continue
        _gate(f"w2a {r.arm} {r.seed}")
        t0 = time.time()
        seed, t_obs = int(r.seed), int(r.step_obs)
        pf = W._path_file(r.arm, seed)
        h = _sha(pf)
        assert h == r.path_sha256, f"path hash mismatch {pf}"
        P = np.load(pf)
        rec = {"on_branch": True, "copy_at_release": r.copy_at_release, "windings": json.loads(r.windings)}
        x, y = W.own_sample(seed)
        s_true = np.abs(P[:, W.VI]).sum(axis=1)
        Pm = P.copy()
        Pm[1:, W.ZI] = np.nan                                             # no hidden state after release ...
        tf = int(r.t_follow)
        Pm[tf, W.ZI] = P[tf, W.ZI]                                        # ... except the follow check's (t_0.8)
        om = W.predict_one(r.arm, seed, Pm, rec, fz[seed], land, x, y)
        Pc = Pm.copy()
        Pc[:, W.VI] = extrapolate_linear(P[:, W.VI], t_obs - 1)           # nothing at or after t_obs
        oc = W.predict_one(r.arm, seed, Pc, rec, fz[seed], land, x, y)
        row = {"test": "W2-A", "arm": r.arm, "seed": seed, "path_sha256_ok": True, "t_obs": t_obs,
               "s_obs": float(r.s_obs), "s_true_at_t_obs": float(s_true[t_obs]),
               "masked": _fields(om, W2A_KEYS), "causal": _fields(oc, W2A_KEYS), "secs": round(time.time() - t0, 1)}
        _write(f, row)
        print(json.dumps({k: row[k] for k in ("arm", "seed", "t_obs", "secs")}), flush=True)
        del P, Pm, Pc
        _rss()


def replay_gelu():
    from . import gelu_transfer as G
    land, fz = G._land(), G._frozen()
    f = REPLAY["gelu"]
    done = _done(f, ("arm", "seed"))
    for arm in G.ARMS:
        d = pd.read_csv(RESULTS / "gelu_transfer" / f"observed_runs_{arm}.csv")
        for r in d[d.scored.astype(bool)].itertuples():
            if (arm, int(r.seed)) in done:
                continue
            _gate(f"gelu {arm} {r.seed}")
            t0 = time.time()
            seed, t_obs = int(r.seed), int(r.step_obs)
            pf = G._path_file(arm, seed)
            assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
            Wp = np.load(pf)["W"]
            rec = {"on_branch": True, "copy_at_release": int(r.copy_at_release)}
            P = G.own_problem(seed)
            Wm = Wp.copy()
            Wm[1:, [0, 1, 3]] = np.nan                                    # no hidden state after release ...
            tf = int(r.t_follow)
            Wm[tf, [0, 1, 3]] = Wp[tf, [0, 1, 3]]                         # ... except the follow check's (t_0.8)
            om = G.predict_one(arm, seed, Wm, rec, fz[seed], land, P)
            Wc = Wm.copy()
            Wc[:, 2] = extrapolate_linear(Wp[:, 2], t_obs - 1)
            oc = G.predict_one(arm, seed, Wc, rec, fz[seed], land, P)
            row = {"test": "GELU-T", "arm": arm, "seed": seed, "path_sha256_ok": True, "t_obs": t_obs,
                   "s_obs": float(r.s_obs), "s_true_at_t_obs": float(abs(Wp[t_obs, 2])),
                   "masked": _fields(om, GELU_KEYS), "causal": _fields(oc, GELU_KEYS),
                   "secs": round(time.time() - t0, 1)}
            _write(f, row)
            print(json.dumps({k: row[k] for k in ("arm", "seed", "t_obs", "secs")}), flush=True)
            _rss()


def adam_replay(seed, a, n_steps, lr=1e-2):
    """Track A's registered Adam run (track_a.train_one) to n_steps with a LOCAL torch Generator: θ, Adam's first moment
    and bias-corrected second moment at every step."""
    import torch
    from torch.nn import functional as F
    from .fold1d import activation, logits, make_data
    f = activation("sin_family", a)
    x, y = make_data(200, int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    th = torch.empty(4).uniform_(-1.0, 1.0, generator=gen).double().clone().requires_grad_(True)
    x, y = x.double(), y.double()
    opt = torch.optim.Adam([th], lr=lr)
    W = np.empty((n_steps + 1, 4))
    M = np.zeros((n_steps + 1, 4))
    V = np.full((n_steps + 1, 4), np.nan)
    W[0] = th.detach().numpy()
    for t in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(logits(th, x, f), y).backward()
        opt.step()
        W[t] = th.detach().numpy()
        st = opt.state[th]
        k = int(st["step"])
        M[t] = st["exp_avg"].numpy()
        V[t] = st["exp_avg_sq"].numpy() / (1 - 0.999 ** k)
    return W, M, V


def _ta_inputs(W, t_rule, rows_v, Mr, Vr, adam):
    """Masked Track A inputs: θ's hidden coordinates NaN except at t_R; Adam moments NaN except the named rows."""
    n = len(W)
    Wm = W.copy()
    keep = np.zeros(n, bool)
    keep[int(t_rule)] = True
    Wm[np.ix_(~keep, [0, 1, 3])] = np.nan
    M = np.full((n, 4), np.nan)
    V = np.full((n, 4), np.nan)
    K = np.zeros(n)
    if adam:
        M[int(t_rule)] = Mr[int(t_rule)]
        for t in rows_v:
            V[int(t)] = Vr[int(t)]
        K = np.arange(n, dtype=float)
    return Wm, M, V, K


def _track_a_one(test, opt, r, m, predict):
    """One Track A / 2A run: hash, (Adam) replay check, masked and causal predictions."""
    pf = m._path_file(opt, int(r.seed))
    assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
    W = np.load(pf)["W"]
    seed, t_obs, t_rule = int(r.seed), int(r.step_obs), int(r.t_rule)
    s = np.abs(W[:, 2])
    t_top = first_ge(s, float(r.s_frozen), 1)
    t_sw = first_ge(s, float(r.s_run))
    adam = opt == "adam"
    rows_v = [t_rule] + ([t_sw] if test == "track2a" else [])
    Mr = Vr = None
    replay_ok = None
    if adam:
        n = max(rows_v)
        Wr, Mr, Vr = adam_replay(seed, m.A, n)
        replay_ok = bool(np.array_equal(Wr, W[:n + 1]))
        assert replay_ok, f"Adam replay differs from the saved path ({test} {seed})"
    Wm, M, V, K = _ta_inputs(W, t_rule, rows_v, Mr, Vr, adam)
    om = predict(seed, Wm, M, V, K, float(r.s_frozen))
    Wc = Wm.copy()
    Wc[:, 2] = extrapolate_linear(W[:, 2], t_obs - 1)
    oc = predict(seed, Wc, M, V, K, float(r.s_frozen))
    keys = T2A_KEYS if test == "track2a" else TA_KEYS
    return {"test": "Test 2A" if test == "track2a" else "Track A", "opt": opt, "seed": seed, "path_sha256_ok": True,
            "adam_replay_bit_identical": replay_ok, "t_obs": t_obs, "t_rule": t_rule, "t_top": t_top,
            "t_sw_cf": t_sw, "s_obs": float(r.s_obs), "masked": _fields(om, keys), "causal": _fields(oc, keys)}


def replay_track_a():
    from . import track_a as TA
    d = pd.read_csv(RESULTS / "track_a" / "observed_runs.csv")
    f = REPLAY["track_a"]
    done = _done(f, ("opt", "seed"))
    for opt, g in d.groupby("opt"):
        use = g[g.crossed.astype(bool) & g.rule_before_cross.fillna(False).astype(bool) & np.isfinite(g.r_traj)
                & np.isfinite(g.r_obs)]
        for r in use.itertuples():
            if (opt, int(r.seed)) in done:
                continue
            _gate(f"track_a {opt} {r.seed}")
            t0 = time.time()
            row = _track_a_one("track_a", opt, r, TA,
                               lambda seed, W, M, V, K, sf, opt=opt: TA.predict_one(opt, seed, W, M, V, K, sf))
            row["secs"] = round(time.time() - t0, 1)
            _write(f, row)
            print(json.dumps({k: row[k] for k in ("opt", "seed", "t_obs", "secs")}), flush=True)
            _rss()


def replay_track2a():
    from . import track2a as T2
    m = T2.pipeline()
    d = pd.read_csv(RESULTS / "track2a" / "observed_runs.csv")
    f = REPLAY["track2a"]
    done = _done(f, ("opt", "seed"))
    for r in d[d.scored.astype(bool)].itertuples():
        if ("adam", int(r.seed)) in done:
            continue
        _gate(f"track2a {r.seed}")
        t0 = time.time()
        row = _track_a_one("track2a", "adam", r, m,
                           lambda seed, W, M, V, K, sf: T2.predict_2a(seed, W, M, V, K, sf, m=m))
        row["secs"] = round(time.time() - t0, 1)
        _write(f, row)
        print(json.dumps({k: row[k] for k in ("opt", "seed", "t_obs", "secs")}), flush=True)
        _rss()


# ------------------------------------------------------------------------------------------ 3b. TRUNCATED replays
# The MASKED inputs, and in addition EVERY row (hidden and output) after a claimed last-read step L set to NaN.  If the
# prediction's fields reproduce the committed values, the prediction reads nothing after step L; together with the code
# (which reads step L itself: s_traj = s[t_traj], s[t_sw] in ṡ, s[t_top] in the rule point) L is its last-read step.
#   L_traj (r_traj): W2-A max(t_traj, t_sw) (T, T': s_switch is the own-path switch; D: t_sw < t_traj in every run);
#                    GELU-T t_traj; Track A max(t_traj, t_top); 2A max(t_traj, t_top, t_sw) (P at t_sw).
#   L_cf (r_cf):     W2-A and GELU-T t_sw (with_traj=False); Track A and 2A max(t_top, t_sw).
TRUNC_FIELDS = {
    "w2a": (("t_sw", "s_switch", "t_traj", "s_traj", "r_traj", "follows_branch"),
            ("t_sw", "s_switch", "sdot", "r_cf", "chi_window_max", "follows_branch")),
    "gelu": (("t_traj", "s_traj", "r_traj", "follows_branch"),
             ("t_sw", "s_switch", "sdot", "r_cf", "chi_window_max", "follows_branch")),
    "track_a": (("t_rule", "s_run", "t_traj", "s_traj", "r_traj"), ("t_rule", "s_run", "sdot", "r_cf")),
    "track2a": (("t_rule", "s_run", "t_sw", "t_traj", "s_traj", "r_traj"), ("t_rule", "s_run", "t_sw", "sdot", "r_cf"))}


def _trunc_row(test, grp, r, L_traj, L_cf, ot, oc, key, t0, **extra):
    ft, fc = TRUNC_FIELDS[key]
    return {"test": test, grp: r.arm if grp == "arm" else r.opt, "seed": int(r.seed), "path_sha256_ok": True,
            "t_obs": int(r.step_obs), "L_traj": int(L_traj), "L_cf": int(L_cf), **extra,
            "trunc_traj": _fields(ot, ft), "trunc_cf": _fields(oc, fc), "secs": round(time.time() - t0, 1)}


def truncated_w2a():
    from . import width2_asym as W
    land, fz = W._land(), W._frozen()
    f = TRUNC["w2a"]
    done = _done(f, ("arm", "seed"))
    for r in _scored_w2a():
        if (r.arm, int(r.seed)) in done:
            continue
        _gate(f"trunc w2a {r.arm} {r.seed}")
        t0 = time.time()
        seed = int(r.seed)
        pf = W._path_file(r.arm, seed)
        assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
        P = np.load(pf)
        rec = {"on_branch": True, "copy_at_release": r.copy_at_release, "windings": json.loads(r.windings)}
        x, y = W.own_sample(seed)
        Pm = P.copy()
        Pm[1:, W.ZI] = np.nan
        tf = int(r.t_follow)
        Pm[tf, W.ZI] = P[tf, W.ZI]
        L_traj, L_cf = max(int(r.t_traj), int(r.t_sw)), int(r.t_sw)
        assert tf <= min(L_traj, L_cf)
        ot = W.predict_one(r.arm, seed, truncate_after(Pm, L_traj), rec, fz[seed], land, x, y)
        oc = W.predict_one(r.arm, seed, truncate_after(Pm, L_cf), rec, fz[seed], land, x, y, with_traj=False)
        row = _trunc_row("W2-A", "arm", r, L_traj, L_cf, ot, oc, "w2a", t0,
                         branch_steps_computed_trunc=ot.get("branch_steps_computed"))
        _write(f, row)
        print(json.dumps({k: row[k] for k in ("arm", "seed", "L_traj", "secs")}), flush=True)
        del P, Pm
        _rss()


def truncated_gelu():
    from . import gelu_transfer as G
    land, fz = G._land(), G._frozen()
    f = TRUNC["gelu"]
    done = _done(f, ("arm", "seed"))
    for arm in G.ARMS:
        d = pd.read_csv(RESULTS / "gelu_transfer" / f"observed_runs_{arm}.csv")
        for r in d[d.scored.astype(bool)].itertuples():
            if (arm, int(r.seed)) in done:
                continue
            _gate(f"trunc gelu {arm} {r.seed}")
            t0 = time.time()
            seed = int(r.seed)
            pf = G._path_file(arm, seed)
            assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
            Wp = np.load(pf)["W"]
            rec = {"on_branch": True, "copy_at_release": int(r.copy_at_release)}
            P = G.own_problem(seed)
            Wm = Wp.copy()
            Wm[1:, [0, 1, 3]] = np.nan
            tf = int(r.t_follow)
            Wm[tf, [0, 1, 3]] = Wp[tf, [0, 1, 3]]
            L_traj, L_cf = int(r.t_traj), int(r.t_sw)
            assert tf <= min(L_traj, L_cf)
            ot = G.predict_one(arm, seed, truncate_after(Wm, L_traj), rec, fz[seed], land, P)
            oc = G.predict_one(arm, seed, truncate_after(Wm, L_cf), rec, fz[seed], land, P, with_traj=False)
            row = _trunc_row("GELU-T", "arm", r, L_traj, L_cf, ot, oc, "gelu", t0,
                             traj_T_trunc=ot.get("traj_T"))
            _write(f, row)
            print(json.dumps({k: row[k] for k in ("arm", "seed", "L_traj", "secs")}), flush=True)
            _rss()


def _track_a_trunc_one(test, opt, r, m, predict):
    pf = m._path_file(opt, int(r.seed))
    assert _sha(pf) == r.path_sha256, f"path hash mismatch {pf}"
    W = np.load(pf)["W"]
    seed, t_rule, t_traj = int(r.seed), int(r.t_rule), int(r.t_traj)
    s = np.abs(W[:, 2])
    t_top = first_ge(s, float(r.s_frozen), 1)
    t_sw = first_ge(s, float(r.s_run))
    adam = opt == "adam"
    rows_v = [t_rule] + ([t_sw] if test == "track2a" else [])
    Mr = Vr = None
    if adam:
        n = max(rows_v)
        Wr, Mr, Vr = adam_replay(seed, m.A, n)
        assert np.array_equal(Wr, W[:n + 1]), f"Adam replay differs from the saved path ({test} {seed})"
    Wm, M, V, K = _ta_inputs(W, t_rule, rows_v, Mr, Vr, adam)
    L_traj = max(t_traj, t_top) if test == "track_a" else max(t_traj, t_top, t_sw)
    L_cf = max(t_top, t_sw)
    ot = predict(seed, truncate_after(Wm, L_traj), M, V, K, float(r.s_frozen))
    oc = predict(seed, truncate_after(Wm, L_cf), M, V, K, float(r.s_frozen))
    return L_traj, L_cf, ot, oc, {"t_top": t_top, "t_sw_cf": t_sw}


def truncated_track_a():
    from . import track_a as TA
    d = pd.read_csv(RESULTS / "track_a" / "observed_runs.csv")
    f = TRUNC["track_a"]
    done = _done(f, ("opt", "seed"))
    for opt, g in d.groupby("opt"):
        use = g[g.crossed.astype(bool) & g.rule_before_cross.fillna(False).astype(bool) & np.isfinite(g.r_traj)
                & np.isfinite(g.r_obs)]
        for r in use.itertuples():
            if (opt, int(r.seed)) in done:
                continue
            _gate(f"trunc track_a {opt} {r.seed}")
            t0 = time.time()
            L_traj, L_cf, ot, oc, ex = _track_a_trunc_one(
                "track_a", opt, r, TA, lambda seed, W, M, V, K, sf, opt=opt: TA.predict_one(opt, seed, W, M, V, K, sf))
            row = _trunc_row("Track A", "opt", r, L_traj, L_cf, ot, oc, "track_a", t0, **ex)
            _write(f, row)
            print(json.dumps({k: row[k] for k in ("opt", "seed", "L_traj", "secs")}), flush=True)
            _rss()


def truncated_track2a():
    from . import track2a as T2
    m = T2.pipeline()
    d = pd.read_csv(RESULTS / "track2a" / "observed_runs.csv")
    f = TRUNC["track2a"]
    done = _done(f, ("opt", "seed"))
    for r in d[d.scored.astype(bool)].itertuples():
        if ("adam", int(r.seed)) in done:
            continue
        _gate(f"trunc track2a {r.seed}")
        t0 = time.time()
        L_traj, L_cf, ot, oc, ex = _track_a_trunc_one(
            "track2a", "adam", r, m, lambda seed, W, M, V, K, sf: T2.predict_2a(seed, W, M, V, K, sf, m=m))
        row = _trunc_row("Test 2A", "opt", r, L_traj, L_cf, ot, oc, "track2a", t0, **ex)
        _write(f, row)
        print(json.dumps({k: row[k] for k in ("opt", "seed", "L_traj", "secs")}), flush=True)
        _rss()


# ------------------------------------------------------------------------------------------ 2. + summary
def _replay_rows(key):
    p = REPLAY[key]
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


def _repro_counts(rows, committed, key, variant):
    """Per field: runs whose `variant` replay equals the committed value (committed: {(group, seed): row})."""
    out = {}
    grp = "opt" if key in ("track_a", "track2a") else "arm"
    for fld in REPRO[key]:
        eq = [rel_equal(r[variant].get(fld), _num(committed[(r[grp], r["seed"])].get(fld))) for r in rows]
        out[fld] = {"n": len(eq), "n_equal": int(sum(eq))}
    return out


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def _causal_stats(rows, committed, grp, s_switch_key):
    """POST HOC causal variant against the registered values (scored runs)."""
    tt, tt_reg, to, rt_c, rt, ro, ro_c, ss_rel = [], [], [], [], [], [], [], []
    n_nopred = 0
    for r in rows:
        c = committed[(r[grp], r["seed"])]
        cz = r["causal"]
        if cz.get("r_traj") is None or cz.get("t_traj") is None:
            n_nopred += 1
            continue
        s_sw_c = cz.get(s_switch_key)
        s_sw = _num(c.get(s_switch_key))
        tt.append(float(cz["t_traj"]))
        tt_reg.append(float(c["t_traj"]))
        to.append(float(r["t_obs"]))
        rt_c.append(float(cz["r_traj"]))
        rt.append(float(c["r_traj"]))
        ro.append(float(c["r_obs"]))
        ro_c.append(float(r["s_obs"]) / float(s_sw_c) - 1)
        ss_rel.append(abs(float(s_sw_c) / float(s_sw) - 1))
    tt, tt_reg, to, rt_c, rt, ro, ro_c, ss_rel = map(np.asarray, (tt, tt_reg, to, rt_c, rt, ro, ro_c, ss_rel))
    n = len(tt)
    if n == 0:
        return {"n": 0, "n_no_prediction": n_nopred}
    rel = np.abs(rt_c / rt - 1)
    return {"n": n, "n_no_prediction": n_nopred,
            "n_t_traj_changed": int(np.sum(tt != tt_reg)),
            "t_traj_minus_t_obs": read_beyond(tt, to),
            "n_r_traj_bit_or_rtol_equal_registered": int(sum(rel_equal(a, b) for a, b in zip(rt_c, rt))),
            "max_rel_change_r_traj": float(rel.max()), "median_rel_change_r_traj": float(np.median(rel)),
            "max_rel_change_s_switch": float(ss_rel.max()),
            "median_ratio_r_obs_over_r_traj_causal_same_s_switch": float(np.median(ro_c / rt_c)),
            "spearman_r_traj_causal_vs_r_obs_same_s_switch": spearman(rt_c, ro_c) if n >= 3 else None,
            "median_ratio_r_obs_registered_over_r_traj_causal": float(np.median(ro / rt_c)),
            "median_ratio_registered_r_obs_over_r_traj": float(np.median(ro / rt))}


def _w2a_counts():
    out = {}
    for arm in ("T", "D", "Tp"):
        d = pd.read_csv(RESULTS / "width2_asym" / f"observed_runs_{arm}.csv")
        S = d[d.scored.astype(bool)]
        on = d[d.on_branch.astype(bool)]
        own = arm in ("T", "Tp")
        last_traj = S.t_traj                                   # r4 hit: v_0..v_t_hit
        last_all = last_read(S.t_traj, S.t_sw)                 # own branch / switch / ṡ window read up to t_sw
        neg = S[S.kappa < 0]
        a = {"n_runs": int(len(d)), "n_on_branch": int(len(on)), "n_scored": int(len(S)),
             "own_path_switch": own,
             "r_traj_recursion_last_read_vs_t_obs": read_beyond(last_traj, S.step_obs),
             "n_t_traj_equal_t_obs": int((S.t_traj == S.step_obs).sum()),
             "L1_ratio_split": ratio_split(S.t_traj, S.step_obs, S.s_traj, S.s_obs, S.r_traj, S.r_obs),
             "t_sw_vs_t_obs": read_beyond(S.t_sw, S.step_obs),
             "r_traj_any_input_last_read_vs_t_obs": read_beyond(
                 last_read(S.t_traj, S.t_sw) if own else S.t_traj, S.step_obs),
             "any_prediction_last_v_read_vs_t_obs": read_beyond(last_all, S.step_obs),
             "n_t_sw_before_t_traj": int((S.t_sw < S.t_traj).sum()),
             "branch_steps_computed_minus_1_equals_max_t_traj_t_sw": int(((S.branch_steps_computed - 1)
                                                                          == last_all).sum()),
             "r_cf_sdot_window": sdot_window(S.t_sw, S.step_obs),
             "n_kappa_negative_scored": int(len(neg)),
             "kappa_negative_t_sw_vs_t_obs": read_beyond(neg.t_sw, neg.step_obs),
             "observed_lag_steps_t_obs_minus_t_sw": {"min": float((S.step_obs - S.t_sw).min()),
                                                     "median": float((S.step_obs - S.t_sw).median()),
                                                     "max": float((S.step_obs - S.t_sw).max())},
             "follow_check_t_follow_vs_t_obs": read_beyond(S.t_follow, S.step_obs),
             "n_on_branch_not_following_excluded": int((on.follows_branch.fillna(False).astype(bool) == False).sum()),  # noqa: E712
             "n_on_branch_crossed_follow_check_at_or_after_crossing": int(
                 (on.crossed.fillna(False).astype(bool) & (on.t_follow >= on.step_obs)).sum()),
             "n_hold_G_positive_all_runs": int(d.hold_G_positive.astype(bool).sum()),
             "n_branch_lost_scored": int(S.branch_lost_at.notna().sum()),
             "registered": {k: json.loads((RESULTS / "width2_asym" / "scores.json").read_text())["arms"][arm][k]
                            for k in ("L1", "L3")}}
        out[arm] = a
    return out


def _gelu_counts():
    out = {}
    for arm in ("random", "branch"):
        d = pd.read_csv(RESULTS / "gelu_transfer" / f"observed_runs_{arm}.csv")
        S = d[d.scored.astype(bool)]
        on = d[d.on_branch.astype(bool)]
        neg = S[S.kappa < 0]
        out[arm] = {"n_runs": int(len(d)), "n_on_branch": int(len(on)), "n_scored": int(len(S)),
                    "r_traj_recursion_last_read_vs_t_obs": read_beyond(S.t_traj, S.step_obs),
                    "n_t_traj_equal_t_obs": int((S.t_traj == S.step_obs).sum()),
                    "L1_ratio_split": ratio_split(S.t_traj, S.step_obs, S.s_traj, S.s_obs, S.r_traj, S.r_obs),
                    "kappa_negative_t_sw_minus_t_traj": {"min": _mn(neg.t_sw - neg.t_traj),
                                                         "max": _mx(neg.t_sw - neg.t_traj)},
                    "s_traj_minus_s_obs_over_s_switch_among_after": _after_ds(S),
                    "segment_end_traj_T_min_scored": int(S.traj_T.min()),
                    "n_traj_T_greater_than_t_obs": int((S.traj_T > S.step_obs).sum()),
                    "t_sw_vs_t_obs": read_beyond(S.t_sw, S.step_obs),
                    "r_cf_sdot_window": sdot_window(S.t_sw, S.step_obs),
                    "n_kappa_negative_scored": int(len(neg)),
                    "kappa_negative_t_sw_vs_t_obs": read_beyond(neg.t_sw, neg.step_obs),
                    "kappa_positive_t_sw_vs_t_obs": read_beyond(S[S.kappa > 0].t_sw, S[S.kappa > 0].step_obs),
                    "follow_check_t_follow_vs_t_obs": read_beyond(S.t_follow, S.step_obs),
                    "n_on_branch_not_following_excluded": int((~on.follows_branch.fillna(False).astype(bool)).sum()),
                    "n_hold_G_positive_all_runs": int(d.hold_G_positive.astype(bool).sum()),
                    "registered": {k: json.loads((RESULTS / "gelu_transfer" / "scores.json").read_text())
                                   ["arms"][arm][k] for k in ("L1", "L3")}}
    return out


def _after_ds(S):
    A = S[S.t_traj > S.step_obs]
    v = ((A.s_traj - A.s_obs) / A.s_switch).to_numpy(float)
    return {"n": int(len(v)), "min": float(v.min()) if len(v) else None, "median": _med(v),
            "max": float(v.max()) if len(v) else None}


def _track_a_counts(rows, test):
    """Track A per optimiser / 2A: the scored set is rebuilt as registered (Track A: crossed, rule point strictly before
    the crossing, finite r_traj and r_obs for L1/L3; 2A: observed_runs.csv `scored`)."""
    out = {}
    if test == "track_a":
        d = pd.read_csv(RESULTS / "track_a" / "observed_runs.csv")
        groups = {o: g[g.crossed.astype(bool) & g.rule_before_cross.fillna(False).astype(bool)
                       & np.isfinite(g.r_traj) & np.isfinite(g.r_obs)] for o, g in d.groupby("opt")}
        sc = json.loads((RESULTS / "track_a" / "scores.json").read_text())
        reg = {o: {k: sc[o][k] for k in ("L1", "L3")} for o in groups}
    else:
        d = pd.read_csv(RESULTS / "track2a" / "observed_runs.csv")
        groups = {"adam": d[d.scored.astype(bool)]}
        sc = json.loads((RESULTS / "track2a" / "scores.json").read_text())
        reg = {"adam": {k: sc[k] for k in ("A1", "A3")}}
    rr = {(r["opt"], r["seed"]): r for r in rows}
    for o, S in groups.items():
        R_ = [rr.get((o, int(s))) for s in S.seed]
        have = [r for r in R_ if r is not None]
        a = {"n_scored": int(len(S)),
             "r_traj_recursion_last_read_vs_t_obs": read_beyond(S.t_traj, S.step_obs),
             "n_t_traj_equal_t_obs": int((S.t_traj == S.step_obs).sum()),
             "L1_ratio_split": ratio_split(S.t_traj, S.step_obs, S.s_traj, S.s_obs, S.r_traj, S.r_obs),
             "rule_point_t_R_vs_t_obs": read_beyond(S.t_rule, S.step_obs),
             "registered": reg[o]}
        if test == "track2a":
            a["t_sw_vs_t_obs"] = read_beyond(S.t_sw, S.step_obs)
            a["r_cf_sdot_window"] = sdot_window(S.t_sw, S.step_obs)
            a["n_P_read_at_t_sw_after_t_R"] = int((S.t_sw > S.t_rule).sum())
            a["t_sw_minus_t_R"] = {"min": float((S.t_sw - S.t_rule).min()),
                                   "median": float((S.t_sw - S.t_rule).median()),
                                   "max": float((S.t_sw - S.t_rule).max())}
        if len(have) == len(S):                                   # path-derived (replay rows)
            to = np.array([r["t_obs"] for r in have], float)
            a["rule_definition_t_top_vs_t_obs"] = read_beyond([r["t_top"] for r in have], to)
            a["r_cf_t_sw_vs_t_obs_from_path"] = read_beyond([r["t_sw_cf"] for r in have], to)
            a["r_cf_sdot_window_from_path"] = sdot_window([r["t_sw_cf"] for r in have], to)
            t_top = [r["t_top"] for r in have]
            t_swp = [r["t_sw_cf"] for r in have]
            tt = S.t_traj.to_numpy(float)
            a["r_traj_any_input_last_read_vs_t_obs"] = read_beyond(
                last_read(tt, t_top) if test == "track_a" else last_read(tt, t_top, t_swp), to)
            a["r_cf_any_input_last_read_vs_t_obs"] = read_beyond(last_read(t_top, t_swp), to)
            a["n_t_obs_replay_equal_csv"] = int((to == S.step_obs.to_numpy(float)).sum())
            if o == "adam":
                a["n_adam_replay_bit_identical"] = int(sum(bool(r["adam_replay_bit_identical"]) for r in have))
        out[o] = a
    return out


def _replay_summary(key, rows, committed, grp, s_switch_key):
    if not rows:
        return {"n_rows": 0, "note": "replay not run (paths needed)"}
    out = {}
    for g in sorted({r[grp] for r in rows}):
        rg = [r for r in rows if r[grp] == g]
        out[g] = {"n_rows": len(rg), "n_path_sha256_verified": int(sum(bool(r["path_sha256_ok"]) for r in rg)),
                  "masked_reproduces": _repro_counts(rg, committed, key, "masked"),
                  "masked_follows_branch_true": int(sum(bool(r["masked"].get("follows_branch")) for r in rg))
                  if key in ("w2a", "gelu") else None,
                  "causal_POST_HOC": _causal_stats(rg, committed, grp, s_switch_key),
                  "causal_reproduces_registered": _repro_counts(rg, committed, key, "causal")}
    return out


def _trunc_summary(key, rows, committed, grp):
    """TRUNCATED replay (rows after L set to NaN, on top of MASKED): per field, runs reproducing the committed value;
    where L lies against t_obs."""
    if not rows:
        return {"n_rows": 0, "note": "truncated replay not run (paths needed)"}
    ft, fc = TRUNC_FIELDS[key]
    out = {}
    for g in sorted({r[grp] for r in rows}):
        rg = [r for r in rows if r[grp] == g]
        rep_ = {}
        for variant, flds in (("trunc_traj", ft), ("trunc_cf", fc)):
            rep_[variant] = {fld: {"n": len(rg), "n_equal": int(sum(
                rel_equal(r[variant].get(fld), _num(committed[(r[grp], r["seed"])].get(fld))) for r in rg))}
                for fld in flds}
        to = [r["t_obs"] for r in rg]
        out[g] = {"n_rows": len(rg), "n_path_sha256_verified": int(sum(bool(r["path_sha256_ok"]) for r in rg)),
                  "reproduces": rep_,
                  "n_all_traj_fields_reproduced": int(sum(all(rel_equal(r["trunc_traj"].get(fl), _num(
                      committed[(r[grp], r["seed"])].get(fl))) for fl in ft) for r in rg)),
                  "n_all_cf_fields_reproduced": int(sum(all(rel_equal(r["trunc_cf"].get(fl), _num(
                      committed[(r[grp], r["seed"])].get(fl))) for fl in fc) for r in rg)),
                  "L_traj_vs_t_obs": read_beyond([r["L_traj"] for r in rg], to),
                  "L_cf_vs_t_obs": read_beyond([r["L_cf"] for r in rg], to)}
    return out


def _committed(key):
    if key == "w2a":
        d = pd.concat([pd.read_csv(RESULTS / "width2_asym" / f"observed_runs_{a}.csv") for a in ("T", "D", "Tp")])
        return {(r["arm"], int(r["seed"])): r for r in d.to_dict("records")}
    if key == "gelu":
        d = pd.concat([pd.read_csv(RESULTS / "gelu_transfer" / f"observed_runs_{a}.csv") for a in ("random", "branch")])
        return {(r["arm"], int(r["seed"])): r for r in d.to_dict("records")}
    d = pd.read_csv(RESULTS / key / "observed_runs.csv")
    return {(r["opt"], int(r["seed"])): r for r in d.to_dict("records")}


def _rows(path):
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def build():
    """All counts from the committed CSVs, scores and replay rows (no path is read; no file is written)."""
    rows = {k: _replay_rows(k) for k in REPLAY}
    trunc = {k: _rows(TRUNC[k]) for k in TRUNC}
    hashes = {k: _sha(RESULTS / k / "predictions.csv") == (RESULTS / k / "predictions.sha256").read_text().split()[0]
              for k in ("width2_asym", "gelu_transfer", "track2a", "track_a")}
    res = {"note": "AUDIT of prediction inputs (2026-09-30). The CAUSAL replay is a POST HOC diagnostic; it replaces "
                   "no registered prediction or verdict.",
           "committed_predictions_hash_matches": hashes,
           "code_refs": [dict(zip(("test", "prediction", "input", "class", "file", "line", "token"), c))
                         for c in CODE_REFS],
           "code_refs_mismatches": code_refs_check(),
           "md_refs_not_in_code_refs": md_refs_check(AUDIT_MD.read_text()) if AUDIT_MD.exists() else None,
           "replay_row_counts": {k: len(v) for k, v in rows.items()},
           "truncated_row_counts": {k: len(v) for k, v in trunc.items()},
           "W2-A": {"arms": _w2a_counts(),
                    "replay": _replay_summary("w2a", rows["w2a"], _committed("w2a"), "arm", "s_switch"),
                    "truncated": _trunc_summary("w2a", trunc["w2a"], _committed("w2a"), "arm")},
           "GELU-T": {"arms": _gelu_counts(),
                      "replay": _replay_summary("gelu", rows["gelu"], _committed("gelu"), "arm", "s_switch"),
                      "truncated": _trunc_summary("gelu", trunc["gelu"], _committed("gelu"), "arm")},
           "Track A": {"opts": _track_a_counts(rows["track_a"], "track_a"),
                       "replay": _replay_summary("track_a", rows["track_a"], _committed("track_a"), "opt", "s_run"),
                       "truncated": _trunc_summary("track_a", trunc["track_a"], _committed("track_a"), "opt")},
           "Test 2A": {"opts": _track_a_counts(rows["track2a"], "track2a"),
                       "replay": _replay_summary("track2a", rows["track2a"], _committed("track2a"), "opt", "s_run"),
                       "truncated": _trunc_summary("track2a", trunc["track2a"], _committed("track2a"), "opt")}}
    return _clean(res)


def summarize():
    """build() -> results/prediction_inputs_audit.json."""
    res = build()
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "code_refs"}, indent=1))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "replay":
        _nice()
        {"w2a": replay_w2a, "gelu": replay_gelu, "track_a": replay_track_a, "track2a": replay_track2a}[sys.argv[2]]()
    elif cmd == "truncated":
        _nice()
        {"w2a": truncated_w2a, "gelu": truncated_gelu, "track_a": truncated_track_a,
         "track2a": truncated_track2a}[sys.argv[2]]()
    elif cmd == "summarize":
        summarize()
    else:
        raise SystemExit(f"unknown command {cmd}")
