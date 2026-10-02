"""EXPLORATORY diagnostics (exploration seeds 2,953,000–2,953,002 only): per-unit state along the scale-only runs.
For each seed and condition, at checkpoints: s, per-unit |v_k| / s and ‖(w_k, c_k)‖, the full-gradient max-norm, and
(for the Phase 2A literal rule) the per-step s increment that comes from output-weight sign changes vs the rest.

    python p2b_diag.py > p2b_diag.log   (writes p2b_diag.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2b_explore as E  # noqa: E402

CKPT = (100, 1_000, 10_000, 100_000, 1_000_000)
out = []
for seed in E.SEEDS[:3]:
    for cond, T in (("gd_wn_r9", 1_000_000), ("gd_wn_r6", 200_000), ("adam_wn_r64", 80_000), ("gd_scale_r9", 2_000)):
        th, _ = E.init_row(seed)
        step = E.make_stepper(E.CONDS[cond])
        ds_flip = ds_other = 0.0
        rec = []
        for t in range(1, T + 1):
            new = step(th)
            ds = np.abs(new[12:16]).sum() - np.abs(th[12:16]).sum()
            if (np.sign(new[12:16]) != np.sign(th[12:16])).any():
                ds_flip += ds
            else:
                ds_other += ds
            th = new
            if t in CKPT or t == T:
                _, g = E.loss_grad(th)
                W = th[:8].reshape(4, 2); c = th[8:12]; v = th[12:16]; s = np.abs(v).sum()
                rec.append({"t": t, "s": float(s), "rho2": float(E.rho2_rows(th[None])[0]),
                            "v_share": (np.abs(v) / s).round(6).tolist(),
                            "hidden_norm": np.sqrt((W ** 2).sum(1) + c ** 2).round(6).tolist(),
                            "W": W.round(4).tolist(), "c": c.round(4).tolist(),
                            "grad_max": float(np.abs(g).max()), "grad_v_max": float(np.abs(g[12:16]).max())})
            if np.abs(th[12:16]).sum() >= E.SCALES["3s*"] and cond != "gd_wn_r9":
                break
        r = {"seed": seed, "cond": cond, "steps": t, "s_increment_at_sign_changes": ds_flip,
             "s_increment_otherwise": ds_other, "records": rec}
        out.append(r)
        print(json.dumps({k: r[k] for k in ("seed", "cond", "steps", "s_increment_at_sign_changes", "s_increment_otherwise")}
                         | {"last": rec[-1]}), flush=True)
(HERE / "p2b_diag.json").write_text(json.dumps({"label": "EXPLORATORY diagnostics (exploration seeds; never registered)",
                                                "runs": out}, indent=1))
