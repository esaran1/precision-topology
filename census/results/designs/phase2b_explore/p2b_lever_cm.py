"""EXPLORATORY (NOT a registration): confirmation run of the COST-MATCHED global arm for the 2B page, on the
non-registered exploration seeds 2,953,000-2,953,019 only (never to be registered).

Rule (fixed from committed exploration runs, before this confirmation): steps-to-3s* ratio vs std is modelled as a power
law in the global factor through (1, 1) and (12.18, r_g), r_g = median_seeds t_glob(3s*)/t_std(3s*) = 9.4541 (runs_lever,
bbf5406). Target r_o = median_seeds t_out16(3s*)/t_std(3s*) = 12.1773. G_cm = 12.18 ** (ln r_o / ln r_g), rounded to 2
decimals = 16.14. Runs: Adam lr 0.01/G_cm on every parameter, 40,000 steps (T_C), via p2b_lever.run (cond globcm).
    python p2b_lever_cm.py run SEED | summarise
"""
import json, math, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import p2b_lever as PL

SEEDS = PL.SEEDS
t3 = lambda c, s: json.loads((PL.RUNS / f"{c}_{s}.json").read_text())["t_scale"]["3s*"]  # noqa: E731
R_G = float(np.median([t3("glob", s) / t3("std", s) for s in SEEDS]))
R_O = float(np.median([t3("out16", s) / t3("std", s) for s in SEEDS]))
G_CM = round(12.18 ** (math.log(R_O) / math.log(R_G)), 2)
PL.T_RUN = 40_000
PL.CKPT = (1_000, 2_000, 5_000, 10_000, 20_000, 40_000)
_lr = PL.lr_vec
PL.lr_vec = lambda c: np.full(17, PL.X.ADAM_LR / G_CM) if c == "globcm" else _lr(c)


def summarise():
    std = {s: json.loads((PL.RUNS / f"std_{s}.json").read_text()) for s in SEEDS}
    cm = {s: json.loads((PL.RUNS / f"globcm_{s}.json").read_text()) for s in SEEDS}
    out16 = {s: json.loads((PL.RUNS / f"out16_{s}.json").read_text()) for s in SEEDS}
    pts = ["scale_1.25s*", "scale_2s*", "scale_3s*", "loss_0.1", "loss_0.03", "loss_0.01", "step_40000"]
    res = {"label": "EXPLORATORY (seeds 2,953,000-2,953,019; never to be registered)", "R_G": R_G, "R_O": R_O,
           "G_cm": G_CM, "cost_ratio_3s*_vs_std": PL._q([cm[s]["t_scale"]["3s*"] / std[s]["t_scale"]["3s*"] for s in SEEDS]),
           "cost_ratio_3s*_vs_out16": PL._q([cm[s]["t_scale"]["3s*"] / out16[s]["t_scale"]["3s*"] for s in SEEDS]),
           "paired_vs_std": {}}
    for p in pts:
        for q in ("rho2", "acc_shuffled", "acc_reversed"):
            d = [cm[s]["at"][p][q] - std[s]["at"][p][q] for s in SEEDS]
            res["paired_vs_std"][f"{p}|{q}"] = {"median": float(np.median(d)), "up": sum(x > 0 for x in d),
                                                 "down": sum(x < 0 for x in d)}
    res["onset_T40000"] = [cm[s]["onset"]["T40000"] and cm[s]["onset"]["T40000"]["s_over_s_star"] for s in SEEDS]
    (PL.HERE / "p2b_lever_cm.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("R_G", "R_O", "G_cm", "cost_ratio_3s*_vs_std", "cost_ratio_3s*_vs_out16")}, default=float))
    for k, v in res["paired_vs_std"].items():
        print(k, v)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        PL.run("globcm", int(sys.argv[2]))
    else:
        summarise()
