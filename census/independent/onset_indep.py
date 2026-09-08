"""1b: independent onset location for family A at B=2000. Currently 1.60.

Reimplemented from the definition: the smallest grid value with rate >= 0.5
that has a strictly smaller value below 0.5. Training reimplemented too --
plain numpy Adam from the update rule, not torch.
"""
from __future__ import annotations
import numpy as np
from reimpl import net, separates, f_sin

INNER_HALF, OUTER_LO, OUTER_HI = 0.8, 1.2, 2.0


def make_data_indep(n_per_class, seed):
    """Data from the task definition: uniform on each class region."""
    rng = np.random.default_rng(seed)
    inner = rng.uniform(-INNER_HALF, INNER_HALF, n_per_class)
    mag = rng.uniform(OUTER_LO, OUTER_HI, n_per_class)
    sign = rng.choice([-1.0, 1.0], n_per_class)
    x = np.concatenate([inner, mag * sign])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return x, y


def adam_train(a, seed, steps=2000, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    """Adam from its update rule (Kingma & Ba), bias correction included."""
    x, y = make_data_indep(200, seed)
    rng = np.random.default_rng(seed + 10**6)
    th = rng.uniform(-1.0, 1.0, 4)
    m = np.zeros(4); v = np.zeros(4)
    for t in range(1, steps + 1):
        z = net(th, x, a)
        s = 1.0 / (1.0 + np.exp(-z))          # sigma(z)
        d = (s - y) / len(x)                   # dL/dz for mean BCE-with-logits
        w1, bb1, w2, bb2 = th
        u = w1 * x + bb1
        fu = f_sin(u, a)
        fp = 1.0 + a * np.cos(u)
        g = np.array([(d * w2 * fp * x).sum(), (d * w2 * fp).sum(),
                      (d * fu).sum(), d.sum()])
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1**t); vh = v / (1 - b2**t)
        th = th - lr * mh / (np.sqrt(vh) + eps)
    return th


def rate(a, seeds=40, steps=2000):
    return sum(separates(adam_train(a, s, steps), a) for s in range(seeds)) / seeds


if __name__ == "__main__":
    grid = [1.70, 1.60, 1.55, 1.50, 1.45, 1.40, 1.35]
    onset, brk = None, False
    for a in grid:
        r = rate(a)
        print(f"  independent a={a}: rate={r:.3f}", flush=True)
        if r >= 0.5:
            onset = a
        elif onset is not None:
            brk = True; break
    print(f"\nINDEPENDENT ONSET at B=2000: {onset} (bracketed={brk})")
    print(f"PRODUCTION ONSET:              1.60")
    print(f"AGREE: {onset == 1.60}")
