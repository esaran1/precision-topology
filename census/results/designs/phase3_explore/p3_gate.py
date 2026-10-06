"""EXPLORATORY arithmetic only (no data drawn): Wilson 95% intervals of the exploration landing rates and the binomial
chance of meeting each candidate gate at the Wilson low and at the point estimate; smallest N for P ≥ 0.95 at the low."""
import math


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def p_ge(N, m, p):
    return sum(math.comb(N, j) * p ** j * (1 - p) ** (N - j) for j in range(m, N + 1))


def n_for(m, p, target=0.95):
    N = m
    while p_ge(N, m, p) < target:
        N += 10
    return N


rows = [("T4 random hold, own samples 7,410,000-019 (20 x 20)", 53, 400),
        ("T4 random hold, population (box 1)", 47, 200),
        ("any T-type 31/22 unplaced, own samples", 179, 400),
        ("Q branch-point start, own samples", 20, 20),
        ("S branch-point start, own samples", 20, 20),
        ("Q random hold, own samples", 0, 400),
        ("Q random hold, population (box 3)", 1, 150)]
for name, k, n in rows:
    lo, hi = wilson(k, n)
    print(f"{name}: {k}/{n} = {k / n:.4f}, Wilson 95% [{lo:.4f}, {hi:.4f}]")
lo = wilson(53, 400)[0]
for N in (400, 500, 600, 700, 800):
    print(f"T4 gate >= 60 of N={N}: P at low {lo:.4f} = {p_ge(N, 60, lo):.4f}; at 0.1325 = {p_ge(N, 60, 0.1325):.4f}")
for m in (60, 40):
    print(f"T4: smallest N (step 10) with P(>= {m}) >= 0.95 at the Wilson low: {n_for(m, lo)}")
lt = wilson(179, 400)[0]
print(f"T-type set gate >= 60 of 200: P at low {lt:.4f} = {p_ge(200, 60, lt):.5f}")
lq = wilson(20, 20)[0]
for p in (lq, 0.95, 0.99):
    print(f"branch-point gate >= 108/120 at p = {p:.4f}: P = {p_ge(120, 108, p):.4f}")
