"""Verify OUR linked-tori construction is a genuine link.

Three checks, for the paper's section 2:
  (1) Gauss linking integral of the two CORE circles = +-1
  (2) neither core self-intersects (they are round circles, so this is exact)
  (3) tube radius < reach, so the thickened solid tori are disjoint and embedded
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


def cores(n: int, major_radius: float = 1.0):
    """The two core circles of src/data.py::linked_tori, verbatim."""

    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    A = major_radius * np.column_stack((c, s, np.zeros_like(th)))
    B = major_radius * np.column_stack((1.0 + c, np.zeros_like(th), s))
    dA = major_radius * np.column_stack((-s, c, np.zeros_like(th)))
    dB = major_radius * np.column_stack((-s, np.zeros_like(th), c))
    dt = 2.0 * np.pi / n
    return A, B, dA * dt, dB * dt


def gauss_linking(n: int = 3000, major_radius: float = 1.0) -> float:
    """Lk = 1/(4 pi) * int int  (a-b) . (da x db) / |a-b|^3."""

    A, B, dA, dB = cores(n, major_radius)
    total = 0.0
    for i in range(0, len(A), 200):
        a = A[i:i + 200, None, :]
        da = dA[i:i + 200, None, :]
        r = a - B[None, :, :]
        cross = np.cross(np.broadcast_to(da, r.shape), dB[None, :, :])
        total += float((np.einsum("ijk,ijk->ij", r, cross)
                        / np.linalg.norm(r, axis=2) ** 3).sum())
    return total / (4.0 * np.pi)


def core_separation(n: int = 4000, major_radius: float = 1.0) -> float:
    A, B, _, _ = cores(n, major_radius)
    best = np.inf
    for i in range(0, len(A), 400):
        d = np.linalg.norm(A[i:i + 400, None, :] - B[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def self_min(n: int = 4000, major_radius: float = 1.0) -> float:
    """Minimum distance between non-adjacent points of one core."""

    A, _, _, _ = cores(n, major_radius)
    best = np.inf
    for i in range(0, len(A), 400):
        d = np.linalg.norm(A[i:i + 400, None, :] - A[None, :, :], axis=2)
        idx = np.arange(i, min(i + 400, n))
        for r, ii in enumerate(idx):
            circ = np.minimum(np.abs(np.arange(n) - ii), n - np.abs(np.arange(n) - ii))
            m = circ > n // 20
            if m.any():
                best = min(best, float(d[r][m].min()))
    return best


def main() -> None:
    R = 1.0
    TUBE = 0.2               # the value used throughout (amplification.py, basin.py)
    print("=== our linked_tori construction (src/data.py) ===\n")
    print("  class 0: core = R*(cos t, sin t, 0),  frame (normal, binormal) ="
          " ((cos t, sin t, 0), (0,0,1))")
    print("  class 1: core = R*(1+cos t, 0, sin t), frame = ((cos t,0,sin t), (0,1,0))")
    print("  offsets: rho*sqrt(U) in the (normal, binormal) disc, uniform in area\n")

    lk = gauss_linking()
    sep = core_separation()
    smin = self_min()
    reach = R                # for a unit circle of radius R the reach is R
    print(f"  (1) Gauss linking integral of the cores : {lk:+.6f}  -> "
          f"{'LINKED (|Lk| = 1)' if abs(abs(lk) - 1.0) < 0.02 else 'NOT +-1'}")
    print(f"  (2) core self-min distance (non-adjacent): {smin:.6f}  -> "
          f"{'no self-intersection' if smin > 1e-6 else 'SELF-INTERSECTS'}")
    print(f"      (each core is a round circle, so this is exact by construction)")
    print(f"  (3) core-to-core minimum distance        : {sep:.6f}  (= R exactly)")
    print(f"      reach of each core circle            : {reach:.6f}")
    print(f"      tube radius used                     : {TUBE:.6f}")
    print(f"      disjoint tubes need 2*rho < core separation: "
          f"2*{TUBE} = {2*TUBE} < {sep:.4f}  -> {2*TUBE < sep}")
    print(f"      embedded tube needs rho < reach:       {TUBE} < {reach}  -> {TUBE < reach}")
    print(f"\n  data.py enforces this: `0 < tube_radius < major_radius/2` "
          f"(= {R/2}), which is exactly 2*rho < R.")
    pd.DataFrame([{"linking_number": lk, "core_separation": sep,
                   "core_self_min": smin, "reach": reach, "tube_radius": TUBE,
                   "tubes_disjoint": bool(2 * TUBE < sep),
                   "tube_embedded": bool(TUBE < reach)}]).to_csv(
        RESULTS / "our_link_verification.csv", index=False)
    print("\nwritten results/our_link_verification.csv")


if __name__ == "__main__":
    main()
