"""Phase 3 ONLY (design draft; not registered): the ACTIVITY TEST at release, with its tolerance fixed now.

Introduced after the 2A-PS post hoc diagnosis (2f0fb33): there a numerically dead unit (|v₂| = 1.179e−93) passed the
registered rule v ≠ 0.0, entered the scale direction â = sign(v)/√|A|, and leaked the scale direction into the full-η
part of the update (Δs at step 1 0.1028 instead of 9.408e−6).  It applies to Phase 3 only and changes nothing
registered.

Rule: unit i is ACTIVE iff |v_i| ≥ THETA·s, s = ‖v‖₁ (inclusive), with THETA = 1e−8.
- Justification: the dead 2A-PS unit has |v|/s ≈ 4e−94 (85 orders below THETA); the smallest genuine active share in
  the Phase 3 exploration is 0.036 (a single unit's share at the switch of three-function copy 44, p3_branches_multi;
  every release share is ≥ 0.1), 6.5 orders above THETA.  THETA is far above float64 underflow residue and far below
  any share a held release or a frozen adiabatic path carries.
- Use in Phase 3: at release a run is on a copy only if every unit the copy uses is active; a run with an inactive unit
  is classified 'neither (inactive unit)' and counted.  Any scale direction (â, e.g. a forecaster's or a diagnostic's
  split of the v update into scale and share parts) is built from the active units only.
"""
import numpy as np

THETA = 1e-8


def active_units(v, theta=THETA):
    v = np.asarray(v, float)
    s = float(np.abs(v).sum())
    if s == 0.0:
        return []
    return [i for i in range(len(v)) if abs(v[i]) >= theta * s]


def scale_direction(v, theta=THETA):
    """â = sign(v_A)/√|A| on the active set A, 0 elsewhere."""
    v = np.asarray(v, float)
    a = np.zeros_like(v)
    A = active_units(v, theta)
    if A:
        a[A] = np.sign(v[A]) / np.sqrt(len(A))
    return a


def scale_only_step(v, g, eta, rho, a_hat):
    """The 2A-PS scale-only update (ρ on the part along â, full η on the rest): v − η[ρ(â·g)â + (I − ââᵀ)g]."""
    v, g, a = (np.asarray(t, float) for t in (v, g, a_hat))
    par = (a @ g) * a
    return v - eta * (rho * par + (g - par))
