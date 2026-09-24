"""Width 2 (Route A), conditional minimisation at fixed ‖w₂‖₁ = s (design §3), validated Block-1 style.

Problem: minimise L*(θ, ṽ; s) = min_b mean ℓ(s·Σṽᵢu(αᵢx + βᵢ) + b, y) over θ ∈ ℝ⁴ and ṽ on the ℓ₁ circle,
ṽ = (t, σ(1 − |t|)), t ∈ [−1, 1], σ ∈ {±1}.  b is profiled exactly (safeguarded Newton); the gradient in (θ, t)
follows from the envelope theorem.

Validation (not certification):
  1. every restart is retained with its loss, gradient norm, G₊ (directional), hidden placement P₊ and flags; the
     constant predictor (loss = entropy of ȳ) is an explicit candidate; audit: no discarded candidate below the
     retained one (stop condition);
  2. restart convergence: the retained minimum unchanged to 1e−9 across 500 → 1,000 → 2,000 → 4,000 restarts;
  3. a stricter search (10× steps, gradient norm 1e−10) and an independent CMA-ES search with v unconstrained and
     normalised afterwards; stop if either finds a lower loss beyond 1e−9;
  4. local certificates (interval Hessian PD on a small box) -- in width2_certify (to be added before any run).

Nothing here is run on real objectives until the report; tests exercise every check on constructed cases.
"""

from __future__ import annotations

import math

import numpy as np

from .width2_geometry import TWO_PI, Act, gaps, phi

BOX_ALPHA = 10.0


def population():
    from .blockB_landscape import population_data
    x, y = population_data()
    return x.numpy().astype(float), y.numpy().astype(float)


def training_set(seed):
    from .fold1d import make_data
    x, y = make_data(200, seed)
    return x.numpy().astype(float), y.numpy().astype(float)


def _sig(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def _softplus(z):
    return np.logaddexp(0, z)


# ------------------------------------------------------------------------------------------ parametrisation
def vtilde(t, sigma):
    t = float(np.clip(t, -1, 1))
    return np.array([t, sigma * (1 - abs(t))])


def profile_b(z0, y, iters=200):
    """b* = argmin_b mean ℓ(z0 + b, y): strictly convex; safeguarded Newton inside a bracket."""
    ybar = y.mean()
    if ybar in (0.0, 1.0):
        raise ValueError("both classes must be present")
    lo = math.log(ybar / (1 - ybar)) - z0.max(); hi = math.log(ybar / (1 - ybar)) - z0.min()
    b = 0.5 * (lo + hi)
    for _ in range(iters):
        s = _sig(z0 + b); g = s.mean() - ybar
        if abs(g) <= 1e-15:
            break
        if g < 0:
            lo = b
        else:
            hi = b
        h = (s * (1 - s)).mean()
        bn = b - g / max(h, 1e-300)
        b = bn if lo < bn < hi else 0.5 * (lo + hi)
        if hi - lo < 1e-15 * max(1, abs(b)):
            break
    return b


def loss_grad(p, s, sigma, x, y, act):
    """L*(p; s) and its gradient in p = (α₁, β₁, α₂, β₂, t), b profiled (envelope theorem)."""
    a1, b1, a2, b2, t = p
    t = float(np.clip(t, -1, 1)); v = vtilde(t, sigma)
    t1, t2 = a1 * x + b1, a2 * x + b2
    u1, u2 = act.u(t1), act.u(t2)
    z0 = s * (v[0] * u1 + v[1] * u2)
    b = profile_b(z0, y)
    z = z0 + b
    L = float((_softplus(z) - y * z).mean())
    r = _sig(z) - y
    d1, d2 = act.du(t1), act.du(t2)
    g = np.array([
        (r * s * v[0] * d1 * x).mean(), (r * s * v[0] * d1).mean(),
        (r * s * v[1] * d2 * x).mean(), (r * s * v[1] * d2).mean(),
        (r * s * (u1 - sigma * np.sign(t) * u2)).mean() if abs(t) < 1 else 0.0])
    return L, g, b


def bfgs(fg, x0, gtol=1e-8, maxit=2000):
    """BFGS with Armijo backtracking (n-dimensional; own_threshold._bfgs generalised)."""
    xk = np.asarray(x0, float).copy(); n = len(xk)
    f, g = fg(xk)
    H = np.eye(n) * 0.1
    it = 0
    for it in range(maxit):
        if np.abs(g).max() <= gtol:
            break
        d = -H @ g
        if d @ g >= 0:
            H = np.eye(n) * 0.1; d = -H @ g
        t = 1.0
        while True:
            fn, gn = fg(xk + t * d)
            if fn <= f + 1e-4 * t * (d @ g) or t < 1e-14:
                break
            t *= 0.5
        if t < 1e-14:
            break
        sv, yv = t * d, gn - g
        xk, f, g = xk + sv, fn, gn
        sy = sv @ yv
        if sy > 1e-300:
            r = 1.0 / sy; I = np.eye(n)
            H = (I - r * np.outer(sv, yv)) @ H @ (I - r * np.outer(yv, sv)) + r * np.outer(sv, sv)
    return xk, float(f), g, it


# ------------------------------------------------------------------------------------------ placement notions
def directional_gplus(p, sigma, act):
    return gaps(p[:4], vtilde(p[4], sigma), act)["G+"]


def hidden_placement(theta, act, n_grid=401):
    """P₊(θ) = max over ‖ṽ‖₁ = 1 of G₊(Σṽᵢuᵢ).  G₊ is concave and positively homogeneous in ṽ, so on each of the
    four edges of the ℓ₁ circle it is concave: grid, then golden-section refine on the best edge.  Returns the lower
    end of the exact enclosure at the maximiser (a certified lower bound of P₊ at that ṽ)."""
    edges = [((1, 0), (0, 1)), ((0, 1), (-1, 0)), ((-1, 0), (0, -1)), ((0, -1), (1, 0))]
    best = (-math.inf, None)
    for e0, e1 in edges:
        e0, e1 = np.array(e0, float), np.array(e1, float)
        f = lambda w: gaps(theta, (1 - w) * e0 + w * e1, act)["G+"][0]
        ws = np.linspace(0, 1, 9); vals = [f(w) for w in ws]
        k = int(np.argmax(vals)); lo, hi = ws[max(k - 1, 0)], ws[min(k + 1, 8)]
        gr = (math.sqrt(5) - 1) / 2
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        for _ in range(40):
            if f(c) > f(d):
                hi = d
            else:
                lo = c
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        w = 0.5 * (lo + hi); val = f(w)
        if val > best[0]:
            best = (val, (1 - w) * e0 + w * e1)
    return best


# ------------------------------------------------------------------------------------------ the validated search
def constant_predictor_loss(y):
    p = y.mean()
    return float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))


def _start(rng, act, box=BOX_ALPHA):
    blo, bhi = (0.0, TWO_PI) if act.periodic else (-box, box)
    return (np.array([rng.uniform(0, box), rng.uniform(blo, bhi), rng.uniform(0, box), rng.uniform(blo, bhi),
                      rng.uniform(-1, 1)]), int(rng.choice([-1, 1])))


def degenerate(p, gnorm, gtol, box=BOX_ALPHA):
    """Flags.  Excluding (not eligible to be retained): non-finite; not converged (gradient norm above 100·gtol).
    Informational only: α beyond the starting box (the search is unconstrained; the box bounds the starts)."""
    flags = []
    if not np.all(np.isfinite(p)):
        flags.append("nonfinite")
    if gnorm > 100 * gtol:
        flags.append("not_converged")
    if abs(p[0]) > box or abs(p[2]) > box:
        flags.append("info:alpha_beyond_start_box")
    return flags


EXCLUDING = {"nonfinite", "not_converged"}


def search(s, x, y, act, restarts=2000, seed=0, gtol=1e-8, maxit=2000, box=BOX_ALPHA):
    """Global search at scale s.  Returns (retained, candidates) with every restart retained."""
    rng = np.random.default_rng(seed)
    cands = []
    for k in range(restarts):
        p0, sigma = _start(rng, act, box)
        fg = lambda p: loss_grad(p, s, sigma, x, y, act)[:2]
        p, L, g, it = bfgs(fg, p0, gtol=gtol, maxit=maxit)
        p[4] = np.clip(p[4], -1, 1)
        gn = float(np.abs(g).max())
        cands.append({"k": k, "p": p, "sigma": sigma, "loss": L, "gnorm": gn, "iters": it,
                      "flags": degenerate(p, gn, gtol, box)})
    cands.append({"k": -1, "p": None, "sigma": 0, "loss": constant_predictor_loss(y), "gnorm": 0.0, "iters": 0,
                  "flags": ["constant_predictor"]})
    return retain(cands), cands


def retain(cands):
    """The retained candidate: lowest loss among non-degenerate, converged candidates (the constant predictor is
    a legitimate candidate)."""
    ok = [c for c in cands if not (set(c["flags"]) & EXCLUDING)]
    return min(ok, key=lambda c: c["loss"])


def audit(cands, retained, tol=1e-9):
    """Stop condition: any discarded candidate with loss below the retained one by more than tol."""
    bad = [c for c in cands if c is not retained and c["loss"] < retained["loss"] - tol]
    return {"ok": not bad, "n_below": len(bad), "worst": min((c["loss"] for c in bad), default=None)}


def convergence_ladder(s, x, y, act, ladder=(500, 1000, 2000, 4000), seed=0, tol=1e-9, **kw):
    """Retained minimum at increasing restart counts (nested: the first n restarts of one seeded stream)."""
    r, cands = search(s, x, y, act, restarts=max(ladder), seed=seed, **kw)
    const = cands[-1]
    vals = []
    for n in ladder:
        sub = cands[:n] + [const]
        vals.append(retain(sub)["loss"])
    return {"ladder": list(ladder), "retained": vals, "ok": max(vals) - min(vals) <= tol}


def independent_search(s, x, y, act, starts=40, seed=7, sigma0=1.0):
    """CMA-ES on an unconstrained v (normalised to the ℓ₁ circle afterwards): a different parametrisation and a
    different optimiser.  Returns the best loss and point (in the (θ, t, σ) parametrisation)."""
    from .cmaes import cma_es
    rng = np.random.default_rng(seed)

    def f(q):
        v = q[4:6]; n1 = abs(v[0]) + abs(v[1])
        if n1 < 1e-12:
            return 1e9
        v = v / n1
        t1, t2 = q[0] * x + q[1], q[2] * x + q[3]
        z0 = s * (v[0] * act.u(t1) + v[1] * act.u(t2))
        b = profile_b(z0, y); z = z0 + b
        return float((_softplus(z) - y * z).mean())
    best = (math.inf, None)
    for _ in range(starts):
        q0 = np.concatenate([rng.uniform(-BOX_ALPHA, BOX_ALPHA, 4), rng.uniform(-1, 1, 2)])
        r = cma_es(f, q0, sigma0, max_generations=400, seed=int(rng.integers(1 << 31)))
        if r.best_f < best[0]:
            best = (r.best_f, r.best_x)
    return {"loss": best[0], "q": best[1]}


def stricter_check(retained_loss, other_loss, tol=1e-9):
    """Stop condition: another search finds a lower loss than the retained one by more than tol."""
    return other_loss >= retained_loss - tol


# ------------------------------------------------------------------------------------------ thresholds
def r2_grid(gamma2_hat, lo=0.02, hi=1.00, step=0.01):
    """The registered scan (amendment): R₂ ∈ [0.02, 1.00] step 0.01, as s = 2R₂/Γ̂₂."""
    R = np.round(np.arange(lo, hi + step / 2, step), 10)
    return R, 2 * R / gamma2_hat


def first_sign_change(R, gplus):
    """R₂,glob: the first grid point whose retained minimiser has G₊ > 0, with the bracket (previous, that point).
    Returns ('not applicable', None) if G₊ never changes sign over the scan (all <= 0 or all > 0), and reports a
    second sign change if one exists."""
    pos = np.asarray(gplus) > 0
    if pos.all() or (~pos).all():
        return {"status": "not applicable", "bracket": None, "sign_changes": 0}
    changes = np.flatnonzero(pos[1:] != pos[:-1])
    first = int(np.flatnonzero(pos)[0])
    return {"status": "defined", "bracket": (float(R[first - 1]), float(R[first])) if first > 0 else None,
            "sign_changes": int(len(changes))}
