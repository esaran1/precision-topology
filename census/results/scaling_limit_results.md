# The scaling limit: results against `scaling_limit_prediction.md`

Registered 2026-09-21 in `results/scaling_limit_prediction.md`, committed
(`264ffe0`) **before** the Block B scan on `h` was run and before any
finite-`eps` correction was fitted. Artifacts: `results/scaling_limit_switches.csv`,
`src/scaling_limit_blockB.py`.

---

## Summary

| step | verdict |
|---|---|
| expansion `f_a(pi+s) - pi = eps^{3/2} h(sigma) + O(eps^{5/2})` | **VERIFIED** symbolically and numerically |
| dip depth `-> (4 sqrt2 / 3) eps^{3/2}` | **VERIFIED** to 6 digits |
| `w1* ~ sqrt(eps)`, `Ghat -> K eps^{3/2}` | **VERIFIED**, `K = 0.579454926` |
| **S-1** `kappa_0 = K/(4 sqrt2/3)` matches measured `kappa` | **PASS**, 0.38% at `a = 1.02` (tol 3%) |
| **S-2** switch points exist on `h`, `a`-free | **PASS** |
| **S-3** finite-`a` deviation positive, increasing in `eps` | **PASS for `R_glob`**, **FAILS for `R_solve`** |
| **S-4** correction is `O(eps^1)`, `c1 > 0` | **PASS for `R_glob`** |
| Step 5 exact loss invariance at the `eps` used | **FAILS** — reported, not repaired |

## The verified chain

1. `f_a(pi+s) = pi + s - (1+eps) sin s`, so exactly
   `f_a(pi+s) - pi = -eps*s + (1+eps)s^3/6 - (1+eps)s^5/120 + ...`
2. With `s = sqrt(eps) sigma` this is
   **`eps^{3/2} [ -sigma + (1+eps) sigma^3/6 ] - (1+eps) eps^{5/2} sigma^5/120 + ...`**
   so the limit is `h(sigma) = -sigma + sigma^3/6`. **The `eps sigma^3/6` term inside
   the bracket is an explicit relative `O(eps)` correction** — this is where the
   measured drift comes from, and it is identified in closed form rather than fitted.
3. `h'(sigma) = -1 + sigma^2/2 = 0` at `sigma = +-sqrt2`, `h(+-sqrt2) = -+2sqrt2/3`,
   dip depth **`4 sqrt2 / 3 = 1.885618083`** exactly. The exact
   `D(a) = 2(sqrt(a^2-1) - arccos(1/a))` gives `D/eps^{3/2}` = 1.877190 / 1.884770 /
   1.885533 / 1.885610 at `eps = 1e-2 ... 1e-5`. Converges.
4. **`K = 0.579454926`** (nested grid refinement, stable to 9 digits from iteration 4),
   attained at `u = w1/sqrt(eps) = 1.605516`, `v = 1.204199`. Both active
   constraints sit at **window edges** (`x = -0.8` and `x = -1.2`), not at `h`'s
   critical points: the optimal placement is edge-limited, not curvature-limited.

## S-1: the geometric constant, predicted from `h` alone — PASS

`kappa_0 = K / (4 sqrt2 / 3) = **0.307302**` against measured
`kappa(a) = Ghat(a)/D(a)`:

| `a` | 1.02 | 1.05 | 1.10 | 1.30 | 1.60 |
|---|---:|---:|---:|---:|---:|
| `kappa(a)` | 0.306122 | 0.306911 | 0.307118 | 0.312060 | 0.315485 |
| vs `kappa_0` | **-0.38%** | -0.13% | -0.06% | +1.55% | +2.67% |

**0.38% at `a = 1.02`, inside the registered 3%.** `kappa` was known to be nearly
constant; it is now known to equal a number computed from a cubic with no
reference to `f_a` at all.

**Why `kappa` drifts so little** (new): `Ghat/eps^{3/2}` falls 0.572 -> 0.480 over
`eps = 0.02 -> 0.60`, a 16% drift, and `D/eps^{3/2}` falls comparably. The two
`O(eps)` corrections **largely cancel in their ratio**, leaving `kappa` within
2.7% over the whole range. The near-constancy is a cancellation, not a coincidence.

## S-2, S-3, S-4: the switch points

Block B run on `h` with the frozen procedure (same `RESTARTS`, `STEPS`, `SCREEN`,
`KEEP`, degeneracy test, bracket-then-refine), substituting only the activation
and `W = w2 eps^{3/2}`, `R_inf = |W| K / 2`.

**Validity gate (rule 7) — the coarse grid had to be refined.** Step 0.05 in `W`
is 0.0145 in `R_inf`, seven times coarser than 0.05 in `w2` at finite `a`
(0.0021 in `R`). The coarse scan gave `R_solve^inf = 0.3183`, which sits 3.8%
above every finite-`a` value and would have read as a clean S-3 failure.
**Refined to step 0.005** the values are:

| | coarse (0.05) | **refined (0.005)** |
|---|---:|---:|
| `R_glob^inf` | 0.2025 | **0.19991** (`W = 0.690`) |
| `R_spin^inf` | 0.2170 | **0.19991** (`W = 0.690`) |
| `R_solve^inf` | 0.3183 | **0.30711** (`W = 1.060`) |

**`R_glob^inf = R_spin^inf` exactly: no hysteresis window in the scaling limit
either.** This independently reproduces Block B's registered falsifier at
`a = 1.30` from the limit activation alone, with no training run and no `f_a`.

### `R_glob`: S-3 and S-4 both PASS

| `a` | 1.30 | 1.35 | 1.40 | 1.45 | 1.50 | 1.60 |
|---|---:|---:|---:|---:|---:|---:|
| `R_glob(a)` | 0.2145 | 0.2162 | 0.2186 | 0.2194 | 0.2225 | 0.2285 |
| dev from limit | +7.28% | +8.15% | +9.36% | +9.73% | +11.28% | **+14.29%** |

All six **positive**, **strictly monotone** in `eps`, `corr(eps, dev) = +0.9866`.
Registered direction confirmed. S-4: log-log slope **0.947** (band [0.5, 2.0]),
and `R_glob(a) = 0.19991 (1 + 0.23099 eps)` fits all six with
`max |residual| = 0.00133`, **inside the 0.002 grid spread**. One parameter, six
points, residuals at grid resolution.

### `R_solve`: S-3's registered direction FAILS — and the reason is a stronger result

| `a` | 1.30 | 1.35 | 1.40 | 1.45 | 1.50 | 1.60 |
|---|---:|---:|---:|---:|---:|---:|
| dev from limit | -0.14% | -0.05% | -0.55% | -0.25% | -0.58% | -0.20% |

All six **negative** (registered: positive) and **not monotone**
(`corr = -0.295`). **S-3 is falsified for `R_solve` as registered.**

But the deviations are `-0.29%` on average, against `R_glob`'s `+7` to `+14%`.
`R_solve` is not *converging* to its scaling-limit value — it is **already there
at every `a` tested**, to within 0.6%. That is why `R_solve` had CV 0.0020 across
`a`: it is an `eps`-independent constant of the limit problem, reached long before
`eps` is small. The registered prediction assumed a visible approach; there is
none to see because there is no gap to close.

Recorded as **registered prediction failed, with the failure explained**: the
direction is wrong because the magnitude is at noise level, not because the
scaling account is wrong. The honest statement is that S-3 was a poorly chosen
test for `R_solve` — it presumed a convergence that a constant sequence cannot
exhibit.

### `R_spin`: mixed

`corr(eps, dev) = +0.954` but not all-positive (`a = 1.30` is at `+7.28%`, equal
to `R_glob` since the two coincide there) and not monotone, because `R_spin` is
the noisier of the two (it depends on continuation, not a global search).
Not scored as a pass.

## Step 5 (loss invariance): FAILS at the `eps` this paper uses — reported as such

Writing `b2 = -w2 pi + c`, the network is exactly

    N(x) = (w2 eps^{3/2}) h(u x + v) + c + w2 r(sqrt(eps)(u x + v))

so the loss is a function of `(W, u, v, c, eps)` and becomes `eps`-free only in
the limit. The neglected term's size relative to the leading one:

| `eps` | 0.02 | 0.10 | 0.30 | 0.50 | 0.60 |
|---|---:|---:|---:|---:|---:|
| `|w2 r| / |W h|` | 0.005 | 0.025 | **0.063** | 0.119 | **0.164** |

At `a = 1.30` the correction is 6.3%; at `a = 1.60`, 16.4%. Direct test: holding
`w2 eps^{3/2}` fixed, the conditional gap drifts `+0.001388` (`a = 1.30`) to
`-0.055321` (`a = 1.60`). **The invariance is asymptotic and is not a good
approximation at `eps = 0.3-0.6`.** The task windows are fixed in `x` while the
fold narrows as `sqrt(eps)`, so `sigma` spans a fixed range while `s` grows.

What survives, and is enough for the paper's purpose: the `eps` dependence enters
through a **single relative `O(eps)` correction**, identified in closed form as
the `eps sigma^3/6` term in Step 2. That is why a one-parameter law in `eps` fits
`R_glob` to grid resolution.

## What is derived and what is still fitted

**Derived, no training and no fit**: `h`, the dip depth `4 sqrt2/3`, `K`,
`kappa_0 = 0.307302` (0.38% from measurement), `R_glob^inf = R_spin^inf` (hence
no hysteresis), `R_solve^inf = 0.30711` (0.29% from the six-`a` mean), and the
`O(eps)` **order** of the correction.

**Fitted**: the coefficient `c1 = 0.231`. An attempt to predict it from
`K(eps) = sup gap of (-sigma + (1+eps)sigma^3/6)` gives `K(eps)/K(0) ~ 1 - 0.491 eps`,
which would imply `c1 ~ 0.49` — **a factor 2.4 too large**. The argument is
incomplete: it treats the switch as occurring at fixed placement quality, but
`R_glob` is where the *conditional minimiser's* gap changes sign, and that
minimiser trades gap against loss rather than maximising gap. Reported as fitted,
not derived.

## Bearing on O1

O1 said `R` is the variable that carries the threshold, on the evidence that
`CV(R) = 0.0311` against `CV(|w2|) = 0.2821`. The scaling limit now says **why**:
near the fold the activation is `eps^{3/2} h(sigma)` up to a relative `O(eps)`
term, so `|w2|` and `eps` enter the network only through `W = w2 eps^{3/2}`,
whose scale-free form is `R = |w2| Ghat/2` with `Ghat = K eps^{3/2}`. `R` is the
**only** combination of output weight and activation amplitude that the limit
problem sees.

The measured `+10.0%` drift in crossing `R` across `a` is then not noise in O1
but the `O(eps)` correction, the same one that moves `R_glob` by `+7` to `+14%`
over the same range and in the same direction.
