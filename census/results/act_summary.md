# Track 3A summary: scale gating at width 1 for GELU, SiLU and Mish

Producer: `src/act_general.py` (steps 0–4) and `src/act_posthoc.py` (post hoc descriptives). Tests:
`tests/test_act_general.py` (21). Outputs are in `results/act_general/`. Registrations:
`results/act_criterion_registration.md` (b707e86, plus amendments a251e80 and cf2d68b) and
`results/act_training_registration.md` (ad99053). Labels: **[registered]** means a verdict under a committed
registration; **[validated]** means a validated (not certified) computation; **[post hoc]** means looked at after the
fact. Task: width 1, I = [−0.8, 0.8] (class 0), O = ±[1.2, 2.0] (class 1), the 800-point population, s = |w₂|.

## Step 0: generalised machinery [validated]

The new module adds the activations as an Act-like object: closed-form u, u′, u″ and u‴, tested against autograd. The
sup|u″| bound is validated, not certified. It is the maximum over a 9-point sub-grid, plus (h/2)·(maximum of |u‴| on
the sub-grid + (h/2)·B4), where B4 is a grid bound on |u⁗| with a 5% margin.

**Check:** with f_a (a = 1.30), the generalised width-1 search plus bisection puts the switch in [4.95796, 4.95835].
That is inside the certified bracket [4.95, 4.9625], and Track 1's s* = 4.95801 is inside it too
(`validate_fa.json`).

## Step 1: a single unit can place and solve [validated]

| act | Ĝ = sup G₊ (NM, 400 starts; DE agrees to 1e−6) | maximiser (w₁, b₁, σ) | dip of u (t, depth) |
|---|---|---|---|
| GELU | 0.037547 | (0.864, −0.904, +1) | (−0.752, −0.170) |
| SiLU | 0.050912 | (1.583, −1.636, +1) | (−1.278, −0.278) |
| Mish | 0.054772 | (1.526, −1.561, +1) | (−1.192, −0.309) |

- At every maximiser, a midpoint output bias gives a network that is sign-correct on the windows (exact extrema); a
  wrong bias does not (`ghat.json`).
- Placement needs σ = +1: the fixed-size dip sits under I, and both outer windows lie higher.
- **Differences from the sine case:**
  - The dip is fixed in size and there is no periodicity, so there is no 2π winding (k = 0).
  - The ramp is not odd and is unbounded, so Δμ is unbounded (≈ 0.6|w₁|) and the class-mean maximiser is not attained.
  - Placement is possible deep in the negative tail with an astronomically small gap (below).

## Step 2: the criterion [registered]

The rule: a switch is predicted iff the validated conditional minimiser at s = 0.05 and at s = 0.1 is unplaced (by its
double-precision G₊ enclosure) and Ĝ > 0.

| act | s = 0.05 | s = 0.1 | verdict |
|---|---|---|---|
| GELU | unplaced (validated) | unplaced (validated) | **switch predicted** |
| SiLU | unplaced (validated) | undecided: enclosure [−2.9e−14, 1.2e−154]; audit fails | **undetermined** |
| Mish | unplaced (validated) | undecided: enclosure [−6.2e−14, 2.7e−153]; audit fails | **undetermined** |

**[post hoc, disclosed in amendment 2]** All six small-scale minimisers are one-sided ramps:
- w₁ = 223–775, with the kink just right of x = 0.8 and I's right edge at the dip. The right outer window rides the
  ramp, while I and the left outer window sit in the negative tail.
- Their loss is ≈ 0.47738, about ¾·H(⅓), which is the loss of separating the right outer window only.
- In arbitrary precision (mpmath) their G₊ is **positive**: 10^(−334,391) and 10^(−37,838) for GELU, 10^(−369) and
  10^(−154) for SiLU, 10^(−367) and 10^(−153) for Mish. In double precision it is 0.
- So in exact arithmetic the small-scale minimiser is "placed" by a vanishing tail margin. The registered verdicts
  above rest on the double-precision enclosure, as registered.

## Step 3: switch brackets [validated]

Method: a scan over s = 10^(k/8) from 0.1 to 1,000, then bisection to 1%. At both bracket ends: a restart ladder
200 → 800 (unchanged within 1e−9, same status), an independent CMA-ES search not lower by more than 1e−9, and an audit.
All checks pass at both ends of every bracket.

| act | bracket [s_lo, s_hi] | G₊ at the ends | s* on the tracked branch | R = s*Ĝ/2 |
|---|---|---|---|---|
| GELU | **[6.6117, 6.6714]** | −5.1e−4 / +3.8e−4 | 6.64563 | 0.1248 |
| SiLU | **[3.6190, 3.6517]** | −8.7e−4 / +5.4e−4 | 3.63919 | 0.0926 |
| Mish | **[3.1908, 3.2197]** | −1.3e−3 / +2.8e−4 | 3.21466 | 0.0880 |

SiLU and Mish were bracketed even though their criterion verdict is undetermined.

**[post hoc]** In exact arithmetic the retained branch has three regimes as s increases:
1. At small s it is tail-placed (G > 0 but vanishing).
2. It becomes genuinely unplaced between the scan points s = 0.56 and 0.75 (GELU) and s = 0.32 and 0.42 (SiLU). The
   scan minimum of G is −0.138 (GELU, at s ≈ 2.4) and −0.218 (SiLU, at s ≈ 1.3).
3. It crosses G = 0 continuously at the bracket (both ends are on the same branch; G is O(1e−4) at each end), then rises
   towards Ĝ.

The meaningful unplaced-to-placed switch is at the bracket. In exact arithmetic there is also an earlier sign change,
from tail-placed to unplaced.

## Step 4: κ and training [registered]

**κ** was computed by Track 1's method (k = 0) at s* and frozen with SHA-256 before any registered run. P came from
calibration seeds 851,000–851,009:

| act | κ_Adam |
|---|---|
| GELU | +0.1473 |
| SiLU | −0.1444 |
| Mish | −0.1539 |

**Protocol:**
- Adam, lr 0.01, U(−1, 1)⁴ drawn in float32 then cast to double, budget 32,000 steps.
- Every-step crossing detection with `phase2b_ordering.state` using each activation.
- Primary seeds: 850,000–850,039 (fresh; none appears anywhere in `results/`, `src/` or `tests/`).

| act | arm | crossing runs | T-a (≥ 90% at s ≥ s_lo) | T-b (median r vs median κχ, tol) |
|---|---|---|---|---|
| GELU | **primary, 40 seeds** | 23 (+2 placed at init) | **UNRESOLVED** | **UNRESOLVED** |
| SiLU | **primary, 40 seeds** | 25 | **UNRESOLVED** | **UNRESOLVED** |
| Mish | **primary, 40 seeds** | 25 | **UNRESOLVED** | **UNRESOLVED** |
| GELU | secondary, 200 seeds (registered) | 132 | **FAIL** (47.7%) | **FAIL**: obs −0.0183 vs pred +0.0023 (tol 0.01) |
| SiLU | secondary, 200 seeds (registered) | 135 | **FAIL** (62.2%) | **FAIL**: obs +0.0353 vs pred −0.0324 (tol 0.01) |
| Mish | secondary, 200 seeds (registered) | 135 | **FAIL** (70.4%) | **FAIL**: obs +0.0658 vs pred −0.0393 (tol 0.01) |

- The registered sensitivity (crossing at dense G ≥ 1e−8) gives identical verdicts.
- The secondary arm (primary seeds plus 850,040–850,199) was registered before any run, because calibration showed a
  crossing rate near 50%.

**[post hoc]** (`posthoc_training.csv`):
- About half the runs never cross. They stall near a trivial point: median final |w₂| is 0.16, 0.07 and 0.11 for
  GELU, SiLU and Mish.
- Among crossing runs the residual r = s_cross/s_glob − 1 is broad. Its interquartile range is [−0.11, +0.08] (GELU),
  [−0.04, +0.14] (SiLU) and [−0.02, +0.16] (Mish), and only about 52% have |r| ≤ 0.10.
- 18 of 132 GELU crossings are genuine early placements at s < ½·s_glob.
- κ·χ predicts residuals of order 0.002–0.04, which is far below this spread. Spearman(r, χ) is negative:
  −0.35, −0.29 and −0.23.
- The registered test uses the population threshold, not each run's own-sample threshold. Training-set sampling
  (200 points per run) plausibly dominates the residual. This is not tested here.

## Not done

- Own-sample (per training set) thresholds were not computed.
- The brackets are validated, not certified. The sup|u″| bound is validated, not certified.
- No certified statement is made about the tail-placed regime.
