# Part 4: does the proof describe the object the code verifies?

The proof was written from the theorem statement; the verification was written
weeks earlier from the implementation. This checks that they describe the same
object. Traced in source, not prose.

Date: 2026-09-14.

---

## Verdict

**They match, with one real finding and two corrections to the draft.** The
finding (`maximum_gap`'s restricted search domain) makes the verification
**conservative rather than wrong**, and its direction is what settles it.

## Q1: the windows — MATCH, exactly

`src/fold1d.py` fixes `INNER_MAX = 0.8`, `OUTER_MIN, OUTER_MAX = 1.2, 2.0`, and
`solves()` checks dense grids of `[-0.8, 0.8]` and `±[1.2, 2.0]`. These are
exactly `I` and `O` as the proof states them.

**One distinction the proof must keep visible**: `solves()` tests strict sign
(`< 0` and `> 0`), i.e. the **`m -> 0` limit**, not a fixed margin. The theorem
is stated for `m > 0` and degenerates as `m -> 0` (the bound becomes
`|w2| >= 0`). So the empirical criterion sits exactly where the bound carries no
force — which is the seam already documented in T37 and §5, and is why the
infimum of `|w2|` over solutions is 0.

## Q2: the margin — MATCH

`src/fold1d_theorem.py:102` computes
`margin = min(-inner_logits.max(), outer_logits.min())`, which is precisely
"`N <= -m` on `I` and `N >= +m` on `O`, largest such `m`". It is a **logit**
margin (pre-sigmoid), signed so positive means separating. Same quantity,
same sign convention.

Each solver is verified against **its own achieved margin**, not a fixed
constant. This matters: assuming `m = 1` inflates the requirement ~1.5x and
manufactures apparent violations (recorded in `src/r_variable.py`).

## Q3: `G` and orientation — MATCH, via a symmetry worth stating

`class_gap` computes `min_O f(w1 x + b1) - max_I f(w1 x + b1)`: the
**`w2 > 0` orientation only**. The proof says "mirror the roles for `w2 < 0`".

This is not a gap in the verification, because **`G*` is orientation-
symmetric**, verified numerically at `a` = 1.25, 1.5, 2.0, 3.0 (equal to 1e-6)
and provable in one line: `f_a` is **odd** (`f(-t) = -f(t)`, checked), and both
`I` and `O` are symmetric about 0, so `(w1, b1) -> (-w1, -b1)` maps the
`+` orientation gap onto the `-` orientation gap exactly. The uniform bound
`2m/G*` therefore holds for both signs.

**The pointwise bound `2m/G(w1,b1)` does need the orientation clause**, and the
proof states it. Keep that sentence.

## Q4: `maximum_gap`'s domain — the finding

The proof quantifies `G*` as a max over **all** `(w1, b1)`. The implementation
searches a **restricted neighbourhood**:

    b1 in pi + arccos(1/a) * [0.6, 1.4]      (33 values)
    w1 in (0.001, 3.0]                       (`resolution` values)

High-resolution local refinement finds gaps **0.3–0.7% larger** than
`maximum_gap` returns, consistently across `a` = 1.10, 1.60, 3.00. So
**`maximum_gap` slightly under-estimates `G*`.**

**Direction settles it.** The bound is `|w2| >= 2m/G*`. A **larger** true `G*`
gives a **smaller** true bound, which is **easier** to satisfy. Under-estimating
`G*` therefore makes our verification **conservative**: we check solvers against
a bound that is slightly too strong.

Worst case, inflating `G*` by 0.7% and rechecking all 66 solvers:

| | as verified | with `G*` inflated 0.7% |
|---|---:|---:|
| violations | **0 of 66** | **0 of 66** |
| minimum slack | 1.0394 | 1.0467 |

Refinement never found a gap **smaller** than the restricted scan, so
`maximum_gap` never over-estimates — the failure mode that would make the bound
too weak to constrain anything does not occur.

**Consequence for `kappa`**: `kappa = G*/D` rises by the same 0.3–0.7%,
[0.305, 0.328] -> [0.306, 0.330]. The shift is in the third decimal, and every
downstream conclusion already holds over `kappa` in [0.1, 1.0], a tenfold
range. **No reported number moves.**

**What to do**: state in the paper that `G*` is obtained by numerical
maximisation over a neighbourhood of the analytic optimum, that the estimate is
a lower bound on the true `G*`, and that this makes the verification
conservative. Do not claim `maximum_gap` returns the global maximum.

## Q5: the width-1 restriction — consistent, but it was implicit

The bound is for a **single hidden unit**. At width `> 1` several units can
split the two sign changes, and the argument gives only
`sum_i |w2_i| G_i >= 2m`, which no single coordinate need satisfy.

Checked across the drafts: **the bound is never invoked at width > 1.** §2's
width-3 experiments cite only Ren & Lim's Theorem 3.7, never ours. But §3
stated the restriction only implicitly ("the network is a single hidden unit"),
which is not enough given that §2 uses width 3 on the facing pages. **An
explicit scope sentence has been added.**

## Two draft corrections found while checking

1. **The asymptotic constant** was still `8/3` in §3, with the stale error
   figures (42.7% / 136.1%) computed against it. Corrected to
   `4*sqrt(2)/3` with the correct errors (**0.9% at `a` = 1.02, 66.9% at
   `a` = 3.0**), and the closed form `D(a) = 2(sqrt(a^2-1) - arccos(1/a))`
   added.
2. **The local logarithmic slope** was quoted as **1.4364** (4.2% from 3/2).
   Recomputed over the stated range `eps` in [0.03, 0.60]: **1.4233**, i.e.
   **5.1%** from 3/2. Neither that range, the [0.025, 0.6] range, nor an
   endpoint-only slope reproduces 1.4364. Corrected.
