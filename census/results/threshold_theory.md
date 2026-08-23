# A1: what the theory's possibility direction actually requires

Read from the recorded theorem inventory (`notes/icml_paper_notes.md`,
`notes/icml_paper_reconciliation.md`, verified against arXiv:2606.31856v1 in
the reconciliation pass). This determines how the rest of Part A is read.

## The two directions, exact hypotheses

**Impossibility (Theorem 4.7, verbatim hypotheses as recorded).**
`M^m, N^n ⊂ R^d` disjoint closed oriented submanifolds, `m + n + 1 = d`,
`link(M, N) ≠ 0`; `F` any width-`d` feedforward network of affine maps and
continuous coordinate-wise **monotonic** activations. Conclusion: `F(M)`,
`F(N)` are not linearly separable. The paper adds explicitly that the
crucial property is monotonicity, not differentiability or boundedness.

**Possibility (Section 5.1 / Appendix F.2).** Hypotheses: the data are
**compact**, and the scalar activation has a **strict local extremum on an
open interval**. The construction: affine shifts and rescalings move the
relevant coordinate into that interval; the non-monotonicity then supplies
a coordinate fold. **Nothing else is required** — no minimum fold extent,
no minimum depth, no shape condition, no unboundedness. The affine layers
carry the quantitative burden: the pre-affine map shrinks the data
coordinate into the extremum window, and downstream affine maps amplify
the folded output arbitrarily.

## Applied to our families

- `f_a(x) = x + a·sin(x)`: for **every** `a > 1`, `f'` is negative on open
  intervals around `x = π (mod 2π)`, so `f_a` has strict local extrema —
  the F.2 hypothesis holds for all `a > 1` without qualification. At
  `a = 1` exactly, `f'` ≥ 0 with isolated zeros: monotone, no strict local
  extremum — the impossibility side applies.
- `g_α`: for **every** `α < 0` there is a strict local minimum at 0; at
  `α = 0` (ReLU), monotone.

**The theory therefore asserts a clean dichotomy exactly at the analytic
thresholds**: separating width-3 networks exist for every `a > 1` and every
`α < 0`, and cannot exist for `a ≤ 1`, `α ≥ 0`.

## The determination

**Explanation 1 — a genuine representational margin requirement — is ruled
out by the theory.** The possibility direction asserts existence with no
margin condition, so the observed offset ((1.05, 1.10] and [−0.25, −0.10))
is **definitionally a gap between possible and reachable** under the
paper's hypotheses (exact arithmetic, unbounded affine weights, compact
data). The live explanations are reachability (2), wrong unit as a
description of *reachability* rather than of possibility (3), grid
resolution (4), and the dense-criterion artifact (5).

One caveat, stated because it is the only escape hatch for explanation 1:
the theory's construction uses exact real arithmetic and unrestricted
weights. A float32 network with finite weights might face a *practical*
representational limit if the required amplification exceeded what
training-scale weights express. Checking the numbers closes the hatch for
our region of interest:

| a | non-injective window width | fold depth | amplification to O(1) |
|---|---:|---:|---:|
| 1.001 | 0.089 | 6.0e−05 | ~17,000× |
| 1.005 | 0.200 | 6.7e−04 | ~1,500× |
| 1.01 | 0.282 | 1.9e−03 | ~530× |
| **1.02** | **0.397** | **5.3e−03** | **~190×** |
| 1.05 | 0.620 | 2.1e−02 | ~48× |
| 1.10 | 0.859 | 5.7e−02 | ~18× |

At `a = 1.02` the non-injective window is 0.4 wide (not small — the data
coordinate fits in it after modest shrinking) and the required
amplification is ~190×, achievable with two layers of weights of order 14,
comfortably inside float32 and inside the weight scales trained networks
in this project actually reach (trained weights of magnitude 3–5 are
routine). Even `a = 1.005` needs only ~1,500× (three layers of weights of
order 12). **Float32 practicality does not rescue explanation 1 anywhere
near the observed transition.**

## Consequence for A4

Since the theory says a separating network *exists* at `a = 1.02`, the F.2
construction is itself buildable there, the same way the Part 2a witness
was built: freeze a layer that shrinks the fold coordinate into the
non-injective window of `f_{1.02}`, let training supply the rest. A4
therefore runs two things: the four search families (reachability), and an
**F.2-style constructed witness at `a = 1.02`** (existence made concrete).
If the construction separates densely, existence at 1.02 is settled by
exhibition rather than by theory citation, and whatever the searches do,
the offset is about reachability.
