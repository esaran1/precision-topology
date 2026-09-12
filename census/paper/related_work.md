# Related work

Every statement attributed to a paper below was read from that paper's own
text, obtained as PDF and extracted to `paper/sources/*.txt`. Theorem numbers
and quoted statements were checked against the extracted text, not against
memory or third-party summaries. Where a claim could not be verified in the
source, it is not made.

**Provenance of each source** (`paper/sources/`):

| Short name | Paper | Verified from |
|---|---|---|
| Ren & Lim | *Low-dimensional topology of deep neural networks*, ICML 2026, PMLR 306 | `ren_lim.txt` (24 pp.) |
| Hanin & Sellke | *Approximating Continuous Functions by ReLU Nets of Minimal Width*, arXiv 1710.11278 | `hanin_sellke_width.txt` |
| Naitzat, Zhitnikov & Lim | *Topology of Deep Neural Networks* | `naitzat_betti.txt` |
| Guss & Salakhutdinov | *On Characterizing the Capacity of Neural Networks using Algebraic Topology* | `guss_salakhutdinov_topology.txt` |
| Cohen et al. | *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability*, ICLR 2021 | `cohen_edge_of_stability.txt` |
| Ahn, Zhang & Sra | *Understanding the Unstable Convergence of Gradient Descent*, arXiv 2204.01050v2 | `ahn_zhang_sra_uphill.txt` |
| Shalev-Shwartz, Shamir & Shammah | *Failures of Gradient-Based Deep Learning* | `shalev_shwartz_failures.txt` |
| Soudry et al. | *The Implicit Bias of Gradient Descent on Separable Data*, JMLR 19 (2018) | `soudry_implicit_bias.txt` |

A correction worth recording: three arXiv identifiers recalled from memory
while assembling this list retrieved **the wrong papers** (1802.04443 returned
Guss & Salakhutdinov rather than Hanin & Sellke; 2006.05467 returned Tanaka et
al. on pruning; 1702.05659 returned Janocha & Czarnecki on loss functions).
Each was caught by reading the first 200 characters of the extracted text. No
identifier in the table above is from recall; each was confirmed against the
retrieved document's own title page.

---

## 1. Ren & Lim — the paper this work extends

**What it proves.** Restricting every layer to width `d` (so `R^d` is the
representation space), the paper tracks linking number through layers.
**Theorem 4.7** (General impossibility theorem) states: for disjoint closed
oriented submanifolds `M^m, N^n ⊂ R^d` with complementary dimension
`m + n + 1 = d` and `link(M, N) ≠ 0`, any width-`d` feedforward network with
affine transformations and coordinate-wise monotonic activations fails to make
`F(M)` and `F(N)` linearly separable, so "perfect classification is
impossible." **Theorem 3.7** is the `R^3`/ReLU special case; **Theorem 4.5**
the higher-dimensional one. The mechanism is stated exactly: invertible affine
layers preserve link up to sign, monotonic activations preserve link up to sign
(Lemma 4.6 / C.7), and reaching `link = 0` therefore requires a rank-deficient
layer — which forces an intersection. **Theorem 5.2** shows a width-`n` ReLU
ResNet escapes via the single identity `|x| = x + 2·ReLU(−x)`; **Theorem 5.3**
gives a two-token attention layer whose output has a strict local minimum at
`0`, a smoothed V-shape approximating `|x|`. **Theorem D.1** gives the matching
upper bound at width `d + 1`, via Hanin & Sellke.

**Precise relationship to our work.** We work at the smallest instance of their
framework. Their §6.4 explicitly instantiates Theorem 4.7 at
`(m, n, d) = (0, 1, 2)` — a point inside an annulus — calling it "the ambient
dimension `d = 2` instance of our linking framework." Our `sign(|x| − 1)` task
is the `d = 1` fold: the obstruction is not linking but the sign-change count
of a monotone scalar map, which is why our impossibility argument is elementary
where theirs is topological. We adopt their central dichotomy — monotone
activations obstructed, folding activations not — and ask a question their
theorems do not address: **among activations that are all equally unobstructed
in principle, which ones does gradient descent actually reach a solution
with, and at what cost?**

**"This is just Ren & Lim in one dimension."** Our §3 impossibility result is
indeed their result specialized, and we say so; we claim no novelty for it. The
work begins after that point. Their theorems are existence statements about
what a network *can* represent and are silent on the optimizer, on the
parameter norm a solution requires, and on the budget needed to find one. Our
contributions are quantities that do not appear in their paper: a lower bound
`|w₂| ≥ 2m/G ≥ 2m/(κ·D(a))` tying the required output weight to the fold depth
of the activation; the measurement that the *findable* threshold sits strictly
inside the *representable* one; and the budget law `ε_onset ~ B^(−α/β)` linking
training budget to how far past the threshold a solution is reachable.

**Where we differ from them empirically, and it matters.** Their §6.3 reports
that ReLU's best run "never exceeds the ~90% topological ceiling at any depth"
while "GELU mean accuracy stays at 89–91% for depths 3–12 and the best run
achieves 100% at depths 5–12, confirming that nonmonotonic activations escape
the constraint." This is the positive control we set out to reproduce. We
reproduce the direction but find the advantage is **configuration-specific**
rather than automatic, and we report the settings in which it does not appear.
Their own framing already concedes the opening: the accuracy ceiling is imposed
"**under our training protocol**." Our work is about how much of the observed
gap is protocol.

**One scope correction we can make to the shared framework.** Their Table 1 and
Theorem 5.2/5.3 treat folding mechanisms as interchangeable. Family B
(`max(t, αt)`, leaky-ReLU-like) is **positively homogeneous**, and we derive
that homogeneity forces `β = 1` regardless of `α` — which makes it a
counterexample to the budget law rather than a limiting case of it. This is a
distinction invisible at the level of "can the architecture fold."

---

## 2. Hanin & Sellke — the width bound underneath Theorem D.1

**What it proves.** **Theorem 1**: `d_in + 1 ≤ w_min(d_in, d_out) ≤ d_in +
d_out`, where `w_min` is the minimal width at which ReLU nets of that width are
uniform universal approximators on compact sets. The upper-bound construction
gives depth `O(diam(K)/ω_f^{-1}(ε))^{d_in+1}` in terms of the target's modulus
of continuity.

**Relationship.** Ren & Lim's Theorem D.1 invokes this result for `d_out = 1`,
where the bound is tight at `d + 1`. We cite it only for that role. We note one
caveat a careful reader should carry: sufficiency at exactly `d + 1` is the
scalar-output case; the general upper bound is `d_in + d_out`.

**"This is just a width-bound paper."** It is, and we make no approximation-
theoretic claim. The relevance is that it fixes the *representability* boundary
precisely, which is what lets us state our finding as a gap: representability
and findability have different thresholds, and the second is strictly inside
the first.

---

## 3. Naitzat, Zhitnikov & Lim — topology change through layers

**What it shows.** Empirically, over persistent homology of many point clouds:
well-trained networks reduce Betti numbers of both class components, "nearly
always ... to their lowest possible values: `β_k(f(M_i)) = 0` for `k ≥ 1` and
`β_0(f(M_i)) = 1`", and the reduction "is significantly faster for ReLU
activation compared to hyperbolic tangent," attributed to ReLU being
non-homeomorphic. The stated goal is to explain why nonsmooth beats smooth and
why depth helps.

**Relationship.** This is the intrinsic-invariant predecessor; Ren & Lim
position their own work against it as studying an *extrinsic* invariant
(linking depends on embedding, Betti numbers do not). Our relationship is
methodological rather than topological: Naitzat et al. establish that
activation choice changes the *rate* at which a network simplifies data
topology. We measure the analogous rate question in a setting small enough to
have an exact answer — where the required parameter norm and the required
budget are both computable — and find the rate is governed by a divergence
exponent `β` that is a property of the activation's fold geometry.

**"This is just Betti-number reduction again."** No: their ReLU-over-tanh
finding is a comparison between an obstructed and an unobstructed activation,
which is the same axis Ren & Lim formalize. Our comparisons are *within* the
unobstructed class, where Betti-number arguments give no ordering at all — every
family we study can fold, so every one drives the invariant to the same place.
The ordering we measure is by cost, not by capability.

---

## 4. Guss & Salakhutdinov — topological capacity and phase transitions

**What it shows.** It reframes architecture selection as matching architecture
to the homological complexity of data, and gives "the first empirical
characterization of the topological capacity of neural networks," reporting
that "at every level of dataset complexity, neural networks exhibit topological
phase transitions" — sharp changes in achievable error as hidden dimension
crosses a data-dependent threshold, measured over datasets with known support
homology from `H(D) = {Z^1, 0}` up to `{Z^30, Z^30}`.

**Relationship.** This is the closest prior work *in spirit* to our onset
measurement: both locate a sharp architectural threshold empirically and both
find it is not where naive capacity counting puts it. The differences are the
swept axis and the resolution. They sweep width against data complexity and
report transitions; we hold the architecture fixed at width 1, sweep the
activation's fold depth continuously, and locate the threshold to within a grid
step with a bracketing criterion, then show the threshold *moves with training
budget* according to a power law.

**"This is just their phase transition."** The distinguishing prediction is the
budget dependence. A capacity phase transition is a property of the
architecture–data pair and should not move when you train longer. Our onset
moves, as `B^(−α/β)`, across four constructed families with analytically known
`β`. That is a statement about optimization, not capacity, and it is the reason
we are careful throughout to call the boundary a *findability* onset rather than
a capacity threshold.

---

## 5. Cohen et al. — Edge of Stability

**What it shows.** Empirically, full-batch gradient descent on neural networks
typically enters a regime where "the maximum eigenvalue of the training loss
Hessian hovers just above the value `2/(step size)`, and the training loss
behaves non-monotonically over short timescales, yet consistently decreases over
long timescales." Progressive sharpening is the preceding phase in which
sharpness rises until it reaches that value.

**Relationship.** We use this to interpret *where our training terminates*.
Our terminal points sit at sharpness near `2/η`, which is why they are not
ordinary interior minima, and it is the reason our terminology moved from
"minima" to terminal points of the trajectory.

**A scope limit we must respect, from their own text.** Cohen et al. establish
EoS for **full-batch gradient descent** and state explicitly that "the sharpness
does not flatline at any value during SGD (as it does during gradient descent)."
Our budget law is measured under Adam. We therefore cite EoS as an account of
the Adam-trajectory termination we observe, and do **not** claim our law is an
EoS consequence — the cross-optimizer tests reported in §6 are the reason that
restraint is not merely formal.

---

## 6. Ahn, Zhang & Sra — unstable convergence

**What it proves.** **Theorem 1**, quoted from the source: for a subset `X`
where `f` is `C²`, "Suppose that for each stationary point `p ∈ X`, it holds
that either `λ_min(∇²f(p)) < 0` or `λ_max(∇²f(p)) > 2/η`. Then under Assumption
1, there is a measure-zero subset `N` s.t. for all initializations `θ_0 ∈ X \ N`,
the GD dynamics `θ_{t+1} = θ_t − η∇f(θ_t)` do not converge to any of the
stationary points in `X`." The proof runs through the Stable Manifold Theorem.

**Relationship.** This is the result we rely on to say that certain
configurations are not reachable *as limits* by gradient descent, and the
disjunction matters: a point is excluded if it is a saddle **or** if it is too
sharp for the step size. Both branches occur in our measurements, and we report
the counts separately rather than merging them, because they have different
meanings — one is a landscape property, the other depends on `η`.

**"This is just a restatement of unstable convergence."** Their theorem says
which stationary points GD cannot converge to; it says nothing about which
*tasks* become unsolvable, nor at what parameter norm, nor how the boundary
moves with budget. It is an input to our §4 exclusions, cited for exactly the
statement above, and the rest of the analysis does not follow from it.

---

## 7. Shalev-Shwartz, Shamir & Shammah — failures of gradient-based learning

**What it shows.** Four families of simple problems where gradient methods fail
or struggle, with theory explaining the source. **Theorem 1** bounds the
variance of the gradient signal with respect to the target,
`Var(H, F, w) ≤ G(w)²/|H|`, for orthogonal target families — so when the
hypothesis class is large, the gradient carries almost no information about
which target is in play, and the failure is information-theoretic rather than
about landscape geometry.

**Relationship.** This is the canonical "expressible but not learnable"
reference, and it establishes the *category* our finding belongs to. The
mechanism is different in an important way. Their hardness is a signal-to-noise
result: the gradient does not point anywhere useful because the target is hidden
among many orthogonal alternatives. Ours is a *cost* result on a task with a
single, fixed, one-dimensional target where the gradient direction is
informative throughout — the solution is simply expensive to reach, requiring a
parameter norm that diverges as the activation approaches the fold threshold,
and the budget law quantifies the exchange rate.

**"This is just another expressivity–learnability gap."** It is one, and we
position it as such rather than claiming the genre. What is new here is that
the gap is *quantitative and predictable*: the location of the findability
boundary is a power-law function of budget whose exponent is `−α/β`, with `β`
derived analytically per family and verified numerically, and `α` measured. We
also report plainly that the law is single-optimizer.

---

## 8. Soudry et al. — implicit bias and norm growth

**What it proves.** On unregularized logistic regression with homogeneous
linear predictors on separable data, gradient descent converges in *direction*
to the max-margin (hard-margin SVM) solution, with the result generalizing to
other monotone decreasing losses with infimum at infinity, to multiclass, and
to training one weight layer of a deep network in a restricted setting.
Crucially for us: the convergence "is very slow, and only logarithmic in the
convergence of the loss itself," which is why continuing to optimize past zero
training error still changes the predictor.

**Relationship.** This is the mechanism behind the growth of `|w₂|` we measure.
On a separable classification objective with logistic-type loss the norm is
expected to grow without bound while the direction settles, so a terminal `|w₂|`
is a statement about *where the trajectory got to by the budget*, not about a
stationary point. This is exactly why our §5 is phrased in terms of budget and
population thresholds rather than convergence, and why `α` — the growth exponent
of terminal `|w₂|` with budget — is a coherent quantity to measure at all.

**"Then your `α` is just logarithmic norm growth, already known."** This is the
most serious reduction of our result and deserves the direct answer. Soudry et
al. predict the norm grows; they do not predict the *exponent* we measure, and
the exponent is not universal in our data — it is the same across activation
families within a budget range (which their theory would explain) but **not
range-stable**, falling from 1.5109 over 1k–4k to 0.7193 over 40k–160k. A
purely logistic-tail account predicts a single asymptotic behaviour, so the
drift is a fact about our measured range that we report rather than smooth over.
What carries the content is not `α` alone but the ratio `−α/β`: `β` varies by
construction across families at fixed optimizer and fixed loss, and the onset
exponent tracks `1/β` with a five-point through-origin slope of **1.0984
[0.958, 1.239]** against independently measured `α = 1.1173` [0.999, 1.236].
The agreement is to 1.7% in point estimate with intervals of roughly ±10%, so it
is consistency within uncertainty and not a precision claim.

---

## Where this work sits

The literature above divides cleanly. Ren & Lim, Hanin & Sellke, Naitzat et al.
and Guss & Salakhutdinov establish **what a constrained architecture can
represent**, in topological or approximation-theoretic terms. Cohen et al.,
Ahn–Zhang–Sra, Shalev-Shwartz et al. and Soudry et al. establish **how gradient
methods behave**, including where they cannot converge and how they fail.

Our contribution is a quantitative bridge at the smallest nontrivial instance of
the first group's framework: within the class of activations that the
representability theorems declare equally unobstructed, the boundary of what
gradient descent actually finds sits strictly inside the boundary of what the
architecture can express, the gap is set by a fold-geometry quantity `D(a)`
through a proved lower bound on `|w₂|`, and the boundary's location moves with
training budget as a power law whose exponent is predicted by an analytically
derived divergence rate. The principal limitation, stated in the abstract rather
than deferred: the law is established under Adam, and three cross-optimizer
tests either failed or proved unmeasurable.
