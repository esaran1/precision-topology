# §7.6 When does the phenomenon occur? — replacement text

*Reframes the MNIST transfer result as a characterisation rather than a failed
transfer. **Every number is unchanged**; this is a framing change only. Source:
`results/mnist_budget_law_results.md`, registered in
`mnist_budget_law_prediction.md`.*

---

**The phenomenon has a precondition, and it is checkable in advance.** Sections
3–6 study a task where the classes are **not separable by a monotone map**: no
monotone activation can produce the two sign changes `sign(|x|-1)` requires, so
a fold is *necessary* and its depth sets the price. That condition is what
makes fold depth the governing variable. Where it does not hold, there is no
reason to expect the same behaviour — and we tested a setting where it does
not.

We built the same experiment on MNIST: an MLP `784 -> 256 -> [w] -> 128 -> 10`
with our own activation family `f_a(x) = x + a sin x` at a bottleneck of width
`w`, so the fold parameter is continuous and its monotonicity threshold is
exactly `a = 1`. The design carries its own control — our account predicts that
monotonic `a` should fail at a narrow bottleneck and that the `a`-dependence
should vanish at a wide one — and both were registered before measurement.

**The control failed, and the way it failed identifies the precondition.** At
no bottleneck width from 1 to 8, and no capability threshold from 0.80 to 0.97,
do monotonic activations fail where non-monotonic ones succeed. From width 2 up
the two classes of activation are indistinguishable, every gap below 0.2
accuracy points and smaller than the seed-to-seed spread. At width 1, where a
gap finally appears, **it has the wrong sign**: monotonic **0.8658** against
non-monotonic **0.7715**, a difference of **+0.0943 in favour of monotonic**
(`p = 0.0002`, two-sided permutation, 20,000 resamples).

**The reversal is the informative part.** Folding is not a free capability. A
fold identifies distant inputs — it maps two separated regions onto the same
output range — which is exactly what you want when the classes are nested and
exactly what you do not want otherwise. Through a 1-unit bottleneck MNIST is
not asking for a fold; it is asking to compress ten classes through a scalar,
and a monotone map does that better than one that folds distinct digits on top
of each other. **Where a fold is not needed to separate the classes, it is a
cost rather than a capability**, and our activation family pays it.

Three checks confirm the finding is about the task and not our setup. The fold
is **active**: between **10.2% and 34.7%** of bottleneck units sit past
`|x| > arccos(-1/a)`, where `f_a` is non-monotone, so the test is not vacuous.
It is **not a budget artifact**: quadrupling the budget to 8,000 steps leaves
the width-1 reversal intact (**0.9041** against **0.8487**). And it is **not an
ID-placement error**: the intrinsic-dimension estimators span **8.03 to 13.83**,
a factor of **1.72** — tighter than the 1.99x systematic disagreement that
closed our earlier width-axis attempt — and we probed from `w = 1`, far below
every estimate, up to `w = 8`.

**What this licenses.** The result does not show the budget law is false outside
the toy task. It shows MNIST-MLP **cannot pose the question**, because the
precondition is absent — and that is a scope statement a reader can act on
rather than a negative result. Combined with the requirements identified when
the width-axis attempt failed (§7.5), a qualifying setting must supply:

1. classes **not separable by a monotone map through the bottleneck** — the
   condition this experiment isolates, and the one MNIST fails;
2. a reproducible activation advantage meeting the four criteria of §7.1,
   stated against tanh rather than ReLU;
3. a flat bottleneck whose width is a single unambiguous number;
4. a stable intrinsic dimension, at least three nonlinear estimators agreeing
   within the sweep's resolution;
5. bottleneck input dimension well above twice that intrinsic dimension.

Condition 1 is new here and it is the cheapest to check: it can be settled by
inspecting the task before any network is trained. That is the practical
content of this section — not that a transfer failed, but that we can now say
in advance which settings are worth attempting.
