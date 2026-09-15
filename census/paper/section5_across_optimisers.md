# §5.5 Across optimisers — replacement text

*Supersedes the hedge "the optimiser difference is undetectable at this
coverage, not absent." Every number recomputed 2026-09-14 from `r_pooled.csv`
and `r_adamw.csv`.*

---

**Across optimisers.** Solvability is governed not by the weight scale a run
reaches but by `R = |w2| G*(a)/2`, the margin that scale can support. That
distinction is testable by changing the optimiser, because different optimisers
reach different weight scales on the same task. Pooled over three of them at
`a = 1.25` — 600 runs — **no run below `R = 0.30` solved (0 of 361), and every
run above `R = 0.50` did (154 of 154)**. In the transition band the three agree
within sampling error: **all nine pairwise Fisher exact tests are null, minimum
`p = 0.674`**, and the fitted 50% crossings are **Adam 0.3350 [0.3203, 0.3939],
AdamW 0.3454 [0.3342, 0.3597], SGD 0.3709 [0.3587, 0.3753]**, with all three
intervals overlapping on a common window.

**The third optimiser was chosen to make that agreement mean something.** AdamW
applies decoupled weight decay, shrinking `|w2|` at every step — it acts
directly against the quantity this account says drives solvability, which a
third adaptive method resembling Adam would not. The suppression is measurable:
over the shared 1k–40k budget range the terminal-scale growth exponents are
**`alpha` = 1.2627 ± 0.0593 (Adam), 0.9802 ± 0.1269 (AdamW), 0.7188 ± 0.0238
(SGD)**, and median terminal `|w2|` at 40,000 steps is **97.5, 30.6 and 19.4** —
a **5x spread**. We registered the direction (`alpha_adamw < alpha_adam`) before
measuring; it holds. **Three optimisers arrive at different places on the `|w2|`
axis, by different dynamics, and solve at the same value of `R`.** The optimiser
enters only through where on the `R` axis its runs land.

**Two design points, stated because a reader checking our numbers will reach
them.** First, the criterion: on point estimates alone, AdamW's crossing differs
from the pooled value by 0.0251 and Adam's by 0.0355, **both nominally
exceeding the 0.0241 threshold we first registered**. That threshold, applied to
point estimates alone, cannot distinguish agreement from disagreement here — it
would flag the new arm on the same basis it already flags the existing one. We
therefore fixed, before collecting the AdamW data, that a material shift
requires **interval separation and a significant regime-wise test**. Neither
occurs: the intervals overlap and the smallest of nine `p`-values is 0.674. The
conclusion rests on the tightened criterion, not the loose one.

Second, coverage. Adam's runs **jump across** the transition band — median `R`
goes 0.234 at 4,000 steps to 0.578 at 8,000 — leaving only **10 runs inside it**,
which is why the two-optimiser comparison could not resolve the question. An
eight-seed pilot located AdamW's in-band window and two budgets (5,000 and
6,000) were added for that reason, giving **57 in-band runs** against Adam's 10
and SGD's 18. **The pilot measured where runs land, not whether they solve**, so
the choice fixes coverage rather than outcome.

**Scope.** This is a statement about optimisers, not about settings: one task,
`a = 1.25`, width 1. The threshold's *value* remains family-specific (§6), and
nothing here extends `R` beyond the activation families in which the product
structure was verified.
