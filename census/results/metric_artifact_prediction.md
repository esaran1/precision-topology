# Registration: is the R threshold a metric artifact?

**Written after reading Schaeffer, Miranda & Koyejo directly (source text in
`paper/sources/schaeffer_mirage.txt`, 14 pages), and before computing any
continuous-metric quantity against R.**

Date: 2026-09-12.

---

## What the paper actually claims, in three sentences

It argues that for a **fixed task and fixed model family, analyzing fixed
model outputs**, claimed emergent abilities arise from the researcher's choice
of metric rather than from a change in model behaviour with scale:
**nonlinear or discontinuous metrics produce apparent emergence, while linear
or continuous metrics produce smooth, predictable change**. The evidence is
threefold — three confirmed predictions on the InstructGPT/GPT-3 family, a
meta-analysis of BIG-Bench finding that emergent abilities appear under at most
5 of 39 metrics with **>92% under just two** (Multiple Choice Grade, which is
discontinuous, and Exact String Match, which is nonlinear), and a constructive
demonstration that choosing metrics can manufacture never-before-seen apparent
emergence in vision networks. The authors state explicitly that **"nothing in
this paper should be interpreted as claiming that large language models cannot
display emergent abilities"** — the claim is about specific previously reported
cases, not an impossibility result.

**Verified against the source**: the 92% figure, the two metric definitions,
the "fixed model outputs" scope condition, and the disclaimer are all quoted
from the extracted text, not from memory.

## Why the objection applies to us with full force

Our solve criterion is **binary and exact**: a run solves iff it produces 0
eval errors (and, for regional claims, 0 dense errors). That is structurally
the same shape as Exact String Match — an indicator on perfect performance.
Under Schaeffer's mechanism, imposing such an indicator on a smoothly improving
underlying quantity manufactures a sharp transition. The R threshold (0 of
2,160 below R = 0.30, 402 of 403 above 0.50) is exactly the pattern their
critique targets, and raising it requires disputing none of our numbers.

## The test (1a)

For every run in the pooled R dataset, regress the **continuous** outcome on R:

1. **Eval error count** (0 to 2,000) — the natural continuous analogue of the
   binary criterion, and the closest analogue of their per-token error rate.
2. **Achieved logit margin** — continuous and signed, negative for
   non-separating runs.
3. **Dense-verification margin** where stored.

Binned finely in R, reporting mean and quantiles, with the binary solve rate
overlaid on the same axis.

## The quantitative criterion, fixed now

Define, for any monotone outcome metric `M(R)`, the **transition width**

> `W_M` = the range of R over which `M` moves from **10% to 90%** of its total
> observed swing (normalizing `M` to [0,1] by its own range over the data).

`W_binary` is the same statistic for the binary solve rate, measured on the
same runs and the same bins. Then:

- **Genuine threshold** if `W_continuous <= 2 * W_binary`. The continuous
  metric transitions on essentially the same scale as the binary one, so the
  sharpness is a property of the system.
- **Metric artifact** if `W_continuous >= 5 * W_binary`. The underlying
  quantity moves smoothly across a region where the binary metric jumps.
- **Intermediate** if between. The honest statement is then that the
  underlying quantity changes rapidly but not discontinuously, and we report
  **the ratio `W_continuous / W_binary` as the quantitative measure of how
  much sharpness the metric contributes.**

The factor thresholds are set now, before seeing any continuous curve, and the
ratio is reported whatever it turns out to be.

## Registered interpretations (1b)

1. **Continuous metrics also transition sharply at R ~ 0.37**: the threshold is
   a property of the system, not the metric. This is a positive contribution to
   a live debate, and is stated as such — a capability threshold verified
   non-artifactual in a setting where the check is decisive.
2. **Continuous metrics vary smoothly through R = 0.37**: our threshold is
   partly a metric artifact in Schaeffer's sense. The R result survives only as
   *"R predicts where the binary criterion flips"*, and the "sharp capability
   threshold" framing is **withdrawn**.
3. **Intermediate**: report the ratio and state plainly how much of the
   sharpness is the metric.

**If outcome 2 or a high intermediate ratio obtains, it is reported
immediately and prominently, and the R claim is rewritten to the smaller form
before anything else is built on it.** This is the objection most likely to be
raised; being caught overclaiming here would damage the whole paper.

## The impossibility argument, scoped in advance (1c)

Below `a = 1` no width-1 network solves, at any R, and this is a **theorem**:
a monotone map composed with an affine map produces at most one sign change,
while `sign(|x| - 1)` requires two. No metric choice alters that.

**What this establishes**: at least one boundary in this system is genuine and
not metric-induced — the analytic threshold at `a = 1`.

**What it does NOT establish**: that the R threshold at ~0.37, which sits well
inside the non-monotonic regime where solutions provably exist, is equally
genuine. The first argument must not be used to carry the second, and the
write-up will say so explicitly.

## Item 2 criterion, registered

Per-family AUC for separating solved from unsolved, using R against `|w2|`
alone, `G*(a)` alone, and any other plausible single variable.

> **Dominance counts if R's AUC exceeds the best competitor's by >= 0.05 with
> non-overlapping 95% bootstrap intervals, in a family with >= 20 runs inside
> its transition band.**

Families with fewer than 20 transition-band runs are reported but not counted
as evidence either way. If R dominates in every qualifying family, the
**product structure** is universal even though the threshold value is not —
a weaker but defensible claim than the one the family test rejected. If it
fails in any qualifying family, that is reported with the margin.
