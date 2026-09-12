# What the draft still lacks

*Accurate rather than optimistic, per the brief. Written 2026-09-11 after §2
and §7 were drafted and verified.*

## Current state

| section | file | status |
|---|---|---|
| §1 Introduction | — | **not written** |
| §2 The link setting | `paper/link_section_draft.md` | drafted, verified |
| §3 Setting and theorem | `paper/results_draft.md` | drafted, verified |
| §4 Unreached regions | `paper/results_draft.md` | drafted, verified |
| §5 Budget law | `paper/results_draft.md` | drafted, verified |
| §6 Geometric relationship | `paper/results_draft.md` | drafted, verified |
| §7 CIFAR | `paper/cifar_section_draft.md` | drafted, verified |
| §8 Related work | `paper/related_work.md` | drafted from primary sources |
| §9 Limitations | — | **not written** |
| §10 Conclusion | — | **not written** |
| Abstract | — | **not written** |

Seven of eleven components exist. The four that do not are the ones a reviewer
reads first.

---

## 1. Abstract — write from scratch

**Material that exists**: every number it would quote is drafted and verified.
Nothing new needs measuring.

**What must be decided, not merely written**: the abstract has to carry the
single-optimizer scope. The standing decision (user direction, 2026-09-06) is
that the Adam-only limitation belongs **in the abstract rather than in
limitations**, because three cross-optimizer tests failed or proved
unmeasurable. Drafting it means choosing which two of the four results lead —
the 5,580-run zero, the `|w2|` lower bound, the budget law, the CIFAR
budget-dependence — and no such choice has been made.

**Estimated difficulty**: moderate. The constraint is that the strongest
result (the zero) is a negative in a setting that does not test the theorem it
sits next to, and the most quantitative result (the budget law) is
single-optimizer. Neither is a clean headline on its own.

## 2. §1 Introduction — write from scratch

**Material that exists**:
- The ordering argument, settled this session: link setting as motivation, 1D
  task as the instrument, negative-then-mechanism.
- The framing in `notes/generality_framing.md`.
- The contribution list is implicitly determined by §2–§7.

**What must be written**: the opening claim, the contribution bullets, and the
paragraph that explains why a four-parameter network is worth a reader's time.
The last of these is the load-bearing one and is exactly the argument the
ordering decision exists to support — provable impossibility for the actual
object, exactly computable regions, and enough measurement access to exclude
the standard accounts by direct measurement.

**Risk**: the introduction is where overclaiming would most easily re-enter.
`notes/overclaim_sweep.md` has a word list (confirms / proves / demonstrates /
validates) that should be run over the draft once written.

## 3. §9 Limitations — write from scratch, but the content exists

**Material that exists, and is unusually complete**:
- `results/interpretive_audit.md`: five of seven load-bearing claims drift to
  "something adjacent" under adversarial reading, each with the required
  rewording already specified.
- `results/rejection.md`: R1–R6 rejection arguments with answers.
- `AUDIT.md`: the exposures that survive.
- `notes/instrument_artifacts.md`: five instances of instrument error read as
  object property.
- The solid-tori / 1-D-curves mismatch (`results/generator_diff.md`).
- The single-optimizer scope (`results/cross_optimizer.md`).

**What must be written**: selection and compression. This is assembly, not new
analysis, but it is a genuine editorial decision — the honest limitations list
is long enough that including all of it would swamp the contributions.

**Specific items that must appear**, none of which can be dropped:
1. The link experiments do not test Theorem 3.7 (solids, not curves).
2. The budget law is Adam-only; three cross-optimizer tests failed.
3. `alpha` is measured, not derived, and is **not range-stable** (1.5109 over
   1k–4k falling to 0.7193 over 40k–160k).
4. Family B is an unexplained counterexample to the law, not a limiting case.
5. The CIFAR flip's crossing regime is **undetermined**; it is the extreme
   case, not the headline.
6. The width prediction is untested, not refuted.
7. Registered prediction failures: P-W1 at width 8, P-step, P-1a/P-1c,
   P-4a/4b/4c.

## 4. §10 Conclusion — write from scratch

**Material that exists**: §2.7 and the closing of `related_work.md` both
already contain a version of the "where this sits" argument.

**What must be written**: a short section. The one decision it embeds is
whether the paper claims a *phenomenon established in two settings* or a
*phenomenon established in one setting and probed in another*. §2 and §7
measure different things (a categorical zero versus a budget-dependent
magnitude), so the honest version is closer to the second, and the conclusion
should not quietly upgrade it.

---

## Structural work beyond the missing sections

### 5. Section renumbering

§3–§6 in `results_draft.md` are written as if they follow §2 directly, which
they now do. But `results_draft.md`'s internal cross-references were written
before §2 and §7 existed. Every "§3" / "§4" reference in all three files needs
one pass for consistency once the numbering is final. **Not yet done.**

### 6. The three drafts are separate files

`results_draft.md`, `link_section_draft.md`, and `cifar_section_draft.md` need
to be merged into one document with consistent heading levels, or a build
order fixed. Trivial, but not done.

### 7. Figures

`src/figures.py` regenerates five figures for §5–§6. **§2 and §7 have no
figures.** Candidates that the artifacts already support: the dose-response
(3/8/17%), the width sweep (the two-effect structure of §2.5), and the flip
(tanh/ReLU crossing). None exists yet. The §2.5 figure is the most valuable,
because the two-effect structure is the part most likely to be misread as a
clean width-6 boundary.

### 8. Theorem statement and proof placement

`results/fold1d_theorem.md` holds the theorem and proof. §3 states it in
`D(a)` with the asymptotic form labelled. Whether the full proof goes in the
body or an appendix is undecided.

### 9. Ledger items still open

- **"Five link families" is undefined** (see `link_cifar_verification.md`).
  Reconstructing from artifacts gives four geometries, or seven counting
  winding numbers separately. Neither is five. T1 and
  `results/winding_results.md` should define the term or drop it. The §2 draft
  sidesteps this by enumerating rather than counting.
- The interpretive audit's five rewordings are applied in the §3–§6 draft but
  should be re-checked against §2 and §7 now that those exist.

---

## What is *not* missing

Worth stating so the remaining work is not overestimated:

- **No new experiments are required** for any section listed above. Every
  number the paper needs has been measured and verified.
- **Related work is done from primary sources**, with theorem numbers checked
  against the papers' own text and a re-runnable quotation checker.
- **Verification infrastructure exists**: `src/verify_ledger.py`,
  `src/figures.py`, `src/zero_decomposition.py`,
  `paper/verify_quotations.py`, and the two verification records.

The remaining work is writing and editorial judgment, not measurement.
