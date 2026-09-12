# Registration: data collapse, threshold independence, held-out prediction

**Written before any of Items 1-3 were computed.** Nothing below was adjusted
after seeing a residual curve. These are reanalyses of existing artifacts — no
training — so the registration is about *what counts as a pass*, fixed now.

Date: 2026-09-12.

---

## Item 1: data collapse

### The scaling variable

If the onset obeys `eps_onset ~ B^{-theta}`, then for family A with measured
`theta = 0.7340`, the natural scaling variable is

    u = (a - 1) * B^{theta}

and the law at the distributional level asserts

    P(solve | a, B) = F(u)

for a single monotone `F` shared across budgets.

### What is computed

1. Pool every measured `(a, B, rate)` cell. Fit a two-parameter logistic in
   `log u` by least squares on the rate, and report the RMS residual.
2. **Scan `theta` from 0 to 2** in fine steps, refitting `F` at each, and
   report residual as a function of `theta`. The minimizer is an **independent
   estimate of the exponent from all data**, not from the 50% crossings.
3. Report a 95% interval on the minimizing `theta` by **bootstrap over cells**
   (resample the pooled `(a,B,rate)` cells with replacement, refit, take
   percentiles) — not a curvature approximation, since the residual curve need
   not be quadratic.
4. Report residuals at **deliberately wrong exponents `theta = 0` and
   `theta = 1.5`** so that collapse selectivity is visible rather than assumed.
5. Repeat per family (A, q4, q2, q1, q0.667), each with its own
   `theta = alpha/beta_family`.
6. **The pooled version**: with `u_family = (a-1) * B^{alpha/beta_family}`, do
   all families collapse onto one curve?

### Pass conditions, fixed now

- **Collapse holds** if the minimizing `theta` is within the onset fit's
  interval AND the residual at the minimum is materially below the residuals at
  `theta = 0` and `theta = 1.5`. "Materially" is fixed as **a factor of 2**.
- **Collapse fails** if the residual is flat in `theta` (no selectivity) or the
  minimum residual is not below that factor.
- The minimizing `theta` is reported **whether or not** it agrees with
  `-0.7340`. A disagreement is a finding requiring diagnosis, not a reason to
  prefer one estimator.

### Registered outcomes

1. **Clean collapse, `theta` agrees with the onset fit** — the law is validated
   at the distributional level, using every point rather than 4-6 crossings.
   Two estimators on different information agreeing is stronger corroboration
   than either alone.
2. **Collapse fails, onsets still move** — the law describes the *median*, not
   the distribution: rate curves change shape with budget rather than
   translating. This is a real narrowing and **belongs in the abstract**, not
   in a later section.
3. **Collapse works but `theta` disagrees with the onset fit** — the two
   estimators disagree; diagnose rather than choose.

## Item 2: threshold independence

The 50% crossing is a choice. If the curves *translate*, every crossing level
gives the same exponent.

Extract onsets at **25%, 50%, 75%** from the same rate curves by linear
interpolation in `a` between bracketing cells, refit `log eps_onset` against
`log B` at each level, and report all three exponents with **bracketing counts
stated before the exponents**.

- **Pass**: the three exponents agree within their fit intervals. The 50%
  choice is then immaterial and is reported as verified.
- **Fail**: systematic drift across levels means the curves change shape, and
  Item 1's collapse must fail too. **The two tests are linked**, and this
  prediction is registered now: *if Item 2 shows systematic drift, Item 1 will
  show poor collapse, and vice versa.* Reporting both makes the finding legible
  either way.

Cells where a level cannot be bracketed are excluded and the exclusion counted.

## Item 3: held-out prediction

Fit the onset exponent on the **four smallest budgets only** (2k, 4k, 8k, 32k
for family A), predict `eps_onset` at **128k**, and compare against the
measured value.

The prediction interval is computed from the fit's standard error on the
four-point fit **before** the held-out cell is consulted, and is recorded in
this file's amendment block at the moment it is computed.

- **Pass**: measured 128k onset inside the predicted interval.
- **Fail**: outside. Reported as a failed held-out prediction without
  softening.

Repeated for any constructed family with >= 4 bracketed cells.

---

## Amendment (2026-09-12, before the held-out cell was consulted)

Item 3's four-point fit and its prediction interval are recorded here at the
time of computation, prior to reading the 128k measurement. See
`results/collapse_results.md` for the recorded values and the comparison.
