# Block status as of 2026-09-22, end of experiment window

Per the plan, **no new experiments start after tonight**. Anything unfinished is
reported as unfinished rather than rushed. This is the authoritative status list.

---

## Complete, reported, in the ledger

| block | status | ledger | headline |
|---|---|---|---|
| **A** | complete | T57, `SESSION_LOG` #2 | O1: `CV(R) = 0.0311` (certified **0.0324**) vs `CV(\|w₂\|) = 0.2821` |
| **B** | complete | T60 | single continuous switch at `R_glob`; `R_spin` dropped at all six `a` |
| **C** | complete | `blockBC_results.md` | runs relax onto the branch (1.09 → 0.008) and track it |
| **E** | **complete, all three parts** | T59 | see below |
| **F** | complete | T63 | lag account falsified as a mechanism |
| **G** | complete | T61 | transfers to four unseen windows |
| **H** | complete (dose-response) | `blockH_rate_results.md` | validity gate **failed**; reported as such |
| **K** | complete | T62 | `β_B = 1` and overshoot restored; AUC claim removed |
| κ certification | complete | T64 | certified `κ ∈ [0.307747, 0.329464]` |
| one `Ĝ` | complete | T65 | all `R` switched to certified |
| ν / `α(ε)` | complete | T66 | separable form misspecified; `α = α(ε)` |
| pathwise distance (#8) | complete | T67 | at most **0.8%** closer than start |
| exact extrema (#13, #2) | complete | T64 | **870/870** solved runs certified exactly |
| MEP details (#9) | complete | `mep_method_appendix.md` | appendix wording fixed |

### Block E — all three parts, explicitly

1. **Saturation vs trapping**: **saturation**. Gradients fall to **0.07% of
   control**; a 5× budget recovers placement **2/15 → 7/15**, which a trapped run
   cannot do. The endpoint is an ill-conditioned flat valley near the
   constant predictor (`w₁ ≈ −0.003`, Hessian condition number > 10⁷).
2. **`hold_high` against `R_solve`**: held `R = 0.24663` lies **strictly between**
   `R_glob = 0.21446` and `R_solve = 0.30667`; **33/37 placed, 0/40 solved**, and
   the conditional minimiser at that exact `|w₂|` also places and **cannot solve**
   — so 0/40 is the landscape's prediction, not a shortfall.
3. **Growth-rate dose-response**: **validity gate failed** (`m=1` places 20/40 vs
   control 37/40, p = 4.3e-5) because the natural `|w₂|` trajectory is
   **non-monotone**. The decline with rate is present (0.550/0.500/0.375/0.000,
   endpoint p = 8.4e-09) but is **not a valid rate test**.

**Consequence**: the annealing framing is **not adopted** — it fails both
preconditions independently.

## Not done, and not starting

| item | status | why |
|---|---|---|
| **Corrugated-Hopf generator check** | **RESOLVED 2026-09-22 (T68)** — unblocked by the repository reference; their code read directly. Their generator implements **neither** of our readings nor its own Appendix G.1: no thickening at all, and `curve1` **self-intersects** (7.8e−15, proven analytically). See `hopf_generator_resolved.md`. *(Original blocked note:)* | `generator_diff.md`: the repository does not hold the published appendix's torus construction text (radii, offsets, sampling law). The diff **cannot be performed** without the appendix text or the reviewer's details. What *was* done: our generator characterised from **200,000 samples** and checked for internal consistency. Reconstructing the appendix from memory to diff against would manufacture agreement or disagreement with equal ease, and was deliberately declined. |
| **Ren–Lim S²⊔S² budget sweep** | **COMPLETE 2026-09-22 (T69), exploratory** — 23 min estimated, registered, run. The gap **widens** with budget (P = 0.9998), the registered falsifier: the budget law does **not** extend. See `blockS2_results.md`. *(Original not-started note:)* | No budget/epoch axis exists in any link artifact — `linking_width3.csv` and `linking_projected.csv` have no `epochs` column. The budget law (T44, `budget_law_results.md`) is established **only for the 1D fold task**. Extending it to the link setting is a genuinely new experiment: 5,120 existing link runs span depth/width/activation/seed at a **single** budget. Per the plan, **not started**. |
| **Block H, valid version** | design recorded, not run | Requires replaying each seed's own `\|w₂\|(t)` time-warped by `m`, so `m = 1` is the natural trajectory by construction. Recorded in `blockH_rate_results.md`. |
| **Block D** | superseded | D-1 became **vacuous** when `R_spin = R_glob` (T60); the two models it compared are identical. |
| **Blocks I, J** | not started | Never registered; not on the critical path. |

## What the paper can claim about the two unfinished items

**Corrugated-Hopf generator**: state that our generator's measured properties are
reported from samples, and that a line-by-line diff against the published appendix
is **pending the appendix text**. Do not claim agreement or disagreement.

**S²⊔S² budget sweep**: the budget law is a **1D-fold result**. The paper should
**not** imply it has been tested in the link setting. If a reviewer asks whether
the budget law generalises, the honest answer is that it is untested there, and the
1D result plus the R-threshold mechanism is what is offered.

## Reproducibility: RESOLVED 2026-09-22

**The `R50` script was confirmed absent from all history** — `git log -S` on both
spellings, every blob ever committed, untracked files, stashes and `git fsck`
dangling objects. Replaced by the committed `src/r50_fit.py`:
**`R50 = 0.3682` [0.3603, 0.3730]**, materiality `0.5W` **0.0224**. The old 0.3705
lies inside the new interval, so no verdict changes. See `r50_provenance.md`.

**The verifier now enforces this**: every artifact it reads must have a committed
producing script. The check found **7 gaps** (6 from this session's ad-hoc
analysis, 2 pre-existing: `theorem_perplacement.csv`, `blockB_fine_windows.csv`);
all are closed by `src/session_artifacts.py`. Verified functional by removing the
module — the check flags all 7 and passes when restored.
