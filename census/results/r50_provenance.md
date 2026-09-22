# Item 1: `R50` provenance — the original script does not exist

## The search, exhaustively

| search | result |
|---|---|
| `git log -S "R50" --all` | 20 commits, **all** touching only `.md` / `.csv` |
| `git log -S "r50" --all` | same, plus two dataset-cache commits |
| `src/r_variable.py` at `d2d10a0` (the commit that first produced `R50`) | three functions: `oriented_gap`, `margin_capacity`, `separation_auc`. **No fit.** |
| every blob ever committed, scanned for `R50\|r50` | 20 blobs, **all** `.md` or `.csv` — `CLAIMS.md`, `VERIFIED_NUMBERS.md`, `paper_claims_delta.md`, `ghat_unification.md`, `metric_artifact_results.md`, `r50_certified.csv`, `r_collapse_results.md`, `third_optimizer_prediction.md` |
| untracked `.py` / `.ipynb` on disk | none contain `R50` |
| scripts outside `census/src` | unrelated project files (`pipelines/`, `simulators/`) |
| `git stash list` | empty |
| `git fsck --lost-found` dangling objects | 2 commits (my own `clean-master` work) and 2 blobs, both `phase2b_checkpoints.csv` |

**Conclusion: the script that produced `R50 = 0.3705` was never committed and is
not recoverable.** The value was computed in an uncommitted session and
hand-recorded into `r_collapse_results.md`.

## The replacement

`src/r50_fit.py`, committed, with the definition stated exactly:

- **Population**: the pooled union `r_pooled.csv + r_adamw.csv` = **3,150 runs**
  (12 `a`, 8 budgets, three optimisers), with `R` recomputed from the **certified
  `Ĝ`** (T65).
- **Binning**: 41 quantile edges over `R`; a bin is kept only if it holds **≥ 10**
  runs. Each kept bin contributes (mean `R`, solve rate). 40 bins survive.
- **Model**: two-parameter logistic `P = 1/(1 + exp(−k(R − mid)))`.
- **Fit**: least squares on the binned rates, grid search over `(k, mid)` at
  **900 × 1600**. *The result is grid-sensitive — coarser grids give values up to
  0.005 higher, which is how I first reported 0.3728 — so the resolution is fixed
  in the module and the convergence is documented.*
- **Crossing**: `R50 := mid`, the logistic's own midpoint where `P = 0.5` exactly.
  `R25`, `R75` are the analytic quantiles `mid + log(q/(1−q))/k`.
- **Interval**: cluster bootstrap over runs, 400 resamples, percentile 95%.

## Result

| quantity | value |
|---|---:|
| **`R50`** | **0.3682** |
| 95% CI | **[0.3603, 0.3730]** |
| `R25` / `R75` | 0.3458 / 0.3907 |
| `k` | 48.97 |
| rms residual | 0.0134 |
| `W = R75 − R25` | 0.0449 |
| **materiality `0.5·W`** | **0.0224** (was 0.0241) |
| `R50` under the restricted `Ĝ` | 0.3669 |
| **shift from the `Ĝ` switch** | **+0.0013** |

**`0.3705` is replaced everywhere by `0.3682 [0.3603, 0.3730]`.** The old value
sits **inside** the new interval, so nothing that depended on it changes
qualitatively — but it is no longer quoted, because it cannot be reproduced.

## Downstream

Materiality tightens **0.0241 → 0.0224** (7%). **No T52 verdict changes**: the
family shifts that failed condition 4 were 3–5× the threshold (`R50` = 0.247 /
0.295 vs 0.373), and every stratum that passed did so with margin to spare.
