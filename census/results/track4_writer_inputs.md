# Track 4 writer inputs: citations, notation, checker paragraph, run populations (2026-09-25)

For the paper's writer. Nothing was run except reads of committed files and web lookups. No existing file was
changed. Sections: (1) citations, (2) notation, (3) the AI Use Statement paragraph on the checker, (4) run populations.

---

## 1. Citations: verified exactly, or excluded

**How each paper was verified (2026-09-25).**
- The title and author list were read from the official page's own citation metadata (`citation_title`,
  `citation_author`), with the page's official BibTeX where one exists.
- 2024 papers: proceedings.iclr.cc and proceedings.neurips.cc.
- ICLR 2023 papers: the official iclr.cc conference page. proceedings.iclr.cc has no 2023 volume
  (`/paper_files/paper/2023` returns "Page not found"), and the author accepts iclr.cc as the source (as in WP-13).
- Each was cross-checked against arXiv.
- Every sentence below is written from the abstract on the official page. That abstract was fetched and read in
  full; nothing is quoted.

**Result: all six are INCLUDED. None is excluded.**

| key | venue | official page | arXiv | differences to note |
|---|---|---|---|---|
| `lyu2024dichotomy` | ICLR 2024 | proceedings.iclr.cc (pp. 33897–33936) | 2311.18817 ("Published as a conference paper at ICLR 2024") | The proceedings give "Du, Simon" and "Lee, Jason"; arXiv gives "Du, Simon S." and "Lee, Jason D.". The entry uses the proceedings form, following WP-13's rule. Note that `references.bib` (`jin…`, line 203) spells the same authors "Simon Shaolei Du" and "Jason D. Lee", so choose one form for the whole bibliography. |
| `kunin2024getrich` | NeurIPS 2024 (vol. 37) | proceedings.neurips.cc (pp. 81157–81203, DOI 10.52202/079017-2580) | 2406.06158 ("NeurIPS 2024") | none |
| `glasgow2024sgd` | ICLR 2024 | proceedings.iclr.cc (pp. 52419–52430) | 2309.15111 | The title's "near-Optimal" and "the XOR problem" are lower-case on both pages, exactly as written. |
| `zhu2023minimalist` | ICLR 2023 | iclr.cc/virtual/2023/poster/11908 | 2210.03294 | No proceedings volume, so there are no pages. |
| `rubin2024grokking` | ICLR 2024 | proceedings.iclr.cc (pp. 23881–23904) | 2310.03789 | none |
| `nanda2023progress` | ICLR 2023 | iclr.cc/virtual/2023/poster/11385 (also listed as oral 12572) | 2301.05217 | The current arXiv abstract differs from the one on iclr.cc. The sentence uses only content that appears in both. iclr.cc labels it "top 25% paper"; per WP-13, do not cite any presentation label. |

**BibTeX conventions (the same as WP-13).**
- Fields are read from the proceedings page.
- `editor` and `url` are left out.
- The ICLR proceedings BibTeX has `volume = {2024}`, which is only the year repeated. WP-13's ICLR entries leave it
  out, so these do too.
- ICLR 2023 entries have no pages because there is no volume.

```bibtex
% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/909c8fef63e1cede406ce9e6794f99a2-Abstract-Conference.html
@inproceedings{lyu2024dichotomy,
  title     = {Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking},
  author    = {Lyu, Kaifeng and Jin, Jikai and Li, Zhiyuan and Du, Simon and Lee, Jason and Hu, Wei},
  booktitle = {International Conference on Learning Representations},
  pages     = {33897--33936},
  year      = {2024}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2024/hash/94074dd5a072d28ff75a76dabed43767-Abstract-Conference.html
@inproceedings{kunin2024getrich,
  title     = {Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning},
  author    = {Kunin, Daniel and Ravent{\'o}s, Allan and Domin{\'e}, Cl{\'e}mentine and Chen, Feng and Klindt, David and Saxe, Andrew and Ganguli, Surya},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {81157--81203},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/079017-2580},
  year      = {2024}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/e6d37cc5723e810b793c834bcb6647cf-Abstract-Conference.html
@inproceedings{glasgow2024sgd,
  title     = {{SGD} Finds then Tunes Features in Two-Layer Neural Networks with near-Optimal Sample Complexity: A Case Study in the {XOR} problem},
  author    = {Glasgow, Margalit},
  booktitle = {International Conference on Learning Representations},
  pages     = {52419--52430},
  year      = {2024}
}

% verified: https://iclr.cc/virtual/2023/poster/11908 (proceedings.iclr.cc has no 2023 volume; arXiv 2210.03294 agrees)
@inproceedings{zhu2023minimalist,
  title     = {Understanding Edge-of-Stability Training Dynamics with a Minimalist Example},
  author    = {Zhu, Xingyu and Wang, Zixuan and Wang, Xiang and Zhou, Mo and Ge, Rong},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/682f87a8c306098ec8be29019bd76aa4-Abstract-Conference.html
@inproceedings{rubin2024grokking,
  title     = {Grokking as a First Order Phase Transition in Two Layer Networks},
  author    = {Rubin, Noa and Seroussi, Inbar and Ringel, Zohar},
  booktitle = {International Conference on Learning Representations},
  pages     = {23881--23904},
  year      = {2024}
}

% verified: https://iclr.cc/virtual/2023/poster/11385 (proceedings.iclr.cc has no 2023 volume; arXiv 2301.05217 agrees)
@inproceedings{nanda2023progress,
  title     = {Progress measures for grokking via mechanistic interpretability},
  author    = {Nanda, Neel and Chan, Lawrence and Lieberum, Tom and Smith, Jess and Steinhardt, Jacob},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

**One sentence each, in writer wording.** Each is based on the abstract on the official page.

- **Lyu et al. (2024).** They prove that homogeneous networks trained from large initialisation with small weight
  decay stay near a kernel predictor for a long time, then move sharply to a min-norm or max-margin predictor, which
  changes test accuracy abruptly. Their transition happens along a training trajectory. Ours is a static property of
  the training loss minimised at fixed output scale, certified by interval arithmetic, and training enters it only
  through a computable lag.
- **Kunin et al. (2024).** They derive exact solutions for a minimal model that moves between lazy and rich learning.
  These show that unbalanced layer-wise initialisation variances and learning rates set the degree of feature
  learning, through conserved quantities that shape the learning trajectory. Their solutions describe a training
  trajectory. Our threshold is a static, certified property of the loss at fixed output scale, and training enters it
  only through a computable lag.
- **Glasgow (2024).** For minibatch SGD on a two-layer ReLU network learning XOR, they prove two phases. First, a small
  network's neurons find features independently. Then SGD maintains and balances those features, and the few neurons
  that found them are amplified as their second-layer weights grow. We share the role of output-weight growth in making
  a hidden unit's feature pay off. We differ in locating the output scale at which correct placement becomes the loss's
  preferred solution, as a certified property of the loss rather than through an analysis of SGD's trajectory.
- **Zhu et al. (2023).** They build a simple objective that shows edge-of-stability behaviour and analyse its training
  dynamics rigorously in a large local region. As they do, we use a minimal model to make a training phenomenon exactly
  analysable. Our object, though, is a threshold of the training loss at fixed output scale, not the dynamics of
  gradient descent at large step size.
- **Rubin et al. (2024).** Using the adaptive-kernel theory of feature learning on teacher–student models, they show
  that after grokking the network is analogous to the mixed phase that follows a first-order phase transition, with
  internal representations sharply different from those before it. Our threshold is also a switch between two kinds
  of solution. Here it is between minimisers of a fixed-scale training loss for a single hidden unit, located by
  certified computation rather than by a kernel-limit theory.
- **Nanda et al. (2023).** They reverse-engineer the algorithm that small transformers learn for modular addition, and
  use it to define progress measures that change continuously before the grokking transition. We share the aim of
  explaining an abrupt change by a quantity that moves continuously before it: here, output scale relative to a
  certified threshold. In our case that quantity is computed from the loss, not read off trained weights.

**Do not say** that any of these papers studies a conditional threshold in output scale (the same rule as WP-13).

---

## 2. Notation

Sources read:
- `paper/WRITER_INPUTS_v4.md`;
- `paper/WRITER_INPUTS_v4_patch.md`: WP-12, WP-16, WP-21 and WP-23, plus WP-3, WP-9, WP-11, WP-15, WP-17, WP-19 and WP-20
  for the appendix list;
- `results/math_note_v2.md`: §1, §10 and §11;
- the docstrings of `src/residual_timescale.py` and `src/lag_law.py` (for χ and κ).

### Main-text symbols

| symbol | meaning | defined where |
|---|---|---|
| s | Output scale. Width 1: s = \|w₂\|. Width 2: s = ‖w₂‖₁, with the hidden output normalised so that ‖ṽ‖₁ = 1. The logit is z = s·φ_θ(x) + b. | math note §1 (A = sε^{3/2} with s = \|w₂\|), §10 "Setting" (s = ‖w₂‖₁), §11 (s = \|w₂\|); WP-3 "Setting" |
| R | Output scale in gap units: R = sĜ(a)/2. The paper's certified R uses the rigorous lower bound Ĝ_cert. | math note §1 "Units" (R = sĜ(a)/2 = KA(1 + δ_ε)/2); v4 Block 1 table; WP-7 and `certificate_audit.md` (the rigorous Ĝ_cert in R) |
| Ĝ(a) | The certified maximum class gap: the supremum over first-layer parameters (w₁, b₁) of G at activation parameter a. It is a certified enclosure. | math note "Three objects" (item 1: the gap-maximising placement defines Ĝ); v4 Block 1 table (column "Ĝ (certified)"); WP-7 (rigorous enclosures; G\* = Ĝ(a)) |
| G | Class gap (worst-case gap) of the hidden unit, in orientation w₂ > 0: G(w₁, b₁) = min over the outer windows of f_a(w₁x + b₁), minus max over the inner window. G > 0 means the unit is placed. It is taken over the continuous windows. The data-point version G_n is appendix only. | math note §10 Lemma 2 and §11 "Which gap each step uses"; the checker docstring (`src/verify_certificates.py`); WP-12 |
| R\* / R_glob | The conditional placement threshold: the R at which the gap G of the *global* minimiser of the profiled training loss at fixed output scale changes sign. It is certified as a bracket at a = 1.30–1.60. | math note "Three objects" (item 3); v4 Block 1 ("Certified thresholds (1c)" and the paper-facing thresholds table) |
| R_solve | The R at which the global conditional minimiser's solve margin changes sign: from there on it is sign-correct everywhere, not just placed. It is certified as a bracket. | v4 Block 1, the note under the thresholds table; math note §2(b) (limit R_solve^∞) |
| R_own | A run's own-sample threshold: R_glob computed on that run's own training set instead of the 800-point population, before training. | v4 Block 1 (1d) and "Sample-specific thresholds" (S1–S3); v4 "Prospective own-seed test" ("computed from its own … training set before training"); WP-1 (U_own) |
| χ | Timescale ratio: the growth rate of output scale (d log s/dt per step at the crossing) over the branch's relaxation rate, lr·λ_min(D^{−1/2} H D^{−1/2}). Here H is the Hessian of the loss in (w₁, b₁, b₂) at fixed w₂, evaluated at the branch minimiser. D is Adam's diag(√v̂ + ε); D = I for SGD. | The writer inputs call it "the ratio" or "timescale ratio" and never use the symbol χ: WP-16 ("The residual's timescale", "EXT"), WP-21, WP-23. The operational definition is in `src/residual_timescale.py` (docstring). The symbol χ appears only in `src/lag_law.py` (see note 2). |
| κ(a) | Lag constant, in r = κ(a)·χ, where r = (s_c − s\*)/s\* is the relative residual of the crossing above the switch. | **Not defined in any writer input or in `math_note_v2.md`.** The only definition is the docstring of `src/lag_law.py`, which is **untracked** (not committed). It cites "math note §13", which does not exist: `math_note_v2.md` ends at §12. An untracked `results/lag_law/` directory appeared while this file was being written, apparently from a concurrent Track 1A run; nothing in it is committed. See notes 1–3. |

**Notes on the main-text symbols (for the author).**
1. **κ is already taken.**
   - κ(a) = Ĝ(a)/D(a) in `scaling_limit_results.md`, `fold1d_theorem.md` (κ(a) ∈ [0.305, 0.328]) and the v3
     `WRITER_INPUTS.md`.
   - The math note's "Three objects" says the gap maximiser "defines Ĝ, K and κ".
   - **κ₀ appears in v4 Block 3's main-text design**: the held-out windows are chosen at quantiles of "the certified
     limiting-cubic κ₀". It also appears in WP-18's baseline RK ("regression on log κ₀").
   - If the main text uses κ(a) for the lag constant, κ₀ in Block 3 needs another symbol, or the lag constant does.
2. **The relationship the writer inputs actually use is not r = κχ.**
   - WP-16, WP-21 and WP-23 use the frozen fitted line: residual = α + β·ratio, with α = 0.0157 and β = 2.658. It has
     an intercept and a single slope across a.
   - The per-a, no-fit κ(a) of `src/lag_law.py` is described there as "derived after the fitted relationship was
     known". No committed result or registration for it exists yet.
   - Any main-text sentence built on r = κ(a)·χ has no writer-input source until Track 1A is committed.
3. **The two definitions of χ differ slightly.**
   - `src/residual_timescale.py` uses d log s/dt at the crossing and each run's own Adam D.
   - `src/lag_law.py` uses (ṡ/s\*) at the switch scale s\*, and a median preconditioner P over runs.
   - State which one the main text means.
4. **"R\*" does not appear in the writer inputs as a name for the threshold.** They use R_glob throughout. A\* is a
   different object: the limit switch in rescaled units. If R\* is used, say R\* ≡ R_glob and keep A\* for the limit.
5. **Wording of the own-seed test's training set.**
   - v4 ("Prospective own-seed test") says R_own comes from each run's "own 200-point training set".
   - Elsewhere, training sets are 400 points: `fold1d.make_data(200, seed)` returns 200 per class. The own-seed code
     calls `data(200, seed)`.
   - Confirm "200-point" against 400 points (200 per class) before printing it.

### Every other symbol in the writer inputs: appendix only

**Activation, windows and data**
- **a, ε** — activation parameter in f_a(t) = t + a sin t, with ε = a − 1. a is needed in the main text. ε is appendix
  only unless the scaling law is stated there.
- **I, O** — the inner window [−0.8, 0.8] (class 0) and the outer windows ±[1.2, 2.0] (class 1).
- **n, n_O, n_I, ȳ** — the number of data points (800 for the population, 400 for a training set), the per-class
  counts, and the mean label (½).
- **Δ** — the asymmetry of the outer window in Track 2. O = [−2.0, −1.2] ∪ [1.2, 2.4], so Δ = 0.4 (WP-15).
- **m** — the ramp term E_O x − E_I x. It is 0 on symmetric windows and 0.44 on Track 2's (WP-15). math note §11 Step 4
  uses it the same way.

**Loss and profiled problem**
- **ℓ(z, y)** — logistic loss, log(1 + eᶻ) − yz.
- **σ** — the logistic sigmoid (§10), and separately the rescaled pre-activation σ = px + q (§1). Two uses.
- **L\*(θ; s)** — the profiled loss, minimised over the output bias b at fixed hidden parameters θ and scale s.
- **b\*** — the minimising output bias.
- **L\*_ε, L\*_0** — the profiled loss in rescaled coordinates, and its ε → 0 limit problem (§1).
- **θ, θ\*** — the hidden parameters. θ\* = (α\*, sign(D\*)·π/2) is the class-mean maximiser (§11). In WP-12's
  proposition, θ\* is that point; in `src/lag_law.py`, θ\*(s) is the tracked branch.
- **φ_θ** — the hidden unit's output as a function of x (§10).

**Rescaled limit (math note §1–§3)**
- **p, q** — rescaled first-layer weight and bias: w₁ = √ε·p and b₁ = π + √ε·q.
- **A** — the rescaled output scale, A = s·ε^{3/2}.
- **A\*** — the limit placement switch, certified A\* ∈ (0.68125, 0.6875]. Its Krawczyk value is 0.6854452.
- **A_solve** — the limit solve switch, in (1.05875, 1.06].
- **A′(0)/A\*** — the first-order shift of the switch, [0.6621547, 0.6621550].
- **h(σ)** — the limit activation profile, −σ + σ³/6.
- **φ_ε** — the rescaled activation, which tends to h.
- **G₀** — the limit gap, min_O h − max_I h.
- **K** — sup G₀, in [0.5794558, 0.5794951]. R = KA(1 + δ_ε)/2.
- **K(ε)** — K at finite ε.
- **K(24)** — the localisation box {|p| ≤ 24, |q| ≤ 2√2 + 48}. The name clashes with the constant K.
- **K_loc** — the local (corner) version of K(ε), in WP-19.
- **δ_ε** — the O(ε) correction in R = KA(1 + δ_ε)/2.
- **U (box)** — the 0.05 box around (p₀, q₀) in the uniqueness chain. The name clashes with the predictor U below.
- **B(24), B_full** — the localisation loss bounds, recomputed by PAVA (WP-11).
- **R_glob^∞, R_solve^∞** — the ε → 0 limits of the two thresholds.
- **c₁** — the first-order coefficient of R_glob in ε, [0.2852300, 0.2852303].
- **k₁** — the K(ε) contribution to c₁, −0.37692.
- **a₁** — the first-order term of A (the product a₁k₁ = −0.250 in v4 Block 2). It is the same quantity as c_s = A′(0)/A* = 0.66215 (final round notation; WP-26).
- **C** — the compact set of WP-19. The name clashes with the predictor C and with WP-3's cut grid C.

**Small- and large-scale limits (math note §10, §11)**
- **Δμ, μ_O, μ_I** — the class-mean gap, and the means of φ_θ over the outer and inner classes.
- **Var(φ_θ)** — the variance of φ_θ over all n points.
- **m₄** — the fourth central moment of φ_θ in the Lemma 1 remainder. The remainder constant is also named K, which
  clashes with the constant K.
- **D(α), D_c(α)** — the cosine contrast E_O cos αx − E_I cos αx, over data points or continuous windows. It clashes
  with Adam's D in χ.
- **α\*, D\*** — the maximiser of |D|: α\* = 1.7913244 on data points, 1.7922917 on continuous windows. D\* is the
  maximum.
- **D_P(a)** — max |D| over the placement domain |α| ≤ a/1.4.
- **G_n** — the gap over data points.
- **Γ_n** — its supremum.
- **η** — the Lipschitz correction between window placement and data placement.
- **h_I, h_O** — grid half-spacings.
- **s₀(a), s₁(a)** — the analytic bracket for the switch.
- **R₀, R₁** — the same bracket in R units. R₁ = log(n/log 2) = 7.05.
- **B_t, B_∞, γ** — Adam's per-step bound: B_∞ = 7.2703, γ = β₁²/β₂.
- **lr, β₁, β₂, ε_Adam** — Adam's hyperparameters.
- **N_min** — the minimum number of steps to reach the threshold scale.
- **α (growth exponent)** — the measured |w₂| ∝ B^α, α = 1.1172. Here B is the step budget.

**Width 2**
- **ṽ, αᵢ, βᵢ, u** — width-2 output weights (‖ṽ‖₁ = 1), the units' first-layer weights and biases, and the unit
  activation.
- **c** — the linear coefficient Σṽᵢαᵢ.
- **R₂** — width-2 R, s·Γ̂₂/2.
- **Γ̂₂** — the width-2 analogue of Ĝ: the computed supremum of the width-2 gap that sets R₂'s units
  (`block6_width2_design.md`). Its stop condition fired in the W0 scan (WP-9).
- **G₊** — the gap in the positive orientation (width-2 tables).
- **s_lo, s_hi** — the Track 2 validated switch bracket, s ∈ [0.4371, 0.4532].
- **s_glob** — Track 2's switch scale, 0.4451.
- **φ₂** — the output learning-rate factor at width 2 (0.0093, 0.01778; WP-15).

**Localisation lemma (WP-3)**
- **W(s, a), W₊, W₋** — the bound on |w₁| for any global conditional minimiser.
- **c, C (cut grid)** — a cut and the 80-cut grid.
- **π(c), δ(π)** — the class-count fraction and its logit bound.

**Predictors and scoring (Blocks 3, 3′, WP-1, WP-18)**
- **U** — the population R_glob as a predictor.
- **C** — C = λ(a)·R_glob, fitted.
- **λ(a)** — the lag factor: 1.115 at a = 1.30 and 1.164 at a = 1.50. It clashes with λ_min.
- **B1, B2, B3** — the baselines: output weight, pooled R, and fraction placed by budget.
- **U_own, C_own** — the own-threshold predictors, unfitted and fitted.
- **ρ_res(a)** — C_own's lag factor.
- **S-early** — the early-branch own threshold.
- **PL, PL5, RG, RK** — the WP-18 post hoc baselines.
- **κ₀** — the limiting-cubic constant used to pick the windows (see note 1).
- **e_own, e_pop** — per-run |log error| against the own and population thresholds.
- **R_cross** — the crossing R.
- **residual** — R_cross/R_own − 1.
- **x₅₀** — the 50% point of a placement curve.
- **T₊, T₋** — the mirror-branch own thresholds.
- **H10, …, V80, G1–G4** — window labels.

**Timescale account (WP-16, WP-21, WP-23)**
- **α, β (fitted line)** — the intercept and slope of residual = α + β·ratio: α = 0.0157, β = 2.658. The names clash
  with α\*, αᵢ, the growth exponent α, and β₁, β₂, βᵢ.
- **φ (lag tests)** — the multiplier on w₂'s learning rate: 0.25, 0.5, 1 or 2. The name clashes with φ_θ and φ_ε.
- **t\*** — the rule "switch φ at the last plateau exit".
- **λ_min, H, D, P** — the relaxation eigenvalue, the joint Hessian, Adam's diagonal and the preconditioner.

**Symbol clashes to resolve before the main text is fixed:**
- κ (lag constant, Ĝ/D, κ₀);
- α (α\*, αᵢ, growth exponent, intercept);
- β (β₁, β₂, βᵢ, slope);
- U (predictor, box);
- C (predictor, compact set, cut grid);
- K (constant, K(24), Lemma 1 remainder);
- D (cosine contrast, Adam diagonal);
- φ (hidden output, lag-test factor, φ₂);
- σ (sigmoid, rescaled pre-activation);
- λ (lag factor, λ_min);
- ε (a − 1, ε_Adam).

---

## 3. AI Use Statement: the independent checker's provenance (one paragraph, under 150 words)

> An independent checker, `verify_certificates.py`, re-verifies the certificates and shares no code with the
> certifying searches. It imports nothing from the project's source tree. It re-implements every objective
> (the profiled loss, the class gap and the limit problem) from its mathematical definition in ball arithmetic
> (python-flint/Arb, 80-bit), whereas the searches used float Lipschitz bounds, numpy outward rounding or mpmath
> intervals. Every decision is an Arb comparison, so rounding is enclosed. For exported certificates it reads only
> the certificate data files, each checked against a committed SHA-256 hash. Its tests include constructed cases it
> must reject. It verifies the finite-a placement brackets, the Ĝ(a) enclosures, the limit-switch bracket and its
> localisation, K = sup G₀ with its domain lemma, the Krawczyk boxes of the first-order calculation, and the
> positive-definite and ring certificates. The outer-exclusion certificates and the solve brackets (finite-a and
> limit) were not run; only the original searches support them.

**Where each statement comes from:**
- **Independence, imports and reading only certificate files**: the checker's header docstring.
- **Arb at 80 bits**: `PREC = 80` in the checker.
- **Hash check**: `files_match_manifest` against `certificates_manifest.csv`.
- **Tests that must reject**: `certificate_audit.md` (11 tests, "including constructed cases it must reject"). WP-17
  adds "tests exercise every new check on constructed pass and fail cases".
- **Coverage**: the WP-17 table and its "Say" paragraph, verbatim in substance.

**Why the paragraph says "for exported certificates".** The finite-a, Ĝ, limit and solve checks read the hash-checked
exported files. The other checks work differently:
- the K check runs its own branch and bound from scratch, reading only the published hi and argmax from
  `limit_K_base.csv`;
- the Krawczyk check starts from the published point in `first_order_c1.csv`, which is not hash-checked through the
  manifest;
- the positive-definite and ring checks take their boxes as parameters.

So "reads only the exported certificate data" is exact for the exported families only. Do not drop the qualifier.

**Not in the files, so not in the paragraph.** The files do not say who wrote the checker, or whether AI tools were
used to write it. Add that yourself if the AI Use Statement needs it.

**Stale statements elsewhere (not changed here).**
- `results/figures/v5/captions.md` ("thresholds") still says K's "independent Arb check is pending (WP-11)". WP-17
  now lists K as **verified**.
- WP-11's "Say" sentence is superseded by WP-17's.

**Do not say** that every certificate, or the sharp value R_glob^∞ = 0.19859, is independently verified (WP-17).

---

## 4. Run populations: "2,400 float32" against "4,800 pooled"

This section reports what the populations are and where each number uses one. It does not choose between them.

### Where each population comes from

| population | file | what it is |
|---|---|---|
| **2,400 float32** | `phase1_decomposition.csv` / `phase1_runs.csv`, rows with `precision == float32`. The same runs are in `fold1d_sweep.csv` (sin_family, a > 1: 1,600 runs at a = 1.02–1.25, 1.5, 2, 3) plus `fold1d_refine.csv` (800 runs at a = 1.30–1.45). | 12 values of a × 200 seeds, Adam lr 0.01, 2,000 steps. **They are identical run for run**: all 2,400 match on the solve outcome and on \|w₂\| exactly (difference 0.0; checked here). 430 solved; failures are 1,664 placement and 306 bias. |
| **2,400 float64** | `phase1_decomposition.csv`, `precision == float64` | 440 solved; failures are 1,633 placement and 327 bias. |
| **4,800 pooled** | `phase1_decomposition.csv` (all rows); `wi_phase1_reconciliation.csv` | both halves: 870 solved, 3,297 placement, 633 bias. |
| (a different 4,800) | `fold1d_sweep.csv` (all rows) | 4,800 rows over **six activation families**: sin_family 2,400 (including a = 0.5, 0.9, 0.95, 1.0), pwl_family 1,600, and gelu, leaky_relu, relu and tanh 200 each. `VERIFIED_NUMBERS.md` line 397 calls this "4,800 per-run rows". It is **not** the pooled Phase 1 population. |

**Fact the author should know before using "4,800 pooled".**
- The float64 half does **not** re-run the float32 runs at higher precision. At the same (a, seed):
  - the sign of w₁ agrees in only **48.6%** of the 2,400 pairs;
  - the failure class agrees in 72.8%;
  - the solve outcome agrees in 85.3%;
  - the median relative difference in \|w₂\| is 0.50.
- This is what independent initialisations look like. It matches the per-dtype RNG problem recorded in the retraction
  (`precision_prediction.md`; commit `1abaf09`: "the two arms of phase1_relog started from COMPLETELY DIFFERENT
  initialisations at the same seed").
- The fix ("draw the init once and cast") went into `src/phase1_relog.py` in that commit. `phase1_runs.csv` was last
  changed in the same commit, and its float64 rows still show independent draws.
- **So the pooled 4,800 are two independent draws of 200 initialisations per a, not 200 seeds × 2 precisions.**
  - The captions' wording "(200 seeds × 2 precisions)" (`captions.md`, "decomposition") describes the labels, not two
    precisions of the same runs.
  - Given the retraction's controlled test (shared initialisation: 0 flips in 180 paired runs), the difference between
    the halves is sampling variation, not arithmetic.

### Main-text numbers that depend on one of these populations

**In `WRITER_INPUTS_v4.md` and the patch:**

| where | number(s) | population it uses now |
|---|---|---|
| v4, "Figure manifest (v4)", `v4_decomposition.pdf` | "400 runs per a (4,800 total: 870 / 3,297 / 633); 0 disagreements" | **4,800 pooled** |
| patch WP-5 (figure manifest after rebuild), `v4_decomposition.pdf` | "n = 400 runs per a" | **4,800 pooled** |
| patch WP-12 (A2) and math note §11.1 | at a = 1.02: "0/200 solves", terminal \|w₂\| "median 1.85 and maximum 3.92" (IDs `A2 a=1.02: solved`, `…median terminal |w2|`, `…max terminal |w2|`) | **2,400 float32**: 200 runs, read from `fold1d_sweep.csv` by `src/harsh_review_a.py`, line 215. The pooled equivalent would be 0/400, with the float64 half at median 1.92 and maximum 3.94. |
| patch WP-12 (A2) | "observed onset a = 1.60" at 2,000 steps; 1.18, 1.06 and 1.03 at 8k, 32k and 128k | **Neither**: these are constants hard-coded in `src/harsh_review_a.py` (line 208), from the onset analyses. The script does not compute them from either population, and their source was not traced here. |

**In the other paper-facing files the main text draws on:**

| where | number(s) | population it uses now |
|---|---|---|
| `results/figures/v5/captions.md`, "decomposition" (main text) | "n = 400 runs per a (200 seeds × 2 precisions); 4,800 runs in total"; 3,297 / 633 / 870 | **4,800 pooled** |
| `results/figures/v5/captions.md`, "metric_check" (main text or appendix) | "2,400 training runs"; solve-rate 10–90% at R ∈ [0.330, 0.429]; continuous error 10–90% at R ∈ [0.055, 0.342]; width ratio 2.9; "76% … (1,815 runs, R < 0.30)" | **2,400 float32** (`src/metric_check.py` reads the sweep plus the refine, sin_family, a > 1) |
| `paper/VERIFIED_NUMBERS.md` §6 (metric-artifact test) | 2,400 runs; 153.6 (n = 995); 36.3 (n = 128); 76%; 1,815 runs | **2,400 float32** |
| `paper/WRITER_INPUTS.md` (v3) §7, still in force where v4 is silent | the table for both halves; "Region-wide certification of solved runs is pooled: 866 of 870 …" | both are given; the certification count is **pooled** |
| `paper/results_draft.md` (lines 100, 181) | "0 of 200 runs" at a = 1.02 | **2,400 float32** (200 runs) |

### Consistency, stated factually

- **The main text currently mixes the two populations.**
  - The decomposition figure and its numbers use the pooled 4,800.
  - The metric check and the a = 1.02 numbers (0/200, median 1.85, maximum 3.92) use the float32 2,400.
- The v3 handoff already recorded this and asked for one or the other throughout: "Either use float32 throughout
  (2,400 = 430 + 1,664 + 306) or both precisions throughout (4,800 = 870 + 3,297 + 633)". The v4 files have not
  resolved it.
- **What each choice would change:**
  - *Float32 throughout*: the decomposition becomes 430 / 1,664 / 306 with n = 200 per a. The metric check and the
    a = 1.02 numbers are unchanged.
  - *Pooled throughout*: a = 1.02 becomes 0/400 (median 1.92 and maximum 3.94 in the float64 half). The metric check
    would need re-binning over 4,800 runs. `metric_check.py` currently reads only the float32 sweep and refine files.
- **Either way, the caption wording "(200 seeds × 2 precisions)"** describes two independent initialisation draws
  labelled by precision, not two precisions of the same runs (see the fact above).
