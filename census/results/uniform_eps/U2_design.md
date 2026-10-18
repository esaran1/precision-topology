# Track U, step U2: design, and a sample of leaf timings (checkpoint 2; NOT a certificate)

**Status.** This page is a design plus timings from an exploratory prototype, `src/u2_sample.py`.
- The prototype works to the producer standard of §20: float arithmetic with a stated rounding allowance, a 10⁻⁹ allowance per
  decision and τ = 10⁻⁷.
- Nothing here has been checked independently. Nothing here is cited as proved.
- No full run has been made, and none will be without the author's approval.
- Sample outputs are in `results/uniform_eps/u2_sample/` (under 0.5 MB in total). `STATUS.md` there records the hold of
  2026-10-16.

## 0. The author's decisions (2026-10-16) on flags F1–F6, and what each requires

**F1. A-window: APPROVED; the claim is WINDOWED (author's decision, 2026-10-16 evening; the earlier all-scale condition is
withdrawn).** For every ε ∈ [0, 0.6] and every A in the window J(ε), the global conditional minimiser is unplaced (G < 0) for
A < A_ε and placed (G > 0) for A > A_ε. Here A_ε is the branch switch, s_br(ε) = A_ε·ε^{−3/2}, and s = A·ε^{−3/2}.

- **The window at ε = 0.6.**
  - **A:** J ≈ [0.8958, 0.9637]. **s:** J ≈ [1.9275, 2.0735], with ε^{−3/2} = 2.15166 at ε = 0.6.
  - These come from the float anchor A_ε^float = 0.93036, computed at ε = 0.5995 (`probes1.jsonl`). That anchor lies inside the
    certified a = 1.60 bracket A ∈ [0.92952, 0.93533], s ∈ [2.0000, 2.0125].
  - The anchor at ε = 0.6 itself moves by about 10⁻⁴ in A. The full run's switch step records the exact window endpoints for
    the revision to quote.
  - At ε = 0 the window is [0.66, 0.71] in A. In s it is unbounded, because s = A·ε^{−3/2} → ∞.

- **The window contains the certified thresholds.** J(ε) is A_ε^float × [0.96287, 1.03582]. The margins below assume the
  worst case: the anchor A_ε^float may lie anywhere inside each certified bracket. (Arithmetic on the committed CSVs only.)

| threshold | A bracket | margin below (bracket lo − window lo) | margin above (window hi − bracket hi) |
|---|---|---|---|
| limit, A\* ∈ (0.68125, 0.6875] (window [0.66, 0.71] at ε = 0) | [0.68125, 0.6875] | 0.02125 | 0.02250 |
| a = 1.01 (`first_order_finite.csv`) | [0.68973, 0.69016] | 0.02519 | 0.02428 |
| a = 1.02 | [0.69403, 0.69490] | 0.02493 | 0.02400 |
| a = 1.03 | [0.69879, 0.69922] | 0.02552 | 0.02460 |
| a = 1.04 | [0.70272, 0.70447] | 0.02440 | **0.02343** (smallest finite-a margin) |
| a = 1.08 (`c1_followup_finite.csv`) | [0.72041, 0.72221] | 0.02501 | 0.02401 |
| a = 1.09 | [0.72536, 0.72581] | 0.02649 | 0.02553 |
| a = 1.10 | [0.72989, 0.73034] | 0.02666 | 0.02569 |
| a = 1.11 | [0.73398, 0.73489] | 0.02637 | 0.02538 |
| a = 1.12 | [0.73856, 0.73902] | 0.02697 | 0.02600 |
| a = 1.30 (`cond_certified_brackets.csv`, glob; A = s·ε^{3/2}) | [0.81337, 0.81542] | 0.02822 | 0.02708 |
| a = 1.35 | [0.83343, 0.83602] | 0.02845 | 0.02727 |
| a = 1.40 | [0.85381, 0.85698] | 0.02865 | 0.02742 |
| a = 1.45 | [0.87165, 0.87542] | 0.02872 | 0.02745 |
| a = 1.50 | [0.89272, 0.89714] | 0.02888 | 0.02756 |
| a = 1.60 | [0.92952, 0.93533] | 0.02891 | 0.02749 |

  Every certified threshold lies inside its window, with margins of at least 0.0212 (limit) and 0.0234 (finite a). The full
  run will also check that every certified switch box Z(ε) lies inside J(ε).

- **OPEN, and not built (author's decision): the global minimiser outside the window.** There is no status branch and
  bound outside J(ε), and no s → 0 or s → ∞ lemma. So U2 says nothing about global sign changes of the gap at scales outside
  the window.
  - **Why the small-scale limit does not carry over:** on the exact 800-point population, §21.5 (Lemma K4) shows that
    |D| = 2 forces |α| ≥ π/q. The small-scale limit is therefore forced onto the alias, and every admissible W is at least
    623,614. A uniform-in-ε "unplaced at small scale" statement would have to be proved for that aliased limit; the
    continuous-window and float-quadrature results of §21.3–§21.4 do not transfer to this objective.
  - The large-scale side, an explicit s_large(ε) with the configuration placed, is likewise not written.

**F2. Continuity across slabs: APPROVED (author's decision)**, provided that:
- every leaf has a STRICT certified uniqueness margin. Outer leaves have D > τ = 10⁻⁷ on top of the 10⁻⁹ allowance; ring boxes
  have a gradient margin > 0 after the allowance; PD sub-boxes have λ_lower > 0; Krawczyk images lie strictly inside their
  boxes;
- slabs are CLOSED intervals [e_k, e_{k+1}] sharing endpoints.

The producer and the checker both test closedness and contiguity: e_{k+1} of one slab equals e_k of the next, exactly, as
floats. Then at a shared endpoint both slabs certify the same unique global minimiser.

**F3. Coordinates: APPROVED (author's decision).** The covered domains contain the localisation set at every ε:
- **Regime R** (slabs inside [0, 0.0462]). Lemma G2 holds uniformly for ε ≤ 0.0462 (§20.2). Every global minimiser lies in
  K(24) relative to its fold centre π + 2πk. The windings b₁ ↦ b₁ + 2πk and the reflection p ↦ −p (exact x-symmetry of
  the data) reduce this to K(24) ∩ {p ≥ 0}, which is the regime-R domain, for every ε in the slab.
- **Regime O** (slabs inside [0.0462, 0.6]). The §5.0 lemma holds for every a: every global minimiser has |w₁| ≤ W(s, a), and
  W increases as s falls and as a rises. So for every (ε, A) in the unit, W(s, a) ≤ W(s_min, a_max), with s_min = A_lo·ε_hi^{−3/2}
  and a_max = 1 + ε_hi.
  - In rescaled coordinates, p = w₁/√ε ≤ W(s_min, a_max)/√ε_lo, which is the domain's p_max.
  - One b₁-period is q-length 2π/√ε ≤ 2π/√ε_lo, so q ∈ [−π/√ε_lo, π/√ε_lo] contains a full period for every ε in the slab.
  - The periodic images of the branch box lie outside that q-window while ε_hi < 4ε_lo, which holds for every planned slab.
  - p ≥ 0 by the reflection.
- **Together.** The slabs cover [0, 0.6] as closed intervals (F2). So at every ε, one of the two lemmas applies and its
  domain is searched.

**F4. Start at ε = 0: APPROVED AND PREFERRED (author's decision)**, with U1 kept as a cross-check, provided:
- **The rescaled objective extends analytically to ε = 0.** In the kernel form φ_ε = −σ + (1+ε)σ³Gt(u) (§2):
  - every kernel (Gt, J, sinc, K1, JJ, K2, K3, SS) is an even entire function of u, so a function of u² = εσ²;
  - φ_ε, φ_σ, φ_σσ and their ε-derivatives are therefore entire in (σ, ε), with φ_0 = h;
  - β/√ε = arccos(1/(1+ε))/√ε is analytic at 0 with value √2. The prototype now uses its series below ε = 10⁻⁶;
  - the second fold (2π − β)/√ε tends to +∞, so its condition is vacuous at ε = 0.
- **check_u2_unit encloses it correctly on intervals containing 0.** Its activation evaluates the kernels as series in
  v = u² = E·S² (with E the ε-ball), never via √E. The closed forms are used only when v is certainly ≥ 1, and the series are
  used, with their tail ball, otherwise.
  - Tests: balls with E ∋ 0 contain the mpmath values at ε = 0 (h, h′, h″) and at random small ε. They also contain U1's
    second-order expansion h + εr ± ε²m₂.
  - The prototype's regime R already accepts e_lo = 0, since every √ε enters only inside u.

**F5. R-form: nothing to build** (the author answers this).

**F6. Checker: APPROVED (author's decision).**
- **Re-search is the PRIMARY route:** check_u2_unit runs its own Arb branch and bound per unit.
- Replay of the exported leaves is used only for units the re-search fails.
- The producer still exports the integer quadtree indices and the ring/PD/switch box lists for every unit, so any failing
  unit can be replayed. The exports are small (§4).

## 1. The target, and what touches its definitions (proposals; the author's decisions are in §0)

**Target (author's U2 text).**
- For every ε ∈ [ε₀, ε_max], with ε₀ = 4.1·10⁻⁶, ε_max = 0.6, a = 1 + ε, and the 800-point quadrature objective:
  - the continued branch is the global conditional minimiser at its switch;
  - so s_glob(ε) = s_br(ε).

**What each slab must establish.** For every ε in the slab and every A in the slab's A-tube J(ε), using rescaled A = s·ε^{3/2}:
- **(i)** the global minimiser of L\*_ε(·; A) is unique, up to p ↦ −p and the windings b₁ ↦ b₁ + 2πk. It lies in the 0.05 box U
  of that A-sub-interval, and it is the unique critical point there.
- **(ii)** the branch's gap changes sign exactly once in J(ε), at A_ε inside the slab's switch box Z.

That gives s_glob(ε) = s_br(ε) = A_ε·ε^{−3/2} for every ε in the slab.

**Flags.** Each of these touches a definition or criterion of the target. Each is a proposal.

- **F1. The A-window (tube).**
  - Theorem U1 states its conclusion for A ∈ [0.66, 0.71]. Its "s_glob" is the global sign change *within that window*. A
    sign change of the global gap far from the branch switch is not excluded by U1, and would not be by U2.
  - At finite ε the switch moves: A_ε rises from 0.6854 at ε = 0 to about 0.930 at ε = 0.6. A fixed window therefore cannot
    serve.
  - **Proposal:** J(ε) = A_ε^float × [0.66/A\*, 0.71/A\*] = A_ε^float × [0.9629, 1.0358], i.e. the same relative window as U1. It is
    split into 40 A-sub-intervals, as U1 was. Here A_ε^float is a float Newton point used only to place the window; the
    certified switch box Z must lie inside it.
  - Cost scales linearly with the window width. The author may prefer a narrower or wider window.
  - **This choice sets what "s_glob" means in U2. It needs the author's decision.**
- **F2. "The continued branch" across slabs.** In each slab the branch is identified only as the unique global minimiser.
  Continuity across slab boundaries, and across the step from the rescaled domain to the original-coordinate domain, then
  follows from that uniqueness: at a shared ε, both slabs certify the same unique global minimiser. No separate continuation
  certificate is proposed.
- **F3. Coordinates.**
  - The author asked for "rescaled coordinates near ε₀ and original coordinates further out". The prototype uses the
    rescaled parametrisation (p, q, A) with the *exact* φ_ε (no expansion) everywhere. That is an exact linear change of
    variables, w₁ = √ε p, b₁ = π + √ε q, s = A ε^{−3/2}.
  - The original coordinates enter only through the domain and localisation lemma:
    - **Regime R** (ε ≤ 0.0462, Lemma G2): K(24) ∩ {p ≥ 0}, with U1's exact isotonic exclusions;
    - **Regime O** (any a; the §5.0 lemma): {0 ≤ w₁ ≤ W(s_min, a_max)} × one b₁-period, mapped to (p, q).
  - **Why:** in fixed (w₁, b₁) the branch moves with ε (dw₁/dε = p/(2√ε)), and ∂_ε L\* is of order 1/ε. In (p, q) the branch
    moves only slowly, and ∂_ε L\* near it is about 0.08 (U1's L₁), so slabs can be about 20 times wider.
  - This choice does not change the statement. The author may still prefer literal (w₁, b₁) cells for regime O, as in the
    finite-a certificates. They would be costlier in ε.
- **F4. Overlap with U1.** U1 covers (0, ε₀], but its outer exclusion is not yet checked in Arb (that is U3). U2's first
  slab can start at ε = 0 at no extra cost: the rescaled regime is valid down to 0, and [4.1·10⁻⁶, 5·10⁻³] already certifies
  in the sample. Doing so would make one certificate cover (0, 0.6]. **Proposal for the author:** start at 0, and keep U1 as
  the separate, sharper small-ε statement.
- **F5. The R-form.** The R-form, R_glob = sĜ/2, is not covered. As in §16.5, it also needs Ĝ(a) to be attained at the
  tied corner for every ε. This is unchanged.
- **F6. The rigour route.** The producer's bounds on the kernels (§3) are grid envelopes, not proved. **Choice: the checker
  route (§5).** The independent Arb checker computes its own enclosures of φ_ε and its derivatives directly, and relies on no
  producer bound. The certificate is then "producer float + independent Arb check", as in Tracks 4 and 5. No kernel lemma is
  needed in the math note.
  - **Why this route:**
    - Arb evaluates sin and cos of balls rigorously, so each kernel's range over a u-ball is a direct ball computation. It
      needs no envelope and no monotonicity claim.
    - A hand-written proof of 7 envelope functions (sup over |u| ≥ U of each kernel) would be a long new lemma with its own
      error risk, and the checker would have to trust it.
    - The checker must re-derive every leaf anyway, so routing the kernel rigour through it costs nothing extra.

## 2. Decomposition

**Slabs.** [ε₀, 0.6] is split into ε-slabs [e_lo, e_hi]. Each slab carries its own A-tube J (F1), split into 40
A-sub-intervals. A **unit** is one (slab × A-sub-interval). Per unit:

1. **Branch side.**
   - θ_c is the float branch point at (A_c, ε_c).
   - The upper bound is BU = L\*(θ_c) + |∂_A L\*|h_A + |∂_ε L\*|h_ε + ½·(an upper bound of the (A, ε) Hessian of L\*(θ_c; ·)).
   - That Hessian bound is ¼·mean(centred (∇_{A,ε} z)²) + mean|z_{Aε}, z_{εε} terms|.
2. **Outer exclusion.** Every θ outside the 0.15 box, in the regime's domain, satisfies L\*(θ; A, ε) > L\*(θ_c; A, ε). This
   is checked on a quadtree over (p, q) from 0.25 cells, with (A, ε) as cell dimensions. A cell is excluded by any of:
   - **(iso), (full):** R only. Exact and ε-uniform (Lemma G2), as in U1.
   - **(pert):** the coupled 4-D Taylor bound on the DIFFERENCE,
     D ≥ (L − L_b) − |g_p|h_p − |g_q|h_q − |g_A − g_A,b|h_A − |g_ε − g_ε,b|h_ε − ½·mean_i T_i − curv_b − 2·10⁻⁹ > τ.
     - T_i bounds Σ_ab |z_ab|h_a h_b over the cell. It uses the Gauss–Newton/Schur argument: the profiled Hessian is at least
       mean((σ − y)∇²z) in all four variables.
     - The ε-terms of T_i are **centred by a per-cell shift c(ε) = φ_ε(q_c)**, which the profiled bias absorbs. That is,
       |φ_ε(σ_i) − φ_ε(q_c)| ≤ sup|φ_σε|·|σ_i − q_c|, and likewise for φ_εε via φ_σεε.
     - **Without this centring, regime O failed at Δε = 0.01.** It got stuck at p ≈ 0, q at the b₁-period edge, where every
       point has the same large |σ| ≈ 14.6, and one unit ran 4.7 h. With it, Δε = 0.03 certifies (§3).
   - **(fav):** the favourable-logit profile. φ_ε's exact range over each σ-interval at ε_c, using its critical points
     u = ±β + 2πk, is widened by h_ε·sup|φ_ε|.
3. **Ring.** On 0.05 ≤ |θ − θ_c|∞ ≤ 0.15, every box shows |∂_p L\*| or |∂_q L\*| > Σ_a M_ka h_a + 10⁻⁹.
   - M_ka = ¼√(Q_k Q_a) + mean sup|z_ka| bounds the profiled Hessian entries in (p, q, A, ε).
   - Boxes start at side 0.0125 and split adaptively to depth 5. The working set is now capped at 256 boxes × 800 points (§4).
4. **PD.** On the 0.05 box U × A-sub-interval × slab, the (p, q, b) Hessian of L at b ∈ b\*'s bracket is enclosed in
   intervals, from mean-value enclosures of φ, φ_σ, φ_σσ. The test is λ_min(mid) − ‖rad‖_F > 0, on adaptive sub-boxes.
5. **Switch (per slab).**
   - **Box Z:** a 4-variable Krawczyk test on (∇_{p,q,b} L, G) in (p, q, b, A), with ε as an interval parameter. The active
     pair (inner x = 0.8, outer x = −1.2) is checked against φ_ε's critical points β/√ε and (2π − β)/√ε over the slab.
   - **Branch boxes:** 3-variable parametric Krawczyk boxes over the A-tube (geometric A-edges, as in U1), with the gap's
     sign certified left and right of Z.
   - Each box must lie in its sub-interval's U.
   - It hit a shape bug in the sample, now fixed. **It has not been timed yet.**

**Leaf claim exported per outer leaf:** (p, q, h_p, h_q, test, D). The unit's header records ε-slab, A-sub-interval, θ_c,
BU and its parts, regime, domain, τ, the 10⁻⁹ allowance, and the population SHA-256. Ring and PD export their box lists in
the same way.

**Time cap and slab splitting** (`certify_split`). Each unit runs under a wall-clock cap, checked inside rounds, not only
between them. A unit that times out or fails has its ε-slab halved, and both halves are retried, recursively, down to a
stated minimum width. A slab that still fails at the minimum width is reported as failed. Tested on constructed cases
(`tests/test_u2_gate.py`); **those tests have not run yet** (§6).

## 3. Sample timings (one process, nice 15, one thread)

**Outer exclusion, middle A-sub-interval k = 20 of 40** (`probes3.jsonl`, `probes4.jsonl`). These are the runs after the
centring fix; "fail" means the unit stopped at its cap or round limit.

| regime | ε-slab | Δε | certified | leaves | seconds |
|---|---|---|---|---|---|
| R | [4.1e-6, 1.0e-3] | 1e-3 | yes | 39,171 | 2.5 |
| R | [4.1e-6, 3.0e-3] | 3e-3 | yes | 39,180 | 2.4 |
| R | [4.1e-6, 5.0e-3] | 5e-3 | yes | 39,183 | 3.5 |
| R | [4.1e-6, 1.0e-2] | 1e-2 | **no** (band p ≈ 18, q ≈ −39) | 40,729 | 167 |
| R | [1e-3, 4e-3] | 3e-3 | yes | 39,180 | 2.4 |
| R | [0.010, 0.015] | 5e-3 | yes | 39,178 | 4.1 |
| R | [0.010, 0.020] | 1e-2 | **no** | 41,041 | 159 |
| R | [0.0262, 0.0362] | 1e-2 | yes | 39,180 | 5.3 |
| R | [0.0352, 0.0452] | 1e-2 | yes | 39,171 | 2.2 |
| O | [0.001, 0.002] | 1e-3 | **no** (domain to q = ±99) | 97,470 | 132 |
| O | [0.010, 0.013] | 3e-3 | yes | 19,696 | 5.6 |
| O | [0.010, 0.020] | 1e-2 | **no** | 32,885 | 365 |
| O | [0.0462, 0.0762] | 0.03 | yes | 6,699 | 2.5 |
| O | [0.0462, 0.1462] | 0.1 | **no** | 4,501 | 143 |
| O | [0.15, 0.18] | 0.03 | yes | 1,762 | 0.7 |
| O | [0.15, 0.25] | 0.1 | **no** | 7,943 | 224 |
| O | [0.30, 0.33] | 0.03 | yes | 1,205 | 0.3 |
| O | [0.30, 0.40] | 0.1 | **no** (live cell next to the box) | 4,625 | 66 |
| O | [0.50, 0.51] | 0.01 | yes | 986 | 0.2 |
| O | [0.50, 0.60] | 0.1 | **no** (live cell next to the box) | 2,876 | 2.8 |

**Full sample unit,** ε ∈ [4.1·10⁻⁶, 3·10⁻³], R, three A-sub-intervals k = 0, 20 and 39. All certified:

| k | outer | ring | PD |
|---|---|---|---|
| 0 | 39,179 leaves in 53.0 s (an outlier; probably contention with another job, not reproduced) | 5,252 boxes, 0.78 s, margin ≥ 2.78·10⁻⁶ | 80 boxes, 0.09 s, λ ≥ 0.0453 |
| 20 | 39,180 leaves in 2.6 s | 5,424 boxes, 1.0 s, margin ≥ 1.17·10⁻⁵ | 80 boxes, 0.13 s, λ ≥ 0.0412 |
| 39 | 39,172 leaves in 3.7 s | 5,568 boxes, 1.76 s, margin ≥ 1.84·10⁻⁶ | 80 boxes, 0.19 s, λ ≥ 0.0373 |

For comparison, U1's first-order bounds stop PD at ε = 5.17·10⁻⁶. The ring and PD links no longer bind.

**Exports:** 3 npz files of 101–109 KB each, **2.6–2.8 bytes per leaf** (compressed), with SHA-256s in `probes4.jsonl`.

**Soundness spot checks** (`validate.json`, before the centring fix):
- 4 slabs, 1,200 random (θ, A, ε) per slab in pert leaves, plus fav leaves: no bound violated; the smallest slack is
  1.2·10⁻³.
- Shifting θ_c by 0.3 makes every case fail, as it should.

**Peak RSS:**
- Outer units: ≤ 1.6 GB, which includes the envelope tables and numpy.
- **Spec 11 (ε ∈ [0.0352, 0.0452], R, k = 0, with ring and PD) reached 3.73 GB and was SIGSTOPped by the watchdog.** It was
  later killed by the coordinator.
- **Diagnosis:** ring_unit (and pd_unit) evaluated *all* pending boxes at once, i.e. (boxes × 800) arrays, about 30 of them.
  With a 10⁻² slab the Lipschitz terms grow, so more boxes split, and 20k+ boxes give about 3–4 GB. The outer unit is
  already chunked (2,000 × 800).
- **Cap:** at most 256 boxes per evaluation (`BOX_CHUNK`), which bounds the ring/PD working set to about 50 MB.
- **Still to do:** confirm which stage it was, and measure the peak after the cap on the same spec. `python -m src.u2_sample
  stages 0.0352 0.0452 R 0 40` reports peak RSS after each stage. It is blocked by the disk hold.

## 4. Preliminary projection (the author asked for it; incomplete; switch and checker not yet timed)

**Slabs.**
- **R:** 5·10⁻³ slabs up to about 0.026, then 10⁻², reaching 0.0462. That is **about 8 slabs**.
- **O:** 0.03 slabs over [0.0462, 0.6]. That is **about 19 slabs**.
- In total **about 27 slabs × 40 = about 1,080 units**. Allow up to 2× for splits at harder A-sub-intervals: only k = 0, 20
  and 39 were sampled, and only on one slab.

**Producer, one worker.**

| part | per unit | units | total |
|---|---|---|---|
| R outer | 2.5–5 s (one 53 s outlier) | about 320 | 0.2–0.5 h |
| O outer | 0.2–2.5 s | about 760 | ≤ 0.5 h |
| ring + PD | about 1–2 s | about 1,080 | ≤ 0.6 h |
| switch | not timed | 27 slabs | U1 took 452 s for 110 boxes in mpmath; the numpy interval version should be faster |

- **Total:** order 1.5–4 h at one worker, including the 2× allowance and taking the switch at U1's cost. At the 3-worker cap,
  about 0.5–1.5 h.
- **Peak memory:** about 1.6 GB per worker, after the cap (to be confirmed).

**Exports.**

| part | leaves or boxes | size at 2.7 B per leaf |
|---|---|---|
| R outer | about 39k per unit, about 12.5M in total | about 34 MB |
| O outer | about 1–7k per unit, about 2.5M in total | about 7 MB (bytes per leaf in O not measured) |
| ring | about 5.5k boxes per unit, about 6M in total | order 20 MB |
| PD | small | small |
| headers | | small |

- **Total on disk:** about 0.1 GB, against free disk of 16.7–23 GB (it has fluctuated). Disk is not the constraint.

**Checker.** Not timed; this is an extrapolation from Track 4.
- Track 4's Arb outer check cost about 19 ms per cell evaluation (82,686 evaluations in 1,577 s).
- Replaying every producer leaf (15M) would take **about 80 h at one process**, about 27 h at 3.
- A fresh Arb branch and bound per unit, as Track 4 did (about 2k evaluations per unit), would take **about 11 h** at one
  process.
- **Recommendation:** use a fresh branch and bound for the outer link, and replay the ring/PD boxes and the switch boxes.
- The checker, not the producer, is the cost driver.

**Feasibility.** Nothing in [ε₀, 0.6] looks infeasible on this sample.
- The failures are all slabs that are too wide, and the splitting rule handles them.
- The regime boundary is the one real constraint: regime O fails below about 0.004 (domain to q = ±99), so the rescaled
  regime R must cover [ε₀, 0.0462].
- The open risks are the switch box over a 0.03 slab and the edge A-sub-intervals.

## 5. Specification of the new ε-interval check function in the independent checker (not written; independent of the producer)

**Name:** `check_u2_unit(name)`, in `src/verify_certificates.py`. It is written fresh: it shares no code with
`src/u2_sample.py` and imports nothing from `src/`.

**Inputs (data only, hash-checked):**
- the population npz;
- the unit header (JSON): ε-slab, A-sub-interval, regime, θ_c, the published claims (τ, the 0.15 / 0.05 radii);
- the leaf arrays.

**Activation.** A new class `PhiEpsBall(E)` with E an Arb ball [e_lo, e_hi]. It gives balls for φ, φ_σ and φ_σσ for a σ-ball
S, valid for every ε in E. It is computed by the checker itself:
- u = √E·S as a ball, then φ = −S + (1 + E)·S³·Gt(u) and φ_σ = −1 + (1 + E)·S²·J(u).
- Gt and J are evaluated on the u-ball as follows:
  - for |u| ≤ 1, by the alternating Taylor series with an explicit Lagrange-type tail ball. The tail is bounded by its first
    omitted term, which holds once the terms decrease. This avoids the cancellation of (u − sin u)/u³ for small balls;
  - for |u| > 1, by the closed forms in Arb.
- φ_σσ = (1 + E)·S·sinc(u).
- Every result's range must be finite. If any intermediate is not finite, the bound becomes ±∞, as with HAct (the NaN rule).
- **No envelope, no monotonicity claim, and no producer constant is used.**

**Per-cell objective.**
- The logit is A·(φ − c) + b with c = the ball φ(q_c, E) at the cell centre; this is the HShift idea of Track 4. A and ε are
  both balls.
- The lower bound of L\* over the cell is the checker's own: the mean-value form and the direct form, as in
  `Objective.lower_bound`, but with the activation above. b\* is bracketed in Arb (tail form).
- Its own subdivision is allowed (in θ and in ε) up to a stated depth. A leaf passes if its rigorous bound exceeds
  U = max over the (A, ε) corners and an Arb ball bound of L\*(θ_c; A, ε).
- Convexity in A is used, as stated in the math note. In ε no convexity is assumed: the checker bounds over the ε-ball
  directly.

**Checks:**
1. **Hashes.**
2. **Domain:**
   - R: K(24) and the B(P) table recomputed by exact PAVA (existing code);
   - O: W(s_min, a_max) from `lemma_W`, mapped to (p, q) with one full b₁-period for every ε in E.
3. **Exact tiling** of the domain minus the closed 0.15 box. The leaves' integer quadtree indices are exported for this.
4. **Every leaf:** the rigorous L\* lower bound > U, or (iso/full) the exact-PAVA conditions.
5. **Ring:** on each exported box, the Arb gradient ball excludes 0 in ∂_p or ∂_q, for all (A, ε) balls.
6. **PD:** Sylvester on the Arb (p, q, b) Hessian over box × bracket.
7. **Switch:** Arb Krawczyk on Z and on every branch box with ε as a ball; active-pair validity with β = acos(1/(1+E))
   as a ball; gap signs.

**Tests (constructed cases):**
- pass: a real exported unit;
- fail: θ_c shifted by 0.3, with a witness;
- fail: a leaf bound edited above the truth;
- unresolved: an ε-ball too wide within the depth cap;
- the activation's balls contain mpmath 60-digit values at 1,000 random (σ, ε);
- NaN propagation.

**Export additions this needs from the producer:**
- the integer quadtree indices (level, i_p, i_q) per leaf, as in the finite-a exports;
- the ring and PD box lists.

### 5.1 As written (2026-10-16; not yet run)

The functions are added to `src/verify_certificates.py`. No existing function was changed, and nothing is imported from
`src/`. Tests are in `tests/test_verify_u2.py`.

**`U2Act(E, c)`: the activation.** It gives φ − c, φ_σ, φ_σσ, φ_ε and φ_εε for every ε in the ball E.
- The kernels Gt, J, S, K1 and K2 are evaluated as functions of v = E·σ², with σ² formed with a nonnegative lower end.
  No √ε is ever taken.
- The series is used when v < 400 (|u| < 20), with the geometric tail bound 2|c_M|V^M. It applies once
  3V/((2M+2)(2M+3)) ≤ ½, since |c_{m+1}/c_m| ≤ 3/((2m+2)(2m+3)) for all five kernels; M is then extended until the tail
  is below 2⁻¹⁰⁰.
- The closed form in u = √v is used when v ≥ 400. A v-ball that straddles 400 is split into the two parts.
- The first draft switched to closed forms at v = 1, and the 60-digit tests caught it: (u − sin u)/u³ on a u-ball near 1
  loses most of its width to dependency. Horner on the v-ball does not.
- Powers are repeated products, never `**`.
- A non-finite intermediate gives a ±∞ range.

**`u2_outer_arb`: the re-search, the primary route.** It runs dyadic cells over the regime's domain:
- R: K(24), with Lemma G2's iso/full tests against B(P) and B_full from the checker's own `localisation_bounds`.
- O: the `lemma_W` domain, with one b₁-period for every ε in the slab, asserted in Arb.

A cell passes by either of two tests:
- **(direct):** `Objective.lower_bound`, with ε as a ball and A as a ball, exceeds the branch's ε-ball upper bound, taken
  as the max over the two A endpoints (convexity in A).
- **(taylor):** at the point ε_c,
  D(ε_c) − h_ε·|∂_ε D(ε_c)| − ½h_ε²·(sup|∂²_ε L\*(θ)| + sup|∂²_ε L\*(θ_c)|) > target.
  - ∂_ε L\* = A·mean((σ(z) − y)φ_ε), enclosed with b\* bracketed.
  - The bound |∂²_ε L\*| ≤ ¼A²·mean((φ_ε − c)²) + A·mean|φ_εε| holds for any constant c.

If both tests fail, the cell is split in θ. The ε-interval can be split n_eps ways. A witness search reports 'fail' when a
point θ outside the box has an Arb upper bound below the branch's Arb lower bound.

**The other parts.**
- **`u2_ring_arb`:** gradient balls over box × A-ball × ε-ball, with b\* bracketed; boxes split up to a depth cap.
- **`u2_pd_arb` and `u2_pd_box_arb`:** Sylvester's criterion on the (p, q, b) Hessian enclosure.
- **`u2_switch_arb`:** replays the producer's published Z box and branch boxes with `krawczyk_arb` (existing,
  unchanged), with ε as a ball.
  - The active pair is checked without β. It needs φ_σσ > 0 on the inner window (convex, so the max is at an end),
    φ_σσ < 0 on the negative outer window, φ_σ > 0 on the positive outer window, and the active ends strictly
    dominating the others.
  - The gap signs are checked left and right of Z, every inside box must lie within Z's θ-range, and the A-tube must be
    tiled exactly.
- **`u2_slabs_closed_ok`:** checks that the slabs are closed and share their endpoints exactly (F2).

**Where it departs from §5.**
- **Switch:** §5 planned to check the active pair through β = arccos(1/(1+E)). The convexity criterion replaces that, so
  no series for β/√ε is needed at ε = 0.
- **Outer exclusion:** the Taylor-in-ε test is added because a pure ε-ball evaluation loses the cancellation in
  mean((σ(z) − y)·∂_εφ).

## 6. Status of the author's items (2026-10-16)

1. **Cap the 3 GB stage.**
   - Ring/PD box evaluation is now chunked, at most 256 boxes per evaluation, with box-count caps. The stage-by-stage RSS
     command is written, and the 3 GB watchdog is kept.
   - **Not yet measured:** the disk hold blocks it (free disk was 16.7 GB at the last gate, against the 20 GB minimum).
2. **Launcher gate.**
   - `src/u2_gate.py`: a separate `python -m src.u2_gate gate "<label>" [--disk-gb 25]` step writes a JSON entry with pid and
     time.
   - `src.u2_sample`'s `__main__` calls `require_gate()` first. It refuses (exit 3) unless the newest entry is:
     - OK;
     - written by a different process before this process started;
     - younger than 10 min;
     - newer than the last recorded launch.
   - Tests are in `tests/test_u2_gate.py`. **Not yet run** (gate FAIL on disk).
3. **Time cap and splitting.**
   - Done in code: `certify_split`, and a cap checked inside each round.
   - Constructed-case tests are written, including a real too-wide unit that must stop within its cap. **Not yet run.**
   - The failing A-sub-intervals at ε-slab width 0.1 (`python -m src.u2_sample ascan 0.1 0.2 O 40 30 ascan_w0.1`: no export,
     about 10 KB of output) **are deferred until the disk hold lifts**.
4. **Kernel bounds:** the checker route was chosen (F6, §5). The specification is written; the checker itself is not.
5. **Ring margin against the rounding allowance:**
   - The ring test subtracts one absolute allowance of 10⁻⁹ per box (`SAFETY`) from |g_k| − Σ M_ka h_a.
   - The recorded margins are net of it: min over boxes 2.78·10⁻⁶ (k = 0), 1.17·10⁻⁵ (k = 20), 1.84·10⁻⁶ (k = 39).
   - So the ratio of margin to allowance, minimised over every box of the three units, is **1.84·10⁻⁶ / 10⁻⁹ ≈ 1,840**.
   - The allowance covers float rounding in the centre gradient. That is about 800 terms of order ≤ 10, so ≤ 10⁻¹²
     relative to the terms, plus the b\* residual ≤ 10⁻¹⁴.
   - The Lipschitz sums are upper bounds computed in floats, with no separate allowance.
   - The thin margin, about 2·10⁻⁶ against gradients of about 10⁻², means the Arb checker's ring boxes will need their own
     subdivision.
6. **This page.**
