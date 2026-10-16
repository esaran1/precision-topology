# Ramp-limit crossover on the asymmetric windows (POST HOC, EXPLORATORY; 2026-10-16)

**Status.** A post hoc, exploratory check: a validated float search, not a certificate. No registered code, artifact,
number or verdict is changed. Producer: `src/ramp_crossover.py` → `results/ramp_crossover.json` (the `ledger` block);
tests: `tests/test_ramp_crossover.py`. Background: math note §22 (7da9bec), Lemmas K7 (the ramp limit) and K8 (the
mirror fold), and the exploratory float check of §22.6.

**Question (author, 2026-10-16).** Below which scale does the ramp-limit minimum beat the registered width-2 retained
loss on the asymmetric windows? Does every registered width-2 quantity lie above that scale?

**Answer.**
- **s_ramp ∈ [0.1576358, 0.1576364]** (relative width 4.4·10⁻⁶), at a = 1.30 on Track 2's registered landscape.
- **Every registered width-2 landscape quantity lies above s_ramp.** The closest is W2-A's hold scale
  s₀ = 0.2225397, which is 1.41 × s_ramp.
- What lies below s_ramp is training data only: the matched initial scales of every T2-3, T2-3b, T2-3c and T2-3d run,
  and some observed crossing scales (table 3). Neither is a landscape quantity, and no criterion compares them with the
  landscape below s_ramp.

## 1. Setting

- **Objective.** Every registered width-2 quantity on the asymmetric windows uses the same objective:
  - a = 1.30, Δ = 0.4, I = [−0.8, 0.8], O = [−2, −1.2] ∪ [1.2, 2.4];
  - the 800-point population `asym_pilot.population(0.4)` (P_an in §22);
  - L\*(θ; s) = min_b mean ℓ(sφ + b, y).
  - So only a = 1.30 is needed.
- **Other windows.**
  - The Δ = 0.8 pilot (`asym_pilot_design.md`) is exploratory, not registered.
  - The width-2 registrations on symmetric windows are out of scope: there the ramp limit ties but never beats
    (Theorem K11).
- **Ramp-limit minimum.**
  - R(s) = min over (κ, b, α, β) of mean ℓ(κx + s·a·sin(αx + β) + b, y). The total slope is c̄ = κ/s.
  - For each (α, β), the minimum over (κ, b) is a convex problem, solved by Newton.
  - **Two independent start sets:** A is the §22.6 grid, 150 × 72 with α ∈ (0, 10]; B is 3,000 seeded uniform draws
    with α ∈ (0, 12]. In each set the 8 lowest distinct starts are polished by 4-D damped Newton (gradient ≤ 1e−14).
  - **The two sets agree at every scale evaluated:** losses within 1e−12, and the same minimiser in α, β and c̄ to
    1e−6.
- **Mirror range (Lemma K8).** A ramp-limit point is realised by a compact width-2 configuration iff |c̄| ≤ α. Inside
  that range it ties with the width-2 minimum; outside it, it can lie below.
- **Crossover.** s_ramp is located by 16 steps of bisection in log s on the sign of α − |c̄|. The starting bracket is
  the committed scan points 0.13335 (outside the mirror range) and 0.17783 (inside).
- **The registered retained loss.**
  - At registered scales: the committed values (`asym_parts/scan.csv`, `bisect.csv`, `validate.csv`).
  - At five new scales: recomputed POST HOC by the unmodified registered code path (`asym_register.eval_scale`: 1,000
    restarts, cap 3,000, 5 polished), with seeds 910–914. The five scales are four near s_ramp, plus s₀.

## 2. The crossover

**Table 1. Retained loss against the ramp-limit minimum. The gap is retained − R(s); a gap > 0 means the ramp limit is
lower.**

| s | source | gap | α | \|c̄\| | α − \|c̄\| |
|---|---|---|---|---|---|
| 0.1 | committed scan | **1.144·10⁻³** | 1.68633 | 2.7147 | −1.028 |
| 0.13335 | committed scan | **3.352·10⁻⁴** | 1.68631 | 2.0108 | −0.324 |
| 0.14975 (0.95·s_ramp,lo) | registered path, post hoc | **3.487·10⁻⁵** | 1.68630 | 1.7800 | −0.0937 |
| 0.15606 (0.99·s_ramp,lo) | registered path, post hoc | **1.388·10⁻⁶** | 1.68629 | 1.7043 | −0.0180 |
| **0.15764 = s_ramp** | bisection | 0 at the root | 1.686294 | 1.686294 | 0 |
| 0.15921 (1.01·s_ramp,hi) | registered path, post hoc | 0 | 1.68629 | 1.6687 | +0.0176 |
| 0.16552 (1.05·s_ramp,hi) | registered path, post hoc | 1.1·10⁻¹⁶ | 1.68629 | 1.6015 | +0.0848 |
| 0.17783 to 7.4989 (13 scan points), the bisection points, and both validated bracket ends | committed | \|gap\| ≤ 1.1·10⁻¹⁶ | 1.6863 to 1.7405 | 1.4843 to 0.0642 | ≥ +0.202 |
| 10 | committed scan | **−2.099·10⁻⁴** (the retained loss is lower) | 1.7562 | 0.0659 | +1.690 |

**Reading.**
- **Below s_ramp.** The ramp-limit minimiser leaves the mirror range and lies below the registered retained loss.
  The gap grows from 0 at s_ramp; it is 1.4·10⁻⁶ at 0.99·s_ramp.
  - The registered search's points there do not attain the infimum. At 0.99·s_ramp and 0.95·s_ramp the retained
    point is effectively width 1: one unit carries the whole weight (|α| = 1.687, 1.692), the other has weight below
    3·10⁻¹⁵, and the slope sits on the mirror boundary |c̄| = |α|. At s = 0.1 the committed point is a partial escape:
    α₁ = 174.9 with weight 0.0245.
  - Every such point is unplaced, as §22.6 found.
- **Above s_ramp.**
  - The ramp-limit minimiser is inside the mirror range and ties with the registered retained loss.
  - **The retained loss never beats the ramp limit between s_ramp and 7.5.** The gap is 0 at 1.01·s_ramp, where the
    mirror margin is 0.0176.
  - At s = 10 the registered retained loss is 2.1·10⁻⁴ **below** the ramp-limit minimum. There a non-ramp width-2
    configuration is better: the opposite direction to the question.
- **Mirror margin at the registered scales:** 0.202 at 0.178, 0.518 at s₀, 1.128 at s_lo, and 1.149 at s_hi.

## 3. Registered quantities against s_ramp

Every scale below is from a committed file. The margin is s/s_ramp,hi − 1.

**Table 2. Registered width-2 landscape quantities (a = 1.30, Δ = 0.4).**

| quantity | s | margin | above? | source (commit) |
|---|---|---|---|---|
| T2-1 bracket s_lo (verified) | 0.437144 | +177% | yes | `asym_scores.json` (74d1dc1); registration `asym_registration.md` (53dce22) |
| T2-1 bracket s_hi (verified) | 0.453158 | +187% | yes | same |
| T2-1 bisection step | 0.486968 | +209% | yes | `asym_parts/bisect.csv` |
| T2-1 scan points, both validation ends | 0.1778 and up | ≥ +12.8% | yes | `asym_parts/scan.csv`, `validate.csv` |
| T2-1 scan points 0.1 and 0.1334 | 0.1, 0.1334 | −37%, −15% | **no** | `scan.csv`; scored "unplaced" (see below) |
| T2-2 comparator (pilot s_lo) | 0.562341 | +257% | yes | `asym_register.PILOT_LO` |
| s_glob = (s_lo + s_hi)/2, which sets k = 0.04209 (T2-3/3b/3c/3d) and φ₂ (3b, 3c) | 0.445151 | +182% | yes | `asym_frozen.json` (9b1eb4e), `asym_t23b_frozen.json` (61f3353), `asym_t23c_frozen.json` (9ba85f5) |
| s_pop2 = √(s_lo·s_hi): T2-3d's criterion (ii) and t2g's comparator 0.44508 | 0.445079 | +182% | yes | `width2_asym/landscape.json` |
| **W2-A hold scale s₀ = 0.5·s_pop2**: also Phase 1C's W2-A arms and Phase 3 | **0.222540** | **+41%** | yes | `width2_asym/landscape.json` (b3f483d); W2-A registration 28b2432; Phase 1C 6433edc; Phase 3 7a9d791 |
| W2-A population switch D | 5.079637 | ×32 | yes | `width2_asym/landscape.json` |
| W2-A population switch D′ | 4.970592 | ×31 | yes | same |
| W2-A population switch T | 0.481586 | +206% | yes | same |
| W2-A population switch T′ | 0.327167 | +108% | yes | same |
| Phase 3 (width 4) population switch Q | 0.482210 | +206% | yes | `phase3/landscape.json` (747d61e); registration 7a9d791 |
| Phase 3 switch S | 0.439780 | +179% | yes | same |
| Phase 3 switch T4 (W2-A's T embedded) | 0.761909 | +383% | yes | same |
| T2-3d's frozen predictor (width-1 own thresholds, 80 seeds) | 4.29 to 6.61 | ≥ ×27 | yes | `asym_t23d_own_frozen.csv` (68323b2) |

**On the table.**
- **The two scan points below s_ramp.** T2-1 reported the retained point at 0.1 and 0.1334 as unplaced. The true
  infimum there is a ramp limit, and it is also unplaced (|c̄| > a/1.4; K13). So T2-1's status at every scale, its
  single switch, and its PASS are unchanged.
- **At s₀.**
  - The registered path gives retained = R(s₀) = 0.574415 exactly (gap 0; mirror margin 0.518).
  - W2-A's branches are fixed-v branch points, not conditional minimisers: they hold the shares fixed, so they cannot
    reach a ramp limit (v₂ → 0). Their losses at s₀ (T 0.574688, D 0.576639, T′ 0.625867, D′ 0.639742) all lie above
    R(s₀). This is as it must be, and it changes nothing.
  - Phase 3 uses s₀ and width-4 branch points in the same way.

**Table 3. Below s_ramp: training observations, not landscape quantities. Flagged as asked.**

| arm | matched initial ‖v‖₁ (all runs) | runs below s_ramp at init | observed s_cross below s_ramp |
|---|---|---|---|
| T2-3 | 0.0063–0.0769, median 0.0412 | 80/80 | 2 of 79 crossing runs (min 0.0215) |
| T2-3b | 0.0050–0.0800, median 0.0425 | 80/80 | 27 of 71 (min 0.0184) |
| T2-3c | 0.0057–0.0776, median 0.0400 | 80/80 | 27 of 80 (min 0.0218) |
| T2-3d | 0.0011–0.0806, median 0.0415 | 80/80 | 21 of 79 (min 0.0164) |

- **What these are.**
  - Every arm starts below s_ramp by design: k = 0.04209 places the initial scale at r₁ = 0.0926 of s_glob.
  - The early crossers (Track 2B's "initial relaxation" crossings) also cross below s_ramp.
- **What this would affect, and what it does not.**
  - None of the four criteria compares a run with the landscape at its starting or crossing scale. T2-3, T2-3b and
    T2-3c compare s_cross with the bracket (s_lo, s_hi). T2-3d compares it with width-1 own thresholds and with
    s_pop2. So no score or verdict moves.
  - The finding bears on **interpretation** only. A sentence that says "the run starts near the width-2 conditional
    minimiser at its initial scale" is not correct there. Below s_ramp the landscape's infimum is a ramp limit, which
    is not attained by any finite point (argued; K12 is proved only on continuous windows).
  - Every point there is unplaced, so "starts unplaced, below the switch" still holds.
- **t2g.** Its EARLY class predicts s₀ = ‖v₀‖₁, which is below s_ramp. But t2g's gate failed (`t2g/gate.json`,
  09b84ad), so nothing was registered from it.

## 4. Limits

- **Float search, not a certificate.**
  - "Every registered quantity is above s_ramp" holds with margin 41% or more, far beyond the bisection width (4.4e−6)
    and the search's agreement tolerance.
  - The location of s_ramp itself depends on the ramp-limit search, which is restricted to α ≤ 12. On a finite point
    set, larger α can alias (§22.3), and this was not searched.
- **The retained loss between scan points** comes from the registered code path re-run post hoc at four scales near
  s_ramp and at s₀. It is not a committed registered value.
- **Run.** One process, nice 15, one thread. Memory gate as its own logged step before each job
  (`ramp_crossover_memory_gate.log`). About 24 minutes; peak RSS 0.37 GB.
