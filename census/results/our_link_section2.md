# §2: our link construction, described from our own code and verified

**Written to replace any statement that we follow "the published
parametrisation."** We do not, and as of 2026-09-22 we know their released code
does not implement its own appendix description either (that finding is in
`notes/renlim_generator_private_note.md`, **private, not for the paper**).

This note describes **our** construction, from `src/data.py::linked_tori`, and
verifies it is a genuine link. Artifact: `results/our_link_verification.csv`,
producer `src/verify_our_link.py`.

---

## The construction, as implemented

Two solid tori in ℝ³, `R = 1` (`major_radius`), tube radius `ρ = 0.2`
(`tube_radius`, the value used throughout):

| | core circle | frame (normal, binormal) |
|---|---|---|
| class 0 | `R·(cos t, sin t, 0)` | `((cos t, sin t, 0), (0, 0, 1))` |
| class 1 | `R·(1 + cos t, 0, sin t)` | `((cos t, 0, sin t), (0, 1, 0))` |

Points are `core(t) + ρ√U·(cos φ · normal + sin φ · binormal)` with
`t, φ ~ U(0, 2π)` and `U ~ U(0,1)`. The `√U` makes the sample **uniform in
cross-sectional area** rather than concentrated near the core.

**There is no oscillation term.** Our tori are smooth.

## Verification

| check | measured | verdict |
|---|---:|---|
| **Gauss linking integral** of the two cores, `Lk = (1/4π)∮∮ (a−b)·(da×db)/\|a−b\|³` | **−1.000000** | **linked**, `\|Lk\| = 1` |
| core self-minimum distance (non-adjacent points) | **0.314420** | **no self-intersection** — each core is a round circle, so this is exact by construction |
| core-to-core minimum distance | **1.000000** (`= R` exactly) | — |
| reach of each core circle | **1.000000** | — |
| tube radius `ρ` | **0.200000** | — |
| tubes disjoint: `2ρ < core separation` | `0.4 < 1.0` | **yes** |
| tubes embedded: `ρ < reach` | `0.2 < 1.0` | **yes** |

`data.py` enforces the disjointness condition in code:
`0 < tube_radius < major_radius/2`, i.e. exactly `2ρ < R`, raising otherwise.

**So the thickened solid tori are disjoint, each is embedded, and their cores
form a Hopf link with linking number −1.**

## What §2 should say, and what it should not

**Say**: we train on two linked solid tori in ℝ³ with `R = 1`, `ρ = 0.2`, sampled
uniformly in cross-sectional area; the cores have linking number −1, each core is
embedded, and `2ρ < R` guarantees the solids are disjoint. All three are verified
numerically (`src/verify_our_link.py`).

**Do not say**: that we follow Ren & Lim's published parametrisation, or that our
construction is theirs. It is not. Ours has **no oscillation term**, and we
established in `corrugation_results.md` that adding one (under either reading of
Appendix G.1) does not change any conclusion in that study — but the smooth
construction is what every other result in this paper was measured on, and §2
should describe it plainly as ours.

**Existing files carrying the superseded phrasing.** `corrugation_prediction.md`
and `corrugation_readings_prediction.md` are **registrations and are append-only**
— their original wording stands, including the sentence "it is part of their
published parametrization". `corrugation_results.md` has the annotation below
appended rather than edited in place, for the same reason.
