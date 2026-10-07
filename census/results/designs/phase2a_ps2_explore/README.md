# 2A-PS2 exploration: EXPLORATORY, NOT REGISTERED

These files produce every exploratory number in `../phase2a_ps2_design.md`.

**Seeds.** Only exploration seeds 2,930,112–2,930,199 were used. They lie inside the 2A-PS exploration range
2,930,000–2,930,199, in the part never drawn by `p2a_explore_g/h/i`. No 2A-PS registered or pilot seed (2,975,xxx,
2,976,xxx) and no candidate 2A-PS2 seed (2,987,xxx, 2,988,xxx) was drawn, and no pilot was run.

**Machine rules.**
- One process, nice 15, one thread.
- Before each job, a separate logged memory gate (`gate.sh` → `memory_gate.log`): free ≥ 25%, swap free ≥ 500 MB,
  disk ≥ 20 GB.
- Stop above 1 GB RSS. Peak was 0.277 GB.
- Compute was about 23 min for `p2_explore` plus about 1 min for the rest (2026-10-07).

**Files**
- **`p2_explore.py`, `.jsonl`, `.log`.** All 88 seeds. It uses the REGISTERED 2A-PS freeze (`freeze_raw`, `evaluate`)
  and the POST HOC helpers (`seed_M`, `seed_branch`, `wider_class`, `quasi_static_rho_time`) unchanged. Per seed:
  - (a) the activity test at the release (|v_k| ≥ 1e−8·s), against the registered v ≠ 0 rule;
  - (b) ρ·t\* = ∫ds/(3·(−dL_M/ds)) from s₀ to 0.9999·s_F on the seed's own M, and a calibration run at ρ = 2⁻¹²
    (NOT a registered rate). The calibration run uses the scale-only rule from the exact release and checks ρ₂ every
    64 steps, so its crossing time is approximate. It is capped at ⌈256/ρ⌉ steps;
  - (c) the class under the registered rule, the posthoc wider search (360 minima, `default_rng([seed, 20261005])`),
    and S's ρ₂ profile on 1.00–1.25·s_F.
  
  Parts (b) and (c) were run on scoreable seeds only.
- **`p2_summary.py`, `.log`.** Arithmetic only, over `p2_explore.jsonl` and the committed 2A-PS `observed.jsonl`. It
  gives the end-of-observation step divided by t_F per class (seeds 2,975,052 and 127, the dead-unit seeds, are
  excluded), and binomial gate and power numbers.
- **`p2_seedscan.py`, `.json`, `.log`.** Candidate ranges: registered 2,987,000–2,987,599 and pilot
  2,988,000–2,988,059. Three checks, all clean:
  - 2A-PS's `scan_tree` regex over src, tests, results, paper, notes, independent, data and dist: no hit;
  - every integer `range(...)` literal in src and tests (663) for overlap: none. The constants not written as range
    literals were checked by eye (all < 10⁷ and far from 2,987,000; `blockA5d_k1`'s offset is 3,000,000 + k);
  - every `*seed*` column of the 72 parquet seed columns: no hit.

**Headline numbers (exploratory)**
- Status of the 88 seeds: untraceable 34, no s\* 13, ineligible 5, scoreable 36 (registered class: clean 16, none 13,
  mixed 7).
- (a) The activity test drops a unit on 3 of 54 releases: 2,930,163 and 167 (clean, scoreable) and 168 (no s\*). Their
  shares are 2e−89 to 2e−83. The smallest live share is 0.129.
- (b) t_F/t\* at 2⁻¹² is 1.0024–1.0114 (n 34). ρ·t\* is 10.9–90.6 (median 23.8). End of observation divided by t\*:
  clean ≤ 1.44, none ≤ 1.92, mixed ≤ 1.71. In 2A-PS (2⁻¹⁴), end divided by t_F: clean ≤ 1.148, none ≤ 1.943 (6 of
  27 above 1.5), mixed ≤ 1.355.
- (c) Wider search: registered clean → clean 16/16, mixed → mixed 7/7, none → mixed 13/13, so there are 0 wide-none
  seeds. S's ρ₂ < q at 1.01·s_F on 0/36, and S reaches q inside [s_F, 1.25·s_F] on 36/36.
- Crossing at 2⁻¹²: registered none 1/13, mixed 7/7, clean 14/14; wide mixed 8/20.
