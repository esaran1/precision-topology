# U2 sample timing: status at the HOLD (2026-10-16, about 13:45)

This is a scratch timing sample. Nothing here is a certificate.

## Process state
- **pid 68620** (`src.u2_sample probes specs4.json probes4`) is **stopped (state T)**.
  - My own RSS watchdog stopped it first. `probes4.log` reads `WATCHDOG: SIGSTOP at rss=3727344KB`: it reached **3.73 GB**,
    above the 3 GB rule. The coordinator's SIGSTOP came after that.
  - It must not be resumed. The right next step is to kill it once the author agrees. Its results up to that point are already
    on disk.
- It stopped inside spec 11: ε ∈ [0.0352, 0.0452], regime R, the first of three A-sub-intervals (k = 0), with
  ring, PD and switch also requested. No line for spec 11 was written, so **spec 11 is incomplete and must be redone**.
  Specs 12–14 never started.
- My background waiter shells have been stopped. No monitor of mine is running.

## Which outputs can be trusted
Every complete JSONL line was written in one piece after its spec finished.
- `probes1.jsonl`: 9 lines, complete. **Superseded**: these ran before the centred ε-curvature fix. The outer results are still
  valid bounds, but the bounds were looser.
- `validate.json`: complete (4 cases). It used the pre-fix code.
- `probes2.jsonl`: 2 lines, complete. These are the pre-fix O-regime failures at Δε = 0.01.
  - Spec 2 ([0.0462, 0.0762], O) ran **17,013 s (4.7 h) on one unit, failed, and was killed**. It appears only in
    `probes2.log` and is a timing result in its own right.
- `probes3.jsonl`: 30 lines, complete. This is the post-fix width ladder.
- `probes4.jsonl`: lines 0–10, complete.
  - Line 10 is ε ∈ [4.1e-6, 3e-3], R: the outer, ring and PD parts finished for all 3 A-sub-intervals, with 3 exported
    npz files.
  - Its switch part raised an IndexError, a scalar-shape bug that has since been fixed in `src/u2_sample.py`.
    **The switch part has not been timed yet.**
- Exports: `u2s_R_4e-6_3e-3_k000/k020/k039.npz`, about 100 KB each, with a `.meta.json` each. The SHA-256 of each is in
  probes4 line 10. The total for `u2_sample/` is under 0.5 MB.

## Still to do (only when the author or coordinator says so)
1. Kill pid 68620.
2. Find out which stage pushed RSS past 3 GB in spec 11 (outer refinement, or ring/PD), and add a cap.
3. Time the switch Krawczyk (after the fix).
4. Run the checker per-leaf proxy, `scratchpad/checker_timing.py`.
5. Write `U2_design.md` and the projection.

## Update (2026-10-16, about 15:55), after the author's items 1–6
- **pid 68620** was killed by the coordinator. **Spec 11 is to be redone** with the capped ring and PD.
- **New code** (uncommitted, compiled, not yet run):
  - `src/u2_gate.py`: the gate is enforced in code;
  - `src/u2_sample.py`:
    - ring and PD evaluate at most 256 boxes at a time (`BOX_CHUNK`);
    - the time cap is checked inside each round;
    - new: `certify_split`, the `stages` command and the `ascan` command;
    - `__main__` refuses to start without a separate gate entry;
  - `tests/test_u2_gate.py`.
- **`../U2_design.md`** has been written.
- **Nothing has run since the hold.** At the last gate, at 15:49, free disk was 16.7 GB, so the gate FAILED.
  - Blocked: the measurement for item 1, the tests for items 2–3, and the scan for item 3 at ε-slab width 0.1.
  - From now on the launcher needs `python -m src.u2_gate gate "<label>" --disk-gb 25`, run as its own step first.

## Update (2026-10-16, about 19:30)
- **`U2_design.md` §0** now records the author's decisions on F1–F6, with the window margins and the domain-coverage
  argument.
  - The author's condition on F1 (UNPLACED below the window and PLACED above it, for all s > 0) needs a new part (iv). It is
    designed but neither sampled nor written: finite status B&B plus small-s and large-s lemmas.
- **Code** (uncommitted, not run):
  - the critical-point helper is now valid at ε = 0;
  - ring boxes are exported with their margins, for the checker timing on the thin-margin boxes;
  - `scratchpad/checker_timing.py` and `specs5.json` (all 40 A-sub-intervals on 4 slabs) are ready.
- **The disk wait gave up after 3 h.** Checks every 15 min from 16:40 to 19:25 read 16.2–19.0 GB, never ≥ 25 GB.
- **Nothing has run since the hold.**

## Update (2026-10-16, about 20:41)
**The author's items A and B are done, and restart step (1) passed. Step (2) is partly done and its job is stopped.**
- **Item A:** U2_design.md §0 F1 now reads as a windowed claim. The parts outside the window are OPEN, with §21.5 as the
  reason. The window's endpoints at ε = 0.6 are given in A and in s.
- **Item B:** `check_u2_unit` and its helper functions are written. They are additions only to `src/verify_certificates.py`
  (no existing function changed). Their tests are in `tests/test_verify_u2.py`. U2_design.md §5.1 describes them.
- **Step (1):** the gate passed at 20:29 (free disk 20.4 GB). The new tests then passed: `tests/test_u2_gate.py` plus
  `tests/test_verify_u2.py`, 28 passed.
  - Two fixes were found on the way. Both are from my own first drafts:
    - the series/closed-form switch point moved from v = 1 to v = 400, because of the dependency loss at moderate u;
    - a gate-test ordering bug.
- **Step (2):** the `stages` run on spec 11 is pid 61607.
  - The outer exclusion certified in 2.8 s. The ring failed at this slab width (Δε = 0.01) quickly: 1,536 boxes in 0.25 s,
    with no memory growth. PD passed (80 boxes).
  - **Peak RSS is 0.95 GB for the process, imports included, against 3.73 GB before the cap.** Current RSS was 0.46 GB.
  - **The disk watchdog SIGSTOPped it at free disk 19.38 GiB**, during the switch stage (ε-split), about 6.5 min in.
  - **Do not resume it without a new gate.** It can simply be killed: the outer, ring and PD figures are in
    `stages_spec11b.log`.
- **Steps (3) onward have not started.**

## Threshold change for this phase (the author's instruction, 2026-10-18 14:00)
- **Scope:** sampling, timing and projection only. This phase writes only a few MB. It does not apply to the full run.
- **New thresholds:** a job may start when the gate (its own step) shows free disk ≥ 19 GB (`--disk-gb 19`). The watchdog
  SIGSTOPs my own job when free disk falls below 18 GB.
- **Reason:** free disk has sat just under 20 GB (19.6 GB) for two days.
- Every gate label in this phase says "disk threshold 19 GB per author 2026-10-18".
- **State at the change:** item 0 is complete in code (unit-level resumability with `src/u2_store.py`, its tests, and the
  item-3 diagnosis code). Nothing has run since the last report.

## Threshold change (author, 2026-10-18 15:40)
- **New thresholds, for this phase only:** start when the gate shows free disk ≥ 15 GB (`--disk-gb 15`). The watchdog
  pauses my own job below 15 GB.
- **Reason:** the author lowered the threshold and said to proceed with the work. Free disk was about 17 GB.
- **State at the change:** suite part 1 had passed at 14:13 (1497 passed, 1 skipped, exit 0). The certificates test was
  waiting on the 19 GB gate.
