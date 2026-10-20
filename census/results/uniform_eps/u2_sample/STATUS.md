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

## Update (2026-10-18, about 21:40): sampling and timing complete (NO full run)
- **Done:** the memory measurement, the item-3 diagnosis and implementation, the ε = 0.1 scan, all 40 A-sub-intervals on
  four slabs, the checker timings (including the thin-margin ring boxes), the widening sample and the projection. All are
  in U2_design.md §7.
- **Disk pauses:** the checker timing was paused twice by the watchdog (free disk < 15 GB). Each time it was resumed after
  a separate passing gate. The timing jobs that were stopped on purpose (checker outer at 12,000 evaluations; the earlier
  uncapped runs) are recorded as such in §7.5.
- **Trustworthy unit results:** `units/probes6`, `units/swwidth1`–`swwidth6`, `units/diag1`, `units/ascan_w0.1`,
  `units/stages_*`, plus `checker_timing4.json` (O outer), `checker_timing5.json` (links) and `checker_thin.json`.
  - `swwidth1`–`swwidth5` used earlier versions of the switch code; their labels say which.
  - `checker_timing.json`, `checker_timing2.json` and `checker_timing3.json` are runs of the checker before it split ε per
    cell. They were superseded and are not used.
- **No process of mine is running.**

## Full producer run (author's instruction, 2026-10-18 22:50): run1
- **Disk thresholds, all set by the author:**
  - 19 GB at 2026-10-18 14:00 ("just continue");
  - 15 GB at 15:40 ("you can go to 15 gb");
  - kept at 15 GB at 22:50 ("keep 15").
- **Rule for this run:** a worker starts only after a separate gate step shows ≥ 15 GB free. The watchdog SIGSTOPs a
  worker when free disk falls below 15 GB (15,728,640 KB).
- **The earlier pauses at 15 GB came from the threshold itself, not from checks that were too rare.** The watchdog polls
  RSS and free disk **every 2 s**, so free disk can fall only by what is written in 2 s before the pause. The sample jobs
  write at most a few MB, so this interval is short enough.
- **Specs:** `run1_specs.json`, 27 closed ε-slabs sharing their endpoints, covering [0, 0.6]:
  - R: [0, 0.03] in steps of 0.005, then [0.03, 0.04] and [0.04, 0.0462];
  - O: from 0.0462 in steps of 0.03, the last slab [0.5862, 0.6].
  - Each slab: 40 A-sub-intervals (outer + ring + PD, with exports) and the sheared switch.
- **Workers:** 3, round-robin over the slabs (worker w runs slabs i ≡ w mod 3). Each worker launch has its own gate
  entry. Units are in `units/run1/`, summaries in `run1_w{0,1,2}.jsonl`.
- **Every record and export is labelled "PRODUCER-ONLY: not independently checked".**
- **2026-10-18 23:07: the three workers were stopped (SIGTERM) and restarted with a new rule for the outer link.** A
  failing outer unit is now retried on closed ε-sub-slabs, halving down to 1/16 of the slab.
  - Reason: on slab [0.005, 0.01] the outer exclusion failed on every A-sub-interval tried so far, at the large-|σ| band
    p ≈ 18.75, q ≈ −40.3.
  - The two unit results stored as failures before this rule are kept, renamed to `*.superseded`. The certified units
    were kept and are skipped on restart.
- **2026-10-19 03:25: checker timing processes chkA, chkB and chkC stopped on purpose (SIGTERM).** The projection had already
  exceeded the author's 48 h limit at 3 workers.
  - Still running when stopped: run1_s06 k20 (≥ 2.9 h), run1_s25 k20 (≥ 2.9 h) and run1_s26 k39 (≥ 1.2 h). Their elapsed
    times are reported as lower bounds.
  - The completed units are in chkA.json, chkB.json and chkC.json.
- **2026-10-19 04:45: run1 complete. Every unit and every slab passed the producer's tests (PRODUCER-ONLY: not
  independently checked).**
  - The checker pieces (a), (b) and (c) are written and tested.
  - Checker timings are in `chkA.json`, `chkB.json`, `chkC.json`, `chkPD.json`, and `chkSW*.json` (switch replay
    diagnoses).
  - **The checker re-projection is above 48 h at 3 workers, so the checker was NOT run** over the producer output.
  - No process of mine is running.

## 2026-10-19: the author's four steps (option (a), deadline 2026-10-30)
- **The 1,080-unit run1 result is still PRODUCER ONLY, NOT INDEPENDENTLY CHECKED.**
- **Step 1 (soundness):** 570 points at 60 digits. No producer problem was found. The mismatches were a checker
  piece-classification bug, now fixed.
- **Step 2 (width):** the checker's M_comp blew up through Horner on wide v-balls (math note §23.5). Fixed by the series
  threshold 16. A NaN-dropping power and union bug in the checker kernels was also fixed. **Every checker result from
  before the fix is VOID** and has been redone.
- **Step 3 (replay):** implemented, treating leaves as geometry only, with coverage checked by the checker. It measured
  slower than re-search, so re-search is primary. 13 units (s01 k000–k012) have no valid leaf export, about 1.3 MB to
  regenerate.
- **Step 4 (timing):** six units and six switch sub-slabs all pass in the checker after the fixes.
  - The projection is ≈ 290 core-h, ≈ 96 h wall at 3 workers, above 48 h.
  - **STOP, for a cluster allocation.**

## 2026-10-19 (evening): NaN audit, regenerated leaves, cluster package
- **NaN audit:** no earlier certificate check could have executed the affected code (see `certificate_audit.md`). No
  verdict changes; no re-run was needed.
- **Leaves:** 26 per-sub-slab leaf files were regenerated for s01 k000–k012. The stale first-attempt files are renamed
  `*.stale`.
- **Cluster package:** `results/uniform_eps/u2_cluster/` (MANIFEST sha256
  3a7a2f778b8d3bf420c84e88da0bf8f39e73651aece75d527f762ff9fa6ab33d).
  - The 5 local reference units (s000_k020, s006_k020, s013_k020, s025_k020, s026_k039) all passed in the checker, in
    642–1,374 s each at ≤ 0.045 GB.
  - The 1,080-unit result stays **PRODUCER ONLY, NOT INDEPENDENTLY CHECKED** until the cluster run and its verify-back
    pass.
