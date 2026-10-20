# U2 cluster run: command sequence for the author (one 32-core Linux machine)

The inputs are run1's exports. They are **PRODUCER-ONLY: not independently checked**, and stay so labelled until the
verify-back step below says ALL PASSED.

**0. Before transfer (on the laptop).**
- The coordinator pushes the commit `<COMMIT>`, timestamps `MANIFEST.sha256` with OpenTimestamps, and makes the bundle:
```
cd census && tar czf u2_bundle.tgz -T results/uniform_eps/u2_cluster/BUNDLE_FILES.txt
```
- Copy `u2_bundle.tgz` to the VM.

**1. Clone at the commit, and unpack the bundle.**
```
git clone <REPO_URL> precision-topology && cd precision-topology && git checkout <COMMIT>
cd census && tar xzf /path/to/u2_bundle.tgz
```

**2. The pinned environment.** Python 3.11; the wheel hashes are recorded on the VM, because wheels are
platform-specific.
```
python3.11 -m venv .venv && . .venv/bin/activate
pip download --no-deps -d wheels numpy==2.4.6 python-flint==0.9.0
pip hash wheels/*.whl > results/uniform_eps/u2_cluster/vm_wheel_hashes.txt
pip install --no-index --find-links wheels numpy==2.4.6 python-flint==0.9.0
python -V; uname -a > results/uniform_eps/u2_cluster/vm_platform.txt
```

**3. Verify the manifest by hand.** The run script verifies it again and refuses on any mismatch.
```
grep -v '^#' results/uniform_eps/u2_cluster/MANIFEST.sha256 | sha256sum -c --quiet && echo MANIFEST OK
```

**4. The memory and disk gate, as its own step**, immediately before each launch (≤ 10 min before; one launch per gate
entry). Use `--swap-mb 0` if the VM has no swap.
```
python -m src.u2_gate gate "VM reference phase" --disk-gb 15 --swap-mb 0
```

**5. The 5-unit reference comparison, which must match the laptop's verdicts EXACTLY.**
- Exit 4 means STOP: report the output and do not continue.
```
nohup python -m src.u2_cluster_run --phase reference --workers 5 > ref.log 2>&1 &
tail -f ref.log        # ends with "reference verdicts match exactly"
```

**6. The full run.** Gate first, then 30 workers (2 cores left for the system).
```
python -m src.u2_gate gate "VM full phase" --disk-gb 15 --swap-mb 0
nohup python -m src.u2_cluster_run --phase full --workers 30 > full.log 2>&1 &
```
- Expected about 290 core-hours, about 10 h wall at 30 workers. Peak memory about 0.05 GB per worker.
- **To resume** after any stop: run a new gate step, then the same `--phase full` command. Finished jobs (same
  manifest) are skipped; each result file is written atomically, so a stop loses at most the running jobs.

**7. Bring the results back.**
```
tar czf u2_results.tgz results/uniform_eps/u2_cluster/results_full results/uniform_eps/u2_cluster/results_reference_linux \
    results/uniform_eps/u2_cluster/vm_wheel_hashes.txt results/uniform_eps/u2_cluster/vm_platform.txt full.log ref.log
```

**8. Local verify-back (on the laptop).**
- Unpack into `census/`, run a gate step, then the verify-back:
```
python -m src.u2_gate gate "verify-back" --disk-gb 15
python -m src.u2_cluster_verify
```
- It must print `"verdict": "ALL PASSED"`: every one of the 1,080 units and all 269 switch sub-slabs passed, against
  this manifest, with the reference match.
- Only then may the U2 result be relabelled as independently checked.
