# Trial 2 - ATHENA steering implemented inside an openpi checkout

This is the second of the two implementation trials in this repository (see the
[top-level README](../README.md)). It was developed **inside a clone of
[openpi](https://github.com/Physical-Intelligence/openpi)** rather than as a
standalone package, so that the steering logic could reach inside π0's denoising
loop instead of only sitting between policy calls.

**Start with [`scripts/steering/PROGRESS.md`](scripts/steering/PROGRESS.md)** — it is
this trial's primary document: status, restart checklist, environment and setup notes,
what has been run, and what is next.

## Layout

```
openpi-diff/          patches against upstream openpi (see below)
scripts/
  serve_pi05_libero.sh          policy-server launcher
  steering/
    PROGRESS.md                 status / handoff document
    dual_instruction_denoise.py core: run one noise sample through two
                                instruction prefixes and compare/mix
    fidelity_signal.py          mid-rollout "is the gripper going to the right
                                object" signal from simulator state
    correction_instruction.py   builds the corrected instruction
    athena_tasks.py             task/object definitions
    serve_steered_policy.py|.sh steered policy server
    run_*.py                    rollout and batch drivers
    diagnose_*.py               determinism / recovery / render diagnostics
    test_*.py, verify_*.py      unit and live checks
    pipeline_healthcheck.py     10-stage pre-flight check of the whole stack
    results/                    batch-result CSVs + RESULTS.md
```

## Reconstructing the working tree

The third-party openpi/LIBERO source is deliberately **not** vendored here. To
reproduce this trial's environment:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 15a9616
git submodule update --init --recursive

# reapply this project's changes to upstream files
git apply /path/to/trial-2-openpi-steering/openpi-diff/pi0.py.patch
git apply /path/to/trial-2-openpi-steering/openpi-diff/pyproject.toml.patch

# then drop this trial's own files in place
cp -r /path/to/trial-2-openpi-steering/scripts/steering scripts/
cp /path/to/trial-2-openpi-steering/scripts/serve_pi05_libero.sh scripts/
```

### What the patches change

- **`openpi-diff/pi0.py.patch`** (+96 lines to `src/openpi/models/pi0.py`) — adds a
  `PrefixCache` NamedTuple and two methods, `compute_prefix_cache()` and
  `denoise_step()`, which split `sample_actions` into its prefix-embedding step and
  its per-step denoising so external code can drive denoising from an ordinary Python
  loop — e.g. running the same initial noise through two different instruction
  prefixes on the same image. `sample_actions` itself is left untouched, so normal
  serving behaviour and performance are unchanged.
- **`openpi-diff/pyproject.toml.patch`** (1 line) — adds a uv override that keeps
  `rerun-sdk` from being installed, so the environment resolves on this cluster.
  Upstream `uv.lock` is regenerated as a result and is not committed here.

## Results

Batch-result CSVs live in [`scripts/steering/results/`](scripts/steering/results/),
with a written summary in
[`results/athena_feedback_logs/RESULTS.md`](scripts/steering/results/athena_feedback_logs/RESULTS.md).
Per-step trace CSVs (`bowl_traces/`, `trace_rollout*.csv`) and raw rollout logs were
left on the cluster scratch filesystem and are excluded by `.gitignore`.
