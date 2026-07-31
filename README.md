# ATHENA-inspired test-time steering for π0.5 on LIBERO - two implementation trials

This repository contains **two separate implementation trials of the same research
project**: adapting ATHENA-style test-time steering (originally a method for count
fidelity in text-to-image diffusion) into an inference-time correction loop for the
π0.5 vision-language-action policy on LIBERO manipulation tasks. In both trials no
model weights are modified — every intervention happens at inference time.

The two trials are kept side by side, not merged. Trial 1 was written first; Trial 2
is an independent re-implementation written after a restart of the work, in a
different location and against a different integration point. Both produced real
results, and each carries its own progress/results documentation.

| | Trial 1 | Trial 2 |
|---|---|---|
| Folder | [`trial-1-athena-robotics/`](trial-1-athena-robotics/) | [`trial-2-openpi-steering/`](trial-2-openpi-steering/) |
| Original location | `/scratch1/nalagand/athena_robotics` | `/scratch2/nalagand/openpi` |
| Form | standalone `athena/` package + experiment scripts, driving a stock π0.5 policy server over the openpi client API | edits layered directly into an openpi checkout, plus a `scripts/steering/` driver suite |
| Where it intervenes | between policy calls — observe, verify, rewrite the instruction, re-query, select/blend action chunks | inside the model's denoising loop — a split prefix-cache / per-step denoise API added to `openpi.models.pi0` |
| Its own documentation | [`PROGRESS.md`](trial-1-athena-robotics/PROGRESS.md), [`README.md`](trial-1-athena-robotics/README.md) | [`scripts/steering/PROGRESS.md`](trial-2-openpi-steering/scripts/steering/PROGRESS.md), [`README.md`](trial-2-openpi-steering/README.md) |
| Its results | run artifacts (`results/`, `logs/`) were left on scratch and are not committed; findings are written up in `PROGRESS.md` | batch-result CSVs under [`scripts/steering/results/`](trial-2-openpi-steering/scripts/steering/results/) plus [`results/athena_feedback_logs/RESULTS.md`](trial-2-openpi-steering/scripts/steering/results/athena_feedback_logs/RESULTS.md); per-step traces and raw logs were left on scratch |

Read each trial's own progress document for its specific findings, status, and open
questions. This top-level README does not summarize or compare their results.

## What is and isn't in this repository

- **Trial 1** is committed in full (its `athena/` package, `scripts/`, `tests/`, and docs).
- **Trial 2** was developed inside a clone of
  [openpi](https://github.com/Physical-Intelligence/openpi) at commit `15a9616`. Only
  this project's own added and modified files are committed — the openpi and LIBERO
  third-party source trees are **not** vendored here. The two upstream files that were
  modified are committed as patches under
  [`trial-2-openpi-steering/openpi-diff/`](trial-2-openpi-steering/openpi-diff/); see
  that trial's README for how to reapply them.
- Run artifacts (videos, episode JSONL, per-step traces, logs, checkpoints) and
  virtualenvs are excluded by `.gitignore` and remain on the cluster scratch
  filesystems, which are not backed up.

## Environment

Both trials ran on the USC CARC cluster against a π0.5 policy server (openpi,
`pi05_libero` checkpoint) with LIBERO in simulation, under SLURM GPU allocations.
Each trial's documentation carries its own setup instructions — note that the two
trials point at **different** openpi checkouts and virtualenvs, and Trial 2 requires
its patched `pi0.py` to be applied before its steering scripts will run.
