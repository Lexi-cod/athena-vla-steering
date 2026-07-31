# ATHENA-style test-time steering for π0.5 on LIBERO-90

Inference-time **predict → verify → re-steer** for vision-language-action
manipulation. Adapts ATHENA (test-time steering for count fidelity in
text-to-image diffusion) to robotic manipulation, per
`/scratch1/nalagand/Executive Summary (4).pdf`.

**No model weights are modified.** All intervention happens at inference time.

> **Working on this? Read [`PROGRESS.md`](PROGRESS.md) first** — it holds the
> stage-by-stage status, resume commands, design rationale and open questions.

## The idea

ATHENA decodes an intermediate diffusion latent, counts objects, builds a
*control prompt* with the corrected count, and mixes denoiser outputs — early,
before image structure becomes hard to revise. The robotic analogue:

| ATHENA | Here |
|--------|------|
| intermediate latent decode | simulator/sensor observation mid-episode |
| object-count estimator | object detector + grasp check |
| control prompt | rewritten language instruction naming the right object |
| denoiser mixing (ε_orig, ε_ctrl) | two action chunks, selected or blended |
| adaptive mixing weight | escalating correction strength |
| correct early, before structure sets | correct before the gripper commits |

## Quick start

```bash
source /scratch1/nalagand/openpi/examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:/scratch1/nalagand/openpi/third_party/libero:/scratch1/nalagand/athena_robotics"
export MUJOCO_GL=egl

python tests/test_perception_live.py                    # validate the stack
python scripts/run_experiment.py --preset adaptive --task-subset pilot --dry-run
sbatch scripts/run_athena.sbatch baseline adaptive      # needs a GPU
python scripts/analyze.py --confusable-only
```

## Variants

| Preset | Behaviour |
|--------|-----------|
| `baseline` | vanilla π0.5, no verification |
| `static` | one correction, never revised |
| `adaptive` | escalating corrections, strength adapted to whether the last helped |
| `dual_select` | two policy passes, pick the chunk aimed at the target |
| `dual_blend` | two policy passes, convex-combine the chunks |

Plus ablations for vision quality, early-vs-late intervention, retry budget and
threshold strictness. See `athena/config.py`.

## Reporting results

**Always report the confusable subset separately.** Only 32 of 90 LIBERO-90
tasks contain distractors the policy can plausibly confuse; on the other 58
there is no wrong-object error to catch, so aggregate numbers understate the
effect. `python scripts/analyze.py --confusable-only`.
