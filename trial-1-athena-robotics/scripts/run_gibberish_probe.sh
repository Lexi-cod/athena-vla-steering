#!/usr/bin/env bash
# Gibberish-prompt probe: does the policy run the same routine when the
# instruction carries no meaning?
#
# The nonsense string is pronounceable, contains no real content words and no
# object nouns, and is close in token count to a genuine LIBERO instruction --
# so it isolates *semantics* rather than confounding "no meaning" with
# "no tokens" the way an empty prompt would. The same string is used for every
# task, so any difference between runs is attributable to the scene.
set -uo pipefail

GIB="blicket dax fep wug zorp tulver nace"
ROOT=/scratch1/nalagand/athena_robotics
cd "$ROOT"

source /scratch1/nalagand/openpi/examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:/scratch1/nalagand/openpi/third_party/libero:$ROOT"
export MUJOCO_GL=egl

echo "=============== 1/3  mugs, GIBBERISH (libero_90 tasks 67,68) ==============="
python -u scripts/run_characterization.py \
    --suite libero_90 --tasks 67 68 --num-trials 5 \
    --run-id gibberish_mugs --port 8000 --save-video \
    --prompt-override "$GIB"

echo "=============== 2/3  orange juice, LANGUAGE control (libero_object 9) ======"
python -u scripts/run_characterization.py \
    --suite libero_object --tasks 9 --num-trials 5 \
    --run-id language_oj --port 8000 --save-video

echo "=============== 3/3  orange juice, GIBBERISH (libero_object 9) ============="
python -u scripts/run_characterization.py \
    --suite libero_object --tasks 9 --num-trials 5 \
    --run-id gibberish_oj --port 8000 --save-video \
    --prompt-override "$GIB"

echo "=============== ALL PROBE RUNS COMPLETE ==============="
