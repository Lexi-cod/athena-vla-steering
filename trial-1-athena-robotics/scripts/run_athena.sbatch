#!/bin/bash
#SBATCH --job-name=athena_steer
#SBATCH --partition=gpu
#SBATCH --account=biyik_1165
#SBATCH --gpus-per-task=v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch1/nalagand/athena_robotics/logs/athena_%j.out
#SBATCH --error=/scratch1/nalagand/athena_robotics/logs/athena_%j.err

# Runs one or more presets against a SINGLE policy server (server startup is
# slow, so amortise it across presets).
#
#   sbatch scripts/run_athena.sbatch baseline adaptive
#   PRESET_ARGS="--task-subset pilot --num-trials-per-task 5" \
#       sbatch scripts/run_athena.sbatch baseline adaptive
#
# Every run is resumable: episodes.jsonl is append-only and already-recorded
# (task_id, episode_idx) pairs are skipped, so re-submitting continues where a
# preempted job stopped.

set -euo pipefail

OPENPI_DIR=/scratch1/nalagand/openpi
ATHENA_DIR=/scratch1/nalagand/athena_robotics
PORT=8000
PRESETS=("$@")
if [ ${#PRESETS[@]} -eq 0 ]; then
    PRESETS=(baseline adaptive)
fi
EXTRA_ARGS=${PRESET_ARGS:-}

mkdir -p "$ATHENA_DIR/logs" "$ATHENA_DIR/results"

# NOTE: `${VAR:-}` everywhere -- with `set -u`, a bare $PYTHONPATH aborts the
# script when the variable is unset. That is what killed job 9358798.
export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_DIR/third_party/libero:$ATHENA_DIR"
export MUJOCO_GL=egl
# Keep tmp off the login-node/shared /tmp.
export TMPDIR="${TMPDIR:-/scratch1/$USER/tmp}"
mkdir -p "$TMPDIR"

echo "=========================================="
echo "job      : ${SLURM_JOB_ID:-interactive}"
echo "host     : $(hostname)"
echo "presets  : ${PRESETS[*]}"
echo "extra    : ${EXTRA_ARGS}"
echo "started  : $(date)"
echo "=========================================="
nvidia-smi || true

cd "$OPENPI_DIR"

# ---------------------------------------------------------------------------
# 1. Policy server
# ---------------------------------------------------------------------------
SERVER_LOG="$ATHENA_DIR/logs/server_${SLURM_JOB_ID:-local}.log"
echo "starting policy server -> $SERVER_LOG"
uv run scripts/serve_policy.py --env LIBERO > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "cleanup: killing policy server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for server on port $PORT..."
for i in $(seq 1 360); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server exited early. Tail of $SERVER_LOG:"
        tail -40 "$SERVER_LOG" || true
        exit 1
    fi
    if (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
        echo "server up after $((i*5))s"
        break
    fi
    sleep 5
    if [ "$i" -eq 360 ]; then
        echo "ERROR: server did not come up within 30 min. Tail of $SERVER_LOG:"
        tail -40 "$SERVER_LOG" || true
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 2. Run each preset in the LIBERO venv
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$OPENPI_DIR/examples/libero/.venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_DIR/third_party/libero:$ATHENA_DIR"

cd "$ATHENA_DIR"
STATUS=0
for preset in "${PRESETS[@]}"; do
    echo ""
    echo "===== preset: $preset ($(date)) ====="
    # shellcheck disable=SC2086
    if python scripts/run_experiment.py --preset "$preset" --port "$PORT" $EXTRA_ARGS; then
        echo "===== preset $preset OK ====="
    else
        echo "===== preset $preset FAILED (continuing) ====="
        STATUS=1
    fi
done

echo ""
echo "===== summary ====="
python scripts/analyze.py || true

echo "finished at $(date) with status $STATUS"
exit $STATUS
