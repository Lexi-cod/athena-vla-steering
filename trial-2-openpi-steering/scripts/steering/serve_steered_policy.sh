#!/usr/bin/env bash
# Launches the steering-aware pi05_libero policy server (serve_steered_policy.py) with the same
# safe LD_LIBRARY_PATH fix as scripts/serve_pi05_libero.sh -- see that script's comment for why this
# is needed (jaxlib-CUDA-shadowing segfault on first real GPU kernel launch, confirmed on this node).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CLEAN_LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/cuda-[0-9]' | paste -sd: -)

CHECKPOINT_DIR="${PI05_LIBERO_CHECKPOINT_DIR:-/scratch1/nalagand/openpi_cache/openpi-assets/checkpoints/pi05_libero}"
PORT="${STEERED_POLICY_PORT:-8001}"

echo "Launching steered pi05_libero policy server (checkpoint: $CHECKPOINT_DIR, port: $PORT)"
LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH" PI05_LIBERO_CHECKPOINT_DIR="$CHECKPOINT_DIR" STEERED_POLICY_PORT="$PORT" \
  uv run scripts/steering/serve_steered_policy.py
