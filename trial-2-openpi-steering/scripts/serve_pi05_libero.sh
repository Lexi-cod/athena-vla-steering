#!/usr/bin/env bash
# Launches the pi05_libero policy server with a safe LD_LIBRARY_PATH.
#
# CARC login nodes load `module load cuda/12.4.0` (see ~/.bashrc), which prepends that
# module's lib64 dir to LD_LIBRARY_PATH. Those older shared libs (cuBLAS/cuDNN) get
# dlopen'd ahead of the ones bundled in the nvidia-cuda12-* pip wheels that jaxlib 0.5.3
# actually ships/expects, causing a segfault on the first real GPU kernel launch
# (jax.devices() and array creation succeed since they don't touch cuBLAS/cuDNN; any
# real matmul/conv does, and crashes). Confirmed on both a P100 and a V100 node, so this
# is an env issue, not a GPU-architecture one. Strip any spack `cuda-*` lib64 dir before
# launching so jaxlib's own bundled CUDA libs are found first.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CLEAN_LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/cuda-[0-9]' | paste -sd: -)

# Default to the /scratch1 copy, not /home1: restoring from /home1 (NFS) measured ~6.5 MiB/s
# (~16.5 min for this 12GB checkpoint) vs. ~650 MiB/s (~10s) from /scratch1 — /home1 also has a
# much smaller quota. Override with PI05_LIBERO_CHECKPOINT_DIR if needed.
CHECKPOINT_DIR="${PI05_LIBERO_CHECKPOINT_DIR:-/scratch1/nalagand/openpi_cache/openpi-assets/checkpoints/pi05_libero}"

echo "Launching pi05_libero policy server (checkpoint: $CHECKPOINT_DIR)"
LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH" uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="$CHECKPOINT_DIR" \
  "$@"
