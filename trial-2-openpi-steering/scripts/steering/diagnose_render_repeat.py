"""Is the render nondeterminism a first-run/warm-up artifact, or every-run?

diagnose_paired_determinism.py established that two identical unsteered arms have bit-identical
physics at replan 0 (joints, objects, eef all 0.0 diff) but images differing by 137 grey levels on
~6% of pixels. This narrows the cause: render THREE identical episodes' first frame in one process,
with no policy calls at all (pure sim + render), and compare.

  A != B == C  -> the FIRST episode in a process renders differently (cold GL context / dirty
                  buffer). Fix: discard a warm-up episode, or warm the renderer after construction.
  A != B != C  -> every render differs; needs a per-arm fresh process or a renderer-level fix.
  A == B == C  -> rendering is deterministic on its own; the earlier diff came from something the
                  policy loop does (and the paired driver must be fixed there instead).

No server needed -- this never calls the policy.

  PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero ATHENA_TASK=middle_bowl \
    examples/libero/.venv/bin/python scripts/steering/diagnose_render_repeat.py
"""

import hashlib
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from athena_tasks import get_task
from run_fidelity_rollout import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    NUM_STEPS_WAIT,
    RESIZE_SIZE,
    SEED,
)

TASK_KEY = os.environ.get("ATHENA_TASK", "middle_bowl")
INIT_STATE = int(os.environ.get("DIAG_INIT_STATE", 10))
N_RUNS = int(os.environ.get("DIAG_N_RUNS", 3))


def settle_and_render(env, initial_state):
    """reset -> set_init_state -> NUM_STEPS_WAIT dummy steps -> return the agentview frame."""
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))


def main():
    np.random.seed(SEED)
    task = get_task(TASK_KEY)
    bm = benchmark.get_benchmark_dict()[task.task_suite]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == task.task_name)
    libero_task = bm.get_task(task_id)
    initial_states = bm.get_task_init_states(task_id)
    init_state = initial_states[INIT_STATE % len(initial_states)]

    bddl = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION
    )
    env.seed(SEED)

    print(f"\n=== RENDER REPEATABILITY  task={task.key}  init_state={INIT_STATE}  runs={N_RUNS} ===")
    print("Pure sim + render, identical settle sequence, NO policy calls.\n")

    frames = [settle_and_render(env, init_state) for _ in range(N_RUNS)]
    for i, f in enumerate(frames):
        print(f"  run {i}: sha={hashlib.md5(f.tobytes()).hexdigest()[:12]}")

    print()
    for i in range(len(frames) - 1):
        d = np.abs(frames[i].astype(np.int16) - frames[i + 1].astype(np.int16))
        npx = int((d > 0).sum())
        verdict = "IDENTICAL" if npx == 0 else "DIFFERS"
        print(
            f"  run {i} vs run {i + 1}: {verdict}  max={d.max():>3d}  mean={d.mean():.4f}  "
            f"differing px={npx}/{d.size} ({100.0 * npx / d.size:.2f}%)"
        )

    print("\n=== VERDICT ===")
    hashes = [hashlib.md5(f.tobytes()).hexdigest() for f in frames]
    if len(set(hashes)) == 1:
        print("All renders IDENTICAL -> rendering itself is deterministic.")
        print("=> the paired-arm image diff comes from the policy loop, not the renderer.")
    elif len(set(hashes[1:])) == 1:
        print("First render differs, later renders agree -> COLD-START artifact.")
        print("=> fix: run one throwaway warm-up episode before the paired trials.")
    else:
        print("Every render differs -> renderer is nondeterministic across resets.")
        print("=> fix: run each paired arm in its own process, or a fresh env per arm.")


if __name__ == "__main__":
    main()
