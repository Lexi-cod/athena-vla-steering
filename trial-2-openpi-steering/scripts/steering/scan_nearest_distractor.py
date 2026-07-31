"""Scans all init states for the red-mug task and reports which distractor is nearest at settle.

No policy server needed -- just resets the sim to each init state, steps the settle wait, and reads
the ground-truth nearest distractor via compute_fidelity_signal. Used to find init states where
porcelain_mug_1 (not white_yellow_mug_1) is nearest, for a targeted corrected-only-instruction re-test
-- see run_corrected_only_rollout.py / conversation.
"""

import pathlib

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from fidelity_signal import compute_fidelity_signal
from run_fidelity_rollout import (
    DISTRACTOR_OBJECTS,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    NUM_STEPS_WAIT,
    SEED,
    TARGET_OBJECT,
    TASK_NAME,
    TASK_SUITE,
)


def main():
    bm = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == TASK_NAME)
    task = bm.get_task(task_id)
    initial_states = bm.get_task_init_states(task_id)

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(SEED)

    print(f"Scanning {len(initial_states)} init states for nearest distractor at settle...\n")
    counts = {}
    for i, init_state in enumerate(initial_states):
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(NUM_STEPS_WAIT):
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
        _, _, (nearest_distractor, nearest_dist) = compute_fidelity_signal(obs, TARGET_OBJECT, DISTRACTOR_OBJECTS)
        counts[nearest_distractor] = counts.get(nearest_distractor, 0) + 1
        print(f"  init_state {i:2d}  nearest_distractor={nearest_distractor}  dist={nearest_dist:.4f}")

    env.close()
    print("\n=== summary ===")
    for name, count in counts.items():
        print(f"  {name}: {count}/{len(initial_states)}")


if __name__ == "__main__":
    main()
