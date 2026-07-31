"""Diagnostic: dump FULL per-step fidelity-signal trajectories for a handful of rollouts.

Not the 30-rollout validation batch -- this is a small, targeted rerun (a few rollouts) to
inspect the divergence/recovery/relapse pattern in detail: does "recovery" (signal crossing
back above RECOVER_THRESH) correspond to target_dist actually decreasing, or just to
nearest_dist (distance to whichever distractor is currently closest) increasing because the
arm is transiting between the two distractor mugs without heading back toward the target?

Persists full per-step logs to results/ as CSV (one file per rollout) since run_fidelity_batch.py
only keeps the 3 threshold-crossing summary steps, not the full trace.
"""

import csv
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _websocket_client_policy

from run_fidelity_rollout import HOST, PORT, SEED, TASK_NAME, TASK_SUITE, run_episode

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# Chosen to cover: earliest divergence (6), a never-recovered case (15, 25), and both
# grasped_object outcomes seen in the 00:52 batch (porcelain_mug_1 vs white_yellow_mug_1).
ROLLOUT_INDICES = [0, 6, 15, 5, 25]


def main():
    np.random.seed(SEED)

    bm = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == TASK_NAME)
    task = bm.get_task(task_id)
    task_description = task.language
    initial_states = bm.get_task_init_states(task_id)

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(SEED)

    client = _websocket_client_policy.WebsocketClientPolicy(HOST, PORT)

    RESULTS_DIR.mkdir(exist_ok=True)

    for i in ROLLOUT_INDICES:
        init_state = initial_states[i % len(initial_states)]
        env.reset()
        print(f"\n=== rollout {i} (init_state_idx={i % len(initial_states)}) ===", flush=True)
        result = run_episode(env, client, init_state, task_description, verbose=False)
        log, done, grasped = result["log"], result["done"], result["grasped_object"]

        out_path = RESULTS_DIR / f"trace_rollout{i}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "signal", "target_dist", "nearest_distractor", "nearest_dist"])
            writer.writerows(log)

        print(f"outcome={'SUCCESS' if done else 'FAILURE'} grasped_object={grasped} steps={len(log)}")
        print(f"full trace written to {out_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
