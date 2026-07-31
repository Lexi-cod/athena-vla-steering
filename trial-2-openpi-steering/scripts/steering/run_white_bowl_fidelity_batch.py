"""Baseline (unsteered) fidelity batch for the white-bowl object-vs-location task -- see CLAUDE.md
2026-07-11 section, step 3. Unlike every other task tested this session (mug = object-identity
confusion, black-bowl = spatial-position confusion), KITCHEN_SCENE7_put_the_white_bowl_on_the_plate has
only ONE bowl (`white_bowl_1`) -- no competing same-category object at all. The known failure mode
("goes for the plate instead") is a pure object-vs-destination-location confusion.
`distractor_objects=["plate_1"]` treats incorrect/premature interaction with the destination as the
"distractor" for fidelity-signal purposes -- a different semantic use of the signal than the mug/bowl
tasks, but mechanically the same computation (distance to target vs. distance to the confusable point).
`grasped_object` (max-displacement heuristic) is less informative here since a plate may not move much
even if bumped -- treat `outcome` (the actual BDDL success predicate) and the signal trajectory as the
primary evidence for this task, not the displacement heuristic alone.

No baseline data existed for this task before this run -- this establishes it, to identify known-failing
init states for the corrected-only test that follows (orange-juice replication deprioritized, see
CLAUDE.md -- it would only replicate the mug task's object-identity category, already well-covered).

Run against the steered-policy server (port 8001) with `steer` omitted/False -- identical code path to
the original unsteered server (see serve_steered_policy.py's steer=False branch); no separate unsteered
server needed.
"""

import csv
import datetime
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _websocket_client_policy

from fidelity_signal import analyze_signal_trajectory
from run_fidelity_rollout import LIBERO_ENV_RESOLUTION, SEED, run_episode

HOST = "0.0.0.0"
PORT = 8001  # steered server; steer omitted -> identical to the unsteered path

TASK_NAME = "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate"
TASK_SUITE = "libero_90"
TARGET_OBJECT = "white_bowl_1"
DISTRACTOR_OBJECTS = ["plate_1"]  # the destination location, not a competing same-category object

NUM_ROLLOUTS = int(os.environ.get("FIDELITY_NUM_ROLLOUTS", 20))
ROLLOUT_OFFSET = int(os.environ.get("FIDELITY_ROLLOUT_OFFSET", 0))
DIVERGE_THRESH = -0.1
RECOVER_THRESH = -0.05

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


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
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(SEED)

    client = _websocket_client_policy.WebsocketClientPolicy(HOST, PORT)

    print(f"Task: {task_description}")
    print(f"Target object: {TARGET_OBJECT}  Distractors: {DISTRACTOR_OBJECTS}")
    print(f"Running {NUM_ROLLOUTS} rollouts (offset {ROLLOUT_OFFSET}), unsteered baseline\n")

    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for i in range(ROLLOUT_OFFSET, ROLLOUT_OFFSET + NUM_ROLLOUTS):
        init_state = initial_states[i % len(initial_states)]
        env.reset()
        result = run_episode(
            env, client, init_state, task_description,
            target_object=TARGET_OBJECT, distractor_objects=DISTRACTOR_OBJECTS, verbose=False,
        )
        log, done = result["log"], result["done"]

        trajectory = analyze_signal_trajectory(log, diverge_thresh=DIVERGE_THRESH, recover_thresh=RECOVER_THRESH)
        signals = np.array([row[1] for row in log])

        row = {
            "rollout": i,
            "init_state_idx": i % len(initial_states),
            "first_divergence_step": trajectory["first_divergence_step"],
            "recovery_step": trajectory["recovery_step"],
            "relapse_step": trajectory["relapse_step"],
            "outcome": "SUCCESS" if done else "FAILURE",
            "min_signal": float(signals.min()),
            "mean_signal": float(signals.mean()),
            "final_signal": float(signals[-1]),
            "grasped_object": result["grasped_object"],
        }
        rows.append(row)

        print(
            f"  rollout {i:2d}  first_divergence={row['first_divergence_step']}  "
            f"recovery={row['recovery_step']}  relapse={row['relapse_step']}  "
            f"outcome={row['outcome']}  min_signal={row['min_signal']:+.4f}  "
            f"mean_signal={row['mean_signal']:+.4f}  grasped={row['grasped_object']}",
            flush=True,
        )

    env.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"white_bowl_fidelity_batch_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-rollout results written to {csv_path}")

    print("\n=== aggregate summary ===")
    n = len(rows)
    n_success = sum(1 for r in rows if r["outcome"] == "SUCCESS")
    print(f"rollouts: {n}")
    print(f"overall success rate: {n_success}/{n} ({100 * n_success / n:.0f}%)")


if __name__ == "__main__":
    main()
