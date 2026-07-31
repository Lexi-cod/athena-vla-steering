"""Measure the render-induced noise floor of the paired control (PROGRESS.md section 3).

The renderer is nondeterministic across resets: identical physics state renders different pixels
(see diagnose_render_repeat.py). So a "paired same-noise-seed" trial is NOT observation-controlled,
and two arms can disagree for reasons that have nothing to do with steering.

This quantifies that. It is the paired control with the intervention REMOVED: both arms are
UNSTEERED, same init state, same per-replan noise_seed -- so every disagreement is pure pipeline
noise. It imports run_one_episode from run_paired_athena_task so the loop is byte-for-byte the one
the real paired controls use.

Output: the outcome-disagreement rate. That is the floor a steering effect must clear to be
detectable. E.g. if identical arms disagree on 2/12 outcomes, the bowl control's "1 flip-to-success
+ 1 flip-to-failure" is exactly what noise alone produces, and is not evidence of anything.

  PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero ATHENA_TASK=middle_bowl NOISE_FLOOR_STATES=12 \
    examples/libero/.venv/bin/python scripts/steering/run_render_noise_floor.py
"""

import csv
import datetime
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _wcp

from athena_tasks import get_task
from run_fidelity_rollout import LIBERO_ENV_RESOLUTION, SEED
from run_paired_athena_task import run_one_episode

TASK_KEY = os.environ.get("ATHENA_TASK", "middle_bowl")
PORT = int(os.environ.get("STEERED_POLICY_PORT", 8001))
NUM_STATES = int(os.environ.get("NOISE_FLOOR_STATES", 12))

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
LOG_DIR = RESULTS_DIR / "athena_feedback_logs"


def main():
    np.random.seed(SEED)
    task = get_task(TASK_KEY)
    bm = benchmark.get_benchmark_dict()[task.task_suite]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == task.task_name)
    libero_task = bm.get_task(task_id)
    task_description = libero_task.language
    initial_states = bm.get_task_init_states(task_id)
    all_objects = [task.target_object, *task.distractor_objects]

    bddl = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION
    )
    env.seed(SEED)
    client = _wcp.WebsocketClientPolicy("0.0.0.0", PORT)

    print(f"Task[{task.key}]: {task_description}")
    print("NOISE FLOOR: both arms UNSTEERED, identical init state + noise_seed.")
    print(f"Any disagreement is pipeline noise. {NUM_STATES} paired trials.\n", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    rows = []
    n_outcome_disagree = 0
    n_grasp_disagree = 0
    for s in range(NUM_STATES):
        init_state = initial_states[s % len(initial_states)]

        env.reset()
        a = run_one_episode(
            env, client, init_state, task_description, task, all_objects, steer_enabled=False, base_seed=s
        )
        env.reset()
        b = run_one_episode(
            env, client, init_state, task_description, task, all_objects, steer_enabled=False, base_seed=s
        )

        outcome_disagree = a["done"] != b["done"]
        grasp_disagree = a["grasped_object"] != b["grasped_object"]
        n_outcome_disagree += int(outcome_disagree)
        n_grasp_disagree += int(grasp_disagree)

        rows.append(
            {
                "init_state": s,
                "run_a_outcome": "SUCCESS" if a["done"] else "FAILURE",
                "run_a_grasped": a["grasped_object"],
                "run_a_target_disp": round(a["target_displacement"], 4),
                "run_b_outcome": "SUCCESS" if b["done"] else "FAILURE",
                "run_b_grasped": b["grasped_object"],
                "run_b_target_disp": round(b["target_displacement"], 4),
                "outcome_disagree": outcome_disagree,
                "grasp_disagree": grasp_disagree,
            }
        )
        print(
            f"  state {s:2d}  A=({'SUCCESS' if a['done'] else 'FAILURE'},{a['grasped_object']})  "
            f"B=({'SUCCESS' if b['done'] else 'FAILURE'},{b['grasped_object']})  "
            f"outcome_disagree={outcome_disagree}  grasp_disagree={grasp_disagree}",
            flush=True,
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"render_noise_floor_{task.key}_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    summary = (
        f"## render_noise_floor[{task.key}] {timestamp}\n"
        f"- task: {task_description}\n"
        f"- BOTH arms unsteered, identical init state + per-replan noise_seed\n"
        f"- paired trials: {n}\n"
        f"- OUTCOME disagreement (success vs failure): {n_outcome_disagree}/{n} "
        f"({100.0 * n_outcome_disagree / n:.0f}%)\n"
        f"- GRASPED-OBJECT disagreement: {n_grasp_disagree}/{n} ({100.0 * n_grasp_disagree / n:.0f}%)\n"
        f"- interpretation: a steering effect must exceed this floor to be detectable\n"
        f"- csv: {csv_path.name}\n\n"
    )
    with (LOG_DIR / "RESULTS.md").open("a") as f:
        f.write(summary)

    print(f"\nPer-trial -> {csv_path}\n\n=== summary (noise floor) ===\n{summary}")


if __name__ == "__main__":
    main()
