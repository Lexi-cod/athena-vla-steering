"""Cross-task validation of the fidelity signal on a MIXED-outcome task.

Unlike LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate (100% failure in the 30-rollout batch),
KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate has partial success in prior annotation (~3/5)
and the same clean target/distractor structure: three visually-identical bowls (akita_black_bowl_1/2/3,
front/middle/back), target = middle (akita_black_bowl_2), distractors = front + back. This checks the
other half of signal validity: does it stay positive/near-neutral on rollouts that actually succeed, or
does it also spuriously dip negative on them? Full per-step traces are saved (not just the 3 summary
steps) so successful rollouts can be inspected in detail, not just categorized.
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
from run_fidelity_rollout import HOST, LIBERO_ENV_RESOLUTION, PORT, SEED, run_episode

TASK_NAME = "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate"
TASK_SUITE = "libero_90"
TARGET_OBJECT = "akita_black_bowl_2"
DISTRACTOR_OBJECTS = ["akita_black_bowl_1", "akita_black_bowl_3"]

NUM_ROLLOUTS = int(os.environ.get("FIDELITY_NUM_ROLLOUTS", 20))
ROLLOUT_OFFSET = int(os.environ.get("FIDELITY_ROLLOUT_OFFSET", 0))
DIVERGE_THRESH = -0.1
RECOVER_THRESH = -0.05

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
TRACES_DIR = RESULTS_DIR / "bowl_traces"


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
    print(f"Running {NUM_ROLLOUTS} rollouts (offset {ROLLOUT_OFFSET}), "
          f"diverge_thresh={DIVERGE_THRESH}, recover_thresh={RECOVER_THRESH}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    TRACES_DIR.mkdir(exist_ok=True)

    rows = []
    for i in range(ROLLOUT_OFFSET, ROLLOUT_OFFSET + NUM_ROLLOUTS):
        init_state = initial_states[i % len(initial_states)]
        env.reset()
        result = run_episode(
            env, client, init_state, task_description,
            target_object=TARGET_OBJECT, distractor_objects=DISTRACTOR_OBJECTS, verbose=False,
        )
        log, done = result["log"], result["done"]

        trace_path = TRACES_DIR / f"trace_rollout{i}.csv"
        with open(trace_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "signal", "target_dist", "nearest_distractor", "nearest_dist"])
            writer.writerows(log)

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
    csv_path = RESULTS_DIR / f"bowl_fidelity_batch_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-rollout results written to {csv_path}")
    print(f"Full per-step traces written to {TRACES_DIR}/trace_rollout*.csv")

    print("\n=== aggregate summary ===")
    n = len(rows)
    n_success = sum(1 for r in rows if r["outcome"] == "SUCCESS")
    print(f"rollouts: {n}")
    print(f"overall success rate: {n_success}/{n} ({100 * n_success / n:.0f}%)")

    successes = [r for r in rows if r["outcome"] == "SUCCESS"]
    failures = [r for r in rows if r["outcome"] == "FAILURE"]

    def signal_stats(subset, label):
        if not subset:
            print(f"{label}: n/a (0 rollouts)")
            return
        mins = [r["min_signal"] for r in subset]
        means = [r["mean_signal"] for r in subset]
        n_diverged = sum(1 for r in subset if r["first_divergence_step"] is not None)
        n_recovered = sum(1 for r in subset if r["recovery_step"] is not None)
        print(f"{label} (n={len(subset)}): "
              f"mean(min_signal)={np.mean(mins):+.4f}  mean(mean_signal)={np.mean(means):+.4f}  "
              f"diverged={n_diverged}/{len(subset)}  recovered-after-diverging={n_recovered}/{len(subset)}")

    print("\n--- SUCCESS rollouts: does the signal stay positive/near-neutral, or dip and genuinely recover? ---")
    signal_stats(successes, "successes")
    print("\n--- FAILURE rollouts: same clean-divergence pattern as the red-mug task? ---")
    signal_stats(failures, "failures")


if __name__ == "__main__":
    main()
