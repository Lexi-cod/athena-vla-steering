import csv
import datetime
import logging
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _websocket_client_policy

from fidelity_signal import analyze_signal_trajectory
from run_fidelity_rollout import (
    DISTRACTOR_OBJECTS,
    HOST,
    LIBERO_ENV_RESOLUTION,
    PORT,
    SEED,
    TARGET_OBJECT,
    TASK_NAME,
    TASK_SUITE,
    run_episode,
)

# Overridable via env vars for smaller ad-hoc validation runs without touching the default
# 30-rollout batch behavior, e.g. FIDELITY_NUM_ROLLOUTS=10 FIDELITY_ROLLOUT_OFFSET=30 to run
# 10 fresh rollouts (indices 30-39, not overlapping any previously-inspected rollout).
NUM_ROLLOUTS = int(os.environ.get("FIDELITY_NUM_ROLLOUTS", 30))
ROLLOUT_OFFSET = int(os.environ.get("FIDELITY_ROLLOUT_OFFSET", 0))
DIVERGE_THRESH = -0.1
RECOVER_THRESH = -0.05

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def main():
    logging.basicConfig(level=logging.INFO)
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
    print(f"Running {NUM_ROLLOUTS} rollouts, diverge_thresh={DIVERGE_THRESH}, recover_thresh={RECOVER_THRESH}\n")

    rows = []
    for i in range(ROLLOUT_OFFSET, ROLLOUT_OFFSET + NUM_ROLLOUTS):
        init_state = initial_states[i % len(initial_states)]
        env.reset()
        result = run_episode(env, client, init_state, task_description, verbose=False)
        log, done = result["log"], result["done"]

        trajectory = analyze_signal_trajectory(log, diverge_thresh=DIVERGE_THRESH, recover_thresh=RECOVER_THRESH)
        final_signal = log[-1][1]

        row = {
            "rollout": i,
            "init_state_idx": i % len(initial_states),
            "first_divergence_step": trajectory["first_divergence_step"],
            "recovery_step": trajectory["recovery_step"],
            "relapse_step": trajectory["relapse_step"],
            "outcome": "SUCCESS" if done else "FAILURE",
            "final_signal": final_signal,
            "grasped_object": result["grasped_object"],
        }
        rows.append(row)

        print(
            f"  rollout {i:2d}  first_divergence={row['first_divergence_step']}  "
            f"recovery={row['recovery_step']}  relapse={row['relapse_step']}  "
            f"outcome={row['outcome']}  final_signal={final_signal:+.4f}  "
            f"grasped={row['grasped_object']}"
        )

    env.close()

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"fidelity_batch_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-rollout results written to {csv_path}")

    print("\n=== aggregate summary ===")
    n = len(rows)
    n_success = sum(1 for r in rows if r["outcome"] == "SUCCESS")
    n_early_diverge = sum(1 for r in rows if r["first_divergence_step"] is not None)
    n_recovery = sum(1 for r in rows if r["recovery_step"] is not None)
    n_relapse = sum(1 for r in rows if r["relapse_step"] is not None)

    print(f"rollouts: {n}")
    print(f"overall success rate: {n_success}/{n} ({100 * n_success / n:.0f}%)")
    print(f"rollouts with early divergence (signal < {DIVERGE_THRESH}): {n_early_diverge}/{n} "
          f"({100 * n_early_diverge / n:.0f}%)")
    print(f"  of those, with a recovery (signal > {RECOVER_THRESH} afterward): {n_recovery}/{n_early_diverge or 1} "
          f"({100 * n_recovery / (n_early_diverge or 1):.0f}%)")
    print(f"    of those recoveries, with a later relapse (signal < {DIVERGE_THRESH} again): "
          f"{n_relapse}/{n_recovery or 1} ({100 * n_relapse / (n_recovery or 1):.0f}%)")

    # Key finding: does recovering from an early dip actually predict eventual success?
    diverged = [r for r in rows if r["first_divergence_step"] is not None]
    recovered = [r for r in diverged if r["recovery_step"] is not None]
    not_recovered = [r for r in diverged if r["recovery_step"] is None]
    recovered_no_relapse = [r for r in recovered if r["relapse_step"] is None]
    recovered_then_relapsed = [r for r in recovered if r["relapse_step"] is not None]

    def success_rate(subset):
        if not subset:
            return "n/a (0 rollouts)"
        s = sum(1 for r in subset if r["outcome"] == "SUCCESS")
        return f"{s}/{len(subset)} ({100 * s / len(subset):.0f}%)"

    print("\n--- does recovery from an early dip predict eventual success? ---")
    print(f"success rate | diverged, never recovered:        {success_rate(not_recovered)}")
    print(f"success rate | diverged, recovered, no relapse:   {success_rate(recovered_no_relapse)}")
    print(f"success rate | diverged, recovered, then relapsed: {success_rate(recovered_then_relapsed)}")
    print(f"success rate | never diverged (signal stayed >= {DIVERGE_THRESH} throughout): "
          f"{success_rate([r for r in rows if r['first_divergence_step'] is None])}")


if __name__ == "__main__":
    main()
