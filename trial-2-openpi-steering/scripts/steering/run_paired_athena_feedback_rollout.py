"""Paired same-noise-seed control for ATHENA-Feedback steer-AWAY on the red-mug task.

Rigor upgrade on the athena_feedback_batch_20260720 result (0/12 success, target never touched). That
batch compared away-steering against the prior unsteered baseline as INDEPENDENT stochastic draws. Here,
for each init state we run BOTH arms with the IDENTICAL deterministic noise sequence per replan
(noise_seed = init_state_idx * 100_000 + replan_idx, derived server-side), isolating exactly what the
away-steering changes while holding the flow-matching noise fixed:

  - UNSTEERED arm: plain task prompt, steer omitted (server's no-op path), noise_seed per replan.
  - AWAY arm:      same signal-gated ATHENA away-steering as run_athena_feedback_rollout.py (steer=True,
                   steer_mode="away", control_prompt naming the nearest distractor, when signal < -0.1),
                   SAME noise_seed per replan.

Both arms compute the fidelity signal each replan (the unsteered arm only for logging; it never steers).
Trajectories still diverge once the two arms produce different actions -- that divergence IS the measured
effect; what's controlled is the stochastic input to each replan's denoising.

Runs init states 0..N-1 (default 12 -- the same states as the 12-rollout away batch).
"""

import collections
import csv
import datetime
import os
import pathlib

from correction_instruction import build_control_instruction
from fidelity_signal import compute_fidelity_signal
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from run_athena_feedback_rollout import DISTRACTOR_DISPLAY_NAMES
from run_athena_feedback_rollout import GAMMA
from run_athena_feedback_rollout import TARGET_DISPLAY_NAME
from run_athena_feedback_rollout import WINDOW_STEPS
from run_fidelity_rollout import DISTRACTOR_OBJECTS
from run_fidelity_rollout import LIBERO_DUMMY_ACTION
from run_fidelity_rollout import LIBERO_ENV_RESOLUTION
from run_fidelity_rollout import NUM_STEPS_WAIT
from run_fidelity_rollout import REPLAN_STEPS
from run_fidelity_rollout import RESIZE_SIZE
from run_fidelity_rollout import SEED
from run_fidelity_rollout import TARGET_OBJECT
from run_fidelity_rollout import TASK_NAME
from run_fidelity_rollout import TASK_SUITE
from run_fidelity_rollout import _quat2axisangle

HOST = "0.0.0.0"
PORT = int(os.environ.get("STEERED_POLICY_PORT", 8001))
MAX_STEPS = 400
DIVERGE_THRESH = -0.1

NUM_STATES = int(os.environ.get("PAIRED_NUM_STATES", 12))
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
LOG_DIR = RESULTS_DIR / "athena_feedback_logs"


def run_one_episode(env, client, initial_state, task_description, all_objects, *, steer_enabled, base_seed, verbose=False):
    """One episode. steer_enabled=False -> unsteered arm; True -> signal-gated ATHENA away arm.
    Both send noise_seed = base_seed*100_000 + replan_idx so the paired arms share the noise sequence."""
    client.reset()
    obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    t = 0
    done = False
    initial_object_pos = None
    replan_idx = 0
    n_steered = 0

    while t < MAX_STEPS + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            if t == NUM_STEPS_WAIT:
                initial_object_pos = {name: np.asarray(obs[f"{name}_pos"]).copy() for name in all_objects}
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE))

        if not action_plan:
            signal, _, (nearest_distractor, _) = compute_fidelity_signal(obs, TARGET_OBJECT, DISTRACTOR_OBJECTS)
            steer = steer_enabled and signal < DIVERGE_THRESH

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
                "prompt": str(task_description),
                "noise_seed": base_seed * 100_000 + replan_idx,
            }
            if steer:
                element["steer"] = True
                element["steer_mode"] = "away"
                element["control_prompt"] = build_control_instruction(
                    str(task_description),
                    target_display=TARGET_DISPLAY_NAME,
                    distractor_display=DISTRACTOR_DISPLAY_NAMES[nearest_distractor],
                )
                element["gamma"] = GAMMA
                element["window_steps"] = WINDOW_STEPS
                n_steered += 1

            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[:REPLAN_STEPS])
            replan_idx += 1

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())
        if done:
            break
        t += 1

    final_object_pos = {name: np.asarray(obs[f"{name}_pos"]) for name in all_objects}
    displacement = {name: float(np.linalg.norm(final_object_pos[name] - initial_object_pos[name])) for name in all_objects}
    grasped_object = max(displacement, key=displacement.get)
    return {
        "done": done,
        "grasped_object": grasped_object,
        "target_displacement": displacement[TARGET_OBJECT],
        "n_steered": n_steered,
        "n_replans": replan_idx,
    }


def main():
    np.random.seed(SEED)

    bm = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == TASK_NAME)
    task = bm.get_task(task_id)
    task_description = task.language
    initial_states = bm.get_task_init_states(task_id)
    all_objects = [TARGET_OBJECT, *DISTRACTOR_OBJECTS]

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file), camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION
    )
    env.seed(SEED)
    client = _websocket_client_policy.WebsocketClientPolicy(HOST, PORT)

    print(f"Task: {task_description}  (target {TARGET_OBJECT})")
    print(f"Paired same-noise-seed: UNSTEERED vs ATHENA-away  gamma={GAMMA}  window={WINDOW_STEPS}")
    print(f"Running {NUM_STATES} init states (0..{NUM_STATES - 1})\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    rows = []
    n_flip_to_success = 0
    n_behavior_changed = 0
    for state_idx in range(NUM_STATES):
        init_state = initial_states[state_idx % len(initial_states)]

        env.reset()
        uns = run_one_episode(env, client, init_state, task_description, all_objects, steer_enabled=False, base_seed=state_idx)
        env.reset()
        away = run_one_episode(env, client, init_state, task_description, all_objects, steer_enabled=True, base_seed=state_idx)

        flipped_to_success = away["done"] and not uns["done"]
        behavior_changed = away["grasped_object"] != uns["grasped_object"] or abs(
            away["target_displacement"] - uns["target_displacement"]
        ) > 1e-3
        n_flip_to_success += int(flipped_to_success)
        n_behavior_changed += int(behavior_changed)

        row = {
            "init_state": state_idx,
            "unsteered_outcome": "SUCCESS" if uns["done"] else "FAILURE",
            "unsteered_grasped": uns["grasped_object"],
            "unsteered_target_disp": round(uns["target_displacement"], 4),
            "away_outcome": "SUCCESS" if away["done"] else "FAILURE",
            "away_grasped": away["grasped_object"],
            "away_target_disp": round(away["target_displacement"], 4),
            "away_n_steered": away["n_steered"],
            "flipped_to_success": flipped_to_success,
            "behavior_changed": behavior_changed,
        }
        rows.append(row)
        print(
            f"  init {state_idx:2d}  unsteered=({row['unsteered_outcome']},{row['unsteered_grasped']},"
            f"d={row['unsteered_target_disp']})  away=({row['away_outcome']},{row['away_grasped']},"
            f"d={row['away_target_disp']},steered={away['n_steered']})  "
            f"flip={flipped_to_success}  changed={behavior_changed}",
            flush=True,
        )

    env.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"paired_athena_feedback_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    n_uns_success = sum(1 for r in rows if r["unsteered_outcome"] == "SUCCESS")
    n_away_success = sum(1 for r in rows if r["away_outcome"] == "SUCCESS")
    summary = (
        f"## paired_athena_feedback (same-noise-seed) {timestamp}\n"
        f"- task: {task_description}  (target {TARGET_OBJECT})\n"
        f"- UNSTEERED vs ATHENA-away  gamma={GAMMA}  window={WINDOW_STEPS}  diverge_thresh={DIVERGE_THRESH}\n"
        f"- paired trials: {n} (init states 0..{n - 1})\n"
        f"- unsteered success: {n_uns_success}/{n}   away success: {n_away_success}/{n}\n"
        f"- flipped to success (away succeeded where unsteered didn't, same seed): {n_flip_to_success}/{n}\n"
        f"- behavior changed (different grasp or target-disp delta >1e-3, same seed): {n_behavior_changed}/{n}\n"
        f"- csv: {csv_path.name}\n\n"
    )
    with open(LOG_DIR / "RESULTS.md", "a") as f:
        f.write(summary)
    print(f"\nPer-trial results written to {csv_path}")
    print("\n=== summary (paired, same-noise-seed) ===")
    print(summary)


if __name__ == "__main__":
    main()
