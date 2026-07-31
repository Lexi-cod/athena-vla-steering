"""Paired same-noise-seed comparison for the middle-black-bowl task (retroactive control on the
existing 1/12 success, 5/12 behavior-changed corrected-only result -- see CLAUDE.md 2026-07-11 section).

METHODOLOGICAL GAP THIS ADDRESSES: the existing bowl corrected-only result compares an unsteered baseline
batch against a corrected-only batch run as INDEPENDENT stochastic samples (different noise draws each
time). Since (a) this task has a non-trivial ~40% unsteered baseline success rate and (b) run-to-run
stochasticity in which wrong object gets grasped is already documented elsewhere this session, some or all
of the observed "1/12 success, 5/12 changed" effect could just be noise, not a real effect of the
corrected instruction.

FIX: for each known-failing init state, run TWO episodes -- one with the plain task instruction
(unsteered-equivalent), one with the fixed spatial corrected instruction, no blending -- using the
IDENTICAL deterministic noise seed at every replan step in both episodes (noise_seed = init_state_idx *
100_000 + replan_idx, sent to serve_steered_policy.py, which derives the noise array deterministically --
see that file's noise_seed handling, added for this test). This isolates whatever the corrected
instruction itself changes, holding the flow-matching noise sequence fixed, rather than comparing across
independent samples. Trajectories can and will still diverge once the two prompts produce different
actions (this is expected and is exactly the effect being measured) -- what's controlled is the STOCHASTIC
INPUT to each replan's denoising, not the resulting physical trajectory.

Runs all 12 known-failing init states from bowl_fidelity_batch_20260709_113120.csv.
"""

import collections
import csv
import datetime
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from run_fidelity_rollout import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    NUM_STEPS_WAIT,
    REPLAN_STEPS,
    RESIZE_SIZE,
    SEED,
    _quat2axisangle,
)

HOST = "0.0.0.0"
PORT = 8001  # steered server; steer omitted -> identical to the unsteered path
MAX_STEPS = 400

TASK_NAME = "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate"
TASK_SUITE = "libero_90"
TARGET_OBJECT = "akita_black_bowl_2"
DISTRACTOR_OBJECTS = ["akita_black_bowl_1", "akita_black_bowl_3"]
SPATIAL_CORRECTION_SUFFIX = ", the one positioned between the other two bowls, not the front bowl or the back bowl"

# All 12 known-failing init states from bowl_fidelity_batch_20260709_113120.csv.
KNOWN_FAILING_INIT_STATES = [0, 1, 4, 6, 7, 10, 11, 13, 14, 15, 18, 19]

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def run_one_episode(env, client, initial_state, all_objects, *, prompt, base_seed):
    client.reset()
    obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    t = 0
    done = False
    initial_object_pos = None
    replan_idx = 0

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
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
                "prompt": prompt,
                "noise_seed": base_seed * 100_000 + replan_idx,
            }
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
    return {"done": done, "displacement": displacement, "grasped_object": grasped_object}


def main():
    np.random.seed(SEED)

    bm = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == TASK_NAME)
    task = bm.get_task(task_id)
    task_description = task.language
    initial_states = bm.get_task_init_states(task_id)
    all_objects = [TARGET_OBJECT, *DISTRACTOR_OBJECTS]
    corrected_prompt = str(task_description) + SPATIAL_CORRECTION_SUFFIX

    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(SEED)

    client = _websocket_client_policy.WebsocketClientPolicy(HOST, PORT)

    print(f"Task: {task_description}")
    print(f"Corrected prompt: {corrected_prompt!r}")
    print(f"Running paired same-noise-seed trials on {len(KNOWN_FAILING_INIT_STATES)} known-failing "
          f"init states: {KNOWN_FAILING_INIT_STATES}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    n_flip_to_success = 0
    n_behavior_changed = 0

    for state_idx in KNOWN_FAILING_INIT_STATES:
        init_state = initial_states[state_idx]

        env.reset()
        unsteered = run_one_episode(env, client, init_state, all_objects, prompt=str(task_description), base_seed=state_idx)

        env.reset()
        corrected = run_one_episode(env, client, init_state, all_objects, prompt=corrected_prompt, base_seed=state_idx)

        flipped_to_success = corrected["done"] and not unsteered["done"]
        behavior_changed = corrected["grasped_object"] != unsteered["grasped_object"]
        n_flip_to_success += int(flipped_to_success)
        n_behavior_changed += int(behavior_changed)

        row = {
            "init_state": state_idx,
            "unsteered_outcome": "SUCCESS" if unsteered["done"] else "FAILURE",
            "unsteered_grasped": unsteered["grasped_object"],
            "corrected_outcome": "SUCCESS" if corrected["done"] else "FAILURE",
            "corrected_grasped": corrected["grasped_object"],
            "flipped_to_success": flipped_to_success,
            "behavior_changed": behavior_changed,
        }
        rows.append(row)
        print(
            f"  init_state {state_idx:2d}  unsteered=({row['unsteered_outcome']}, {row['unsteered_grasped']})  "
            f"corrected=({row['corrected_outcome']}, {row['corrected_grasped']})  "
            f"flipped_to_success={flipped_to_success}  behavior_changed={behavior_changed}",
            flush=True,
        )

    env.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"paired_noise_seed_bowl_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-trial results written to {csv_path}")

    n = len(rows)
    print("\n=== summary (paired, same-noise-seed) ===")
    print(f"trials: {n}")
    print(f"flipped to success (corrected succeeded, unsteered didn't, same seed): {n_flip_to_success}/{n}")
    print(f"behavior changed (different grasped_object, same seed): {n_behavior_changed}/{n}")


if __name__ == "__main__":
    main()
