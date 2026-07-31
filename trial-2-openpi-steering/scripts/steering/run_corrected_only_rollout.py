"""Isolates whether the disambiguating ("corrected") instruction works at all on its own --
no blending, no original instruction, straight single-instruction sample_actions() using ONLY the
corrected prompt for the entire episode. See conversation: the gamma sweep in run_steered_rollout.py
never flipped grasped_object to the target across gamma in {1, 2, 4}; this test separates two very
different explanations:
  - corrected-only ALSO never reaches the target -> perceptual/spatial-grounding bottleneck, not a
    blending or gamma/window tuning problem.
  - corrected-only DOES succeed -> the disambiguating language works fine alone, and the dual-
    instruction blending mechanism itself is diluting/cancelling the correction -- a design bug, not
    a fundamental limit.

IMPORTANT: uses natural-language distractor names ("white mug" / "yellow and white mug"), NOT the raw
internal object ids ("porcelain_mug_1" / "white_yellow_mug_1") that run_steered_rollout.py's gamma
sweep actually fed the model via compute_fidelity_signal's return value -- see conversation for why
that's a real, separate bug that confounds the gamma-sweep "no effect" result.

The corrected instruction is fixed once per rollout, right after the settle wait (using whichever
distractor is nearest at that point), and reused for every replan for the rest of the episode --
matching "as if it were the only instruction ever given" rather than a per-replan recomputation.
"""

import collections
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from correction_instruction import build_corrected_instruction
from fidelity_signal import compute_fidelity_signal
from run_fidelity_rollout import (
    DISTRACTOR_OBJECTS,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    NUM_STEPS_WAIT,
    REPLAN_STEPS,
    RESIZE_SIZE,
    SEED,
    TARGET_OBJECT,
    TASK_NAME,
    TASK_SUITE,
    _quat2axisangle,
)

HOST = "0.0.0.0"
PORT = 8001  # the steered server; steer omitted (defaults False) -> identical to the unsteered path
MAX_STEPS = 400

# Natural-language names -- matches CLAUDE.md's documented mapping, NOT the raw object ids.
DISTRACTOR_DISPLAY_NAMES = {
    "porcelain_mug_1": "white mug",
    "white_yellow_mug_1": "yellow and white mug",
}


def run_episode(env, client, initial_state, task_description):
    client.reset()
    obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    t = 0
    done = False
    all_objects = [TARGET_OBJECT, *DISTRACTOR_OBJECTS]
    initial_object_pos = None
    corrected_prompt = None

    while t < MAX_STEPS + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            if t == NUM_STEPS_WAIT:
                initial_object_pos = {name: np.asarray(obs[f"{name}_pos"]).copy() for name in all_objects}
                _, _, (nearest_distractor, _) = compute_fidelity_signal(obs, TARGET_OBJECT, DISTRACTOR_OBJECTS)
                corrected_prompt = build_corrected_instruction(
                    str(task_description),
                    target_name=TARGET_OBJECT,
                    distractor_name=DISTRACTOR_DISPLAY_NAMES[nearest_distractor],
                )
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
                "prompt": corrected_prompt,  # ONLY the corrected instruction, whole episode, no blending
            }
            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[:REPLAN_STEPS])

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
        "displacement": displacement,
        "grasped_object": grasped_object,
        "corrected_prompt": corrected_prompt,
    }


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

    # Comma-separated init_state indices to run (repeats allowed -- each repeat is still an
    # independent stochastic rollout since the policy draws fresh flow-matching noise per call).
    # Defaults to 0,1,2 to match the original 3-init-state run.
    init_state_indices = [int(x) for x in os.environ.get("CORRECTED_ONLY_INIT_STATES", "0,1,2").split(",")]

    print(f"Task: {task_description}")
    print(f"Target object: {TARGET_OBJECT}  Distractors: {DISTRACTOR_OBJECTS}")
    print(f"Running {len(init_state_indices)} rollouts on init_state indices {init_state_indices}, "
          f"corrected-instruction-only (no blending, no original instruction)\n")

    for run_idx, state_idx in enumerate(init_state_indices):
        env.reset()
        result = run_episode(env, client, initial_states[state_idx], task_description)
        outcome = "SUCCESS" if result["done"] else "FAILURE"
        print(
            f"  rollout {run_idx} (init_state {state_idx})  corrected_prompt={result['corrected_prompt']!r}\n"
            f"    outcome={outcome}  grasped={result['grasped_object']}  "
            f"displacement={ {k: round(v, 4) for k, v in result['displacement'].items()} }"
        )

    env.close()


if __name__ == "__main__":
    main()
