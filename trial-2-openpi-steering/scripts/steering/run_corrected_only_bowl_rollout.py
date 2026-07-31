"""Priority discriminating test (see CLAUDE.md 2026-07-11 section, "Next steps" #3): does SPATIAL
disambiguating language succeed where OBJECT-IDENTITY disambiguating language failed on the red-mug
task (run_corrected_only_rollout.py: 0/8 successes, target displacement 0.0000m every time, across
both distractors)?

Uses KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate (see run_bowl_fidelity_batch.py /
verify_object_lookup_bowls.py): three visually-identical bowls in a row, target = middle
(akita_black_bowl_2), distractors = front (akita_black_bowl_1) and back (akita_black_bowl_3) -- a
*spatial* confusion (position in a row), structurally different from the red-mug task's *object-identity*
confusion (color/pattern). Unsteered baseline (bowl_fidelity_batch_20260709_113120.csv, 20 rollouts) is
mixed -- 8/20 success -- unlike the red-mug task's ~0% baseline, so this test specifically targets the
12 known-FAILING init states (a mix of both failure directions: init states 0/6/7 grasped the back bowl,
1/4/10 grasped the front bowl).

Same corrected-only-instruction methodology as run_corrected_only_rollout.py (single instruction, no
blending, no original instruction, fixed for the whole episode after settle) but the corrected instruction
here is a SPATIAL description (position in the row), not an object-identity/color description -- built as
a fixed template, not via build_corrected_instruction() (which only knows how to name a specific
distractor object, not describe relative position).
"""

import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from run_fidelity_rollout import (
    LIBERO_ENV_RESOLUTION,
    NUM_STEPS_WAIT,
    REPLAN_STEPS,
    RESIZE_SIZE,
    SEED,
    _quat2axisangle,
)

HOST = "0.0.0.0"
PORT = 8001  # the steered server; steer omitted (defaults False) -> identical to the unsteered path
MAX_STEPS = 400

TASK_NAME = "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate"
TASK_SUITE = "libero_90"
TARGET_OBJECT = "akita_black_bowl_2"
DISTRACTOR_OBJECTS = ["akita_black_bowl_1", "akita_black_bowl_3"]  # front, back

# Fixed spatial disambiguation -- same instruction for every rollout regardless of which distractor is
# nearest, since it disambiguates by POSITION in the row, not by naming a specific object.
SPATIAL_CORRECTION_SUFFIX = ", the one positioned between the other two bowls, not the front bowl or the back bowl"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Known-failing init states from bowl_fidelity_batch_20260709_113120.csv (12 failures total; picked a
# mix of both failure directions -- 0/6/7 grasped the back bowl, 1/4/10 grasped the front bowl).
DEFAULT_INIT_STATES = "0,6,7,1,4,10"


def run_episode(env, client, initial_state, task_description):
    import collections

    client.reset()
    obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    t = 0
    done = False
    all_objects = [TARGET_OBJECT, *DISTRACTOR_OBJECTS]
    initial_object_pos = None
    corrected_prompt = str(task_description) + SPATIAL_CORRECTION_SUFFIX

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
                "prompt": corrected_prompt,
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

    init_state_indices = [int(x) for x in os.environ.get("BOWL_CORRECTED_INIT_STATES", DEFAULT_INIT_STATES).split(",")]

    print(f"Task: {task_description}")
    print(f"Target object: {TARGET_OBJECT}  Distractors: {DISTRACTOR_OBJECTS}")
    print(f"Spatial corrected instruction: {task_description + SPATIAL_CORRECTION_SUFFIX!r}")
    print(f"Running {len(init_state_indices)} rollouts on KNOWN-FAILING init_state indices "
          f"{init_state_indices}, corrected-instruction-only (spatial, no blending)\n")

    n_success = 0
    for run_idx, state_idx in enumerate(init_state_indices):
        env.reset()
        result = run_episode(env, client, initial_states[state_idx], task_description)
        outcome = "SUCCESS" if result["done"] else "FAILURE"
        n_success += int(result["done"])
        print(
            f"  rollout {run_idx} (init_state {state_idx})\n"
            f"    outcome={outcome}  grasped={result['grasped_object']}  "
            f"displacement={ {k: round(v, 4) for k, v in result['displacement'].items()} }"
        )

    env.close()
    print(f"\n=== summary === success rate: {n_success}/{len(init_state_indices)}")


if __name__ == "__main__":
    main()
