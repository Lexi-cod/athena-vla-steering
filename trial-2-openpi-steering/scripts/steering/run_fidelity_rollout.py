import collections
import logging
import math
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from fidelity_signal import compute_fidelity_signal

TASK_NAME = "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate"
TASK_SUITE = "libero_90"
TARGET_OBJECT = "red_coffee_mug_1"
DISTRACTOR_OBJECTS = ["porcelain_mug_1", "white_yellow_mug_1"]

HOST, PORT = "0.0.0.0", 8000
RESIZE_SIZE = 224
REPLAN_STEPS = 5
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10
MAX_STEPS = 400
SEED = 7

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def run_episode(
    env, client, initial_state, task_description, target_object=TARGET_OBJECT,
    distractor_objects=DISTRACTOR_OBJECTS, verbose=True,
):
    """Runs one LIBERO episode against a running policy server, logging the fidelity signal every step.

    target_object/distractor_objects default to this module's red-mug task constants, but can be
    overridden to reuse this driver for a different task's object-confusion signal (e.g. a bowl task) --
    see run_bowl_fidelity_batch.py.

    Returns a dict with:
        log: list of (t, signal, target_dist, nearest_distractor, nearest_dist) rows.
        done: whether the episode succeeded (BDDL goal predicate fired) before MAX_STEPS.
        displacement: {object_name: euclidean displacement from post-settle baseline to episode end}.
        grasped_object: name of the object with the largest displacement.
    """
    client.reset()
    obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    t = 0
    done = False
    log = []
    all_objects = [target_object, *distractor_objects]
    initial_object_pos = None

    if verbose:
        print(f"Task: {task_description}")
        print(f"Target object: {target_object}  Distractors: {distractor_objects}")

    while t < MAX_STEPS + NUM_STEPS_WAIT:
        if t < NUM_STEPS_WAIT:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            if t == NUM_STEPS_WAIT:
                # Objects settle under gravity in the first ~2 steps then stay static;
                # capture the post-settle baseline position for grasp-displacement tracking.
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
                "prompt": str(task_description),
            }
            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[:REPLAN_STEPS])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        signal, target_dist, (nearest_distractor, nearest_dist) = compute_fidelity_signal(
            obs, target_object, distractor_objects
        )
        log.append((t, signal, target_dist, nearest_distractor, nearest_dist))
        if verbose:
            print(
                f"  t={t:3d}  fidelity_signal={signal:+.4f}  target_dist={target_dist:.4f}  "
                f"nearest_distractor={nearest_distractor} ({nearest_dist:.4f})"
            )

        if done:
            if verbose:
                print(f"  SUCCESS at t={t}")
            break
        t += 1

    if not done and verbose:
        print("  Episode ended without success (max steps reached).")

    final_object_pos = {name: np.asarray(obs[f"{name}_pos"]) for name in all_objects}
    displacement = {name: float(np.linalg.norm(final_object_pos[name] - initial_object_pos[name])) for name in all_objects}
    grasped_object = max(displacement, key=displacement.get)

    return {"log": log, "done": done, "displacement": displacement, "grasped_object": grasped_object}


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
    env.reset()

    client = _websocket_client_policy.WebsocketClientPolicy(HOST, PORT)

    result = run_episode(env, client, initial_states[0], task_description, verbose=True)
    log, done = result["log"], result["done"]

    signals = np.array([row[1] for row in log])
    print("\n=== summary ===")
    print(f"steps logged: {len(log)}")
    print(f"fidelity_signal: min={signals.min():.4f} max={signals.max():.4f} mean={signals.mean():.4f}")
    print(f"final fidelity_signal: {signals[-1]:+.4f}")
    print(f"outcome: {'SUCCESS' if done else 'FAILURE (max steps reached)'}")

    print("object displacement (settle baseline -> final, ground-truth sim state):")
    for name in [TARGET_OBJECT, *DISTRACTOR_OBJECTS]:
        marker = "  <-- likely grasped/moved" if name == result["grasped_object"] else ""
        print(f"  {name}: {result['displacement'][name]:.4f} m{marker}")
    print(f"object that actually moved most: {result['grasped_object']}")

    env.close()


if __name__ == "__main__":
    main()
