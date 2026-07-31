"""Localize the paired-control determinism leak (PROGRESS.md section 3).

Symptom: in paired_athena_middle_bowl_20260720_134145.csv, init states 3/8/9/10 have
away_n_steered=0 -- steering never fired -- yet all are marked changed=True, and state 10 flipped
FAILURE->SUCCESS. With identical base_seed and identical per-replan noise_seed and ZERO steering
interventions, both arms should be bit-identical. They are not.

This runs the SAME unsteered arm TWICE with identical seeds, through the exact env.reset() ->
set_init_state() sequence run_paired_athena_task.main() uses, and diffs per replan:

  - the observation fed to the server (image bytes + state vector)
  - the action chunk returned by the server

First divergence tells us where the leak is:
  * actions differ but observations identical  -> server-side RNG not covered by noise_seed
  * observations differ at replan 0            -> env.reset()/set_init_state not restoring state
  * observations differ later, obs 0 identical -> sim stepping is nondeterministic

Run (server must be up on :8001):
  PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero ATHENA_TASK=middle_bowl \
    examples/libero/.venv/bin/python scripts/steering/diagnose_paired_determinism.py
"""

import collections
import hashlib
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

from athena_tasks import get_task
from run_fidelity_rollout import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    NUM_STEPS_WAIT,
    REPLAN_STEPS,
    RESIZE_SIZE,
    SEED,
    _quat2axisangle,
)

TASK_KEY = os.environ.get("ATHENA_TASK", "middle_bowl")
PORT = int(os.environ.get("STEERED_POLICY_PORT", 8001))
INIT_STATE = int(os.environ.get("DIAG_INIT_STATE", 10))  # the state that flipped with 0 steering
MAX_REPLANS = int(os.environ.get("DIAG_MAX_REPLANS", 12))
OBJECTS = []  # filled in main() from the task registry


def _h(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def run_unsteered(env, client, initial_state, task_description, base_seed):
    """Unsteered episode, capped at MAX_REPLANS. Records per-replan fingerprints."""
    client.reset()
    obs = env.set_init_state(initial_state)
    action_plan = collections.deque()
    t = 0
    replan_idx = 0
    trace = []

    while replan_idx < MAX_REPLANS:
        if t < NUM_STEPS_WAIT:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE))

        if not action_plan:
            state_vec = np.concatenate(
                (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": state_vec,
                "prompt": str(task_description),
                "noise_seed": base_seed * 100_000 + replan_idx,
            }
            actions = np.asarray(client.infer(element)["actions"])
            trace.append(
                {
                    "replan": replan_idx,
                    "t": t,
                    "img": _h(img),
                    "wrist": _h(wrist),
                    # keep raw pixels for the first few replans so we can tell trivial renderer
                    # noise (max diff 1-2 grey levels) from a genuinely different frame.
                    "img_raw": img.copy() if replan_idx < 3 else None,
                    "state": state_vec.copy(),
                    # object world positions -- the robot state vector alone can be identical while
                    # the OBJECTS sit elsewhere, which is exactly what a big image diff at replan 0
                    # with state maxdiff 0.0 would mean.
                    "objects": {n: np.asarray(obs[f"{n}_pos"]).copy() for n in OBJECTS},
                    "joints": np.asarray(obs.get("robot0_joint_pos", [])).copy(),
                    "actions": actions.copy(),
                }
            )
            action_plan.extend(actions[:REPLAN_STEPS])
            replan_idx += 1

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    return trace


def main():
    np.random.seed(SEED)
    task = get_task(TASK_KEY)
    bm = benchmark.get_benchmark_dict()[task.task_suite]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == task.task_name)
    libero_task = bm.get_task(task_id)
    task_description = libero_task.language
    initial_states = bm.get_task_init_states(task_id)
    init_state = initial_states[INIT_STATE % len(initial_states)]

    bddl = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION
    )
    env.seed(SEED)
    client = _wcp.WebsocketClientPolicy("0.0.0.0", PORT)
    global OBJECTS
    OBJECTS = [task.target_object, *task.distractor_objects]

    print(f"\n=== DETERMINISM DIAGNOSTIC  task={task.key}  init_state={INIT_STATE}  seed={SEED} ===")
    print(f"Two identical UNSTEERED runs, same noise_seed per replan, {MAX_REPLANS} replans each.\n")

    env.reset()
    a = run_unsteered(env, client, init_state, task_description, base_seed=INIT_STATE)
    env.reset()
    b = run_unsteered(env, client, init_state, task_description, base_seed=INIT_STATE)

    print(f"{'replan':>6} {'img A/B':>28} {'state maxdiff':>14} {'action maxdiff':>15}")
    first_obs_div = None
    first_act_div = None
    for ra, rb in zip(a, b):
        img_same = ra["img"] == rb["img"] and ra["wrist"] == rb["wrist"]
        sdiff = float(np.max(np.abs(ra["state"] - rb["state"])))
        adiff = float(np.max(np.abs(ra["actions"] - rb["actions"])))
        flag = "" if (img_same and sdiff == 0 and adiff == 0) else "  <-- DIVERGES"
        print(
            f"{ra['replan']:>6} {ra['img'][:6]}/{rb['img'][:6]}{'  same' if img_same else '  DIFF':>10}"
            f"{sdiff:>16.3e}{adiff:>15.3e}{flag}"
        )
        if first_obs_div is None and (not img_same or sdiff > 0):
            first_obs_div = ra["replan"]
        if first_act_div is None and adiff > 0:
            first_act_div = ra["replan"]

    print("\n=== OBJECT POSITIONS (first replans) ===")
    for ra, rb in zip(a[:3], b[:3]):
        parts = []
        for n in OBJECTS:
            d = float(np.max(np.abs(ra["objects"][n] - rb["objects"][n])))
            parts.append(f"{n}={d:.2e}")
        print(f"  replan {ra['replan']}: " + "  ".join(parts))

    print("\n=== ROBOT JOINT ANGLES (first replans) ===")
    for ra, rb in zip(a[:3], b[:3]):
        if ra["joints"].size == 0:
            print("  (robot0_joint_pos not in obs)"); break
        d = np.abs(ra["joints"] - rb["joints"])
        print(f"  replan {ra['replan']}: maxdiff={d.max():.3e}  per-joint={np.round(d,5)}")

    print("\n=== IMAGE PIXEL DIFF (first replans) ===")
    for ra, rb in zip(a[:3], b[:3]):
        if ra["img_raw"] is None or rb["img_raw"] is None:
            continue
        d = np.abs(ra["img_raw"].astype(np.int16) - rb["img_raw"].astype(np.int16))
        npx = int((d > 0).sum())
        print(
            f"  replan {ra['replan']}: max={d.max():>3d} grey levels  mean={d.mean():.4f}  "
            f"differing px={npx}/{d.size} ({100.0 * npx / d.size:.2f}%)"
        )

    print("\n=== VERDICT ===")
    if first_obs_div is None and first_act_div is None:
        print("Fully deterministic across both runs -- the leak is NOT here.")
        print("Next suspect: the steered arm's extra server calls advancing shared state,")
        print("or env.reset() differing only after longer horizons. Re-run with larger DIAG_MAX_REPLANS.")
    elif first_act_div is not None and first_obs_div is None:
        print(f"Observations identical but ACTIONS differ from replan {first_act_div}.")
        print("=> server-side nondeterminism NOT covered by noise_seed (jit/RNG in the infer path).")
    elif first_obs_div == 0:
        print("Observations differ at the very FIRST replan.")
        print("=> env.reset()/set_init_state is not restoring identical sim state.")
    else:
        print(f"Observations identical at replan 0, diverge at replan {first_obs_div}.")
        print("=> sim stepping is nondeterministic (or actions diverged first and drove it).")


if __name__ == "__main__":
    main()
