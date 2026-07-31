"""Generalized paired same-noise-seed control for ATHENA steer-AWAY, task-selectable via ATHENA_TASK.

The causal test the raw batches can't give: for each init state, run BOTH arms with the IDENTICAL
deterministic noise per replan (noise_seed = state*100_000 + replan_idx), so any outcome difference is
attributable to the away-steering alone, not to independent stochastic draws:

  - UNSTEERED arm: plain task prompt, steer omitted, noise_seed per replan.
  - AWAY arm:      signal-gated ATHENA away-steering (steer_mode='away', control prompt naming the nearest
                   distractor when signal < DIVERGE_THRESH), SAME noise_seed per replan.

Motivation for the bowl (spatial) task specifically: the raw athena_feedback_middle_bowl batch got 4/12
success, but ALL successes were rollouts where steering barely/never fired (baseline easy cases) while
every heavily-steered rollout failed -- raw success count == baseline. This isolates whether steering
changes the outcome on the SAME (diverging) seed.

  PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero ATHENA_TASK=middle_bowl \
    examples/libero/.venv/bin/python scripts/steering/run_paired_athena_task.py
"""

import collections
import csv
import datetime
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

from athena_tasks import get_task
from correction_instruction import build_control_instruction
from fidelity_signal import compute_fidelity_signal
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
MAX_STEPS = 400
DIVERGE_THRESH = float(os.environ.get("STEER_DIVERGE_THRESH", -0.1))
GAMMA = float(os.environ.get("STEER_GAMMA", 4.0))
WINDOW_STEPS = int(os.environ.get("STEER_WINDOW_STEPS", 4))
NUM_STATES = int(os.environ.get("PAIRED_NUM_STATES", 12))

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
LOG_DIR = RESULTS_DIR / "athena_feedback_logs"


def run_one_episode(env, client, initial_state, task_description, task, all_objects, *, steer_enabled, base_seed):
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
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            if t == NUM_STEPS_WAIT:
                initial_object_pos = {n: np.asarray(obs[f"{n}_pos"]).copy() for n in all_objects}
            continue

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE))

        if not action_plan:
            signal, _, (nearest, _) = compute_fidelity_signal(obs, task.target_object, list(task.distractor_objects))
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
                    str(task_description), task.target_display, task.distractor_display_names[nearest]
                )
                element["gamma"] = GAMMA
                element["window_steps"] = WINDOW_STEPS
                n_steered += 1
            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[:REPLAN_STEPS])
            replan_idx += 1

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1

    final_object_pos = {n: np.asarray(obs[f"{n}_pos"]) for n in all_objects}
    displacement = {n: float(np.linalg.norm(final_object_pos[n] - initial_object_pos[n])) for n in all_objects}
    return {
        "done": done, "grasped_object": max(displacement, key=displacement.get),
        "target_displacement": displacement[task.target_object], "n_steered": n_steered,
    }


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
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION)
    env.seed(SEED)
    client = _wcp.WebsocketClientPolicy("0.0.0.0", PORT)

    print(f"Task[{task.key}]: {task_description}  ({task.failure_case})")
    print(f"Paired same-noise-seed: UNSTEERED vs ATHENA-away  gamma={GAMMA}  window={WINDOW_STEPS}")
    print(f"Running {NUM_STATES} init states\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    rows = []
    n_flip_win = 0   # away succeeded where unsteered failed
    n_flip_lose = 0  # away failed where unsteered succeeded (steering HURT)
    n_changed = 0
    for s in range(NUM_STATES):
        init_state = initial_states[s % len(initial_states)]
        env.reset()
        uns = run_one_episode(env, client, init_state, task_description, task, all_objects, steer_enabled=False, base_seed=s)
        env.reset()
        away = run_one_episode(env, client, init_state, task_description, task, all_objects, steer_enabled=True, base_seed=s)

        flip_win = away["done"] and not uns["done"]
        flip_lose = uns["done"] and not away["done"]
        changed = away["grasped_object"] != uns["grasped_object"] or abs(
            away["target_displacement"] - uns["target_displacement"]) > 1e-3
        n_flip_win += int(flip_win); n_flip_lose += int(flip_lose); n_changed += int(changed)

        row = {
            "init_state": s,
            "unsteered_outcome": "SUCCESS" if uns["done"] else "FAILURE", "unsteered_grasped": uns["grasped_object"],
            "away_outcome": "SUCCESS" if away["done"] else "FAILURE", "away_grasped": away["grasped_object"],
            "away_n_steered": away["n_steered"], "flip_win": flip_win, "flip_lose": flip_lose, "changed": changed,
        }
        rows.append(row)
        print(f"  init {s:2d}  unsteered=({row['unsteered_outcome']},{uns['grasped_object']})  "
              f"away=({row['away_outcome']},{away['grasped_object']},steered={away['n_steered']})  "
              f"flip_win={flip_win} flip_lose={flip_lose} changed={changed}", flush=True)
    env.close()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"paired_athena_{task.key}_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n = len(rows)
    n_uns = sum(r["unsteered_outcome"] == "SUCCESS" for r in rows)
    n_away = sum(r["away_outcome"] == "SUCCESS" for r in rows)
    summary = (
        f"## paired_athena_{task.key} (same-noise-seed) {ts}\n"
        f"- task[{task.key}]: {task_description}  ({task.failure_case})\n"
        f"- UNSTEERED vs ATHENA-away  gamma={GAMMA}  window={WINDOW_STEPS}  diverge_thresh={DIVERGE_THRESH}\n"
        f"- paired trials: {n}\n"
        f"- unsteered success: {n_uns}/{n}   away success: {n_away}/{n}   (net = {n_away - n_uns:+d})\n"
        f"- flipped TO success (steering helped, same seed): {n_flip_win}/{n}\n"
        f"- flipped TO failure (steering HURT, same seed): {n_flip_lose}/{n}\n"
        f"- behavior changed (same seed): {n_changed}/{n}\n- csv: {csv_path.name}\n\n"
    )
    with open(LOG_DIR / "RESULTS.md", "a") as f:
        f.write(summary)
    print(f"\nPer-trial -> {csv_path}\n\n=== summary (paired) ===\n{summary}")


if __name__ == "__main__":
    main()
