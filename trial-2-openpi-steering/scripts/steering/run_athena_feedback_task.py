"""Generalized ATHENA-Feedback steer-AWAY driver, task-selectable via ATHENA_TASK (athena_tasks registry).

Same mechanism as run_athena_feedback_rollout.py (which is red-mug-hardcoded), but any registered failure
case can be run without editing code:

  PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero ATHENA_TASK=middle_bowl STEER_GAMMA=4.0 \
    FIDELITY_NUM_ROLLOUTS=12 examples/libero/.venv/bin/python scripts/steering/run_athena_feedback_task.py

Feedback gating and the away-steering update are identical to run_athena_feedback_rollout.py: on each
replan, steer = STEER_ENABLED and (fidelity_signal < DIVERGE_THRESH); when steering, the control prompt
names whichever distractor is currently nearest (the observed-wrong object), and the server steers the
action velocity AWAY from it (v_orig + gamma*(v_orig - v_control), norm-matched). Records per-rollout CSV
+ a summary block in results/athena_feedback_logs/RESULTS.md.
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
from fidelity_signal import analyze_signal_trajectory, compute_fidelity_signal
from run_fidelity_rollout import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    NUM_STEPS_WAIT,
    REPLAN_STEPS,
    RESIZE_SIZE,
    SEED,
    _quat2axisangle,
)

TASK_KEY = os.environ.get("ATHENA_TASK", "red_mug")
PORT = int(os.environ.get("STEERED_POLICY_PORT", 8001))
MAX_STEPS = 400
DIVERGE_THRESH = float(os.environ.get("STEER_DIVERGE_THRESH", -0.1))
RECOVER_THRESH = -0.05
GAMMA = float(os.environ.get("STEER_GAMMA", 4.0))
WINDOW_STEPS = int(os.environ.get("STEER_WINDOW_STEPS", 4))
NUM_ROLLOUTS = int(os.environ.get("FIDELITY_NUM_ROLLOUTS", 12))
ROLLOUT_OFFSET = int(os.environ.get("FIDELITY_ROLLOUT_OFFSET", 0))
STEER_ENABLED = os.environ.get("STEER_ENABLED", "true").strip().lower() not in ("false", "0", "no")

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
LOG_DIR = RESULTS_DIR / "athena_feedback_logs"


def run_episode(env, client, initial_state, task_description, task, *, steer_enabled, verbose=False):
    client.reset()
    obs = env.set_init_state(initial_state)
    action_plan = collections.deque()
    t = 0
    done = False
    log = []
    steer_log = []
    all_objects = [task.target_object, *task.distractor_objects]
    initial_object_pos = None

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
            signal, _, (nearest_distractor, _) = compute_fidelity_signal(
                obs, task.target_object, list(task.distractor_objects)
            )
            steer = steer_enabled and signal < DIVERGE_THRESH
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
                "prompt": str(task_description),
                "steer": steer,
            }
            control_prompt = None
            if steer:
                control_prompt = build_control_instruction(
                    str(task_description), task.target_display, task.distractor_display_names[nearest_distractor]
                )
                element["steer_mode"] = "away"
                element["control_prompt"] = control_prompt
                element["gamma"] = GAMMA
                element["window_steps"] = WINDOW_STEPS
            steer_log.append((t, steer, control_prompt))
            if verbose:
                print(f"  t={t:3d}  signal={signal:+.4f}  steer={steer}  control_prompt={control_prompt!r}", flush=True)
            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[:REPLAN_STEPS])

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        signal, td, (nd, ndist) = compute_fidelity_signal(obs, task.target_object, list(task.distractor_objects))
        log.append((t, signal, td, nd, ndist))
        if done:
            break
        t += 1

    final_object_pos = {n: np.asarray(obs[f"{n}_pos"]) for n in all_objects}
    displacement = {n: float(np.linalg.norm(final_object_pos[n] - initial_object_pos[n])) for n in all_objects}
    return {
        "log": log, "steer_log": steer_log, "done": done, "displacement": displacement,
        "grasped_object": max(displacement, key=displacement.get),
    }


def main():
    np.random.seed(SEED)
    task = get_task(TASK_KEY)
    bm = benchmark.get_benchmark_dict()[task.task_suite]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == task.task_name)
    libero_task = bm.get_task(task_id)
    task_description = libero_task.language
    initial_states = bm.get_task_init_states(task_id)

    bddl = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION)
    env.seed(SEED)
    client = _wcp.WebsocketClientPolicy("0.0.0.0", PORT)

    mode = "ATHENA steer-AWAY (feedback)" if STEER_ENABLED else "STEER-OFF baseline"
    print(f"Task[{task.key}]: {task_description}  ({task.failure_case})")
    print(f"Target {task.target_object}  Distractors {list(task.distractor_objects)}")
    print(f"Mode: {mode}  gamma={GAMMA}  window={WINDOW_STEPS}  diverge_thresh={DIVERGE_THRESH}")
    print(f"Running {NUM_ROLLOUTS} rollouts (offset {ROLLOUT_OFFSET})\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    rows = []
    for i in range(ROLLOUT_OFFSET, ROLLOUT_OFFSET + NUM_ROLLOUTS):
        init_state = initial_states[i % len(initial_states)]
        env.reset()
        r = run_episode(env, client, init_state, task_description, task,
                        steer_enabled=STEER_ENABLED, verbose=(i == ROLLOUT_OFFSET))
        traj = analyze_signal_trajectory(r["log"], diverge_thresh=DIVERGE_THRESH, recover_thresh=RECOVER_THRESH)
        steered = [s for s in r["steer_log"] if s[1]]
        row = {
            "rollout": i, "init_state_idx": i % len(initial_states),
            "outcome": "SUCCESS" if r["done"] else "FAILURE",
            "grasped_object": r["grasped_object"],
            "target_displacement": round(r["displacement"][task.target_object], 4),
            "first_divergence_step": traj["first_divergence_step"],
            "num_replans": len(r["steer_log"]), "num_replans_steered": len(steered),
            "first_steer_step": steered[0][0] if steered else None,
        }
        rows.append(row)
        print(f"  rollout {i:2d}  {row['outcome']}  grasped={row['grasped_object']}  "
              f"target_disp={row['target_displacement']}  steered={row['num_replans_steered']}/{row['num_replans']}",
              flush=True)
    env.close()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"athena_feedback_{task.key}" + ("" if STEER_ENABLED else "_steeroff")
    csv_path = RESULTS_DIR / f"{tag}_batch_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    n_succ = sum(r["outcome"] == "SUCCESS" for r in rows)
    n_tgt = sum(r["grasped_object"] == task.target_object for r in rows)
    n_moved = sum(r["target_displacement"] > 1e-4 for r in rows)
    n_steer = sum(r["num_replans_steered"] > 0 for r in rows)
    summary = (
        f"## {tag} batch {ts}\n"
        f"- task[{task.key}]: {task_description}  ({task.failure_case})\n"
        f"- mode: {mode}  gamma={GAMMA}  window={WINDOW_STEPS}  diverge_thresh={DIVERGE_THRESH}\n"
        f"- rollouts: {n} (offset {ROLLOUT_OFFSET})\n"
        f"- success: {n_succ}/{n} ({100*n_succ//n}%)   target grasped: {n_tgt}/{n}   target moved(>1e-4): {n_moved}/{n}\n"
        f"- steering triggered >=1x: {n_steer}/{n}\n- csv: {csv_path.name}\n\n"
    )
    with open(LOG_DIR / "RESULTS.md", "a") as f:
        f.write(summary)
    print(f"\nPer-rollout results -> {csv_path}\n\n=== summary ===\n{summary}")


if __name__ == "__main__":
    main()
