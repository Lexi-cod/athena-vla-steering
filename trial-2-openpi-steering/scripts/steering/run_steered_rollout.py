"""LIBERO-side driver for the ATHENA-Feedback dual-instruction steering experiment.

Talks to serve_steered_policy.py (a SEPARATE server from the normal scripts/serve_policy.py /
scripts/serve_pi05_libero.sh path, which stays completely untouched) over the same
WebsocketClientPolicy used by run_fidelity_rollout.py -- just with extra optional keys in the
request dict (`steer`, `corrected_prompt`, `gamma`, `window_steps`).

The fidelity signal (ground-truth object positions from the LIBERO/robosuite sim state) is computed
entirely here, client-side, exactly as in run_fidelity_rollout.py -- never sent over the wire. Each
time the action plan is replanned (every REPLAN_STEPS env-steps), this decides whether that chunk
should be steered: `steer = steer_enabled and (signal < DIVERGE_THRESH)`, and if so, builds the
corrected instruction from whichever distractor the signal currently flags as nearest.

Run modes (env vars, mirroring run_fidelity_batch.py's FIDELITY_NUM_ROLLOUTS/FIDELITY_ROLLOUT_OFFSET):
  STEER_ENABLED=false  -- forces steer=False for every replan regardless of the fidelity signal
                          (the regression-baseline mode: should reproduce the unsteered ~0% success
                          rate and signal trajectories from the already-validated 30-rollout batch).
  STEER_ENABLED=true   -- real fidelity-signal-gated steering (default).
  FIDELITY_NUM_ROLLOUTS, FIDELITY_ROLLOUT_OFFSET -- as in run_fidelity_batch.py.
"""

import collections
import csv
import datetime
import os
import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy as _websocket_client_policy

from correction_instruction import build_corrected_instruction
from fidelity_signal import analyze_signal_trajectory, compute_fidelity_signal
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
from openpi_client import image_tools

HOST = "0.0.0.0"
PORT = int(os.environ.get("STEERED_POLICY_PORT", 8001))
MAX_STEPS = 400
DIVERGE_THRESH = -0.1
RECOVER_THRESH = -0.05
GAMMA = float(os.environ.get("STEER_GAMMA", 1.0))
WINDOW_STEPS = int(os.environ.get("STEER_WINDOW_STEPS", 4))

# Natural-language names -- matches CLAUDE.md's documented mapping. build_corrected_instruction()
# must never be called with the raw internal object id (e.g. "white_yellow_mug_1") -- see CLAUDE.md
# 2026-07-11 section for why that produced gibberish instruction text and confounded the gamma sweep.
DISTRACTOR_DISPLAY_NAMES = {
    "porcelain_mug_1": "white mug",
    "white_yellow_mug_1": "yellow and white mug",
}

NUM_ROLLOUTS = int(os.environ.get("FIDELITY_NUM_ROLLOUTS", 30))
ROLLOUT_OFFSET = int(os.environ.get("FIDELITY_ROLLOUT_OFFSET", 0))
STEER_ENABLED = os.environ.get("STEER_ENABLED", "true").strip().lower() not in ("false", "0", "no")

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def run_episode(env, client, initial_state, task_description, *, steer_enabled, verbose=False):
    """Runs one LIBERO episode against the steered policy server.

    Returns a dict with:
        log: list of (t, signal, target_dist, nearest_distractor, nearest_dist) rows (every env step,
            same format as run_fidelity_rollout.run_episode -- for direct comparability).
        steer_log: list of (t, steer, corrected_prompt) rows, one per replan call.
        done, displacement, grasped_object: as in run_fidelity_rollout.run_episode.
    """
    client.reset()
    obs = env.set_init_state(initial_state)

    action_plan = collections.deque()
    t = 0
    done = False
    log = []
    steer_log = []
    all_objects = [TARGET_OBJECT, *DISTRACTOR_OBJECTS]
    initial_object_pos = None

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
            signal, target_dist, (nearest_distractor, nearest_dist) = compute_fidelity_signal(
                obs, TARGET_OBJECT, DISTRACTOR_OBJECTS
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
            corrected_prompt = None
            if steer:
                corrected_prompt = build_corrected_instruction(
                    str(task_description),
                    target_name=TARGET_OBJECT,
                    distractor_name=DISTRACTOR_DISPLAY_NAMES[nearest_distractor],
                )
                element["corrected_prompt"] = corrected_prompt
                element["gamma"] = GAMMA
                element["window_steps"] = WINDOW_STEPS

            steer_log.append((t, steer, corrected_prompt))
            if verbose:
                print(f"  t={t:3d}  replan  signal={signal:+.4f}  steer={steer}  corrected={corrected_prompt}")

            action_chunk = client.infer(element)["actions"]
            action_plan.extend(action_chunk[:REPLAN_STEPS])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())

        signal, target_dist, (nearest_distractor, nearest_dist) = compute_fidelity_signal(
            obs, TARGET_OBJECT, DISTRACTOR_OBJECTS
        )
        log.append((t, signal, target_dist, nearest_distractor, nearest_dist))

        if done:
            break
        t += 1

    final_object_pos = {name: np.asarray(obs[f"{name}_pos"]) for name in all_objects}
    displacement = {name: float(np.linalg.norm(final_object_pos[name] - initial_object_pos[name])) for name in all_objects}
    grasped_object = max(displacement, key=displacement.get)

    return {
        "log": log,
        "steer_log": steer_log,
        "done": done,
        "displacement": displacement,
        "grasped_object": grasped_object,
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

    print(f"Task: {task_description}")
    print(f"Target object: {TARGET_OBJECT}  Distractors: {DISTRACTOR_OBJECTS}")
    print(f"STEER_ENABLED={STEER_ENABLED}  gamma={GAMMA}  window_steps={WINDOW_STEPS}  "
          f"diverge_thresh={DIVERGE_THRESH}")
    print(f"Running {NUM_ROLLOUTS} rollouts (offset {ROLLOUT_OFFSET})\n")

    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for i in range(ROLLOUT_OFFSET, ROLLOUT_OFFSET + NUM_ROLLOUTS):
        init_state = initial_states[i % len(initial_states)]
        env.reset()
        result = run_episode(env, client, init_state, task_description, steer_enabled=STEER_ENABLED, verbose=False)
        log, steer_log, done = result["log"], result["steer_log"], result["done"]

        trajectory = analyze_signal_trajectory(log, diverge_thresh=DIVERGE_THRESH, recover_thresh=RECOVER_THRESH)
        final_signal = log[-1][1]
        steered_replans = [s for s in steer_log if s[1]]
        first_steer_step = steered_replans[0][0] if steered_replans else None

        row = {
            "rollout": i,
            "init_state_idx": i % len(initial_states),
            "first_divergence_step": trajectory["first_divergence_step"],
            "recovery_step": trajectory["recovery_step"],
            "relapse_step": trajectory["relapse_step"],
            "outcome": "SUCCESS" if done else "FAILURE",
            "final_signal": final_signal,
            "grasped_object": result["grasped_object"],
            "num_replans": len(steer_log),
            "num_replans_steered": len(steered_replans),
            "first_steer_step": first_steer_step,
        }
        rows.append(row)

        print(
            f"  rollout {i:2d}  first_divergence={row['first_divergence_step']}  "
            f"recovery={row['recovery_step']}  relapse={row['relapse_step']}  "
            f"outcome={row['outcome']}  final_signal={final_signal:+.4f}  "
            f"grasped={row['grasped_object']}  steered_replans={row['num_replans_steered']}/{row['num_replans']}  "
            f"first_steer_step={first_steer_step}",
            flush=True,
        )

    env.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "steered" if STEER_ENABLED else "steeroff"
    csv_path = RESULTS_DIR / f"{tag}_batch_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-rollout results written to {csv_path}")

    n = len(rows)
    n_success = sum(1 for r in rows if r["outcome"] == "SUCCESS")
    print("\n=== aggregate summary ===")
    print(f"rollouts: {n}")
    print(f"overall success rate: {n_success}/{n} ({100 * n_success / n:.0f}%)")
    if STEER_ENABLED:
        n_any_steer = sum(1 for r in rows if r["num_replans_steered"] > 0)
        print(f"rollouts where steering triggered at least once: {n_any_steer}/{n}")


if __name__ == "__main__":
    main()
