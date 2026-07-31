"""LIBERO-side driver for the ATHENA-FEEDBACK steer-AWAY experiment (red-mug task).

This is the ATHENA-faithful counterpart to run_steered_rollout.py. Where run_steered_rollout.py steers
TOWARD a corrected/desired prompt (v_orig + gamma*(v_corrected - v_orig) -- the attract/mirror-image of
ATHENA), this drives the server's steer_mode="away" path, which implements ATHENA's actual update
(arXiv 2603.19676, Eq. 5 + Alg. 2/4): the denoiser is repelled from a CONTROL prompt naming the wrong
object the gripper is drifting toward:

    v_steered = v_original + gamma*(v_original - v_control),   then norm-matched to ||v_original||.

Feedback gating (ATHENA-Feedback analog): the fidelity signal (ground-truth object distances from the
sim, computed client-side -- never sent over the wire) is the count-mismatch check. On each replan,
`steer = STEER_ENABLED and (signal < DIVERGE_THRESH)`; when steering, the control prompt names whichever
distractor the signal currently flags as nearest (build_control_instruction), i.e. the observed-wrong
object -- the robot analog of ATHENA-Feedback replacing the target count k with the observed count c.

Env vars:
  STEER_ENABLED=true|false  -- true (default): real signal-gated away-steering. false: force steer=False
                               every replan (regression baseline; should reproduce the ~0% unsteered
                               red-mug success rate, a no-op-equivalence check on this new path).
  STEER_GAMMA               -- steering strength (default 4.0, the ATHENA paper's feedback value).
  STEER_WINDOW_STEPS        -- early-window length in steps (default 4 of 10).
  FIDELITY_NUM_ROLLOUTS, FIDELITY_ROLLOUT_OFFSET -- as in run_steered_rollout.py.

Records: per-rollout CSV (results/athena_feedback_batch_<ts>.csv) + appends a summary block to
results/athena_feedback_logs/RESULTS.md, and (verbose) prints the exact control_prompt sent each replan.
"""

import collections
import csv
import datetime
import os
import pathlib

from correction_instruction import build_control_instruction
from fidelity_signal import analyze_signal_trajectory
from fidelity_signal import compute_fidelity_signal
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
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
RECOVER_THRESH = -0.05
GAMMA = float(os.environ.get("STEER_GAMMA", 4.0))  # ATHENA paper feedback value
WINDOW_STEPS = int(os.environ.get("STEER_WINDOW_STEPS", 4))

# Target's natural-language phrase as it appears in the task instruction ("put the red mug on the left
# plate"), so build_control_instruction can swap it for the distractor phrase to form the control prompt.
TARGET_DISPLAY_NAME = "red mug"
# Distractor natural-language names -- must never pass raw ids like "white_yellow_mug_1" into a prompt
# (CLAUDE.md 2026-07-11 gibberish-instruction bug).
DISTRACTOR_DISPLAY_NAMES = {
    "porcelain_mug_1": "white mug",
    "white_yellow_mug_1": "yellow and white mug",
}

NUM_ROLLOUTS = int(os.environ.get("FIDELITY_NUM_ROLLOUTS", 12))
ROLLOUT_OFFSET = int(os.environ.get("FIDELITY_ROLLOUT_OFFSET", 0))
STEER_ENABLED = os.environ.get("STEER_ENABLED", "true").strip().lower() not in ("false", "0", "no")

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
LOG_DIR = RESULTS_DIR / "athena_feedback_logs"


def run_episode(env, client, initial_state, task_description, *, steer_enabled, verbose=False):
    """Runs one LIBERO episode against the server's ATHENA steer-away path.

    Returns dict: log (per-step signal rows), steer_log (per-replan (t, steer, control_prompt)),
    done, displacement, grasped_object -- same shape as run_steered_rollout.run_episode.
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
            control_prompt = None
            if steer:
                control_prompt = build_control_instruction(
                    str(task_description),
                    target_display=TARGET_DISPLAY_NAME,
                    distractor_display=DISTRACTOR_DISPLAY_NAMES[nearest_distractor],
                )
                element["steer_mode"] = "away"
                element["control_prompt"] = control_prompt
                element["gamma"] = GAMMA
                element["window_steps"] = WINDOW_STEPS

            steer_log.append((t, steer, control_prompt))
            if verbose:
                print(
                    f"  t={t:3d}  replan  signal={signal:+.4f}  steer={steer}  "
                    f"control_prompt={control_prompt!r}",
                    flush=True,
                )

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

    mode_label = "ATHENA steer-AWAY (feedback)" if STEER_ENABLED else "STEER-OFF regression baseline"
    print(f"Task: {task_description}")
    print(f"Target: {TARGET_OBJECT} ('{TARGET_DISPLAY_NAME}')  Distractors: {DISTRACTOR_OBJECTS}")
    print(f"Mode: {mode_label}  gamma={GAMMA}  window_steps={WINDOW_STEPS}  diverge_thresh={DIVERGE_THRESH}")
    print(f"Running {NUM_ROLLOUTS} rollouts (offset {ROLLOUT_OFFSET})\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    rows = []
    for i in range(ROLLOUT_OFFSET, ROLLOUT_OFFSET + NUM_ROLLOUTS):
        init_state = initial_states[i % len(initial_states)]
        env.reset()
        # verbose on the first rollout so the exact control_prompt strings are visible in the log.
        result = run_episode(
            env, client, init_state, task_description, steer_enabled=STEER_ENABLED, verbose=(i == ROLLOUT_OFFSET)
        )
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
            "target_displacement": result["displacement"][TARGET_OBJECT],
            "num_replans": len(steer_log),
            "num_replans_steered": len(steered_replans),
            "first_steer_step": first_steer_step,
        }
        rows.append(row)

        print(
            f"  rollout {i:2d}  outcome={row['outcome']}  grasped={row['grasped_object']}  "
            f"target_disp={row['target_displacement']:.4f}  final_signal={final_signal:+.4f}  "
            f"steered_replans={row['num_replans_steered']}/{row['num_replans']}  "
            f"first_steer_step={first_steer_step}",
            flush=True,
        )

    env.close()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "athena_feedback" if STEER_ENABLED else "athena_feedback_steeroff"
    csv_path = RESULTS_DIR / f"{tag}_batch_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-rollout results written to {csv_path}")

    n = len(rows)
    n_success = sum(1 for r in rows if r["outcome"] == "SUCCESS")
    n_target_grasped = sum(1 for r in rows if r["grasped_object"] == TARGET_OBJECT)
    n_any_steer = sum(1 for r in rows if r["num_replans_steered"] > 0)
    n_target_moved = sum(1 for r in rows if r["target_displacement"] > 1e-4)
    summary = (
        f"## {tag} batch {timestamp}\n"
        f"- task: {task_description}  (target {TARGET_OBJECT})\n"
        f"- mode: {mode_label}  gamma={GAMMA}  window_steps={WINDOW_STEPS}  diverge_thresh={DIVERGE_THRESH}\n"
        f"- rollouts: {n} (offset {ROLLOUT_OFFSET})\n"
        f"- success: {n_success}/{n} ({100 * n_success / n:.0f}%)\n"
        f"- target object grasped (max displacement): {n_target_grasped}/{n}\n"
        f"- target object moved at all (>1e-4 m): {n_target_moved}/{n}\n"
        f"- rollouts where steering triggered >=1x: {n_any_steer}/{n}\n"
        f"- csv: {csv_path.name}\n\n"
    )
    with open(LOG_DIR / "RESULTS.md", "a") as f:
        f.write(summary)

    print("\n=== aggregate summary ===")
    print(summary)


if __name__ == "__main__":
    main()
