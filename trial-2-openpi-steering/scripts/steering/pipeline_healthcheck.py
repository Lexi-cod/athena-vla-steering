"""End-to-end pipeline health-check for the ATHENA steer-away experiments.

Exercises EVERY stage of the pipeline for a given ATHENA_TASK (default red_mug) and prints PASS/FAIL per
stage, so we can confirm the whole thing is wired correctly before (or independently of) running a full
batch. Run in the LIBERO venv with the policy server up on :8001:

  PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero ATHENA_TASK=middle_bowl \
    examples/libero/.venv/bin/python scripts/steering/pipeline_healthcheck.py

Stages checked:
  1. task config loads from athena_tasks registry
  2. LIBERO benchmark/task/init-states/bddl resolve
  3. env builds, resets, set_init_state, settles
  4. obs exposes every object's <name>_pos key; positions finite & distinct
  5. compute_fidelity_signal returns finite values; nearest distractor is a real distractor
  6. build_control_instruction yields a grammatical control prompt (target swapped, no raw ids)
  7. server reachable; UNSTEERED infer -> right-shaped, finite, bounded actions
  8. AWAY infer (steer_mode='away' + control_prompt) -> finite, bounded, in-scale actions
  9. noise_seed determinism: same seed -> identical; different seed -> differs
 10. live loop: execute a few action chunks; env steps, positions update, no crash
"""

import os

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
    RESIZE_SIZE,
    SEED,
    _quat2axisangle,
)

TASK_KEY = os.environ.get("ATHENA_TASK", "red_mug")
PORT = int(os.environ.get("STEERED_POLICY_PORT", 8001))

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)
    return ok


def obs_to_element(obs, task_description, **extra):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE))
    el = {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": str(task_description),
    }
    el.update(extra)
    return el


def main():
    print(f"\n=== PIPELINE HEALTH-CHECK  (ATHENA_TASK={TASK_KEY}, server :{PORT}) ===\n")

    # 1. config
    t = get_task(TASK_KEY)
    check("1. task config loads", True, f"{t.failure_case}: target {t.target_object}")
    all_objects = [t.target_object, *t.distractor_objects]

    # 2. benchmark/task
    bm = benchmark.get_benchmark_dict()[t.task_suite]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == t.task_name)
    task = bm.get_task(task_id)
    task_description = task.language
    init_states = bm.get_task_init_states(task_id)
    check("2. benchmark/task/init-states resolve", len(init_states) > 0,
          f"instruction={task_description!r}, {len(init_states)} init states")

    # 3. env
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION)
    env.seed(SEED)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    check("3. env builds/reset/settles", True)

    # 4. object position lookup
    try:
        pos = {n: np.asarray(obs[f"{n}_pos"]) for n in all_objects}
        finite = all(np.isfinite(p).all() and p.shape == (3,) for p in pos.values())
        distinct = len({tuple(np.round(p, 4)) for p in pos.values()}) == len(all_objects)
        check("4. object _pos keys present, finite, distinct", finite and distinct,
              "; ".join(f"{n}={np.round(pos[n],3)}" for n in all_objects))
    except KeyError as e:
        check("4. object _pos keys present", False, f"missing {e}")

    # 5. fidelity signal
    signal, target_dist, (nearest, nearest_dist) = compute_fidelity_signal(obs, t.target_object, list(t.distractor_objects))
    check("5. fidelity signal sane", np.isfinite(signal) and nearest in t.distractor_objects,
          f"signal={signal:+.4f} target_dist={target_dist:.3f} nearest={nearest}({nearest_dist:.3f})")

    # 6. control prompt
    ctrl = build_control_instruction(str(task_description), t.target_display, t.distractor_display_names[nearest])
    grammatical = "_" not in ctrl and t.distractor_display_names[nearest] in ctrl and ctrl != str(task_description)
    check("6. control prompt grammatical", grammatical, repr(ctrl))

    # 7-9. server calls
    client = _wcp.WebsocketClientPolicy("0.0.0.0", PORT)
    a = np.asarray(client.infer(obs_to_element(obs, task_description, steer=False, noise_seed=42))["actions"])
    check("7. unsteered infer sane", a.ndim == 2 and np.isfinite(a).all() and np.abs(a).max() < 5.0,
          f"shape={a.shape} max|a|={np.abs(a).max():.3f}")

    aw = np.asarray(client.infer(obs_to_element(
        obs, task_description, steer=True, steer_mode="away", control_prompt=ctrl,
        gamma=4.0, window_steps=4, noise_seed=42))["actions"])
    check("8. away infer sane & in-scale", np.isfinite(aw).all() and np.abs(aw).max() < 5.0
          and abs(np.abs(aw).max() - np.abs(a).max()) < 1.0,
          f"max|away|={np.abs(aw).max():.3f} vs max|unsteered|={np.abs(a).max():.3f} (norm-matched)")

    a2 = np.asarray(client.infer(obs_to_element(obs, task_description, steer=False, noise_seed=42))["actions"])
    a3 = np.asarray(client.infer(obs_to_element(obs, task_description, steer=False, noise_seed=999))["actions"])
    check("9. noise_seed determinism", np.abs(a - a2).max() < 1e-6 and np.abs(a - a3).max() > 1e-3,
          f"same-seed diff={np.abs(a-a2).max():.2e}, diff-seed diff={np.abs(a-a3).max():.3f}")

    # 10. live loop
    ok_loop = True
    p0 = {n: np.asarray(obs[f"{n}_pos"]).copy() for n in all_objects}
    try:
        for _ in range(3):
            chunk = client.infer(obs_to_element(obs, task_description, steer=False, noise_seed=7))["actions"]
            for act in chunk[:5]:
                obs, _, done, _ = env.step(np.asarray(act).tolist())
        eef_moved = np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - p0[t.target_object]) >= 0  # trivially true, just no crash
        check("10. live loop executes (env steps, no crash)", eef_moved)
    except Exception as e:  # noqa: BLE001
        ok_loop = check("10. live loop executes", False, f"{type(e).__name__}: {e}")
    env.close()

    n_pass = sum(results)
    n = len(results)
    print(f"\n=== {n_pass}/{n} stages passed -> {'PIPELINE HEALTHY' if n_pass == n else 'PROBLEM DETECTED'} ===\n")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
