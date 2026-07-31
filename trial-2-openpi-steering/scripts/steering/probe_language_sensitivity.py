"""Which KINDS of language actually move pi0.5's actions? (PROGRESS.md TODO 5)

Cheap diagnostic, no rollouts. Rationale: every steering result so far is null, but
test_real_weights.py previously measured cosine 0.815 between "move the arm up" and "move the arm
down" velocities -- i.e. the denoiser reads MOTION language strongly -- while our colour/identity
corrections did exactly nothing. If spatial/directional language has a real lever and attribute
language does not, then corrections should be phrased spatially, which stays inside the
"language, in action space" constraint.

Measurement: hold the observation and the noise_seed FIXED and change only the prompt. The size of
the resulting action difference is how much that language distinction moves the policy. Uses the
running server, so no second checkpoint load (the server already holds ~24.7GB).

Reported per contrast:
  rel_L2 = ||A_1 - A_2|| / ||A_1||   (0 = prompt made no difference at all)
  max_abs = max |A_1 - A_2|

Controls:
  IDENTICAL   -- same prompt twice, same seed. Must be ~0, else the measurement is meaningless.
  MOTION      -- "move the arm up" vs "move the arm down". Known-large from test_real_weights.py;
                 the positive control that proves the probe can detect a real effect.

  PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero ATHENA_TASK=middle_bowl \
    examples/libero/.venv/bin/python scripts/steering/probe_language_sensitivity.py
"""

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
    RESIZE_SIZE,
    SEED,
    _quat2axisangle,
)

TASK_KEY = os.environ.get("ATHENA_TASK", "middle_bowl")
PORT = int(os.environ.get("STEERED_POLICY_PORT", 8001))
INIT_STATE = int(os.environ.get("PROBE_INIT_STATE", 0))
NOISE_SEED = 424242

# Contrast sets per task: (label, prompt_a, prompt_b, kind)
CONTRASTS = {
    "middle_bowl": [
        ("IDENTICAL (control)", "put the middle black bowl on the plate", "put the middle black bowl on the plate", "control"),
        ("MOTION up/down (control)", "move the arm up", "move the arm down", "motion"),
        ("SPATIAL front/back", "put the front black bowl on the plate", "put the back black bowl on the plate", "spatial"),
        ("SPATIAL middle/front", "put the middle black bowl on the plate", "put the front black bowl on the plate", "spatial"),
        ("SPATIAL left/right", "put the left black bowl on the plate", "put the right black bowl on the plate", "spatial"),
        ("DIRECTIONAL reach", "reach to the left", "reach to the right", "motion"),
    ],
    "red_mug": [
        ("IDENTICAL (control)", "put the red mug on the left plate", "put the red mug on the left plate", "control"),
        ("MOTION up/down (control)", "move the arm up", "move the arm down", "motion"),
        ("ATTRIBUTE red/white", "put the red mug on the left plate", "put the white mug on the left plate", "attribute"),
        ("ATTRIBUTE red/yellow", "put the red mug on the left plate", "put the yellow and white mug on the left plate", "attribute"),
        ("SPATIAL left/right plate", "put the red mug on the left plate", "put the red mug on the right plate", "spatial"),
        ("DIRECTIONAL reach", "reach to the left", "reach to the right", "motion"),
    ],
    "orange_juice": [
        ("IDENTICAL (control)", "pick up the orange juice and put it in the basket", "pick up the orange juice and put it in the basket", "control"),
        ("MOTION up/down (control)", "move the arm up", "move the arm down", "motion"),
        ("IDENTITY juice/milk", "pick up the orange juice and put it in the basket", "pick up the milk and put it in the basket", "attribute"),
        ("IDENTITY juice/ketchup", "pick up the orange juice and put it in the basket", "pick up the ketchup and put it in the basket", "attribute"),
        ("DIRECTIONAL reach", "reach to the left", "reach to the right", "motion"),
    ],
}


def main():
    np.random.seed(SEED)
    task = get_task(TASK_KEY)
    bm = benchmark.get_benchmark_dict()[task.task_suite]()
    task_id = next(i for i in range(bm.get_num_tasks()) if bm.get_task(i).name == task.task_name)
    libero_task = bm.get_task(task_id)
    initial_states = bm.get_task_init_states(task_id)

    bddl = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION
    )
    env.seed(SEED)
    env.reset()
    obs = env.set_init_state(initial_states[INIT_STATE % len(initial_states)])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, RESIZE_SIZE, RESIZE_SIZE))
    state_vec = np.concatenate(
        (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    client = _wcp.WebsocketClientPolicy("0.0.0.0", PORT)

    def actions_for(prompt):
        return np.asarray(
            client.infer(
                {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": state_vec,
                    "prompt": prompt,
                    "noise_seed": NOISE_SEED,
                }
            )["actions"]
        )

    print(f"\n=== LANGUAGE SENSITIVITY PROBE  task={task.key}  init_state={INIT_STATE} ===")
    print("Same observation, same noise_seed; ONLY the prompt changes.\n")
    print(f"{'contrast':<30}{'kind':<11}{'rel_L2':>10}{'max_abs':>10}")
    print("-" * 61)

    results = []
    for label, pa, pb, kind in CONTRASTS[task.key]:
        a, b = actions_for(pa), actions_for(pb)
        denom = np.linalg.norm(a)
        rel = float(np.linalg.norm(a - b) / denom) if denom > 0 else float("nan")
        results.append((label, kind, rel))
        print(f"{label:<30}{kind:<11}{rel:>10.4f}{float(np.max(np.abs(a - b))):>10.4f}")

    print("\n=== INTERPRETATION ===")
    ctrl = [r for lbl, k, r in results if k == "control"]
    motion = [r for lbl, k, r in results if k == "motion"]
    spatial = [r for lbl, k, r in results if k == "spatial"]
    attribute = [r for lbl, k, r in results if k == "attribute"]

    if ctrl and ctrl[0] > 1e-6:
        print(f"WARNING: identical-prompt control is {ctrl[0]:.2e}, not ~0 -- probe is unreliable.")
    for name, vals in (("motion", motion), ("spatial", spatial), ("attribute", attribute)):
        if vals:
            print(f"  {name:<10} mean rel_L2 = {np.mean(vals):.4f}   (n={len(vals)})")
    if spatial and attribute:
        print(
            f"\n  spatial/attribute ratio = {np.mean(spatial) / max(np.mean(attribute), 1e-9):.2f}x"
            "  -- >>1 means spatial language is the lever attribute language isn't."
        )


if __name__ == "__main__":
    main()
