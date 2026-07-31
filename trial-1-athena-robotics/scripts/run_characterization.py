"""Baseline-only characterisation of multi-object routines in a LIBERO scene.

Answers: across a whole episode, which objects does the policy grasp, in what
order, and where does it put them -- including objects the instruction never
names.

Two deliberate departures from `athena.runner`:

1. **No steering, no verifier.** This is characterisation. `variant="none"`.

2. **The episode does NOT stop at success.** LIBERO's `done` is just
   `_check_success()` recomputed each step, so it is safe to keep stepping.
   This matters: on task 67 (white mug -> left plate) the episode would
   terminate the instant the white mug lands, hiding anything the policy does
   afterwards -- which is exactly the secondary loop under investigation.
   We record `success_t` (first step the goal held) and `success_final` (goal
   held at the horizon) separately, so the success metric is unchanged while
   the observation window covers the full episode.

Usage:
    python scripts/run_characterization.py --tasks 67 68 65 --num-trials 5 \
        --run-id living5_characterization --port 8000
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

from athena.config import ExperimentConfig
from athena.eventlog import TABLE, InteractionTracker
from athena.perception import OracleDetector, _unwrap
from athena.runner import _get_libero_env, _quat2axisangle
from athena.taskspec import parse_suite
from athena.verifier import resolve_target

logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Gripper-object distance counting as "approached", independent of whether a
# grasp closed. Distances are eef-site -> object-body-centre, so a *grasped*
# object still sits ~0.09 m away; 0.055 (cfg.grasp_radius) would misreport a
# grasped object as untouched. Measured separation in this scene: grasped mugs
# 0.089-0.094, an untouched mug 0.185. 0.12 sits in that gap.
TOUCH_RADIUS = 0.12


def _receptacles_and_objects(env, spec) -> tuple[list[str], list[str]]:
    """Split the scene into movable objects and destination receptacles.

    Receptacles are whatever appears as the second argument of a binary goal
    predicate anywhere in the scene's task family (plate_1/plate_2 here);
    everything else with a body id is a movable object.
    """
    inner = _unwrap(env)
    all_objs = list(getattr(inner, "obj_body_id", {}).keys())
    recept = set()
    for state in inner.parsed_problem.get("goal_state", []):
        if len(state) == 3:
            recept.add(state[2])
    # Track every plate/basket in the scene, not just the one the goal names,
    # so an unprompted placement on the *other* plate is still recorded.
    for o in all_objs:
        if "plate" in o or "basket" in o:
            recept.add(o)
    # A goal may name a site region ("basket_1_contain_region") whose parent
    # body ("basket_1") is a separate entry in obj_body_id. The parent is a
    # receptacle, not something the robot picks up -- exclude it from movables.
    parents = {o for o in all_objs for r in recept if r.startswith(o) and r != o}
    recept |= parents
    objects = [o for o in all_objs if o not in recept]
    return objects, sorted(recept)


def run_episode(*, cfg, client, env, spec, task_id, task_description,
                episode_idx, initial_state, max_steps, prompt_override=None,
                video_dir=None) -> dict:
    t_start = time.perf_counter()
    env.reset()
    obs = env.set_init_state(initial_state)

    inner = _unwrap(env)
    detector = OracleDetector(env)
    objects, receptacles = _receptacles_and_objects(env, spec)
    tracker = InteractionTracker(inner, detector, objects, receptacles)

    target = resolve_target(spec)
    # The policy sees `sent_prompt`; success is still scored against the real
    # BDDL goal, so a gibberish run stays directly comparable to the language
    # run on every metric.
    sent_prompt = prompt_override if prompt_override is not None else str(task_description)

    action_plan: collections.deque = collections.deque()
    frames: list[np.ndarray] = []
    success_t: int | None = None
    success_final = False
    t = 0

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        step_idx = t - cfg.num_steps_wait

        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, cfg.resize_size, cfg.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, cfg.resize_size, cfg.resize_size)
        )

        tracker.step(step_idx)
        if video_dir is not None:
            # Full-resolution agentview, not the 224 policy input.
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ),
                "prompt": sent_prompt,
            }
            chunk = np.asarray(client.infer(element)["actions"])
            action_plan.extend(chunk[: cfg.replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(np.asarray(action).tolist())
        t += 1

        # Record when the goal first held, but keep going.
        if done and success_t is None:
            success_t = step_idx

    final_step = t - cfg.num_steps_wait
    tracker.step(final_step)
    tracker.finalize(final_step)
    success_final = bool(inner._check_success())

    first_grasp_obj, first_grasp_t = (tracker.first_grasp or (None, None))
    final_locs = tracker.final_locations()
    # "Touched" = approached within TOUCH_RADIUS *or* actually grasped, so a
    # grasp can never be missed by the distance threshold.
    grasped_ever = set(tracker.grasp_order())
    touched = {
        o: (v or o in grasped_ever) for o, v in tracker.touched(TOUCH_RADIUS).items()
    }

    named = {o for o in objects if o == target}
    unnamed = [o for o in objects if o not in named]
    unnamed_touched = {o: touched[o] for o in unnamed}
    unnamed_grasped = [o for o in tracker.grasp_order() if o in unnamed]

    video_path = None
    if video_dir is not None and frames:
        video_path = _write_video(video_dir, cfg, task_id, episode_idx,
                                  success_t is not None, frames)

    return {
        "run_id": cfg.run_id,
        "prompt_sent": sent_prompt,
        "prompt_is_override": prompt_override is not None,
        "video": video_path,
        "task_id": task_id,
        "task_name": spec.name,
        "language": spec.language,
        "episode_idx": episode_idx,
        "seed": cfg.seed,
        "target": target,
        "target_receptacle": _target_receptacle(inner),
        "objects_tracked": objects,
        "receptacles": receptacles,
        # -- the full picture --
        "full_sequence": tracker.sequence(),
        "grasp_order": tracker.grasp_order(),
        "grasp_order_min5": tracker.grasp_order(min_duration=5),
        "n_grasp_events": tracker.n_grasp_events,
        "first_grasp_object": first_grasp_obj,
        "first_grasp_timestep": first_grasp_t,
        "first_grasp_correct": (
            None if first_grasp_obj is None else first_grasp_obj == target
        ),
        "final_locations": final_locs,
        "min_gripper_dist": tracker.min_gripper_dist,
        "touched": touched,
        "was_unnamed_object_touched": any(unnamed_touched.values()),
        "unnamed_objects_touched": [o for o, v in unnamed_touched.items() if v],
        "unnamed_objects_grasped": unnamed_grasped,
        "unnamed_object_final_location": {o: final_locs[o] for o in unnamed},
        "misplaced_on_plate": [
            o for o in unnamed if final_locs[o] != TABLE
        ],
        # -- success --
        "success_t": success_t,
        "success_final": success_final,
        "success": success_t is not None,
        "steps": final_step,
        "wall_time_s": time.perf_counter() - t_start,
    }


def _write_video(video_dir, cfg, task_id, episode_idx, success, frames) -> str | None:
    suffix = "success" if success else "failure"
    path = pathlib.Path(video_dir) / (
        f"{cfg.run_id}_task{task_id:02d}_ep{episode_idx:03d}_{suffix}.mp4"
    )
    try:
        imageio.mimwrite(str(path), [np.asarray(f) for f in frames], fps=20)
        return str(path)
    except Exception:
        logger.warning("could not write video %s", path, exc_info=True)
        return None


def _target_receptacle(inner) -> str | None:
    for state in inner.parsed_problem.get("goal_state", []):
        if len(state) == 3:
            return state[2]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--num-trials", type=int, default=5)
    ap.add_argument("--run-id", default="characterization")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--suite", default="libero_90",
                    help="libero_90 | libero_object | libero_spatial | ...")
    ap.add_argument("--prompt-override", default=None,
                    help="send this string to the policy instead of the task "
                         "instruction; success is still scored against the BDDL goal")
    ap.add_argument("--save-video", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = ExperimentConfig(
        run_id=args.run_id, variant="none", host=args.host, port=args.port,
        seed=args.seed, num_trials_per_task=args.num_trials, max_steps=args.max_steps,
        task_suite_name=args.suite,
    )
    if args.prompt_override is not None:
        logger.info("PROMPT OVERRIDE ACTIVE -> %r (goal scoring unchanged)",
                    args.prompt_override)
    np.random.seed(cfg.seed)

    out_dir = pathlib.Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"

    # Resume by skip, same contract as the main runner.
    done_keys = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                d = json.loads(line)
                done_keys.add((d["task_id"], d["episode_idx"]))
            except Exception:
                continue
    if done_keys:
        logger.info("resuming: %d episodes already recorded", len(done_keys))

    video_dir = out_dir / "videos"
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    specs_by_name = {s.name: s for s in parse_suite(cfg.task_suite_name)}
    client = _wcp.WebsocketClientPolicy(cfg.host, cfg.port)
    max_steps = cfg.resolved_max_steps()

    for task_id in args.tasks:
        task = task_suite.get_task(task_id)
        spec = specs_by_name[pathlib.Path(task.bddl_file).stem]
        if all((task_id, e) in done_keys for e in range(cfg.num_trials_per_task)):
            logger.info("task %d already complete; skipping", task_id)
            continue
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.seed)
        logger.info("task %d: %r target=%s", task_id, task_description, resolve_target(spec))
        try:
            for episode_idx in range(cfg.num_trials_per_task):
                if (task_id, episode_idx) in done_keys:
                    continue
                rec = run_episode(
                    cfg=cfg, client=client, env=env, spec=spec, task_id=task_id,
                    task_description=task_description, episode_idx=episode_idx,
                    initial_state=initial_states[episode_idx % len(initial_states)],
                    max_steps=max_steps,
                    prompt_override=args.prompt_override,
                    video_dir=video_dir if args.save_video else None,
                )
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                logger.info(
                    "task=%d ep=%d success=%s(t=%s) grasp_order=%s misplaced=%s",
                    task_id, episode_idx, rec["success"], rec["success_t"],
                    rec["grasp_order"], rec["misplaced_on_plate"],
                )
        finally:
            try:
                env.close()
            except Exception:
                pass

    logger.info("characterization complete: %s", out_path)


if __name__ == "__main__":
    main()
