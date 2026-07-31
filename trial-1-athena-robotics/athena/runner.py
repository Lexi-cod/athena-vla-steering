"""The closed-loop predict -> verify -> re-steer -> execute runner.

Structure mirrors examples/libero/main.py so results stay comparable to the
existing openpi baseline; the additions are the verification hook, the
re-steering hook, and per-episode metric recording.
"""

from __future__ import annotations

import collections
import logging
import math
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

from athena.config import ExperimentConfig
from athena.metrics import EpisodeRecord, MetricsWriter, Timer
from athena.perception import NoisyDetector, OracleDetector
from athena.steering import DualPlanSteering, build_steering
from athena.taskspec import TaskSpec, parse_suite
from athena.verifier import StateVerifier, VerdictKind

logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _quat2axisangle(quat):
    """From robosuite transform_utils."""
    quat = np.asarray(quat, dtype=float).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)  # seed affects object placement even with fixed init states
    return env, task.language, str(task_bddl_file)


def select_task_ids(cfg: ExperimentConfig, specs_by_name: dict[str, TaskSpec],
                    task_suite) -> list[int]:
    """Resolve cfg.task_subset into concrete task ids of the suite."""
    n = task_suite.n_tasks
    if cfg.task_subset == "all":
        return list(range(n))

    scored: list[tuple[int, int]] = []
    for tid in range(n):
        task = task_suite.get_task(tid)
        spec = specs_by_name.get(pathlib.Path(task.bddl_file).stem)
        scored.append((tid, spec.confusion_score if spec else 0))

    confusable = [tid for tid, s in scored if s > 0]
    if cfg.task_subset == "confusable":
        return confusable
    if cfg.task_subset == "pilot":
        ranked = sorted(scored, key=lambda t: (-t[1], t[0]))
        return sorted(tid for tid, s in ranked[: cfg.pilot_size] if s > 0)
    raise ValueError(f"unknown task_subset {cfg.task_subset!r}")


def run(cfg: ExperimentConfig) -> pathlib.Path:
    np.random.seed(cfg.seed)

    out_dir = pathlib.Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "episodes.jsonl"
    writer = MetricsWriter(metrics_path)
    logger.info("run_id=%s variant=%s -> %s", cfg.run_id, cfg.variant, metrics_path)
    if writer.n_done:
        logger.info("resuming: %d episodes already recorded", writer.n_done)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    specs = parse_suite(cfg.task_suite_name)
    specs_by_name = {s.name: s for s in specs}

    task_ids = select_task_ids(cfg, specs_by_name, task_suite)
    logger.info("evaluating %d tasks (subset=%s)", len(task_ids), cfg.task_subset)

    client = _wcp.WebsocketClientPolicy(cfg.host, cfg.port)
    max_steps = cfg.resolved_max_steps()

    video_dir = out_dir / "videos"
    if cfg.save_video or cfg.save_video_on_failure:
        video_dir.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        spec = specs_by_name.get(pathlib.Path(task.bddl_file).stem)
        if spec is None:
            logger.warning("no BDDL spec for task %d (%s); skipping", task_id, task.bddl_file)
            continue

        # Skip the whole task if every episode is already recorded.
        if all(writer.already_done(task_id, e) for e in range(cfg.num_trials_per_task)):
            logger.info("task %d already complete; skipping", task_id)
            continue

        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.seed)
        videos_saved = 0

        try:
            for episode_idx in range(cfg.num_trials_per_task):
                if writer.already_done(task_id, episode_idx):
                    continue
                rec = _run_episode(
                    cfg=cfg,
                    client=client,
                    env=env,
                    spec=spec,
                    task_id=task_id,
                    task_description=task_description,
                    episode_idx=episode_idx,
                    initial_state=initial_states[episode_idx % len(initial_states)],
                    max_steps=max_steps,
                    video_dir=video_dir,
                    allow_video=(videos_saved < cfg.video_max_per_task),
                )
                if rec.error is None and (cfg.save_video or
                                          (cfg.save_video_on_failure and not rec.success)):
                    videos_saved += 1
                writer.write(rec)
                logger.info(
                    "task=%d ep=%d success=%s grasp_correct=%s steers=%d",
                    task_id, episode_idx, rec.success, rec.grasped_correct,
                    rec.n_steer_events,
                )
        finally:
            try:
                env.close()
            except Exception:
                pass

    logger.info("run complete: %s (%d episodes)", metrics_path, writer.n_done)
    return metrics_path


def _run_episode(
    *, cfg, client, env, spec, task_id, task_description, episode_idx,
    initial_state, max_steps, video_dir, allow_video,
) -> EpisodeRecord:
    rec = EpisodeRecord(
        run_id=cfg.run_id,
        variant=cfg.variant,
        task_id=task_id,
        task_name=spec.name,
        language=spec.language,
        episode_idx=episode_idx,
        seed=cfg.seed,
        confusion_score=spec.confusion_score,
        n_confusable_distractors=len(spec.confusable_distractors),
    )

    t_start = time.perf_counter()
    policy_timer, verify_timer = Timer(), Timer()

    try:
        env.reset()
        obs = env.set_init_state(initial_state)

        base_detector = OracleDetector(env)
        if cfg.detector == "noisy":
            detector = NoisyDetector(
                base_detector,
                p_miss=cfg.p_miss,
                p_swap=cfg.p_swap,
                pos_noise_std=cfg.pos_noise_std,
                seed=cfg.seed * 1000 + task_id * 100 + episode_idx,
            )
        else:
            detector = base_detector

        verifier = StateVerifier(
            spec, detector,
            tau_obj=cfg.tau_obj,
            grasp_radius=cfg.grasp_radius,
            ambiguity_margin=cfg.ambiguity_margin,
        )
        steering = build_steering(
            cfg.variant, spec,
            max_retries=cfg.max_retries,
            mode=cfg.dual_mode,
            alpha=cfg.dual_alpha,
        )
        steering.reset()

        rec.target = verifier.target
        current_prompt = str(task_description)
        rec.prompts_used.append(current_prompt)

        action_plan: collections.deque = collections.deque()
        replay_images: list[np.ndarray] = []
        done = False
        t = 0

        while t < max_steps + cfg.num_steps_wait:
            # Let objects settle before doing anything.
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, cfg.resize_size, cfg.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, cfg.resize_size, cfg.resize_size)
            )
            replay_images.append(img)

            # ---------------- verify ----------------
            steer_now = False
            verdict = None
            if cfg.variant != "none" and (t - cfg.num_steps_wait) % cfg.verify_every == 0:
                with verify_timer:
                    if cfg.verify_mode == "late":
                        verdict = verifier.verify_effect()
                    else:
                        verdict = verifier.verify_precondition()
                rec.n_verifications += 1
                if not verdict.ok:
                    rec.n_failed_verifications += 1
                if steering.should_intervene(verdict):
                    steer_now = True

            # Record the first object actually grasped — the object-accuracy metric.
            if rec.first_grasped is None:
                with verify_timer:
                    held = detector.grasped_object()
                if held is not None:
                    rec.first_grasped = held
                    rec.grasped_correct = held == verifier.target
                    rec.grasped_confusable = held in verifier.confusable

            # ---------------- re-steer ----------------
            if steer_now:
                res = steering.steer(verdict, current_prompt)
                if res.prompt != current_prompt:
                    current_prompt = res.prompt
                    rec.prompts_used.append(current_prompt)
                rec.n_steer_events += 1
                rec.max_escalation = max(rec.max_escalation, res.escalation)
                rec.steer_notes.append(res.note)
                # Drop the queued chunk: it was computed under the old prompt,
                # and the whole point is to not execute it.
                action_plan.clear()

            # ---------------- predict ----------------
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
                    "prompt": current_prompt,
                }

                with policy_timer:
                    chunk = np.asarray(client.infer(element)["actions"])
                    rec.policy_calls += 1

                # Dual-plan: a second pass under the original instruction,
                # then combine — ATHENA's two-forward-pass mixing.
                if (
                    isinstance(steering, DualPlanSteering)
                    and current_prompt != str(task_description)
                ):
                    element_orig = dict(element, prompt=str(task_description))
                    with policy_timer:
                        chunk_orig = np.asarray(client.infer(element_orig)["actions"])
                        rec.policy_calls += 1
                    tgt = detector.detect(verifier.target) if verifier.target else None
                    steering.set_geometry(
                        detector.gripper_pos(), tgt.pos if tgt else None
                    )
                    chunk = steering.combine(
                        chunk_orig, chunk, verdict or _neutral_verdict()
                    )

                if len(chunk) < cfg.replan_steps:
                    raise RuntimeError(
                        f"policy returned {len(chunk)} steps, need {cfg.replan_steps}"
                    )
                action_plan.extend(chunk[: cfg.replan_steps])

            # ---------------- execute ----------------
            action = action_plan.popleft()
            obs, reward, done, info = env.step(np.asarray(action).tolist())
            t += 1
            if done:
                break

        rec.success = bool(done)
        rec.steps = t - cfg.num_steps_wait

        # Final grasp check, in case the grasp happened on the last step.
        if rec.first_grasped is None:
            held = detector.grasped_object()
            if held is not None:
                rec.first_grasped = held
                rec.grasped_correct = held == verifier.target
                rec.grasped_confusable = held in verifier.confusable

        if allow_video and (cfg.save_video or (cfg.save_video_on_failure and not rec.success)):
            _write_video(video_dir, cfg, task_id, episode_idx, rec.success, replay_images)

    except Exception as e:  # noqa: BLE001 - one bad episode must not kill the run
        logger.exception("task=%d ep=%d failed", task_id, episode_idx)
        rec.error = f"{type(e).__name__}: {e}"

    rec.policy_time_s = policy_timer.total
    rec.verify_time_s = verify_timer.total
    rec.wall_time_s = time.perf_counter() - t_start
    return rec


def _neutral_verdict():
    from athena.verifier import Verdict
    return Verdict(VerdictKind.NOT_APPLICABLE, "no verdict this step")


def _write_video(video_dir, cfg, task_id, episode_idx, success, frames) -> None:
    if not frames:
        return
    suffix = "success" if success else "failure"
    path = pathlib.Path(video_dir) / (
        f"{cfg.run_id}_task{task_id:02d}_ep{episode_idx:03d}_{suffix}.mp4"
    )
    try:
        imageio.mimwrite(path, [np.asarray(f) for f in frames], fps=10)
    except Exception:
        logger.warning("could not write video %s", path, exc_info=True)
