#!/usr/bin/env python
"""Live smoke test: instantiate a real LIBERO env and exercise perception.

Runs no policy — it drives the arm with scripted actions toward a known object
and checks that OracleDetector reports sane poses, a working gripper position,
and that the verifier fires WRONG_OBJECT when we deliberately reach for a
distractor.

  MUJOCO_GL=egl python tests/test_perception_live.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from libero.libero import benchmark  # noqa: E402

from athena.perception import NoisyDetector, OracleDetector  # noqa: E402
from athena.runner import LIBERO_ENV_RESOLUTION, _get_libero_env  # noqa: E402
from athena.taskspec import parse_suite  # noqa: E402
from athena.verifier import StateVerifier, VerdictKind  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILURES.append(msg)


def main() -> int:
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    specs = {s.name: s for s in parse_suite("libero_90")}

    # Task 13: "put the black bowl at the front on the plate" -- three
    # identical black bowls, the canonical confusion case.
    task_id = 13
    task = suite.get_task(task_id)
    spec = specs[pathlib.Path(task.bddl_file).stem]
    print(f"\n=== task {task_id}: {spec.language} ===")
    print(f"target={spec.obj_of_interest}  confusable={spec.confusable_distractors}")

    env, desc, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed=7)
    init_states = suite.get_task_init_states(task_id)
    env.reset()
    obs = env.set_init_state(init_states[0])

    # Settle the scene.
    for _ in range(20):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

    det = OracleDetector(env)

    print("\n--- detector ---")
    dets = det.detect_all()
    check(len(dets) > 0, f"detect_all returned {len(dets)} objects")
    for name, d in sorted(dets.items()):
        print(f"    {name:34s} pos={np.round(d.pos, 3)}")

    for obj in spec.objects:
        check(obj in dets, f"instantiated object {obj!r} is detectable")

    gp = det.gripper_pos()
    print(f"\n--- gripper ---\n    pos={np.round(gp, 3)}")
    check(gp.shape == (3,), "gripper_pos returns a 3-vector")
    check(bool(np.any(gp != 0)), "gripper_pos is not the zero fallback")
    check(0.0 < float(np.linalg.norm(gp)) < 3.0, "gripper_pos is in a plausible range")

    check(det.grasped_object() is None, "nothing grasped at rest")

    near, dist = det.nearest_object()
    print(f"    nearest={near} dist={dist:.3f}")
    check(np.isfinite(dist), "nearest_object returns a finite distance")

    # --- verifier at rest: gripper is in free space, so nothing to judge ---
    print("\n--- verifier (arm at rest) ---")
    v = StateVerifier(spec, det)
    check(v.target in spec.objects, f"resolved target {v.target!r} is a real object")
    verdict = v.verify_precondition()
    print(f"    {verdict.kind.value}: {verdict.reason}")
    check(verdict.ok, "no false alarm while the gripper is in free space")

    # --- drive the gripper down onto a DISTRACTOR and expect WRONG_OBJECT ---
    print("\n--- verifier (reaching for a distractor) ---")
    distractor = spec.confusable_distractors[0]
    print(f"    driving toward distractor {distractor!r}")
    fired = False
    for step in range(140):
        d_pos = det.detect(distractor).pos
        gp = det.gripper_pos()
        delta = d_pos - gp
        # Simple proportional reach; descend only once roughly overhead.
        act = np.zeros(7)
        act[:3] = np.clip(delta * 12.0, -1.0, 1.0)
        if np.linalg.norm(delta[:2]) > 0.03:
            act[2] = max(act[2], 0.0)
        act[6] = -1.0
        obs, _, _, _ = env.step(act.tolist())

        verdict = v.verify_precondition()
        if verdict.kind is VerdictKind.WRONG_OBJECT:
            print(f"    step {step}: {verdict.kind.value}: {verdict.reason}")
            fired = True
            break

    check(fired, "verifier flags WRONG_OBJECT when reaching for a distractor")
    if fired:
        check(verdict.observed == distractor, "verdict names the distractor reached for")
        check(verdict.intended == v.target, "verdict names the intended target")

    # --- noisy detector degrades as configured ---
    print("\n--- noisy detector ---")
    noisy = NoisyDetector(det, p_miss=0.5, p_swap=0.0, pos_noise_std=0.01, seed=0)
    counts = [len(noisy.detect_all()) for _ in range(40)]
    print(f"    detections/call over 40 calls: mean={np.mean(counts):.1f} of {len(dets)}")
    check(np.mean(counts) < len(dets), "p_miss=0.5 drops objects")

    clean = NoisyDetector(det, p_miss=0.0, p_swap=0.0, pos_noise_std=0.0, seed=0)
    check(len(clean.detect_all()) == len(dets), "noise-free NoisyDetector is lossless")

    env.close()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
