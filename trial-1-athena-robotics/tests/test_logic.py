#!/usr/bin/env python
"""Offline unit tests — no simulator, no GPU, no policy server.

Covers the parts that are easy to break while editing: prompt construction,
escalation, resume, and metric aggregation. Run this before every commit.

  python tests/test_logic.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from athena.metrics import (  # noqa: E402
    EpisodeRecord, MetricsWriter, is_self_recovered, load_records, summarize,
    wilson_interval,
)
from athena.steering import DualPlanSteering, _readable, build_steering  # noqa: E402
from athena.taskspec import content_tokens, parse_suite  # noqa: E402
from athena.verifier import Verdict, VerdictKind, resolve_target  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILURES.append(msg)


def _wrong(intended: str, observed: str, dist: float = 0.04) -> Verdict:
    return Verdict(
        VerdictKind.WRONG_OBJECT, "test",
        intended=intended, observed=observed, distance=dist,
    )


def test_taskspec(specs):
    print("\n--- taskspec ---")
    check(len(specs) == 90, f"parsed 90 libero_90 tasks (got {len(specs)})")
    check(all(s.language for s in specs), "every task has a language instruction")
    check(all(s.obj_of_interest for s in specs), "every task has obj_of_interest")

    n_conf = sum(1 for s in specs if s.confusion_score > 0)
    check(n_conf == 32, f"32 confusable tasks (got {n_conf})")

    bowl = next(s for s in specs if s.language == "put the black bowl at the front on the plate")
    check(
        set(bowl.confusable_distractors) == {"akita_black_bowl_2", "akita_black_bowl_3"},
        "identical bowls are flagged as mutual distractors",
    )
    check(resolve_target(bowl) == "akita_black_bowl_1",
          "resolve_target picks the manipulated object, not the destination")
    check("bowl" in content_tokens("akita_black_bowl_1"), "content_tokens strips indices")
    check("the" not in content_tokens("the black bowl"), "content_tokens drops stopwords")


def test_readable():
    print("\n--- readable names ---")
    check(_readable("akita_black_bowl_2") == "black bowl", "brand token dropped")
    check(_readable("red_coffee_mug_1") == "red coffee mug", "colour token kept")
    check(_readable("plate_1") == "plate", "index stripped")


def test_escalation(specs):
    print("\n--- escalation ---")
    bowl = next(s for s in specs if s.language == "put the black bowl at the front on the plate")

    none = build_steering("none", bowl)
    check(not none.should_intervene(_wrong("akita_black_bowl_1", "akita_black_bowl_2")),
          "baseline never intervenes")

    static = build_steering("static", bowl)
    v = _wrong("akita_black_bowl_1", "akita_black_bowl_2")
    check(static.should_intervene(v), "static intervenes once")
    static.steer(v, bowl.language)
    check(not static.should_intervene(v), "static does not intervene twice")

    adaptive = build_steering("adaptive", bowl, max_retries=3)
    prompts = []
    for i in range(5):
        vv = _wrong("akita_black_bowl_1", "akita_black_bowl_2", dist=0.04 + 0.01 * i)
        if adaptive.should_intervene(vv):
            prompts.append(adaptive.steer(vv, bowl.language).prompt)
    # max_retries caps the *ladder*, not the number of interventions. Once at
    # the top rung the policy keeps re-asserting it (and keeps forcing a replan)
    # for the rest of the episode -- ATHENA does not stop after k corrections.
    # Regression for the retry-budget fix.
    check(len(prompts) == 5,
          f"adaptive keeps intervening past max_retries (got {len(prompts)})")
    check(len(set(prompts)) == 3,
          f"ladder caps at 3 distinct rungs (got {len(set(prompts))})")
    check(adaptive.escalation == 3,
          f"escalation never exceeds max_retries (got {adaptive.escalation})")
    check(prompts[2] == prompts[3] == prompts[4],
          "top rung is re-asserted verbatim once reached")

    # Regression: identical instances must not produce a self-contradiction.
    bad = [p for p in prompts if "do not pick up the black bowl" in p
           and "pick up the black bowl" in p.split("do not pick up the black bowl")[1]]
    check(not bad, "no self-contradictory negation for identical instances")

    # Regression: a qualifier belonging to the destination must not be
    # attached to the target object.
    mug = next(s for s in specs if s.language == "put the red mug on the left plate")
    ad2 = build_steering("adaptive", mug, max_retries=3)
    mug_prompts = []
    for i in range(3):
        vv = _wrong("red_coffee_mug_1", "white_yellow_mug_1", dist=0.04 + 0.01 * i)
        if ad2.should_intervene(vv):
            mug_prompts.append(ad2.steer(vv, mug.language).prompt)
    check(
        not any("mug at the left" in p for p in mug_prompts),
        "destination qualifier ('left plate') is not attached to the mug",
    )
    check(any("white yellow mug" in p for p in mug_prompts),
          "distinct-name distractor is negated by name")

    # Adaptive holds its level when the correction is helping.
    ad3 = build_steering("adaptive", bowl, max_retries=5)
    ad3.steer(_wrong("akita_black_bowl_1", "akita_black_bowl_2", dist=0.10), "")
    lvl = ad3.escalation
    ad3.steer(_wrong("akita_black_bowl_1", "akita_black_bowl_2", dist=0.04), "")
    check(ad3.escalation == lvl, "adaptive holds level when distance is shrinking")


def test_dual_plan(specs):
    print("\n--- dual-plan combination ---")
    bowl = next(s for s in specs if s.confusion_score > 0)

    d = DualPlanSteering(bowl, mode="select")
    d.set_geometry(gripper_pos=np.zeros(3), target_pos=np.array([1.0, 0.0, 0.0]))
    toward = np.tile(np.array([0.5, 0.0, 0.0, 0, 0, 0, -1.0]), (5, 1))
    away = np.tile(np.array([-0.5, 0.0, 0.0, 0, 0, 0, -1.0]), (5, 1))
    picked = d.combine(away, toward, _wrong("a", "b"))
    check(np.allclose(picked, toward), "select picks the chunk aimed at the target")
    picked2 = d.combine(toward, away, _wrong("a", "b"))
    check(np.allclose(picked2, toward), "select rejects the chunk aimed away")

    b = DualPlanSteering(bowl, mode="blend", alpha=0.5)
    mixed = b.combine(away, toward, _wrong("a", "b"))
    check(np.allclose(mixed[0, 0], 0.0), "blend averages the two chunks")

    try:
        DualPlanSteering(bowl, mode="nonsense")
        check(False, "invalid dual mode raises")
    except ValueError:
        check(True, "invalid dual mode raises")


def test_metrics():
    print("\n--- metrics + resume ---")
    path = pathlib.Path(tempfile.mkdtemp()) / "episodes.jsonl"
    w = MetricsWriter(path)
    for t in range(2):
        for e in range(3):
            w.write(EpisodeRecord(
                run_id="t", variant="adaptive", task_id=t, task_name=f"task{t}",
                language="x", episode_idx=e, seed=7,
                success=(e == 0), grasped_correct=(e != 2),
                n_verifications=10, n_failed_verifications=2,
                n_steer_events=(1 if e else 0),
                policy_calls=20, policy_time_s=4.0, verify_time_s=0.2, wall_time_s=10.0,
            ))
    check(w.n_done == 6, "6 episodes indexed")
    check(w.already_done(1, 2), "already_done finds a written episode")
    check(not w.already_done(5, 5), "already_done rejects an unwritten episode")

    # Simulate a killed job leaving a half-written line.
    with path.open("a") as f:
        f.write('{"task_id": 9, "epis')
    w2 = MetricsWriter(path)
    check(w2.n_done == 6, "truncated trailing line is ignored on resume")

    s = summarize(load_records(path))
    check(s["n_episodes"] == 6, "summarize counts episodes")
    check(abs(s["success_rate"] - 1 / 3) < 1e-9, "success_rate correct")
    check(abs(s["object_accuracy"] - 2 / 3) < 1e-9, "object_accuracy correct")
    check(abs(s["intervention_rate"] - 2 / 3) < 1e-9, "intervention_rate correct")
    check(abs(s["verify_overhead_frac"] - 0.02) < 1e-9, "verify overhead correct")
    check(summarize([]) == {"n_episodes": 0}, "summarize handles no records")

    lo, hi = wilson_interval(7, 12)
    check(lo < 7 / 12 < hi, "Wilson interval brackets the point estimate")
    check(all(np.isnan(x) for x in wilson_interval(0, 0)), "Wilson handles n=0")


def test_self_recovery():
    print("\n--- self-recovery (derived) ---")
    # wrong first grasp + success == recovered
    check(is_self_recovered({"grasped_correct": False, "success": True}) is True,
          "wrong grasp + success -> recovered")
    check(is_self_recovered({"grasped_correct": False, "success": False}) is False,
          "wrong grasp + failure -> not recovered")
    check(is_self_recovered({"grasped_correct": True, "success": True}) is False,
          "correct grasp is never 'recovery'")
    check(is_self_recovered({"grasped_correct": None, "success": False}) is None,
          "undefined when nothing was grasped")

    # Must work on records written before the field existed -- that is the
    # whole point of deriving rather than storing it.
    legacy = [
        {"grasped_correct": False, "success": True},    # recovered
        {"grasped_correct": False, "success": False},
        {"grasped_correct": False, "success": False},
        {"grasped_correct": True, "success": True},
        {"grasped_correct": None, "success": False},    # no grasp
    ]
    s = summarize(legacy)
    check(s["n_wrong_first_grasp"] == 3, "counts wrong-first-grasp episodes")
    check(s["n_self_recovered"] == 1, "counts recovered episodes")
    check(abs(s["self_recovery_rate"] - 1 / 3) < 1e-9, "self_recovery_rate correct")
    check(abs(s["success_given_correct_grasp"] - 1.0) < 1e-9,
          "P(success | correct grasp) correct")

    none_wrong = [{"grasped_correct": True, "success": True}]
    check(np.isnan(summarize(none_wrong)["self_recovery_rate"]),
          "self_recovery_rate is NaN when nothing grasped wrong")


def main() -> int:
    specs = parse_suite("libero_90")
    test_taskspec(specs)
    test_readable()
    test_escalation(specs)
    test_dual_plan(specs)
    test_metrics()
    test_self_recovery()

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
