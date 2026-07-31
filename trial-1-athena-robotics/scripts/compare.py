#!/usr/bin/env python
"""Head-to-head comparison of two runs, with deltas and significance.

  python scripts/compare.py baseline adaptive
  python scripts/compare.py baseline adaptive --paired    # same (task, ep) only

`--paired` restricts to episodes both runs actually executed, which removes
task-mix differences when one run is incomplete. Use it while a run is still
in flight.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from athena.metrics import load_records, summarize, wilson_interval  # noqa: E402

# (key, label, higher_is_better | None if neutral)
ROWS = [
    ("success_rate", "success rate", True),
    ("object_accuracy", "object accuracy", True),
    ("wrong_object_rate", "wrong-object rate", False),
    ("self_recovery_rate", "self-recovery rate", None),
    ("success_given_correct_grasp", "P(success | correct grasp)", True),
    ("intervention_rate", "intervention rate", None),
    ("mean_steer_events", "mean steer events", None),
    ("mean_max_escalation", "mean max escalation", None),
    ("mean_policy_calls", "mean policy calls", False),
    ("mean_wall_time_s", "mean wall time (s)", False),
    ("verify_overhead_frac", "verify overhead", False),
    ("mean_steps", "mean steps", False),
]


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (float("nan"), float("nan"))
    z = (p2 - p1) / se
    # Two-sided p via the normal CDF (erf), no scipy dependency.
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, pval)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _delta(a, b, higher_better) -> str:
    if a is None or b is None:
        return "-"
    if isinstance(a, float) and (a != a or b != b):
        return "-"
    d = b - a
    arrow = ""
    if higher_better is not None and abs(d) > 1e-9:
        good = (d > 0) == higher_better
        arrow = "  ++" if good else "  --"
    return f"{d:+.3f}{arrow}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_a", nargs="?", default="baseline")
    p.add_argument("run_b", nargs="?", default="adaptive")
    p.add_argument("--results-dir", default="/scratch1/nalagand/athena_robotics/results")
    p.add_argument("--paired", action="store_true",
                   help="only episodes present in BOTH runs")
    p.add_argument("--confusable-only", action="store_true")
    args = p.parse_args()

    root = pathlib.Path(args.results_dir)
    recs_a = load_records(root / args.run_a / "episodes.jsonl")
    recs_b = load_records(root / args.run_b / "episodes.jsonl")

    if not recs_a or not recs_b:
        print(f"missing data: {args.run_a}={len(recs_a)} {args.run_b}={len(recs_b)}")
        return 1

    if args.confusable_only:
        recs_a = [r for r in recs_a if r.get("n_confusable_distractors", 0) > 0]
        recs_b = [r for r in recs_b if r.get("n_confusable_distractors", 0) > 0]

    if args.paired:
        keys_a = {(r["task_id"], r["episode_idx"]) for r in recs_a}
        keys_b = {(r["task_id"], r["episode_idx"]) for r in recs_b}
        common = keys_a & keys_b
        recs_a = [r for r in recs_a if (r["task_id"], r["episode_idx"]) in common]
        recs_b = [r for r in recs_b if (r["task_id"], r["episode_idx"]) in common]
        print(f"paired on {len(common)} common episodes\n")

    sa, sb = summarize(recs_a), summarize(recs_b)

    w = 28
    print(f"{'metric':<{w}}{args.run_a:>12}{args.run_b:>12}{'delta':>14}")
    print("-" * (w + 38))
    for key, label, hb in ROWS:
        print(f"{label:<{w}}{_fmt(sa.get(key)):>12}{_fmt(sb.get(key)):>12}"
              f"{_delta(sa.get(key), sb.get(key), hb):>14}")
    print("-" * (w + 38))
    print(f"{'episodes':<{w}}{sa['n_episodes']:>12}{sb['n_episodes']:>12}")

    # -- significance on the two headline rates ---------------------------
    print("\n### significance (two-proportion z-test)")
    ka = sum(1 for r in recs_a if r.get("success"))
    kb = sum(1 for r in recs_b if r.get("success"))
    z, pv = two_proportion_z(ka, len(recs_a), kb, len(recs_b))
    la, ha = wilson_interval(ka, len(recs_a))
    lb, hb_ = wilson_interval(kb, len(recs_b))
    print(f"  success   {args.run_a}: {ka}/{len(recs_a)} [{la:.3f},{ha:.3f}]  "
          f"{args.run_b}: {kb}/{len(recs_b)} [{lb:.3f},{hb_:.3f}]  "
          f"z={z:.2f} p={pv:.4f}")

    ga = [r for r in recs_a if r.get("grasped_correct") is not None]
    gb = [r for r in recs_b if r.get("grasped_correct") is not None]
    oa = sum(1 for r in ga if r["grasped_correct"])
    ob = sum(1 for r in gb if r["grasped_correct"])
    z2, pv2 = two_proportion_z(oa, len(ga), ob, len(gb))
    print(f"  obj acc   {args.run_a}: {oa}/{len(ga)}  "
          f"{args.run_b}: {ob}/{len(gb)}  z={z2:.2f} p={pv2:.4f}")

    # -- the diagnostic that decides whether the mechanism works -----------
    print("\n### mechanism check")
    iv = sb.get("intervention_rate", 0.0) or 0.0
    d_obj = (sb.get("object_accuracy") or 0) - (sa.get("object_accuracy") or 0)
    if iv < 0.05:
        print("  VERIFIER BARELY FIRED (intervention rate < 5%).")
        print("  The experiment did not really run — retune grasp_radius/tau_obj")
        print("  before drawing any conclusion about steering.")
    elif d_obj <= 0.02:
        print("  VERIFIER FIRED BUT OBJECT ACCURACY DID NOT MOVE.")
        print("  pi0.5 is likely ignoring the re-steered prompt. The problem is")
        print("  the injection mechanism, not the thresholds. See PROGRESS.md")
        print("  'Open questions'.")
    else:
        print(f"  Verifier fired (rate {iv:.2f}) and object accuracy moved "
              f"{d_obj:+.3f}.")
        print("  Steering is affecting behaviour — proceed to the wider run.")

    # -- per-task breakdown ------------------------------------------------
    print("\n### per-task success / object accuracy")
    by_a: dict[int, list] = {}
    by_b: dict[int, list] = {}
    for r in recs_a:
        by_a.setdefault(r["task_id"], []).append(r)
    for r in recs_b:
        by_b.setdefault(r["task_id"], []).append(r)
    print(f"  {'task':>5} {'n':>3} {'succ A':>8}{'succ B':>8}"
          f"{'obj A':>8}{'obj B':>8}   instruction")
    for tid in sorted(set(by_a) | set(by_b)):
        va, vb = by_a.get(tid, []), by_b.get(tid, [])
        ssa, ssb = summarize(va), summarize(vb)
        lang = (va or vb)[0].get("language", "")[:44]
        print(f"  {tid:>5} {max(len(va), len(vb)):>3} "
              f"{_fmt(ssa.get('success_rate')):>8}{_fmt(ssb.get('success_rate')):>8}"
              f"{_fmt(ssa.get('object_accuracy')):>8}"
              f"{_fmt(ssb.get('object_accuracy')):>8}   {lang}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
