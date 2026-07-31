#!/usr/bin/env python
"""Aggregate episodes.jsonl across runs into the section-5 metric tables.

  python scripts/analyze.py                      # all runs under results/
  python scripts/analyze.py --runs baseline adaptive
  python scripts/analyze.py --by-task --runs adaptive
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from athena.metrics import (  # noqa: E402
    load_records, summarize, summarize_by, wilson_interval,
)

METRICS = [
    ("success_rate", "success"),
    ("object_accuracy", "obj acc"),
    ("wrong_object_rate", "wrong obj"),
    ("self_recovery_rate", "self-rec"),
    ("success_given_correct_grasp", "P(s|ok)"),
    ("intervention_rate", "interv"),
    ("mean_steer_events", "steers"),
    ("mean_policy_calls", "pol calls"),
    ("verify_overhead_frac", "vfy ovh"),
    ("mean_wall_time_s", "wall s"),
]


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:  # NaN
            return "-"
        return f"{v:.3f}"
    return str(v)


def print_table(rows: dict[str, dict], label: str) -> None:
    if not rows:
        print(f"(no data for {label})")
        return
    width = max(len(k) for k in rows) + 2
    header = f"{label:<{width}}" + "".join(f"{h:>11}" for _, h in METRICS) + f"{'n':>7}"
    print(header)
    print("-" * len(header))
    for name, s in rows.items():
        line = f"{name:<{width}}"
        for key, _ in METRICS:
            line += f"{_fmt(s.get(key)):>11}"
        line += f"{s.get('n_episodes', 0):>7}"
        print(line)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="/scratch1/nalagand/athena_robotics/results")
    p.add_argument("--runs", nargs="*", default=None, help="run ids; default all")
    p.add_argument("--by-task", action="store_true", help="break down per task")
    p.add_argument("--confusable-only", action="store_true",
                   help="restrict to episodes whose task has confusable distractors")
    args = p.parse_args()

    root = pathlib.Path(args.results_dir)
    if not root.exists():
        print(f"no results dir: {root}")
        return 1

    run_dirs = sorted(d for d in root.iterdir() if (d / "episodes.jsonl").exists())
    if args.runs:
        run_dirs = [d for d in run_dirs if d.name in args.runs]
    if not run_dirs:
        print(f"no runs with episodes.jsonl under {root}")
        return 1

    all_summaries: dict[str, dict] = {}
    for d in run_dirs:
        recs = load_records(d / "episodes.jsonl")
        if args.confusable_only:
            recs = [r for r in recs if r.get("n_confusable_distractors", 0) > 0]
        if not recs:
            continue
        all_summaries[d.name] = summarize(recs)

        if args.by_task:
            print(f"\n### {d.name} — by task")
            print_table(summarize_by(recs, "task_name"), "task")

    print(f"\n### overall{' (confusable tasks only)' if args.confusable_only else ''}")
    print_table(all_summaries, "run")

    # Self-recovery is the correction term between object-accuracy gains and
    # real success gains, so show raw counts rather than only the rate.
    print("\n### self-recovery (wrong first grasp, succeeded anyway)")
    for name, s in all_summaries.items():
        n_wrong = s.get("n_wrong_first_grasp", 0)
        n_rec = s.get("n_self_recovered", 0)
        rate = f"{n_rec / n_wrong:.3f}" if n_wrong else "-"
        print(f"  {name:<24} {n_rec:3d}/{n_wrong:<3d} = {rate}")

    # Success-rate confidence intervals, since per-task n is small.
    print("\n### success rate with 95% Wilson CI")
    for name, d in ((d.name, d) for d in run_dirs):
        if name not in all_summaries:
            continue
        recs = load_records(d / "episodes.jsonl")
        if args.confusable_only:
            recs = [r for r in recs if r.get("n_confusable_distractors", 0) > 0]
        k = sum(1 for r in recs if r.get("success"))
        n = len(recs)
        lo, hi = wilson_interval(k, n)
        print(f"  {name:<24} {k:4d}/{n:<4d} = {k/n:.3f}  [{lo:.3f}, {hi:.3f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
