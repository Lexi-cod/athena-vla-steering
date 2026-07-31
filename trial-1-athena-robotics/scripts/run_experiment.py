#!/usr/bin/env python
"""CLI entry point: run one experiment preset against a running policy server.

  python scripts/run_experiment.py --preset adaptive --task-subset pilot \
      --num-trials-per-task 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from athena.config import PRESETS, ExperimentConfig, from_preset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="baseline", choices=sorted(PRESETS))
    p.add_argument("--run-id", default=None, help="defaults to the preset name")
    p.add_argument("--task-suite-name", default=None)
    p.add_argument("--task-subset", default=None,
                   choices=["all", "confusable", "pilot"])
    p.add_argument("--pilot-size", type=int, default=None)
    p.add_argument("--num-trials-per-task", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--verify-every", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved config and task selection, then exit")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides = {
        k: v
        for k, v in dict(
            run_id=args.run_id,
            task_suite_name=args.task_suite_name,
            task_subset=args.task_subset,
            pilot_size=args.pilot_size,
            num_trials_per_task=args.num_trials_per_task,
            seed=args.seed,
            host=args.host,
            port=args.port,
            out_dir=args.out_dir,
            verify_every=args.verify_every,
            max_retries=args.max_retries,
            save_video=True if args.save_video else None,
        ).items()
        if v is not None
    }
    cfg = from_preset(args.preset, **overrides)

    print("=== resolved config ===")
    print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))

    if args.dry_run:
        # Import lazily so --dry-run works without a policy server.
        from athena.runner import select_task_ids
        from athena.taskspec import parse_suite
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
        specs = {s.name: s for s in parse_suite(cfg.task_suite_name)}
        ids = select_task_ids(cfg, specs, suite)
        print(f"\n=== {len(ids)} tasks selected ===")
        for tid in ids:
            t = suite.get_task(tid)
            spec = specs.get(pathlib.Path(t.bddl_file).stem)
            score = spec.confusion_score if spec else 0
            print(f"  [{tid:2d}] score={score} {t.language}")
        est = len(ids) * cfg.num_trials_per_task
        print(f"\nepisodes to run: {est}")
        return 0

    from athena.runner import run
    path = run(cfg)
    print(f"\nwrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
