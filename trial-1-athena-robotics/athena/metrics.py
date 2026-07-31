"""Metric recording. One JSONL line per episode, append-only.

Append-only JSONL is what makes runs resumable: a crashed or preempted job
leaves a valid prefix, and the runner skips (task_id, episode_idx) pairs that
already appear in the file.

Metrics follow section 5 of the executive summary:
  task success, object accuracy, step success, recovery rate, latency.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import statistics
import threading
import time


@dataclasses.dataclass
class EpisodeRecord:
    run_id: str
    variant: str
    task_id: int
    task_name: str
    language: str
    episode_idx: int
    seed: int

    success: bool = False
    steps: int = 0

    # object accuracy
    target: str | None = None
    first_grasped: str | None = None
    grasped_correct: bool | None = None       # None if nothing was ever grasped
    grasped_confusable: bool | None = None

    # intervention accounting
    n_verifications: int = 0
    n_failed_verifications: int = 0
    n_steer_events: int = 0
    max_escalation: int = 0
    steer_notes: list[str] = dataclasses.field(default_factory=list)
    prompts_used: list[str] = dataclasses.field(default_factory=list)

    # cost
    policy_calls: int = 0
    policy_time_s: float = 0.0
    verify_time_s: float = 0.0
    wall_time_s: float = 0.0

    # task metadata for slicing results
    confusion_score: int = 0
    n_confusable_distractors: int = 0

    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


class MetricsWriter:
    """Thread-safe append-only JSONL writer with a resume index."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._done: set[tuple[int, int]] = set()
        self._load_done()

    def _load_done(self) -> None:
        """Index (task_id, episode_idx) already recorded, for resume."""
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Truncated final line from a killed job — ignore it.
                    continue
                if "task_id" in rec and "episode_idx" in rec:
                    self._done.add((rec["task_id"], rec["episode_idx"]))

    def already_done(self, task_id: int, episode_idx: int) -> bool:
        return (task_id, episode_idx) in self._done

    @property
    def n_done(self) -> int:
        return len(self._done)

    def write(self, rec: EpisodeRecord) -> None:
        with self._lock:
            with self.path.open("a") as f:
                f.write(rec.to_json() + "\n")
                f.flush()
            self._done.add((rec.task_id, rec.episode_idx))


class Timer:
    """Accumulates elapsed time across many `with` blocks."""

    def __init__(self):
        self.total = 0.0
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.total += time.perf_counter() - self._t0
        return False


# -- aggregation ------------------------------------------------------------


def load_records(path: str | pathlib.Path) -> list[dict]:
    path = pathlib.Path(path)
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(statistics.mean(xs)) if xs else float("nan")


def is_self_recovered(rec: dict) -> bool | None:
    """Did the policy grasp the wrong object first and still succeed?

    Derived, not stored: it is a pure function of `grasped_correct` and
    `success`, so it applies retroactively to every record ever written --
    including runs finished before this function existed.

    Returns None when the episode never grasped anything, since "recovered"
    is undefined there.

    Why it matters: steering fires on the *first* wrong grasp, so any episode
    that would have self-recovered is one where the intervention takes credit
    for a success the policy would have earned unaided. This is the correction
    term between object-accuracy gains and real success-rate gains.
    """
    if rec.get("grasped_correct") is None:
        return None
    if rec["grasped_correct"]:
        return False
    return bool(rec.get("success"))


def summarize(records: list[dict]) -> dict:
    """Aggregate the section-5 metrics over a set of episode records."""
    if not records:
        return {"n_episodes": 0}

    grasped = [r for r in records if r.get("grasped_correct") is not None]
    steered = [r for r in records if r.get("n_steer_events", 0) > 0]
    # Episodes that started with the wrong object in the gripper.
    wrong_start = [r for r in grasped if not r["grasped_correct"]]

    return {
        "n_episodes": len(records),
        # .get() throughout: a partial record must not take down analysis of a
        # whole run.
        "n_tasks": len({r.get("task_id") for r in records}),
        # 1. task success rate
        "success_rate": _mean([bool(r.get("success")) for r in records]),
        # 2. object accuracy — of episodes where something was grasped,
        #    how often was it the right object
        "object_accuracy": _mean([bool(r["grasped_correct"]) for r in grasped]),
        "n_episodes_with_grasp": len(grasped),
        "wrong_object_rate": _mean(
            [not r["grasped_correct"] for r in grasped]
        ),
        # 3. self-recovery: P(success | wrong first grasp). The share of
        #    wrong-object episodes the policy rescues unaided -- the ceiling on
        #    how much credit steering can legitimately claim.
        "self_recovery_rate": _mean([bool(r.get("success")) for r in wrong_start]),
        "n_wrong_first_grasp": len(wrong_start),
        "n_self_recovered": sum(1 for r in wrong_start if r.get("success")),
        # Success conditional on a correct first grasp -- the other half of the
        # picture: if this is well below 1.0, wrong-object grasps are not the
        # dominant failure mode and steering cannot help much.
        "success_given_correct_grasp": _mean(
            [bool(r.get("success")) for r in grasped if r["grasped_correct"]]
        ),
        # 4. recovery / intervention rate
        "intervention_rate": len(steered) / len(records),
        "mean_steer_events": _mean([r.get("n_steer_events", 0) for r in records]),
        "mean_max_escalation": _mean([r.get("max_escalation", 0) for r in records]),
        "verification_failure_rate": _mean(
            [
                r["n_failed_verifications"] / r["n_verifications"]
                for r in records
                if r.get("n_verifications")
            ]
        ),
        # 5. latency / overhead
        "mean_policy_calls": _mean([r.get("policy_calls", 0) for r in records]),
        "mean_policy_time_s": _mean([r.get("policy_time_s", 0.0) for r in records]),
        "mean_verify_time_s": _mean([r.get("verify_time_s", 0.0) for r in records]),
        "mean_wall_time_s": _mean([r.get("wall_time_s", 0.0) for r in records]),
        "verify_overhead_frac": _mean(
            [
                r["verify_time_s"] / r["wall_time_s"]
                for r in records
                if r.get("wall_time_s")
            ]
        ),
        "mean_steps": _mean([r.get("steps", 0) for r in records]),
        "n_errors": sum(1 for r in records if r.get("error")),
    }


def summarize_by(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(str(r.get(key)), []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success rate — better than normal approx
    at the small n we get per task (5-10 episodes)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
