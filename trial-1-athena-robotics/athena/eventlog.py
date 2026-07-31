"""Full-episode object-interaction logging from ground-truth sim state.

Why this exists
---------------
`EpisodeRecord.first_grasped` records only the *first* commitment. That was
enough while the question was "does the policy pick the right object", but it
cannot see a multi-object routine: grasp A, place A, then grasp B and place B.
Characterising that requires the whole ordered event sequence for *every*
tracked object, including objects the instruction never mentions.

Everything here reads ground-truth simulator state:

  - grasp/release  -> robosuite `_check_grasp` (contact-based), via the same
                      `Detector.grasped_object()` the metrics already use, so
                      grasp events stay consistent with `first_grasped`.
  - on-plate       -> LIBERO's own `_eval_predicate(("on", obj, plate))`, i.e.
                      the *identical* function that scores task success. A
                      "placed" event therefore means the same thing the
                      benchmark means.

No policy perception is involved.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# Contacts flicker for a step or two when the gripper brushes an object. We do
# not filter here -- raw transitions are recorded with their hold duration so
# that analysis can threshold after the fact rather than baking a choice in.
TABLE = "table"


@dataclasses.dataclass
class Event:
    kind: str            # grasp | release | placed_on | removed_from
    obj: str
    t: int
    location: str | None = None   # for release/placed_on/removed_from
    duration: int | None = None   # for release: steps held since its grasp

    def as_dict(self) -> dict[str, Any]:
        d = {"kind": self.kind, "obj": self.obj, "t": self.t}
        if self.location is not None:
            d["location"] = self.location
        if self.duration is not None:
            d["duration"] = self.duration
        return d


class InteractionTracker:
    """Polls sim state each step and emits object-interaction transitions.

    Parameters
    ----------
    env         unwrapped LIBERO env (must expose `_eval_predicate`)
    detector    a `Detector` (oracle) for grasp state and gripper position
    objects     movable objects to track (e.g. the three mugs)
    receptacles receptacle objects that count as destinations (e.g. plates)
    plate_poll_every
                on-plate predicates are contact queries and cost more than the
                grasp check; 5 steps is well inside the resolution needed to
                order events that are tens of steps apart.
    """

    def __init__(
        self,
        env,
        detector,
        objects: list[str],
        receptacles: list[str],
        plate_poll_every: int = 5,
    ):
        self.env = env
        self.detector = detector
        self.objects = list(objects)
        self.receptacles = list(receptacles)
        self.plate_poll_every = plate_poll_every

        self.events: list[Event] = []
        self._held: str | None = None
        self._held_since: int | None = None
        self._on: dict[tuple[str, str], bool] = {}
        self.min_gripper_dist: dict[str, float] = {o: float("inf") for o in self.objects}
        self.first_grasp: tuple[str, int] | None = None
        self.n_grasp_events = 0

    # -- ground truth helpers ------------------------------------------------

    def _is_at(self, obj: str, receptacle: str) -> bool:
        """Is `obj` placed at `receptacle`, by LIBERO's own goal predicates.

        Which predicate applies depends on the receptacle: plates use `On`
        (`(On mug_1 plate_1)`), baskets use `In` against a *contain region*
        site (`(In orange_juice_1 basket_1_contain_region)`). Trying both and
        taking either keeps this correct across scenes without hardcoding a
        mapping -- and a hardcoded `on` silently reports "never placed" for
        every basket task, which is worse than a redundant check.
        """
        for pred in ("on", "in"):
            try:
                if bool(self.env._eval_predicate((pred, obj, receptacle))):
                    return True
            except Exception:
                continue
        return False

    def location_of(self, obj: str) -> str:
        for r in self.receptacles:
            if self._is_at(obj, r):
                return r
        return TABLE

    # -- per-step polling ----------------------------------------------------

    def step(self, t: int) -> None:
        self._poll_grasp(t)
        self._poll_distances()
        if t % self.plate_poll_every == 0:
            self._poll_plates(t)

    def _poll_grasp(self, t: int) -> None:
        held = self.detector.grasped_object()
        if held is not None and held not in self.objects:
            held = None  # only track the objects we were asked about
        if held == self._held:
            return

        if self._held is not None:
            self.events.append(
                Event(
                    kind="release",
                    obj=self._held,
                    t=t,
                    location=self.location_of(self._held),
                    duration=(t - self._held_since) if self._held_since is not None else None,
                )
            )
        if held is not None:
            self.events.append(Event(kind="grasp", obj=held, t=t))
            self.n_grasp_events += 1
            if self.first_grasp is None:
                self.first_grasp = (held, t)

        self._held = held
        self._held_since = t if held is not None else None

    def _poll_distances(self) -> None:
        gp = self.detector.gripper_pos()
        dets = self.detector.detect_all()
        for o in self.objects:
            det = dets.get(o)
            if det is not None:
                d = float(np.linalg.norm(det.pos - gp))
                if d < self.min_gripper_dist[o]:
                    self.min_gripper_dist[o] = d

    def _poll_plates(self, t: int) -> None:
        for o in self.objects:
            for r in self.receptacles:
                now = self._is_at(o, r)
                was = self._on.get((o, r))
                if was is None:
                    self._on[(o, r)] = now
                    continue
                if now != was:
                    self.events.append(
                        Event(
                            kind="placed_on" if now else "removed_from",
                            obj=o,
                            t=t,
                            location=r,
                        )
                    )
                    self._on[(o, r)] = now

    def finalize(self, t: int) -> None:
        """Close an open grasp so the sequence is well-formed."""
        if self._held is not None:
            self.events.append(
                Event(
                    kind="release",
                    obj=self._held,
                    t=t,
                    location=self.location_of(self._held),
                    duration=(t - self._held_since) if self._held_since is not None else None,
                )
            )
            self._held = None

    # -- summaries -----------------------------------------------------------

    def sequence(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.events]

    def grasp_order(self, min_duration: int = 0) -> list[str]:
        """Distinct objects in the order they were first grasped.

        `min_duration` filters brush contacts: a grasp counts only if its
        matching release is at least that many steps later (an unreleased
        grasp at episode end always counts).
        """
        kept: list[str] = []
        open_grasp: tuple[str, int] | None = None
        for e in self.events:
            if e.kind == "grasp":
                open_grasp = (e.obj, e.t)
            elif e.kind == "release" and open_grasp is not None:
                if (e.duration or 0) >= min_duration and open_grasp[0] not in kept:
                    kept.append(open_grasp[0])
                open_grasp = None
        if open_grasp is not None and open_grasp[0] not in kept:
            kept.append(open_grasp[0])
        return kept

    def final_locations(self) -> dict[str, str]:
        return {o: self.location_of(o) for o in self.objects}

    def touched(self, radius: float) -> dict[str, bool]:
        return {o: self.min_gripper_dist[o] <= radius for o in self.objects}
