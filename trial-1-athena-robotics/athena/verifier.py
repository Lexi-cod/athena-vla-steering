"""State verification: the robotic analogue of ATHENA's intermediate counter.

ATHENA decodes a mid-diffusion latent and counts objects. We cannot count, and
in this stack pi0.5 emits no subtask text (the policy server returns only
`actions`), so there is no textual plan to inspect either. Instead we infer the
policy's *de facto* intent from its behaviour:

  - which object the gripper is closing on / has grasped, and
  - whether that object is the one the instruction actually names.

That is the check: "is the thing you are about to commit to the right thing?",
asked while the action is still revisable — the same early-intervention
principle ATHENA applies before image structure sets.
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np

from athena.perception import Detector
from athena.taskspec import TaskSpec, content_tokens


class VerdictKind(enum.Enum):
    OK = "ok"                       # nothing wrong, proceed
    NOT_APPLICABLE = "n/a"          # nothing to check yet (nothing grasped/near)
    WRONG_OBJECT = "wrong_object"   # committing to an object the task did not name
    AMBIGUOUS = "ambiguous"         # cannot tell which candidate is intended
    LOW_CONFIDENCE = "low_conf"     # detector unsure
    NOT_DETECTED = "not_detected"   # target not visible at all


@dataclasses.dataclass
class Verdict:
    kind: VerdictKind
    reason: str = ""
    intended: str | None = None     # object the task wants
    observed: str | None = None     # object the policy is actually going for
    confidence: float = 1.0
    distance: float = float("nan")

    @property
    def ok(self) -> bool:
        return self.kind in (VerdictKind.OK, VerdictKind.NOT_APPLICABLE)

    @property
    def actionable(self) -> bool:
        """Is this a failure we know how to re-steer from?"""
        return self.kind in (VerdictKind.WRONG_OBJECT, VerdictKind.AMBIGUOUS)


def resolve_target(spec: TaskSpec) -> str | None:
    """The manipulable object the instruction is about.

    BDDL's `obj_of_interest` lists targets in instruction order; for
    "put the red mug on the left plate" it is (red_coffee_mug_1, plate_1),
    i.e. the *manipulated* object first and the destination second. We take the
    first entry that is an object rather than a fixture.
    """
    for name in spec.obj_of_interest:
        if name in spec.objects:
            return name
    return spec.obj_of_interest[0] if spec.obj_of_interest else None


def destination(spec: TaskSpec) -> str | None:
    """The place target, if the task names one (second obj_of_interest entry)."""
    target = resolve_target(spec)
    for name in spec.obj_of_interest:
        if name != target:
            return name
    return None


class StateVerifier:
    """Checks, before the gripper commits, that the right object is in play."""

    def __init__(
        self,
        spec: TaskSpec,
        detector: Detector,
        tau_obj: float = 0.5,
        grasp_radius: float = 0.055,
        ambiguity_margin: float = 0.02,
    ):
        self.spec = spec
        self.detector = detector
        self.tau_obj = tau_obj
        # How close the gripper must be before we treat it as "committing".
        self.grasp_radius = grasp_radius
        # If the two nearest candidates are within this margin, we cannot tell
        # which one the policy means.
        self.ambiguity_margin = ambiguity_margin

        self.target = resolve_target(spec)
        self.destination = destination(spec)
        # Same-category / same-token distractors are the ones worth confusing.
        self.confusable = set(spec.confusable_distractors)

    # -- precondition -------------------------------------------------------

    def verify_precondition(self) -> Verdict:
        """Called each control step, before executing the queued action.

        Returns OK unless the policy is closing on the wrong object.
        """
        if self.target is None:
            return Verdict(VerdictKind.NOT_APPLICABLE, "task names no object target")

        detections = self.detector.detect_all()

        # Already holding something? That is the strongest evidence of intent.
        held = self.detector.grasped_object()
        if held is not None:
            if held == self.target:
                return Verdict(
                    VerdictKind.OK, "correct object grasped",
                    intended=self.target, observed=held,
                )
            return Verdict(
                VerdictKind.WRONG_OBJECT,
                f"grasped {held}, task wants {self.target}",
                intended=self.target, observed=held,
            )

        # Not holding anything: is the gripper closing on something?
        gp = self.detector.gripper_pos()
        ranked = sorted(
            ((d.distance_to(gp), n, d) for n, d in detections.items()),
            key=lambda t: t[0],
        )
        if not ranked:
            return Verdict(VerdictKind.NOT_DETECTED, "no objects detected")

        dist, nearest, det = ranked[0]
        if dist > self.grasp_radius:
            # Still in free space — nothing has been committed to yet.
            return Verdict(
                VerdictKind.NOT_APPLICABLE, "gripper not near any object",
                intended=self.target, distance=dist,
            )

        if det.confidence < self.tau_obj:
            return Verdict(
                VerdictKind.LOW_CONFIDENCE,
                f"detector confidence {det.confidence:.2f} < {self.tau_obj}",
                intended=self.target, observed=nearest,
                confidence=det.confidence, distance=dist,
            )

        # Two candidates equally close -> we cannot attribute intent.
        if len(ranked) > 1 and (ranked[1][0] - dist) < self.ambiguity_margin:
            other = ranked[1][1]
            if self.target not in (nearest, other):
                return Verdict(
                    VerdictKind.WRONG_OBJECT,
                    f"closing on {nearest}/{other}, task wants {self.target}",
                    intended=self.target, observed=nearest, distance=dist,
                )
            return Verdict(
                VerdictKind.AMBIGUOUS,
                f"{nearest} and {other} equidistant ({dist:.3f}m)",
                intended=self.target, observed=nearest, distance=dist,
            )

        if nearest == self.target:
            return Verdict(
                VerdictKind.OK, "closing on correct object",
                intended=self.target, observed=nearest,
                confidence=det.confidence, distance=dist,
            )

        # Closing on a non-target. Only treat it as an error if it is a
        # plausible confusion (a mug when we want a different mug), not e.g.
        # brushing past the table on the way.
        if nearest in self.confusable or nearest in self.spec.objects:
            return Verdict(
                VerdictKind.WRONG_OBJECT,
                f"closing on {nearest}, task wants {self.target}",
                intended=self.target, observed=nearest,
                confidence=det.confidence, distance=dist,
            )
        return Verdict(
            VerdictKind.NOT_APPLICABLE, f"near non-object {nearest}",
            intended=self.target, distance=dist,
        )

    # -- postcondition ------------------------------------------------------

    def verify_effect(self) -> Verdict:
        """After executing a chunk: did we end up holding the right thing?"""
        held = self.detector.grasped_object()
        if held is None:
            return Verdict(
                VerdictKind.NOT_APPLICABLE, "nothing grasped", intended=self.target
            )
        if held == self.target:
            return Verdict(
                VerdictKind.OK, "holding target", intended=self.target, observed=held
            )
        return Verdict(
            VerdictKind.WRONG_OBJECT,
            f"holding {held}, task wants {self.target}",
            intended=self.target, observed=held,
        )

    # -- diagnostics --------------------------------------------------------

    def grasp_report(self) -> dict:
        """Ground-truth-ish snapshot for metrics (uses the detector in play)."""
        held = self.detector.grasped_object()
        return {
            "held": held,
            "target": self.target,
            "held_is_target": (held == self.target) if held else None,
            "held_is_confusable": (held in self.confusable) if held else None,
        }
