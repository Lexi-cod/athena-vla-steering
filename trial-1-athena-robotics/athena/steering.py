"""Re-steering: the robotic analogue of ATHENA's control prompt + mixing.

ATHENA builds a *control prompt* encoding the corrected count, denoises with
both prompts, and mixes the two noise predictions -- statically, or with a
weight adapted to whether the last correction helped.

Here the prompt is the language instruction handed to pi0.5, and the "denoiser
outputs" are action chunks. So:

  control prompt        -> a rewritten instruction that disambiguates the target
  eps_orig / eps_ctrl   -> action chunks from the original and rewritten prompts
  static mixing         -> one fixed correction, applied once
  adaptive mixing       -> escalating corrections, strength adapted to whether
                           the previous one moved the gripper toward the target

Unlike ATHENA we cannot blend in a smooth latent space for free: interpolating
two action chunks can produce a trajectory that is neither. We therefore
support both true convex blending (`blend`) and hard selection (`select`), and
make selection the default for the dual-plan variant.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np

from athena.taskspec import TaskSpec
from athena.verifier import Verdict, VerdictKind

# Spatial qualifiers LIBERO uses to disambiguate identical objects.
_QUALIFIERS = ("left", "right", "front", "back", "middle", "top", "bottom")


def _readable(obj_name: str) -> str:
    """akita_black_bowl_2 -> 'black bowl'. Keeps the phrase promptable."""
    name = re.sub(r"_\d+$", "", obj_name)
    words = [w for w in name.split("_") if w]
    # Drop brand-ish leading tokens that never appear in instructions.
    drop = {"akita", "porcelain", "wooden"}
    kept = [w for w in words if w not in drop] or words
    return " ".join(kept)


@dataclasses.dataclass
class SteerResult:
    prompt: str
    escalation: int
    note: str


class SteeringPolicy:
    """Base: never intervenes. This is the vanilla-pi0.5 baseline."""

    name = "none"
    uses_dual_plan = False

    def __init__(self, spec: TaskSpec, max_retries: int = 3):
        self.spec = spec
        self.max_retries = max_retries
        self.base_prompt = spec.language
        self.escalation = 0
        self.history: list[SteerResult] = []

    def reset(self) -> None:
        self.escalation = 0
        self.history.clear()

    def should_intervene(self, verdict: Verdict) -> bool:
        return False

    def steer(self, verdict: Verdict, current_prompt: str) -> SteerResult:
        return SteerResult(current_prompt, self.escalation, "no steering")

    # -- how to combine plans when running the policy twice ------------------

    def combine(
        self, action_orig: np.ndarray, action_ctrl: np.ndarray, verdict: Verdict
    ) -> np.ndarray:
        return action_orig


class StaticSteering(SteeringPolicy):
    """One correction, applied once, then never revised.

    Analogue of ATHENA's fixed mixing weight.
    """

    name = "static"

    def should_intervene(self, verdict: Verdict) -> bool:
        return verdict.actionable and self.escalation < 1

    def _control_prompt(self, verdict: Verdict) -> str:
        target = _readable(verdict.intended or "")
        return f"{self.base_prompt}. pick up the {target}"

    def steer(self, verdict: Verdict, current_prompt: str) -> SteerResult:
        self.escalation += 1
        res = SteerResult(
            self._control_prompt(verdict), self.escalation, f"static:{verdict.kind.value}"
        )
        self.history.append(res)
        return res


class AdaptiveSteering(StaticSteering):
    """Escalating corrections, adapted to whether the last one helped.

    Escalation ladder, increasingly explicit:
      1. restate the target             "... pick up the black bowl"
      2. add the spatial qualifier      "... pick up the black bowl at the front"
      3. negate the observed distractor "... do not pick up the black bowl in the middle"

    Like ATHENA's adaptive variant, we check whether the previous correction
    moved the situation toward the goal (gripper closer to the intended object)
    and only escalate when it did not.
    """

    name = "adaptive"

    def __init__(self, spec: TaskSpec, max_retries: int = 3):
        super().__init__(spec, max_retries)
        self._last_distance: float | None = None

    def reset(self) -> None:
        super().reset()
        self._last_distance = None

    def should_intervene(self, verdict: Verdict) -> bool:
        # No budget gate. `escalation` caps the *ladder* (see steer), not the
        # number of interventions: ATHENA keeps applying its adapted weight for
        # the rest of the trajectory rather than stopping after k corrections.
        # Gating here also silently disabled the forced replan, so a stale chunk
        # computed under the old prompt would still execute.
        return verdict.actionable

    def _needs_spatial_qualifier(self, target: str, observed: str | None) -> bool:
        """Is position the only thing separating target from distractor?

        For "put the red mug on the left plate" the mugs differ by colour, and
        the sole spatial word ("left") modifies the *destination plate* -- so
        appending it to the mug phrase ("the red coffee mug at the left") is
        actively wrong. Only reach for the qualifier when the competing objects
        are otherwise indistinguishable by name.
        """
        if observed is None:
            return False
        return _readable(observed) == _readable(target)

    def _qualifier(self, target: str = "", observed: str | None = None) -> str | None:
        """Recover the disambiguating phrase from the instruction itself.

        Returns None unless the target genuinely needs spatial disambiguation.
        """
        if target and not self._needs_spatial_qualifier(target, observed):
            return None
        text = self.base_prompt.lower()
        for q in _QUALIFIERS:
            if re.search(rf"\b{q}\b", text):
                return q
        return None

    def _control_prompt(self, verdict: Verdict) -> str:
        target = _readable(verdict.intended or "")
        level = self.escalation

        if level <= 1:
            return f"{self.base_prompt}. pick up the {target}"

        qual = self._qualifier(verdict.intended or "", verdict.observed)
        if level == 2:
            if qual:
                # Re-attach the qualifier the policy appears to have ignored.
                return f"{self.base_prompt}. pick up the {target} at the {qual}"
            return f"{self.base_prompt}. the target is the {target}"

        # Final escalation: name the distractor and rule it out.
        #
        # Only useful when the distractor *reads differently* from the target.
        # For identical instances (akita_black_bowl_1 vs _2 -> both "black
        # bowl") a negation is self-contradictory -- "do not pick up the black
        # bowl, pick up the black bowl" -- so fall back to asserting the
        # spatial qualifier, which is the only thing that separates them.
        wrong = _readable(verdict.observed) if verdict.observed else None
        if wrong and wrong != target:
            return (
                f"{self.base_prompt}. do not pick up the {wrong}. "
                f"pick up the {target}"
                + (f" at the {qual}" if qual else "")
            )
        if qual:
            return (
                f"{self.base_prompt}. pick up only the {qual} {target}, "
                f"not the other one"
            )
        return f"{self.base_prompt}. pick up the correct {target}"

    def steer(self, verdict: Verdict, current_prompt: str) -> SteerResult:
        # Adaptive part: did the previous correction help?
        helped = None
        d = verdict.distance
        if self._last_distance is not None and not np.isnan(d):
            helped = d < self._last_distance
        if not np.isnan(d):
            self._last_distance = d

        # If the last correction *did* help, hold the current level rather than
        # escalating -- mirrors ATHENA keeping its weight when the count moved
        # the right way.
        if helped:
            note = f"adaptive:hold@{self.escalation}"
        else:
            # Clamp the ladder instead of spending a budget: once at the top
            # rung we keep re-asserting it (and keep discarding stale chunks)
            # for the rest of the episode.
            if self.escalation < self.max_retries:
                self.escalation += 1
                note = f"adaptive:escalate@{self.escalation}:{verdict.kind.value}"
            else:
                note = f"adaptive:reassert@{self.escalation}:{verdict.kind.value}"

        res = SteerResult(self._control_prompt(verdict), self.escalation, note)
        self.history.append(res)
        return res


class DualPlanSteering(AdaptiveSteering):
    """Run pi0.5 twice (original + control prompt) and pick between the chunks.

    Closest analogue to ATHENA's explicit two-forward-pass mixing. `select`
    chooses the chunk whose first step moves the gripper closer to the intended
    object; `blend` takes a convex combination.
    """

    name = "dual"
    uses_dual_plan = True

    def __init__(
        self,
        spec: TaskSpec,
        max_retries: int = 3,
        mode: str = "select",
        alpha: float = 0.5,
    ):
        super().__init__(spec, max_retries)
        if mode not in ("select", "blend"):
            raise ValueError(f"mode must be 'select' or 'blend', got {mode!r}")
        self.mode = mode
        self.alpha = alpha
        self._target_pos: np.ndarray | None = None
        self._gripper_pos: np.ndarray | None = None

    def set_geometry(self, gripper_pos, target_pos) -> None:
        """Runner supplies current geometry so `select` can score chunks."""
        self._gripper_pos = None if gripper_pos is None else np.asarray(gripper_pos)
        self._target_pos = None if target_pos is None else np.asarray(target_pos)

    def combine(
        self, action_orig: np.ndarray, action_ctrl: np.ndarray, verdict: Verdict
    ) -> np.ndarray:
        if self.mode == "blend":
            n = min(len(action_orig), len(action_ctrl))
            return (1 - self.alpha) * action_orig[:n] + self.alpha * action_ctrl[:n]

        # select: prefer the chunk whose commanded delta points at the target.
        if self._target_pos is None or self._gripper_pos is None:
            return action_ctrl
        direction = self._target_pos - self._gripper_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return action_ctrl
        direction = direction / norm

        def score(chunk: np.ndarray) -> float:
            # LIBERO actions are [dx, dy, dz, drx, dry, drz, gripper];
            # the positional delta is the first three dims.
            delta = np.asarray(chunk)[0, :3]
            return float(np.dot(delta, direction))

        try:
            return action_ctrl if score(action_ctrl) >= score(action_orig) else action_orig
        except Exception:
            return action_ctrl


def build_steering(variant: str, spec: TaskSpec, **kw) -> SteeringPolicy:
    """Factory used by configs."""
    table = {
        "none": SteeringPolicy,
        "static": StaticSteering,
        "adaptive": AdaptiveSteering,
        "dual": DualPlanSteering,
    }
    if variant not in table:
        raise ValueError(f"unknown steering variant {variant!r}; have {list(table)}")
    cls = table[variant]
    if cls is not DualPlanSteering:
        kw = {k: v for k, v in kw.items() if k in ("max_retries",)}
    return cls(spec, **kw)
