"""Perception layer: what the verifier is allowed to know about the scene.

Design note
-----------
The executive summary assumes a detector that maps a language name to a pose
("detect_object('red mug')"). We implement that interface, but back it by two
interchangeable sources so the *algorithm* can be evaluated independently of
detector quality (ablation "vision quality: perfect vs noisy detector"):

  OracleDetector  - reads ground-truth poses from the MuJoCo sim state.
  NoisyDetector   - wraps any detector and injects misses, identity swaps and
                    pose noise at controlled rates.

A learned RGB detector plugs in behind the same `Detector` interface later
without touching the verifier or the steering logic.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Detection:
    """One detected object instance."""

    name: str
    pos: np.ndarray          # (3,) world position
    quat: np.ndarray         # (4,) world orientation, wxyz (MuJoCo convention)
    confidence: float = 1.0

    def distance_to(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(self.pos - np.asarray(point)))


class Detector:
    """Interface shared by oracle, noisy and (future) learned detectors."""

    def detect_all(self) -> dict[str, Detection]:
        raise NotImplementedError

    def detect(self, name: str) -> Detection | None:
        return self.detect_all().get(name)

    def gripper_pos(self) -> np.ndarray:
        raise NotImplementedError

    def grasped_object(self) -> str | None:
        raise NotImplementedError


def _unwrap(env):
    """LIBERO wraps the robosuite env; walk down to the one holding sim state."""
    inner = env
    for _ in range(8):
        if hasattr(inner, "object_states_dict") and hasattr(inner, "sim"):
            return inner
        if hasattr(inner, "env"):
            inner = inner.env
        else:
            break
    return inner


class OracleDetector(Detector):
    """Ground-truth perception straight from the simulator.

    This is the doc's "oracle detector" upper-bound condition: it isolates the
    steering algorithm from perception error. It is *not* a claim about what a
    real detector would achieve.
    """

    def __init__(self, env):
        self.env = _unwrap(env)

    # -- objects ------------------------------------------------------------

    def _object_names(self) -> list[str]:
        return list(getattr(self.env, "obj_body_id", {}).keys())

    def detect_all(self) -> dict[str, Detection]:
        out: dict[str, Detection] = {}
        sim = self.env.sim
        for name, body_id in getattr(self.env, "obj_body_id", {}).items():
            out[name] = Detection(
                name=name,
                pos=np.array(sim.data.body_xpos[body_id], dtype=float),
                quat=np.array(sim.data.body_xquat[body_id], dtype=float),
                confidence=1.0,
            )
        return out

    # -- robot --------------------------------------------------------------

    def gripper_pos(self) -> np.ndarray:
        """World position of the end-effector."""
        sim = self.env.sim
        try:
            robot = self.env.robots[0]
            site = robot.controller.eef_name if hasattr(robot, "controller") else None
            if site:
                return np.array(sim.data.site_xpos[sim.model.site_name2id(site)], dtype=float)
        except Exception:
            pass
        # Fallback: gripper body position.
        for cand in ("gripper0_grip_site", "gripper0_eef", "robot0_right_hand"):
            try:
                return np.array(
                    sim.data.site_xpos[sim.model.site_name2id(cand)], dtype=float
                )
            except Exception:
                continue
        try:
            return np.array(
                sim.data.body_xpos[sim.model.body_name2id("robot0_right_hand")],
                dtype=float,
            )
        except Exception:
            return np.zeros(3)

    def grasped_object(self) -> str | None:
        """Name of the object currently held, via robosuite's grasp check."""
        env = self.env
        try:
            gripper = env.robots[0].gripper
        except Exception:
            return None
        for name in self._object_names():
            try:
                obj = env.get_object(name)
                geoms = getattr(obj, "contact_geoms", None)
                if not geoms:
                    continue
                if env._check_grasp(gripper=gripper, object_geoms=geoms):
                    return name
            except Exception:
                continue
        return None

    def nearest_object(
        self, exclude: tuple[str, ...] = (), max_dist: float = 0.20
    ) -> tuple[str | None, float]:
        """Object closest to the gripper — a proxy for what the policy is
        currently reaching toward, since pi0.5 emits no subtask text here."""
        gp = self.gripper_pos()
        best, best_d = None, float("inf")
        for name, det in self.detect_all().items():
            if name in exclude:
                continue
            d = det.distance_to(gp)
            if d < best_d:
                best, best_d = name, d
        if best is None or best_d > max_dist:
            return None, best_d
        return best, best_d


class NoisyDetector(Detector):
    """Wraps a detector and degrades it in controlled ways.

    Implements the doc's "vision quality" ablation:
      p_miss  - probability an object is not reported at all
      p_swap  - probability an object's identity is confused with a
                same-category / nearby instance
      pos_noise_std - Gaussian noise (metres) added to reported positions
    """

    def __init__(
        self,
        base: Detector,
        p_miss: float = 0.0,
        p_swap: float = 0.0,
        pos_noise_std: float = 0.0,
        conf_floor: float = 0.5,
        seed: int = 0,
    ):
        self.base = base
        self.p_miss = p_miss
        self.p_swap = p_swap
        self.pos_noise_std = pos_noise_std
        self.conf_floor = conf_floor
        self.rng = np.random.default_rng(seed)

    def detect_all(self) -> dict[str, Detection]:
        truth = self.base.detect_all()
        names = list(truth.keys())
        out: dict[str, Detection] = {}
        for name, det in truth.items():
            if self.rng.random() < self.p_miss:
                continue
            pos = det.pos
            if self.pos_noise_std > 0:
                pos = pos + self.rng.normal(0, self.pos_noise_std, size=3)
            reported = name
            if self.rng.random() < self.p_swap and len(names) > 1:
                others = [n for n in names if n != name]
                reported = str(self.rng.choice(others))
            conf = float(self.rng.uniform(self.conf_floor, 1.0))
            out[name] = Detection(
                name=reported, pos=pos, quat=det.quat, confidence=conf
            )
        return out

    def gripper_pos(self) -> np.ndarray:
        # Proprioception is not degraded — robots know their own joint angles.
        return self.base.gripper_pos()

    def grasped_object(self) -> str | None:
        held = self.base.grasped_object()
        if held is None:
            return None
        if self.rng.random() < self.p_miss:
            return None
        if self.rng.random() < self.p_swap:
            names = [n for n in self.base.detect_all() if n != held]
            if names:
                return str(self.rng.choice(names))
        return held

    def nearest_object(self, exclude=(), max_dist: float = 0.20):
        return self.base.nearest_object(exclude=exclude, max_dist=max_dist)
