"""Experiment configuration: the variants and ablations from the exec summary."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ExperimentConfig:
    # -- identity -----------------------------------------------------------
    run_id: str = "dev"
    variant: str = "none"          # none | static | adaptive | dual

    # -- policy server ------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # -- environment --------------------------------------------------------
    task_suite_name: str = "libero_90"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    seed: int = 7
    max_steps: int | None = None   # None -> suite default

    # -- task selection -----------------------------------------------------
    # "all"       -> every task in the suite
    # "confusable"-> only tasks with same-token distractors (the failure mode
    #                this method targets); 32/90 tasks in libero_90
    # "pilot"     -> the N most confusable tasks, for fast iteration
    task_subset: str = "all"
    pilot_size: int = 20

    # -- verification -------------------------------------------------------
    verify_every: int = 5          # control steps between checks
    tau_obj: float = 0.5           # detection-confidence threshold
    grasp_radius: float = 0.055    # m; "committing to an object" threshold
    ambiguity_margin: float = 0.02  # m; two candidates this close == ambiguous
    max_retries: int = 3
    # Verify before the gripper closes (early) vs only after a grasp (late).
    # The doc's "delay vs early intervene" ablation.
    verify_mode: str = "early"     # early | late

    # -- perception ---------------------------------------------------------
    detector: str = "oracle"       # oracle | noisy
    p_miss: float = 0.0
    p_swap: float = 0.0
    pos_noise_std: float = 0.0

    # -- dual-plan ----------------------------------------------------------
    dual_mode: str = "select"      # select | blend
    dual_alpha: float = 0.5

    # -- output -------------------------------------------------------------
    out_dir: str = "/scratch1/nalagand/athena_robotics/results"
    save_video: bool = False       # off by default: 900 episodes of mp4 is a lot
    save_video_on_failure: bool = True
    video_max_per_task: int = 2

    def resolved_max_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
            "libero_90": 400,
        }.get(self.task_suite_name, 400)


# Named configurations reproducing the doc's baselines and ablations.
PRESETS: dict[str, dict] = {
    # -- baselines (section 5) ---------------------------------------------
    "baseline": dict(variant="none", detector="oracle"),
    "static": dict(variant="static", detector="oracle"),
    "adaptive": dict(variant="adaptive", detector="oracle"),
    "dual_select": dict(variant="dual", detector="oracle", dual_mode="select"),
    "dual_blend": dict(variant="dual", detector="oracle", dual_mode="blend"),

    # -- ablation: vision quality ------------------------------------------
    "adaptive_noisy_light": dict(
        variant="adaptive", detector="noisy",
        p_miss=0.05, p_swap=0.05, pos_noise_std=0.005,
    ),
    "adaptive_noisy_heavy": dict(
        variant="adaptive", detector="noisy",
        p_miss=0.15, p_swap=0.20, pos_noise_std=0.02,
    ),

    # -- ablation: early vs late intervention -------------------------------
    "adaptive_late": dict(variant="adaptive", detector="oracle", verify_mode="late"),

    # -- ablation: retry budget --------------------------------------------
    "adaptive_r1": dict(variant="adaptive", detector="oracle", max_retries=1),
    "adaptive_r5": dict(variant="adaptive", detector="oracle", max_retries=5),

    # -- ablation: threshold strictness ------------------------------------
    "adaptive_strict": dict(
        variant="adaptive", detector="oracle", grasp_radius=0.08, tau_obj=0.7
    ),
    "adaptive_loose": dict(
        variant="adaptive", detector="oracle", grasp_radius=0.035, tau_obj=0.3
    ),
}


def from_preset(name: str, **overrides) -> ExperimentConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    kw = dict(PRESETS[name])
    kw.setdefault("run_id", name)
    kw.update(overrides)
    return ExperimentConfig(**kw)
