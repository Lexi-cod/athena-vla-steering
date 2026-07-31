import numpy as np


def get_object_keypoints(obs, object_names):
    """Extract world-frame 3D keypoint (position) for each named object from a LIBERO obs dict.

    Uses the built-in "<object_name>_pos" observation keys (world-frame body_xpos),
    not manual sim.data.body_xpos lookups.
    """
    return {name: np.asarray(obs[f"{name}_pos"]) for name in object_names}


def compute_fidelity_signal(obs, target_object, distractor_objects, eef_key="robot0_eef_pos"):
    """Ground-truth "wrong object picked" fidelity signal.

    Compares the end-effector's world-frame distance to the correct target object
    against its distance to the nearest distractor. Both distances are computed in
    the same world frame (obs["robot0_eef_pos"] vs obs["<object>_pos"]) -- NOT via
    the "<object>_to_robot0_eef_pos" key, which is expressed in the gripper's local
    (rotated) frame and is not directly comparable to a simple distance to another
    object's gripper-frame vector.

    Returns:
        signal: min_distance_to_distractor - distance_to_target.
            Positive => eef is closer to the correct target than to any distractor (good).
            Negative => eef is closer to some distractor than to the target (salience-bias risk).
        target_dist: eef -> target_object distance.
        nearest_distractor: (name, distance) of the closest distractor.
    """
    eef_pos = np.asarray(obs[eef_key])
    keypoints = get_object_keypoints(obs, [target_object, *distractor_objects])

    target_dist = float(np.linalg.norm(keypoints[target_object] - eef_pos))

    distractor_dists = {name: float(np.linalg.norm(keypoints[name] - eef_pos)) for name in distractor_objects}
    nearest_name = min(distractor_dists, key=distractor_dists.get)
    nearest_dist = distractor_dists[nearest_name]

    signal = nearest_dist - target_dist
    return signal, target_dist, (nearest_name, nearest_dist)


def analyze_signal_trajectory(log, diverge_thresh=-0.1, recover_thresh=-0.05, grasp_range_thresh=0.08):
    """Detects the divergence / recovery / relapse pattern in one rollout's fidelity_signal trace.

    Args:
        log: list of (t, signal, target_dist, nearest_distractor, nearest_dist) rows, in step order
            (as produced by run_fidelity_rollout.run_episode).
        diverge_thresh: signal below this counts as "clearly negative" (salience-bias risk).
        recover_thresh: signal above this after a divergence counts as "returned to near-neutral".
        grasp_range_thresh: target_dist below this counts as "actually within grasping range of the
            target", in meters. Default 0.08 based on the KITCHEN_SCENE2 middle-black-bowl validation
            batch (20 rollouts): min target_dist reached 0.050-0.065m on all 8 successes vs. never
            below 0.104m on any of the 12 failures -- a clean, zero-overlap gap. May need retuning for
            objects of very different size/geometry.

    Returns:
        dict with:
            first_divergence_step: first t where signal < diverge_thresh, or None if it never happens.
            recovery_step: first t after first_divergence_step where signal > recover_thresh AND
                target_dist has both improved relative to its value at first_divergence_step AND dropped
                below grasp_range_thresh, or None.
                (Requiring target_dist to improve rules out the "recovery" that's actually just the eef
                moving away from an already-approached distractor -- e.g. carrying it off -- which raises
                nearest_dist, and therefore the scalar signal, without the eef ever heading back toward
                the target. Requiring it to also cross grasp_range_thresh additionally rules out cm-scale
                noise: in the same validation batch, 5/6 scalar-and-improvement "recoveries" among failed
                rollouts were 0.5-1.2cm target_dist improvements that never got remotely close to grasp
                range, vs. the one real recovery among successes improving target_dist by 18cm down to
                0.050m. See scripts/steering/results/bowl_traces/trace_rollout*.csv.)
            relapse_step: first t after recovery_step where signal < diverge_thresh again, or None
                (also None if there was no recovery to relapse from).
    """
    first_divergence_step = None
    target_dist_at_divergence = None
    recovery_step = None
    relapse_step = None

    for t, signal, target_dist, *_ in log:
        if first_divergence_step is None:
            if signal < diverge_thresh:
                first_divergence_step = t
                target_dist_at_divergence = target_dist
            continue
        if recovery_step is None:
            if (
                signal > recover_thresh
                and target_dist < target_dist_at_divergence
                and target_dist < grasp_range_thresh
            ):
                recovery_step = t
            continue
        if relapse_step is None and signal < diverge_thresh:
            relapse_step = t

    return {
        "first_divergence_step": first_divergence_step,
        "recovery_step": recovery_step,
        "relapse_step": relapse_step,
    }
