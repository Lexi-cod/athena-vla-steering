"""Task registry for ATHENA steer-away experiments across the different 'wrong object' failure cases.

One AthenaTask per LIBERO task we test ATHENA on, so a single driver (run_athena_feedback_task.py) and
the pipeline health-check can be pointed at any failure case via the ATHENA_TASK env var, instead of
hardcoding red-mug constants. Display names are natural-language phrases (never raw object ids -- see the
CLAUDE.md 2026-07-11 gibberish-instruction bug); for the spatial bowl task they are deliberately
POSITIONAL ("front black bowl" / "back black bowl"), the vocabulary the action denoiser actually responds
to (cf. the 0.815-cosine up/down probe vs. the ~0 color contrast).

The ATHENA control prompt is built by build_control_instruction: the original instruction with
`target_display` swapped for the nearest distractor's display name (the observed-wrong object to steer
away from).
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class AthenaTask:
    key: str
    task_name: str
    task_suite: str
    target_object: str
    distractor_objects: tuple
    target_display: str  # target's phrase AS IT APPEARS in the instruction, for the swap
    distractor_display_names: dict  # raw distractor id -> natural-language phrase
    failure_case: str  # human label of which of the user's three cases this is


TASKS = {
    # Case 1: COLOR / object-identity confusion (same object category, wrong color). Done -> hard null.
    "red_mug": AthenaTask(
        key="red_mug",
        task_name="LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
        task_suite="libero_90",
        target_object="red_coffee_mug_1",
        distractor_objects=("porcelain_mug_1", "white_yellow_mug_1"),
        target_display="red mug",
        distractor_display_names={
            "porcelain_mug_1": "white mug",
            "white_yellow_mug_1": "yellow and white mug",
        },
        failure_case="color/identity (red vs white/yellow mug)",
    ),
    # Case 2: SPATIAL confusion (same object x3, wrong position). ~40% unsteered baseline -> has headroom.
    # Distractor display names are positional, which the denoiser reads better than attribute words.
    "middle_bowl": AthenaTask(
        key="middle_bowl",
        task_name="KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
        task_suite="libero_90",
        target_object="akita_black_bowl_2",
        distractor_objects=("akita_black_bowl_1", "akita_black_bowl_3"),
        target_display="middle black bowl",
        distractor_display_names={
            "akita_black_bowl_1": "front black bowl",
            "akita_black_bowl_3": "back black bowl",
        },
        failure_case="spatial (middle vs front/back bowl)",
    ),
    # Case 3: IDENTITY FIXATION (different object categories; arm goes for one attractor regardless
    # of what was asked). Seven distinct grocery items on a table, only orange_juice_1 is the target;
    # basket_1 is the DESTINATION, not a distractor, so it is deliberately excluded from
    # distractor_objects (including it would make the fidelity signal treat the goal as an error).
    # Verified against the bddl: instruction is "pick up the orange juice and put it in the basket",
    # so target_display "orange juice" appears verbatim and the phrase swap yields grammatical
    # control prompts, e.g. "pick up the milk and put it in the basket".
    # NOTE: unlike the other two, this task has NO measured unsteered baseline yet -- run one before
    # any steering batch (case 1 taught us a ~0% baseline is the wrong hill for a method that
    # amplifies latent competence).
    "orange_juice": AthenaTask(
        key="orange_juice",
        task_name="LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
        task_suite="libero_90",
        target_object="orange_juice_1",
        distractor_objects=(
            "alphabet_soup_1",
            "cream_cheese_1",
            "tomato_sauce_1",
            "ketchup_1",
            "milk_1",
            "butter_1",
        ),
        target_display="orange juice",
        distractor_display_names={
            "alphabet_soup_1": "alphabet soup",
            "cream_cheese_1": "cream cheese",
            "tomato_sauce_1": "tomato sauce",
            "ketchup_1": "ketchup",
            "milk_1": "milk",
            "butter_1": "butter",
        },
        failure_case="identity fixation (orange juice vs 6 other grocery items)",
    ),
}


def get_task(key: str) -> AthenaTask:
    if key not in TASKS:
        raise KeyError(f"unknown ATHENA_TASK {key!r}; known: {sorted(TASKS)}")
    return TASKS[key]
