def build_corrected_instruction(original_instruction: str, target_name: str, distractor_name: str) -> str:
    """Builds a disambiguating instruction nudging the policy away from a specific distractor.

    Starts simple: appends ", not the {distractor_name}" to the original instruction.
    `target_name` is accepted but not yet used in the string -- kept in the signature for forward
    compatibility with richer templates later (e.g. re-stating the target explicitly), and so call
    sites are self-documenting about which object is the target vs. which is the distractor being
    corrected against.
    """
    del target_name  # unused for now, see docstring
    return f"{original_instruction}, not the {distractor_name}"


def build_control_instruction(original_instruction: str, target_display: str, distractor_display: str) -> str:
    """Builds the ATHENA CONTROL prompt p-hat: the UNDESIRED instruction to steer *away* from.

    Unlike build_corrected_instruction (which builds a DESIRED prompt to blend TOWARD, the
    attract/mirror-image of ATHENA), this constructs the ATHENA-faithful control prompt used in the
    steer-away update v_steered = v_orig + gamma*(v_orig - v_control) (see
    run_athena_feedback_denoise). Following ATHENA-Feedback (Algorithm 4, arXiv 2603.19676), the
    control prompt is the original prompt with the target swapped for the *observed wrong* object --
    i.e. a positive instruction to pick up the distractor the gripper is currently drifting toward:

        original:  "put the red mug on the left plate"   (target_display="red mug")
        control:   "put the white mug on the left plate" (distractor_display="white mug")

    The denoiser is then repelled from this control prediction. We do a literal phrase swap so the
    control prompt is grammatical natural language (the CLAUDE.md 2026-07-11 bug was passing raw
    object ids like "white_yellow_mug_1" mid-sentence). If `target_display` is not found verbatim in
    the instruction (wording mismatch), fall back to a bare "pick up the {distractor}" instruction so
    a caller never silently steers away from the *original* prompt (which would be a no-op-ish self
    reference). Callers MUST log the returned string and eyeball it on a live call before trusting a
    batch -- same discipline the earlier gibberish-instruction bug taught.
    """
    if target_display in original_instruction:
        return original_instruction.replace(target_display, distractor_display)
    return f"pick up the {distractor_display}"
