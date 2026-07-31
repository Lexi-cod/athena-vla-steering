"""ATHENA-style dual-instruction denoising.

Runs π0.5's 10-step Euler flow-matching loop manually (via `Pi0.compute_prefix_cache` /
`Pi0.denoise_step`, see src/openpi/models/pi0.py, instead of `sample_actions`'s fused
`jax.lax.while_loop`), blending the predicted velocity from an original instruction with that of a
disambiguating "corrected" instruction (see correction_instruction.py) during an early window of
steps, then running the remaining steps on the original instruction alone.

    v_blended = v_original + gamma * (v_corrected - v_original)

gamma=0 -> pure v_original every step (identical to sample_actions on the original instruction).
gamma=1 -> pure v_corrected during the window.
gamma>1 -> extrapolates past the corrected direction.

See test_dual_instruction.py for the pre-flight checks (regression, tokenization, window-boundary,
gamma-sweep) that should pass before running this against a real task.
"""

import dataclasses

import jax
import numpy as np

from openpi.models import model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils

DEFAULT_NUM_STEPS = 10
DEFAULT_WINDOW_STEPS = 4
DEFAULT_GAMMA = 1.0


def with_prompt(obs: _model.Observation, tokenizer: PaligemmaTokenizer, prompt: str) -> _model.Observation:
    """Returns a copy of `obs` with a different tokenized instruction, same image/state/etc."""
    batch_size = obs.state.shape[0]
    tokens, mask = tokenizer.tokenize(prompt)
    tokenized_prompt = np.broadcast_to(tokens, (batch_size, *tokens.shape)).copy()
    tokenized_prompt_mask = np.broadcast_to(mask, (batch_size, *mask.shape)).copy()
    return _model.Observation(
        images=obs.images,
        image_masks=obs.image_masks,
        state=obs.state,
        tokenized_prompt=jax.numpy.asarray(tokenized_prompt),
        tokenized_prompt_mask=jax.numpy.asarray(tokenized_prompt_mask),
        token_ar_mask=obs.token_ar_mask,
        token_loss_mask=obs.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class DenoiseStepLog:
    step: int
    time: float
    blended: bool


def run_dual_instruction_denoise(
    model,
    obs_original: _model.Observation,
    obs_corrected: _model.Observation,
    noise: jax.Array,
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    gamma: float = DEFAULT_GAMMA,
    compute_prefix_cache=None,
    denoise_step=None,
):
    """Runs the manual Euler denoising loop with early-window velocity blending.

    `compute_prefix_cache`/`denoise_step` may be passed in pre-jitted (e.g. via
    `nnx_utils.module_jit(model.compute_prefix_cache)`) to avoid re-jitting on every call; if omitted,
    they're jitted once here.

    Returns (x_0, step_log) where step_log is a list of DenoiseStepLog, one per denoising step,
    recording whether that step's velocity was blended or pure-original.
    """
    if compute_prefix_cache is None:
        compute_prefix_cache = nnx_utils.module_jit(model.compute_prefix_cache)
    if denoise_step is None:
        denoise_step = nnx_utils.module_jit(model.denoise_step)

    prefix_cache_original = compute_prefix_cache(obs_original)
    prefix_cache_corrected = compute_prefix_cache(obs_corrected)

    dt = -1.0 / num_steps
    x_t, time = noise, 1.0
    step_log = []
    for i in range(num_steps):
        v_original = denoise_step(obs_original, prefix_cache_original, x_t, time)
        blended = i < window_steps
        if blended:
            v_corrected = denoise_step(obs_corrected, prefix_cache_corrected, x_t, time)
            v = v_original + gamma * (v_corrected - v_original)
        else:
            v = v_original
        step_log.append(DenoiseStepLog(step=i, time=time, blended=blended))
        x_t = x_t + dt * v
        time = time + dt
        if i == window_steps - 1:
            # Corrected-instruction cache is only read during the blending window (steps
            # 0..window_steps-1) -- drop the reference once that window closes so its device
            # buffers can be freed instead of sitting alongside the original cache for the
            # remaining (num_steps - window_steps) steps.
            del prefix_cache_corrected

    return x_t, step_log


def _match_norm(v_new: jax.Array, v_ref: jax.Array) -> jax.Array:
    """Rescale `v_new` to have the same per-sample norm as `v_ref` (ATHENA Alg. 2, line 5).

    ATHENA normalizes the steered noise estimate back to the original prediction's magnitude so the
    steered update "remains within the denoiser's expected scale and preserves stable, in-distribution
    sampling" (Sec 4.3.1). Norm is taken over all non-batch axes (here (action_horizon, action_dim)),
    matching ATHENA's single global-norm rescale for a batch of 1.
    """
    axes = tuple(range(1, v_new.ndim))
    ref_norm = jax.numpy.sqrt(jax.numpy.sum(v_ref**2, axis=axes, keepdims=True))
    new_norm = jax.numpy.sqrt(jax.numpy.sum(v_new**2, axis=axes, keepdims=True))
    return v_new * (ref_norm / (new_norm + 1e-8))


def run_athena_feedback_denoise(
    model,
    obs_original: _model.Observation,
    obs_control: _model.Observation,
    noise: jax.Array,
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    gamma: float = DEFAULT_GAMMA,
    compute_prefix_cache=None,
    denoise_step=None,
):
    """ATHENA-faithful steer-AWAY denoising (arXiv 2603.19676, Eq. 5 + Algorithm 2/4).

    Unlike run_dual_instruction_denoise (which steers TOWARD a corrected prompt --
    v_orig + gamma*(v_corrected - v_orig), the attract/mirror-image of ATHENA), this implements
    ATHENA's actual update: the denoiser is evaluated on the original prompt (kept as the base, the
    correct target) and on a CONTROL prompt (the *undesired* condition -- the wrong object the gripper
    is drifting toward, see correction_instruction.build_control_instruction), and the two are combined
    so the trajectory is pushed *away* from the control prediction:

        v_steered = v_original + gamma * (v_original - v_control)      # Eq. (5), == (1+gamma)v_orig - gamma v_control
        v_steered = (||v_original|| / ||v_steered||) * v_steered       # Alg. 2 line 5, norm match

    Steering is applied only during the early window (first `window_steps` of `num_steps`), then pure
    v_original -- the same early-intervention principle as ATHENA (steer from step T down to a cutoff
    t_steer, then plain diffusion) and as run_dual_instruction_denoise.

    gamma=0 -> v_steered = v_original and _match_norm is identity, so this is an exact no-op vs.
    sample_actions on the original prompt (regression property, mirrors the toward path).
    ATHENA paper uses gamma=4-5.

    Returns (x_0, step_log) where step_log records, per step, whether steering was applied.
    """
    if compute_prefix_cache is None:
        compute_prefix_cache = nnx_utils.module_jit(model.compute_prefix_cache)
    if denoise_step is None:
        denoise_step = nnx_utils.module_jit(model.denoise_step)

    prefix_cache_original = compute_prefix_cache(obs_original)
    prefix_cache_control = compute_prefix_cache(obs_control)

    dt = -1.0 / num_steps
    x_t, time = noise, 1.0
    step_log = []
    for i in range(num_steps):
        v_original = denoise_step(obs_original, prefix_cache_original, x_t, time)
        steered = i < window_steps
        if steered:
            v_control = denoise_step(obs_control, prefix_cache_control, x_t, time)
            v = v_original + gamma * (v_original - v_control)  # steer AWAY from the control prompt
            v = _match_norm(v, v_original)
        else:
            v = v_original
        step_log.append(DenoiseStepLog(step=i, time=time, blended=steered))
        x_t = x_t + dt * v
        time = time + dt
        if i == window_steps - 1:
            del prefix_cache_control

    return x_t, step_log
