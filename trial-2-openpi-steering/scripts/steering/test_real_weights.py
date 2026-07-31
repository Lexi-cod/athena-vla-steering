"""One-off confirming check against the REAL trained pi05_libero checkpoint (not random-init, not the
probe-model workaround from test_dual_instruction.py) -- repeats:
  (a) gamma=0 regression vs. sample_actions
  (b) exaggerated up/down instruction test, gamma=1
to confirm trained adaRMS gates produce sensible, non-degenerate dual-instruction blending, before
spending real GPU/task time on the actual red-mug rollout experiment.

Loads the checkpoint directly -- same minimal path `create_trained_policy` uses internally for the JAX
branch (policy_config.py): `train_config.model.load(_model.restore_params(checkpoint_dir / "params",
dtype=bfloat16))`. Bypasses the websocket server, the `Policy` wrapper, and the LIBERO-input transform
pipeline / norm-stats loading entirely -- none of that applies here since compute_prefix_cache/
denoise_step are called directly with synthetic fake_obs()-shaped data and raw tokenized prompts, same
pattern as test_pi0_split.py / test_dual_instruction.py.

Runs on CPU (JAX_PLATFORMS=cpu) by default: the live policy server already holds ~24.7GB of the 32GB
V100 (~8GB free), too little headroom to safely load a second ~6.2GB (bfloat16) checkpoint's
activation/compilation buffers alongside it. A single real-weights model fits comfortably within this
job's 32GB host-memory cap (unlike the earlier two-random-init-models-at-once OOM: `.load()` uses
`nnx.eval_shape` for the shape trace, which doesn't materialize a second concrete random model).
"""

import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from correction_instruction import build_corrected_instruction
from dual_instruction_denoise import run_dual_instruction_denoise, with_prompt
import openpi.models.model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils
from openpi.training import config as _config

NUM_STEPS = 10
WINDOW_STEPS = 4

# Matches scripts/serve_pi05_libero.sh's default / PI05_LIBERO_CHECKPOINT_DIR override -- the same
# checkpoint the live server already has loaded, read-only, no download.
CHECKPOINT_DIR = pathlib.Path(
    os.environ.get(
        "PI05_LIBERO_CHECKPOINT_DIR",
        "/scratch1/nalagand/openpi_cache/openpi-assets/checkpoints/pi05_libero",
    )
)


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main():
    train_config = _config.get_config("pi05_libero")
    print(f"Loading real pi05_libero checkpoint from {CHECKPOINT_DIR} ...", flush=True)
    params = _model.restore_params(CHECKPOINT_DIR / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    print("Checkpoint loaded.", flush=True)

    config = train_config.model
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)
    base_obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(42), (1, config.action_horizon, config.action_dim))

    compute_prefix_cache = nnx_utils.module_jit(model.compute_prefix_cache)
    denoise_step = nnx_utils.module_jit(model.denoise_step)
    sample_actions_jit = nnx_utils.module_jit(model.sample_actions)

    # ---------------------------------------------------------------- (a) gamma=0 regression, real weights
    section("(a) gamma=0 regression vs. sample_actions -- REAL trained weights")
    mug_original = "put the red mug on the left plate"
    mug_corrected = build_corrected_instruction(mug_original, target_name="red mug", distractor_name="white mug")
    print(f"original:  {mug_original!r}")
    print(f"corrected: {mug_corrected!r}")
    obs_original = with_prompt(base_obs, tokenizer, mug_original)
    obs_corrected = with_prompt(base_obs, tokenizer, mug_corrected)

    reference = np.asarray(sample_actions_jit(jax.random.key(1), obs_original, num_steps=NUM_STEPS, noise=noise))
    x_0_gamma0, _ = run_dual_instruction_denoise(
        model, obs_original, obs_corrected, noise,
        num_steps=NUM_STEPS, window_steps=WINDOW_STEPS, gamma=0.0,
        compute_prefix_cache=compute_prefix_cache, denoise_step=denoise_step,
    )
    x_0_gamma0 = np.asarray(x_0_gamma0)
    max_abs_diff = np.max(np.abs(reference - x_0_gamma0))
    print(f"sample_actions checksum:        {reference.sum():.6f}")
    print(f"dual-instruction (gamma=0):     {x_0_gamma0.sum():.6f}")
    print(f"max abs diff: {max_abs_diff:.3e}")
    if np.allclose(reference, x_0_gamma0, atol=1e-3, rtol=1e-3):
        print("PASS: gamma=0 dual-instruction loop matches sample_actions within 1e-3.")
    else:
        print("FLAG: gamma=0 dual-instruction loop diverges from sample_actions beyond 1e-3 -- "
              "compare magnitude against jit-vs-eager noise before concluding this is a real bug.")

    # ---------------------------------------------------------------- (b) exaggerated instruction, real weights
    section("(b) exaggerated-instruction test: 'move the arm up' vs 'move the arm down', gamma=1 -- REAL weights")
    obs_up = with_prompt(base_obs, tokenizer, "move the arm up")
    obs_down = with_prompt(base_obs, tokenizer, "move the arm down")

    baseline_up = np.asarray(sample_actions_jit(jax.random.key(1), obs_up, num_steps=NUM_STEPS, noise=noise))
    baseline_down = np.asarray(sample_actions_jit(jax.random.key(1), obs_down, num_steps=NUM_STEPS, noise=noise))
    steered, step_log = run_dual_instruction_denoise(
        model, obs_up, obs_down, noise,
        num_steps=NUM_STEPS, window_steps=WINDOW_STEPS, gamma=1.0,
        compute_prefix_cache=compute_prefix_cache, denoise_step=denoise_step,
    )
    steered = np.asarray(steered)

    axis = (baseline_down - baseline_up).ravel()
    axis_norm = np.linalg.norm(axis)
    disp = (steered - baseline_up).ravel()
    disp_norm = np.linalg.norm(disp)
    projection_fraction = float(np.dot(disp, axis) / (axis_norm**2)) if axis_norm > 0 else float("nan")
    cosine = float(np.dot(disp, axis) / (disp_norm * axis_norm)) if disp_norm > 0 and axis_norm > 0 else float("nan")

    print(f"||baseline_down - baseline_up|| (full up->down axis): {axis_norm:.4f}")
    print(f"||steered - baseline_up||  (displacement from up):    {disp_norm:.4f}")
    print(f"projection fraction along up->down axis:               {projection_fraction:+.3f}  "
          f"(0 = no movement, 1 = landed exactly on pure-down output)")
    print(f"cosine similarity (displacement, up->down axis):       {cosine:+.3f}  "
          f"(1 = same direction, 0 = orthogonal, -1 = opposite)")
    print(f"max|baseline_up|={np.max(np.abs(baseline_up)):.4f}  max|baseline_down|={np.max(np.abs(baseline_down)):.4f}  "
          f"max|steered|={np.max(np.abs(steered)):.4f}")

    blended_steps = [e.step for e in step_log if e.blended]
    print(f"blended steps: {blended_steps}")

    if not np.isfinite(steered).all():
        print("FAIL: steered output contains NaN/Inf -- degenerate.")
    elif disp_norm > 1e-6 and cosine > 0.1:
        print("PASS: real-weight steered output diverges from baseline, positively aligned with the "
              "original->corrected axis -- non-degenerate blending confirmed on real trained weights.")
    else:
        print("FLAG: real-weight steered output did not move appreciably / was not aligned with the "
              "corrected direction -- investigate before the real mug-task run.")


if __name__ == "__main__":
    main()
