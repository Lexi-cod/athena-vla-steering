"""Pre-flight checks for dual_instruction_denoise.py, before running it against the real red-mug task.

Runs on CPU (JAX_PLATFORMS=cpu) with a randomly-initialized (untrained) pi05_libero-shaped model and
fake_obs(), same setup as test_pi0_split.py -- avoids GPU contention with the live policy server.

IMPORTANT FINDING, discovered while writing this test: pi05's action expert uses adaLN-Zero-style
gating (RMSNorm.__call__ in gemma.py, the `adaptive RMSNorm` branch -- `modulation = nn.Dense(...,
kernel_init=nn.initializers.zeros)(cond)`, then `_gated_residual` does `x + y * gate`). This is a
standard, intentional diffusion-transformer init pattern: every adaRMS-conditioned block (attention
AND FFN) starts as an exact identity function w.r.t. its input, gated to zero, so a freshly
initialized model is architecturally guaranteed to produce IDENTICAL output regardless of the
prefix (image+instruction) or even the flow-matching timestep, until training moves the gates off
zero. Confirmed directly: swapping a real prefix cache for `kv_cache=None` (no prefix at all) or for
pure random noise produces bit-identical `denoise_step` output on a freshly-initialized model.

This means checks (b) and (e) below -- which need the model to actually respond to a changed
instruction -- are structurally untestable on a stock random-init model. They instead use a
TEST-ONLY model built with the adaRMS gate's zero-initializer monkeypatched to a small random init
(`build_probe_model`, patched and restored immediately around `config.create`) purely to unlock a
non-zero gate for mechanical validation -- this does not touch any production file, and is not a
substitute for validating against the real trained checkpoint. Checks (a)/(c)/(d) don't depend on
gating at all and run on the standard (real-init-scheme) model.

  (a) gamma=0 regression: dual-instruction loop w/ gamma=0 vs. plain sample_actions.
  (b) exaggerated-instruction test: "move the arm up" vs "move the arm down", gamma=1, does the
      steered output move measurably toward the pure-corrected-instruction output?
  (c) tokenization check: real mug-task instruction pair, confirm genuinely different token sequences.
  (d) window-boundary check: log which of the 10 steps were blended vs. pure-original.
  (e) gamma sweep: {0, 0.5, 1, 2, 4} on the exaggerated case, report magnitude/stability.
"""

import sys

import flax.linen as nn
import jax
import numpy as np

from correction_instruction import build_corrected_instruction
from dual_instruction_denoise import run_dual_instruction_denoise, with_prompt
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import nnx_utils

NUM_STEPS = 10
WINDOW_STEPS = 4


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def build_probe_model(config, rng):
    """TEST-ONLY: builds a model with adaRMS gates non-zero-initialized, to mechanically test that
    conditioning propagates to the output at all. See module docstring. Monkeypatches
    flax.linen.initializers.zeros for the duration of construction only, then restores it --
    doesn't affect any other model built in this process (e.g. the standard `model` used by checks
    (a)/(c)/(d), which must keep the real zero-init scheme)."""
    original_zeros = nn.initializers.zeros
    nn.initializers.zeros = nn.initializers.normal(stddev=0.02)
    try:
        return config.create(rng)
    finally:
        nn.initializers.zeros = original_zeros


def run_standard_checks():
    """Checks (a), (c), (d) -- don't depend on adaRMS gating, run on the standard (real-init) model."""
    config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    model = config.create(jax.random.key(0))
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)

    base_obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(42), (1, config.action_horizon, config.action_dim))

    compute_prefix_cache = nnx_utils.module_jit(model.compute_prefix_cache)
    denoise_step = nnx_utils.module_jit(model.denoise_step)

    # ---------------------------------------------------------------- (c) tokenization check, first
    # (cheap, and the mug-task strings are reused as a realistic build_corrected_instruction example
    # even though the rest of this script uses the up/down pair for the exaggerated tests below)
    section("(c) tokenization check -- real mug-task instruction pair")
    mug_original = "put the red mug on the left plate"
    mug_corrected = build_corrected_instruction(mug_original, target_name="red mug", distractor_name="white mug")
    print(f"original:  {mug_original!r}")
    print(f"corrected: {mug_corrected!r}")
    tok_orig, mask_orig = tokenizer.tokenize(mug_original)
    tok_corr, mask_corr = tokenizer.tokenize(mug_corrected)
    n_orig, n_corr = int(mask_orig.sum()), int(mask_corr.sum())
    print(f"original  tokens ({n_orig} valid): {tok_orig[:n_orig].tolist()}")
    print(f"corrected tokens ({n_corr} valid): {tok_corr[:n_corr].tolist()}")
    identical = np.array_equal(tok_orig[:n_orig], tok_corr[:n_corr]) if n_orig == n_corr else False
    truncated = n_orig >= config.max_token_len or n_corr >= config.max_token_len
    print(f"token sequences identical: {identical}   (expect False)")
    print(f"either truncated at max_token_len={config.max_token_len}: {truncated}   (expect False)")
    assert not identical, "FAIL: original and corrected instructions tokenized identically"
    assert not truncated, "FAIL: one of the instructions was truncated"
    assert n_corr > n_orig, "FAIL: corrected instruction should tokenize to more tokens than original"
    print("PASS: genuinely different, untruncated token sequences.")

    # ---------------------------------------------------------------- (a) gamma=0 regression
    section("(a) gamma=0 regression vs. sample_actions")
    obs_original = with_prompt(base_obs, tokenizer, mug_original)
    obs_corrected = with_prompt(base_obs, tokenizer, mug_corrected)

    reference = np.asarray(model.sample_actions(jax.random.key(1), obs_original, num_steps=NUM_STEPS, noise=noise))
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
    if np.allclose(reference, x_0_gamma0, atol=1e-4, rtol=1e-4):
        print("PASS: gamma=0 dual-instruction loop matches sample_actions within 1e-4.")
    else:
        print("FAIL: gamma=0 dual-instruction loop diverges from sample_actions.")
        raise SystemExit(1)

    # ---------------------------------------------------------------- (d) window-boundary check
    section("(d) window-boundary check")
    _, step_log = run_dual_instruction_denoise(
        model, obs_original, obs_corrected, noise,
        num_steps=NUM_STEPS, window_steps=WINDOW_STEPS, gamma=1.0,
        compute_prefix_cache=compute_prefix_cache, denoise_step=denoise_step,
    )
    for entry in step_log:
        print(f"  step {entry.step}  time={entry.time:+.3f}  blended={entry.blended}")
    blended_steps = [e.step for e in step_log if e.blended]
    expected = list(range(WINDOW_STEPS))
    print(f"blended steps: {blended_steps}  (expected: {expected})")
    assert blended_steps == expected, "FAIL: blended steps don't match intended window"
    print(f"PASS: exactly steps {expected} (first {WINDOW_STEPS} of {NUM_STEPS}) were blended.")


def run_probe_checks():
    """Checks (b), (e) -- need the model to respond to conditioning, so they run on the TEST-ONLY
    non-zero-gate probe model. Kept in a separate process (see __main__ dispatch below) from
    run_standard_checks() so only one full pi05-sized model is ever resident at once -- this job's
    32GB memory limit doesn't comfortably fit two."""
    config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)
    base_obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(42), (1, config.action_horizon, config.action_dim))

    # ---------------------------------------------------------------- (b) exaggerated-instruction test
    section("(b) exaggerated-instruction test: 'move the arm up' vs 'move the arm down', gamma=1")
    print("Using the TEST-ONLY non-zero-gate probe model (see module docstring) -- a stock")
    print("random-init pi05 model is architecturally guaranteed to ignore all conditioning.")
    probe_model = build_probe_model(config, jax.random.key(0))
    probe_compute_prefix_cache = nnx_utils.module_jit(probe_model.compute_prefix_cache)
    probe_denoise_step = nnx_utils.module_jit(probe_model.denoise_step)

    obs_up = with_prompt(base_obs, tokenizer, "move the arm up")
    obs_down = with_prompt(base_obs, tokenizer, "move the arm down")

    baseline_up = np.asarray(
        probe_model.sample_actions(jax.random.key(1), obs_up, num_steps=NUM_STEPS, noise=noise)
    )
    baseline_down = np.asarray(
        probe_model.sample_actions(jax.random.key(1), obs_down, num_steps=NUM_STEPS, noise=noise)
    )
    steered, _ = run_dual_instruction_denoise(
        probe_model, obs_up, obs_down, noise,
        num_steps=NUM_STEPS, window_steps=WINDOW_STEPS, gamma=1.0,
        compute_prefix_cache=probe_compute_prefix_cache, denoise_step=probe_denoise_step,
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
    print("\nNOTE: even with non-zero gates, weights are otherwise randomly initialized (untrained)")
    print("-- 'up'/'down' carry no learned semantic meaning here. This checks the MECHANICAL claim")
    print("(blending measurably moves the trajectory toward the pure-corrected-instruction output),")
    print("not semantic correctness -- that requires the real trained checkpoint on the actual task.")
    if disp_norm > 1e-6 and cosine > 0.3:
        print("PASS (mechanical): steered output diverges from baseline, positively aligned with the "
              "original->corrected axis.")
    else:
        print("FLAG: steered output did not move appreciably toward the corrected-instruction direction "
              "-- investigate before trusting gamma-based blending.")

    # ---------------------------------------------------------------- (e) gamma sweep
    section("(e) gamma sweep on the exaggerated case")
    print(f"{'gamma':>6}  {'||steered - up||':>18}  {'proj. fraction':>15}  {'cosine':>8}  {'max|action|':>12}")
    for gamma in (0.0, 0.5, 1.0, 2.0, 4.0):
        out, _ = run_dual_instruction_denoise(
            probe_model, obs_up, obs_down, noise,
            num_steps=NUM_STEPS, window_steps=WINDOW_STEPS, gamma=gamma,
            compute_prefix_cache=probe_compute_prefix_cache, denoise_step=probe_denoise_step,
        )
        out = np.asarray(out)
        d = (out - baseline_up).ravel()
        d_norm = np.linalg.norm(d)
        proj = float(np.dot(d, axis) / (axis_norm**2)) if axis_norm > 0 else float("nan")
        cos = float(np.dot(d, axis) / (d_norm * axis_norm)) if d_norm > 0 and axis_norm > 0 else float("nan")
        max_abs_action = float(np.max(np.abs(out)))
        flag = ""
        if not np.isfinite(out).all():
            flag = "  <-- NaN/Inf, DEGENERATE"
        elif max_abs_action > 10 * float(np.max(np.abs(baseline_up))):
            flag = "  <-- magnitude blown up relative to baseline, likely unstable"
        print(f"{gamma:6.1f}  {d_norm:18.4f}  {proj:15.3f}  {cos:8.3f}  {max_abs_action:12.4f}{flag}")

    print("\nDone. Review the gamma sweep table above to judge where output stops looking like a "
          "reasonable nudge and starts looking unstable/degenerate before choosing a default gamma.")


if __name__ == "__main__":
    # Run as two separate process invocations (see run_probe_checks docstring for why) --
    # `python test_dual_instruction.py standard` then `python test_dual_instruction.py probe`.
    phase = sys.argv[1] if len(sys.argv) > 1 else "standard"
    if phase == "standard":
        run_standard_checks()
    elif phase == "probe":
        run_probe_checks()
    else:
        raise SystemExit(f"unknown phase {phase!r}, expected 'standard' or 'probe'")
