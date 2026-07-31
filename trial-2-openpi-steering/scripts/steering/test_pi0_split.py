"""Regression + decomposition-fidelity test for the Pi0.compute_prefix_cache / Pi0.denoise_step split.

Two checks:
  (a) sample_actions still produces the same output as before the split was added (regression) --
      verified separately via git stash (compute_prefix_cache/denoise_step are pure additions, no
      lines of sample_actions changed; see the diff).
  (b) manually looping compute_prefix_cache + denoise_step `num_steps` times, with the same fixed
      noise, reproduces sample_actions's output on the same inputs -- proving the split is a
      faithful decomposition, not just superficially similar code.

Runs on CPU (JAX_PLATFORMS=cpu) with a randomly-initialized (untrained) pi05_libero-shaped model and
fake_obs() -- this is a pure numerics/decomposition check, unrelated to model quality, and avoids GPU
memory contention with a live policy server.
"""

import jax
import numpy as np

from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils

NUM_STEPS = 10


def main():
    config = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    model = config.create(jax.random.key(0))

    obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(42), (1, config.action_horizon, config.action_dim))

    # --- reference: sample_actions (untouched, single jax.lax.while_loop / jit boundary) ---
    reference = model.sample_actions(jax.random.key(1), obs, num_steps=NUM_STEPS, noise=noise)
    reference = np.asarray(reference)
    print(f"sample_actions output: shape={reference.shape} checksum={reference.sum():.6f}")

    # --- manual decomposition: compute_prefix_cache once + denoise_step x NUM_STEPS, module_jit-wrapped ---
    compute_prefix_cache = nnx_utils.module_jit(model.compute_prefix_cache)
    denoise_step = nnx_utils.module_jit(model.denoise_step)

    prefix_cache = compute_prefix_cache(obs)
    dt = -1.0 / NUM_STEPS
    x_t, time = noise, 1.0
    for _ in range(NUM_STEPS):
        v_t = denoise_step(obs, prefix_cache, x_t, time)
        x_t = x_t + dt * v_t
        time = time + dt
    manual = np.asarray(x_t)
    print(f"manual decomposition output: shape={manual.shape} checksum={manual.sum():.6f}")

    max_abs_diff = np.max(np.abs(reference - manual))
    print(f"\nmax abs diff (sample_actions vs manual compute_prefix_cache+denoise_step loop): {max_abs_diff:.3e}")

    if np.allclose(reference, manual, atol=1e-4, rtol=1e-4):
        print("PASS: manual decomposition reproduces sample_actions within 1e-4 tolerance.")
    else:
        print("FAIL: manual decomposition diverges from sample_actions beyond tolerance.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
