"""Fast CPU checks of the ATHENA steer-away core math (no model weights needed).

Validates the two load-bearing properties of run_athena_feedback_denoise's per-step combine:
  1. norm-match: ||v_steered|| == ||v_original|| (ATHENA Alg 2 line 5).
  2. steer-AWAY sign: v_steered moves away from v_control relative to v_original --
     i.e. dist(v_steered_dir, v_control) > dist(v_original_dir, v_control), and the applied
     delta (v_steered - v_original) is anti-aligned with (v_control - v_original).
  3. gamma=0 is an exact no-op (v_steered == v_original).

Run: JAX_PLATFORMS=cpu .venv/bin/python scripts/steering/test_athena_math.py
"""

from dual_instruction_denoise import _match_norm
import jax.numpy as jnp
import numpy as np


def combine(v_original, v_control, gamma):
    v = v_original + gamma * (v_original - v_control)  # steer AWAY
    return _match_norm(v, v_original)


def main():
    rng = np.random.default_rng(0)
    shape = (1, 50, 32)  # (batch, action_horizon, action_dim) sized like pi05
    v_original = jnp.asarray(rng.standard_normal(shape))
    v_control = jnp.asarray(rng.standard_normal(shape))

    # 1. gamma=0 exact no-op
    v0 = combine(v_original, v_control, 0.0)
    assert jnp.max(jnp.abs(v0 - v_original)) < 1e-6, jnp.max(jnp.abs(v0 - v_original))
    print(f"[ok] gamma=0 no-op: max|v0 - v_original| = {float(jnp.max(jnp.abs(v0 - v_original))):.2e}")

    for gamma in (1.0, 4.0, 5.0):
        v = combine(v_original, v_control, gamma)

        # 2a. norm-match
        n_ref = float(jnp.linalg.norm(v_original))
        n_new = float(jnp.linalg.norm(v))
        assert abs(n_ref - n_new) / n_ref < 1e-5, (n_ref, n_new)

        # 2b. away sign: the pre-normalization delta (v_orig - v_control)*gamma is added, so the
        # applied change should be POSITIVELY aligned with (v_original - v_control), i.e. NEGATIVELY
        # aligned with (v_control - v_original) -> steering away from the control velocity.
        delta = v - v_original
        toward_control = v_control - v_original
        cos = float(
            jnp.sum(delta * toward_control)
            / (jnp.linalg.norm(delta) * jnp.linalg.norm(toward_control))
        )
        assert cos < 0, f"gamma={gamma}: expected anti-alignment with control, got cos={cos}"

        # 2c. distance to control grows (steered dir is farther from control than original was)
        d_orig = float(jnp.linalg.norm(v_original - v_control))
        d_steer = float(jnp.linalg.norm(v - v_control))
        assert d_steer > d_orig, f"gamma={gamma}: steered not farther from control ({d_steer} vs {d_orig})"

        print(
            f"[ok] gamma={gamma}: ||v||matched ({n_new:.4f} vs {n_ref:.4f}), "
            f"cos(delta, toward_control)={cos:+.3f} (<0 = away), "
            f"dist_to_control {d_orig:.3f} -> {d_steer:.3f} (grows)"
        )

    print("\nALL ATHENA-math checks passed.")


if __name__ == "__main__":
    main()
