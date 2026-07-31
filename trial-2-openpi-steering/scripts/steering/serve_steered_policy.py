"""Steering-aware policy server for the ATHENA-Feedback dual-instruction experiment.

A SEPARATE server from scripts/serve_policy.py (which stays completely untouched) -- reuses
WebsocketPolicyServer / WebsocketClientPolicy from openpi_client unmodified (both are already generic
over an arbitrary `obs: dict -> dict`), just with a few extra optional keys in the request:

    steer: bool (default False) -- if False, behaves identically to the normal server (delegates
        straight to the standard Policy.infer(), same code path as scripts/serve_policy.py).
    corrected_prompt: str -- required if steer=True. The disambiguating instruction to blend toward
        (see correction_instruction.build_corrected_instruction on the client side).
    gamma: float (default 1.0), window_steps: int (default 4) -- optional per-request overrides of
        the blending strength / early-window length validated in test_dual_instruction.py and
        test_real_weights.py.

Ground-truth object positions / the fidelity signal are NEVER sent over the wire -- computed entirely
client-side (see run_steered_rollout.py), matching the existing wire protocol's realism (server only
ever sees what a real robot would send: images, state, prompt) and keeping this server's own
responsibility narrow: given obs + instruction(s) + steer flag, run the right denoising path.
"""

import dataclasses
import logging
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy  # noqa: F401  (registers numpy msgpack hooks on import)
from openpi.serving import websocket_policy_server
import openpi.models.model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config

from dual_instruction_denoise import (
    DEFAULT_GAMMA,
    DEFAULT_WINDOW_STEPS,
    run_athena_feedback_denoise,
    run_dual_instruction_denoise,
)

HOST = "0.0.0.0"
PORT = int(os.environ.get("STEERED_POLICY_PORT", 8001))
NUM_STEPS = 10

CHECKPOINT_DIR = pathlib.Path(
    os.environ.get(
        "PI05_LIBERO_CHECKPOINT_DIR",
        "/scratch1/nalagand/openpi_cache/openpi-assets/checkpoints/pi05_libero",
    )
)


class SteeredPolicy(_base_policy.BasePolicy):
    """Wraps a standard, fully-transform-correct `Policy` (built via create_trained_policy, exactly
    like scripts/serve_policy.py does) to additionally support fidelity-signal-gated dual-instruction
    steering, via compute_prefix_cache/denoise_step (src/openpi/models/pi0.py) instead of the fused
    sample_actions used by the unsteered path.
    """

    def __init__(self, policy):
        self._policy = policy
        model = policy._model  # noqa: SLF001 -- intentional: reuse the exact loaded model instance
        self._compute_prefix_cache = nnx_utils.module_jit(model.compute_prefix_cache)
        self._denoise_step = nnx_utils.module_jit(model.denoise_step)
        self._rng = jax.random.key(0)

    def infer(self, obs: dict) -> dict:
        steer = obs.pop("steer", False)
        # steer_mode selects which steering math to run when steer=True:
        #   "toward" (default) -- v_orig + gamma*(v_corrected - v_orig), the original dual-instruction
        #       blend TOWARD a corrected/desired prompt (run_dual_instruction_denoise). Backward
        #       compatible: requests that omit steer_mode behave exactly as before.
        #   "away"   -- ATHENA-faithful steer AWAY from a control (undesired) prompt, Eq. 5 +
        #       norm-match (run_athena_feedback_denoise). Requires `control_prompt`.
        steer_mode = obs.pop("steer_mode", "toward")
        corrected_prompt = obs.pop("corrected_prompt", None)
        control_prompt = obs.pop("control_prompt", None)
        gamma = obs.pop("gamma", DEFAULT_GAMMA)
        window_steps = obs.pop("window_steps", DEFAULT_WINDOW_STEPS)
        # Optional: client-supplied integer seed for paired same-noise-seed comparisons (e.g. unsteered
        # vs. corrected-only on the identical initial noise draw, to isolate whether a correction changed
        # the outcome for that specific draw rather than comparing across independent stochastic samples).
        # Deliberately a seed, not a raw noise array, so the client never needs to know action_horizon/
        # action_dim -- the server derives the (1, action_horizon, action_dim) array deterministically.
        noise_seed = obs.pop("noise_seed", None)
        fixed_noise = None
        if noise_seed is not None:
            fixed_noise = jax.random.normal(
                jax.random.key(int(noise_seed)),
                (1, self._policy._model.action_horizon, self._policy._model.action_dim),  # noqa: SLF001
            )

        if not steer:
            # Identical code path to the unsteered server (scripts/serve_policy.py) -- this is what
            # makes the steer=False regression check a meaningful equivalence proof, not just a
            # similar-looking reimplementation. `noise` kwarg is a no-op passthrough when None (matches
            # Policy.infer's existing default), so this is unchanged for every call that doesn't set
            # noise_seed.
            return self._policy.infer(obs, noise=np.asarray(fixed_noise) if fixed_noise is not None else None)

        # Select which second prompt (besides the original) the steering uses, and the denoise fn,
        # based on steer_mode. Both paths precompute two prefix caches (original + second prompt, same
        # image/state) and run the manual Euler loop; they differ only in the per-step velocity combine.
        if steer_mode == "away":
            if not control_prompt:
                raise ValueError("steer=True with steer_mode='away' requires a non-empty control_prompt")
            second_prompt = control_prompt
            denoise_fn = run_athena_feedback_denoise
        elif steer_mode == "toward":
            if not corrected_prompt:
                raise ValueError("steer=True with steer_mode='toward' requires a non-empty corrected_prompt")
            second_prompt = corrected_prompt
            denoise_fn = run_dual_instruction_denoise
        else:
            raise ValueError(f"unknown steer_mode {steer_mode!r} (expected 'toward' or 'away')")

        # Build the two (image/state identical, prompt differing) transformed inputs the same way
        # Policy.infer() does -- see src/openpi/policies/policy.py:68-106. Uses the SAME
        # _input_transform/_output_transform the unsteered path uses, for parity.
        obs_original_raw = jax.tree.map(lambda x: x, obs)
        obs_second_raw = jax.tree.map(lambda x: x, {**obs, "prompt": second_prompt})

        inputs_original = self._policy._input_transform(obs_original_raw)  # noqa: SLF001
        inputs_second = self._policy._input_transform(obs_second_raw)  # noqa: SLF001
        inputs_original = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_original)
        inputs_second = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_second)

        observation_original = _model.Observation.from_dict(inputs_original)
        observation_second = _model.Observation.from_dict(inputs_second)

        if fixed_noise is not None:
            noise = fixed_noise
        else:
            self._rng, noise_rng = jax.random.split(self._rng)
            noise = jax.random.normal(
                noise_rng, (1, self._policy._model.action_horizon, self._policy._model.action_dim)  # noqa: SLF001
            )

        actions, step_log = denoise_fn(
            self._policy._model,  # noqa: SLF001
            observation_original,
            observation_second,
            noise,
            num_steps=NUM_STEPS,
            window_steps=window_steps,
            gamma=gamma,
            compute_prefix_cache=self._compute_prefix_cache,
            denoise_step=self._denoise_step,
        )

        outputs = {"state": inputs_original["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._policy._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {"steered_steps": [e.step for e in step_log if e.blended]}
        return outputs

    def reset(self) -> None:
        self._policy.reset()


def main():
    logging.basicConfig(level=logging.INFO)
    train_config = _config.get_config("pi05_libero")
    logging.info(f"Loading pi05_libero checkpoint from {CHECKPOINT_DIR} ...")
    policy = _policy_config.create_trained_policy(train_config, CHECKPOINT_DIR)
    logging.info("Checkpoint loaded.")

    steered_policy = SteeredPolicy(policy)
    server = websocket_policy_server.WebsocketPolicyServer(
        steered_policy, host=HOST, port=PORT, metadata=policy.metadata
    )
    logging.info(f"Serving steered policy on {HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
