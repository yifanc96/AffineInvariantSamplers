"""Public samplers must accept real-valued starts at the active JAX precision."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import affine_invariant_samplers as samplers


NAMES = [name for name in samplers.__all__ if name.startswith("sampler_")]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("input_dtype", [jnp.float32, jnp.int32])
def test_numeric_initial_arrays_do_not_change_scan_carry_type(name, input_dtype):
    # A global x64 setting previously caused float32 state + float64 noise to
    # change the carry dtype, while integer starts also broke differentiation.
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        sampler = getattr(samplers, name)
        parameters = inspect.signature(sampler).parameters
        initial = jax.random.normal(jax.random.key(7), (8, 2), dtype=jnp.float32)
        if input_dtype == jnp.int32:
            initial = jnp.round(initial * 3).astype(input_dtype)
        kwargs = dict(log_prob_fn=lambda x: -.5 * jnp.sum(x*x, axis=-1),
                      initial_state=initial, num_samples=2, warmup=0, verbose=False)
        for flag in ("find_init_step_size", "find_init_gamma", "adapt_step_size",
                     "adapt_a", "adapt_gamma", "adapt_L"):
            if flag in parameters:
                kwargs[flag] = False
        if "step_size" in parameters:
            kwargs["step_size"] = .1
        if "L" in parameters:
            kwargs["L"] = 2
        if "max_tree_depth" in parameters:
            kwargs["max_tree_depth"] = 2
        if "hessian_fn" in parameters:
            kwargs["hessian_fn"] = lambda x: jnp.eye(2, dtype=x.dtype)
        if "forward_fn" in parameters:
            kwargs.update(forward_fn=lambda x: x, M=jnp.eye(2, dtype=jnp.float64))
        if "adapt" in parameters:
            kwargs["adapt"] = None

        samples, _ = sampler(**kwargs)
        assert samples.shape == (2, 8, 2)
        assert np.all(np.isfinite(samples))
        expected_dtype = (jnp.float32 if name in
                          ("sampler_aldi", "sampler_pickles_unadjusted") else jnp.float64)
        assert samples.dtype == expected_dtype
    finally:
        jax.config.update("jax_enable_x64", previous)

