"""Regressions for positive-width stretch adaptation during warmup."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers import sampler_stretch, sampler_ensemble_dr_stretch


@pytest.mark.parametrize("sampler", [sampler_stretch, sampler_ensemble_dr_stretch])
def test_high_dimension_warmup_keeps_positive_width_and_moves(sampler):
    initial = jax.random.normal(jax.random.key(42), (200, 100))
    log_prob = lambda x: -0.5 * jnp.sum(x * x, axis=-1)
    samples, info = sampler(log_prob, initial, 16, warmup=50, seed=0, verbose=False)
    assert info["final_a"] > 1.
    assert np.isfinite(info["final_a"])
    assert info["acceptance_rate"] > 0.
    assert np.isfinite(np.asarray(samples)).all()
    assert bool(jnp.any(samples[1:] != samples[:-1]))


@pytest.mark.parametrize("sampler", [sampler_stretch, sampler_ensemble_dr_stretch])
def test_disabled_adaptation_preserves_legacy_fixed_parameter(sampler):
    initial = jax.random.normal(jax.random.key(42), (12, 2))
    log_prob = lambda x: -0.5 * jnp.sum(x * x, axis=-1)
    _, info = sampler(log_prob, initial, 4, warmup=0, a=0.5,
                      adapt_a=False, seed=0, verbose=False)
    assert info["final_a"] == 0.5


@pytest.mark.parametrize("sampler", [sampler_stretch, sampler_ensemble_dr_stretch])
@pytest.mark.parametrize("a", [0.5, 1.])
def test_adaptation_requires_a_positive_width(sampler, a):
    with pytest.raises(ValueError, match="adapt_a=True requires a > 1"):
        sampler(lambda x: -0.5 * jnp.sum(x * x, axis=-1),
                jnp.ones((4, 2)), 2, warmup=0, a=a, verbose=False)
