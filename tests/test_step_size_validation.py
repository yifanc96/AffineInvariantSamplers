"""Invalid step sizes must fail before the initial search can double zero forever."""

import inspect

import jax
import jax.numpy as jnp
import pytest

import affine_invariant_samplers as samplers


NAMES = [name for name in samplers.__all__ if name.startswith('sampler_')
         and 'step_size' in inspect.signature(getattr(samplers, name)).parameters]


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('step_size', [0., -0.1, float('inf'), float('nan')])
def test_invalid_step_size_fails_before_search(name, step_size, monkeypatch):
    def fail_if_search_can_start(seed):
        pytest.fail('invalid step_size reached sampler initialization')

    monkeypatch.setattr(jax.random, 'key', fail_if_search_can_start)
    sampler = getattr(samplers, name)
    kwargs = dict(log_prob_fn=lambda x: -0.5 * jnp.sum(x*x, axis=-1),
                  initial_state=jnp.ones((4, 2)), num_samples=1,
                  warmup=0, step_size=step_size, verbose=False)
    if 'forward_fn' in inspect.signature(sampler).parameters:
        kwargs.update(forward_fn=lambda x: x, M=jnp.eye(2))
    with pytest.raises(ValueError, match='step_size must be finite and strictly positive'):
        sampler(**kwargs)

GAMMA_NAMES = [name for name in samplers.__all__ if name.startswith('sampler_')
               and 'find_init_gamma' in inspect.signature(getattr(samplers, name)).parameters]


@pytest.mark.parametrize('name', GAMMA_NAMES)
@pytest.mark.parametrize('gamma', [0., -0.1, float('inf'), float('nan')])
def test_invalid_gamma_fails_before_search(name, gamma, monkeypatch):
    def fail_if_search_can_start(seed):
        pytest.fail('invalid gamma reached sampler initialization')

    monkeypatch.setattr(jax.random, 'key', fail_if_search_can_start)
    with pytest.raises(ValueError, match='gamma must be finite and strictly positive'):
        getattr(samplers, name)(
            lambda x: -0.5 * jnp.sum(x*x, axis=-1), jnp.ones((4, 2)),
            num_samples=1, warmup=0, gamma=gamma, verbose=False)


@pytest.mark.parametrize('name', ['sampler_peaches', 'sampler_side', 'sampler_ensemble_dr_side'])
def test_degenerate_ensemble_search_fails_with_actionable_error(name):
    # Identical complement walkers produce zero proposal directions at every
    # finite scale. Search doubling must not pass infinity into production.
    with pytest.raises(ValueError, match='initial scale search requires at least two distinct walkers in each half'):
        getattr(samplers, name)(
            lambda x: -0.5 * jnp.sum(x*x, axis=-1), jnp.ones((4, 2)),
            num_samples=1, warmup=0, verbose=False)
