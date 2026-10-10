"""Optional trajectory jitter reaches warmup and production without changing defaults."""
from pathlib import Path
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers import sampler_peaches


@pytest.fixture
def x64_mode():
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


@pytest.mark.parametrize('jitter', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_jitter_rejected(jitter):
    with pytest.raises(ValueError, match='trajectory_jitter'):
        sampler_peaches(lambda q: -jnp.sum(q*q, -1), jnp.ones((8, 2)), 4,
                        trajectory_jitter=jitter, verbose=False)


def radical_inverse(n):
    fraction, value = .5, 0.
    while n:
        value += fraction * (n % 2)
        fraction *= .5
        n //= 2
    return value


@pytest.mark.parametrize('jitter', [0., .6, 1.])
def test_production_length_distribution_and_cost(jitter, x64_mode):
    initial = jax.random.normal(jax.random.key(19), (16, 2))
    _, info = sampler_peaches(lambda q: -.5*jnp.sum(q*q, -1), initial, 64,
        warmup=0, L=4, step_size=.125, max_L=8, adapt_step_size=False,
        find_init_step_size=False, trajectory_jitter=jitter, verbose=False)
    # T=.5 and eps=.125 are exact dyadic values. Independently compute the
    # continued base-two sequence, rounding and minimum-one-step rule.
    steps = [max(1, int(np.ceil(4 * (1-jitter+jitter*radical_inverse(i)))))
             for i in range(1, 65)]
    assert info['mean_L'] == np.mean(steps)
    assert info['n_grad_evals'] == 16 * (sum(steps) + 64)


@pytest.mark.parametrize('move', ['h-walk', 'h-side'])
@pytest.mark.parametrize('metric', ['affine-invariant', 'euclidean'])
def test_full_jitter_matches_frozen_sampler_with_only_jitter_changed(move, metric, x64_mode):
    root = Path(__file__).resolve().parents[1]
    reference = root / 'tests/fixtures/chees_metric_reference/peaches.py'
    if not reference.exists():
        reference = root / 'audit/robustness/ensemble_metric/reference/peaches.py'
    source = reference.read_text().replace('jitter=0.6', 'jitter=1.0').replace('jitter = 0.6', 'jitter = 1.0')
    module = types.ModuleType('peaches_full_jitter_reference')
    exec(compile(source, str(reference), 'exec'), module.__dict__)
    initial = jax.random.normal(jax.random.key(91), (16, 2))
    log_prob = lambda q: -.5*jnp.sum(q*q, -1)
    settings = dict(warmup=8, step_size=.1, L=3, max_L=16, seed=9,
                    move=move, chees_metric=metric, verbose=False)
    expected, expected_info = module.sampler_peaches(log_prob, initial, 32, **settings)
    actual, info = sampler_peaches(log_prob, initial, 32, trajectory_jitter=1., **settings)
    np.testing.assert_array_equal(actual, expected)
    assert info == expected_info
