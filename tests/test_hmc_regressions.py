"""Regressions for Hamiltonian sampler precision and adaptation bookkeeping."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers import (
    sampler_malt, sampler_mams, sampler_nuts, sampler_peaches, sampler_peams,
    sampler_peanuts, sampler_pickles,
)

mams = importlib.import_module(sampler_mams.__module__)


@pytest.fixture
def float64_mode():
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def test_mams_step_returns_terminal_velocity_for_length_adaptation():
    positions = jnp.asarray([[1., .3], [-.2, .8], [.7, -1.], [-.5, -.3]])
    potential = lambda x: .5 * jnp.dot(x, x)
    gradient = lambda x: x
    energies = jax.vmap(potential)(positions)
    key = jax.random.key(17)
    state = (positions, positions, energies)
    _, _, _, velocity, _ = mams._mams_step(
        state, potential, gradient, .2, 3, 2, key)

    # Reconstruct the same initial velocity and trajectory from the split keys.
    key_velocity, _, key_trajectory = jax.random.split(key, 3)
    normals = jax.random.normal(key_velocity, positions.shape)
    initial_velocity = normals / jnp.sqrt(jnp.sum(normals**2, axis=1, keepdims=True)
                                          + 1e-30)
    trajectory_keys = jax.random.split(key_trajectory, len(positions))
    _, terminal_velocity, _, _, _ = jax.vmap(
        lambda x, u, energy, rng: mams._mams_trajectory(
            x, u, x, energy, potential, gradient, .2, 3, 2, 1., rng))(
                positions, initial_velocity, energies, trajectory_keys)
    np.testing.assert_allclose(velocity, terminal_velocity, rtol=1e-6, atol=1e-6)
    assert not np.allclose(velocity, initial_velocity)


def test_mams_chees_receives_proposal_velocity(monkeypatch):
    original_step = mams._mams_step
    original_update = mams._chees_update
    velocities = []
    updates = []

    def record_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if not isinstance(result[3], jax.core.Tracer):
            velocities.append(np.asarray(result[3]))
        return result

    def record_update(state, log_alpha, current, proposed, velocity, *args, **kwargs):
        np.testing.assert_allclose(velocity, velocities[-1])
        updates.append(np.asarray(velocity))
        return original_update(state, log_alpha, current, proposed, velocity,
                               *args, **kwargs)

    # Run the two warmup updates eagerly so their adaptation inputs are inspectable.
    monkeypatch.setattr(jax, 'jit', lambda fn, *args, **kwargs: fn)
    monkeypatch.setattr(mams, '_mams_step', record_step)
    monkeypatch.setattr(mams, '_chees_update', record_update)
    initial = jnp.asarray([[1., .3], [-.2, .8], [.7, -1.], [-.5, -.3]])
    samples, _ = sampler_mams(
        lambda x: -.5 * jnp.dot(x, x), initial, 1,
        warmup=2, step_size=.1, L='auto', chees_metric='euclidean',
        adapt_step_size=False, find_init_step_size=False, verbose=False)
    assert len(updates) == 2
    assert samples.shape == (1, 4, 2)


@pytest.mark.parametrize('sampler', [sampler_mams, sampler_nuts, sampler_peanuts])
@pytest.mark.parametrize('warmup', [0, 2])
def test_float64_loop_carries_match_sampler_energy_dtype(sampler, warmup, float64_mode):
    initial = jax.random.normal(jax.random.key(20), (8, 2), dtype=jnp.float64)
    if sampler is sampler_mams:
        log_prob = lambda x: -.5 * jnp.dot(x, x)
        settings = dict(L=2)
    else:
        log_prob = lambda x: -.5 * jnp.sum(x*x, axis=-1)
        settings = dict(max_tree_depth=2)
    samples, info = sampler(
        log_prob, initial, 2, warmup=warmup, step_size=.1,
        adapt_step_size=warmup > 0, find_init_step_size=False,
        verbose=False, **settings)
    assert samples.dtype == jnp.float64
    assert samples.shape == (2, 8, 2)
    assert np.all(np.isfinite(samples))
    assert 0. <= info['acceptance_rate'] <= 1.


@pytest.mark.parametrize('sampler,parameter,value', [
    (sampler_peaches, 'move', 'typo'),
    (sampler_peanuts, 'move', 'typo'),
    (sampler_peams, 'chees_metric', 'typo'),
    (sampler_mams, 'chees_metric', 'typo'),
    (sampler_mams, 'tune', 'typo'),
])
def test_unsupported_options_raise_instead_of_selecting_another_algorithm(
        sampler, parameter, value):
    initial = jnp.ones((8, 2))
    lp = lambda x: -.5 * jnp.sum(x*x, axis=-1)
    with pytest.raises(ValueError, match=parameter):
        sampler(lp, initial, 1, warmup=0, find_init_step_size=False,
                verbose=False, **{parameter: value})


@pytest.mark.parametrize('value', [-1., np.nan, np.inf])
@pytest.mark.parametrize('sampler,parameter', [
    (sampler_malt, 'gamma'), (sampler_pickles, 'gamma'),
    (sampler_mams, 'langevin'), (sampler_peams, 'langevin'),
])
def test_invalid_refresh_parameters_fail_before_sampling(sampler, parameter, value):
    initial = jnp.ones((8, 2))
    lp = lambda x: -.5 * jnp.sum(x*x, axis=-1)
    with pytest.raises(ValueError, match=parameter):
        sampler(lp, initial, 1, warmup=0, find_init_step_size=False,
                verbose=False, **{parameter: value})


def test_mams_tau_tuning_respects_requested_length_cap(monkeypatch):
    monkeypatch.setattr(mams, '_estimate_tau', lambda samples: 100.)
    initial = jax.random.normal(jax.random.key(4), (8, 2))
    samples, info = sampler_mams(
        lambda x: -.5 * jnp.dot(x, x), initial, 2,
        warmup=160, step_size=.1, L='auto', tune='tau', max_L=3,
        adapt_step_size=False, find_init_step_size=False, verbose=False)
    assert info['L'] == 3
    assert samples.shape == (2, 8, 2)
    assert np.all(np.isfinite(samples))


@pytest.mark.parametrize('length', [0, 21])
def test_mams_fixed_length_must_fit_requested_cap(length):
    with pytest.raises(ValueError, match='1 <= L <= max_L'):
        sampler_mams(lambda x: -.5 * jnp.dot(x, x), jnp.ones((8, 2)), 1,
                     L=length, max_L=20, find_init_step_size=False, verbose=False)


@pytest.mark.parametrize('sampler', [sampler_mams, sampler_peams])
def test_zero_partial_refresh_time_uses_full_refresh(sampler):
    initial = jax.random.normal(jax.random.key(8), (8, 2))
    lp = (lambda x: -.5 * jnp.dot(x, x)) if sampler is sampler_mams else \
         (lambda x: -.5 * jnp.sum(x*x, axis=-1))
    samples, info = sampler(lp, initial, 2, warmup=0, step_size=.1,
                            L=2, langevin=0., adapt_step_size=False,
                            find_init_step_size=False, verbose=False)
    assert np.all(np.isfinite(samples))
    assert info['acceptance_rate'] > 0.


@pytest.mark.parametrize('adapt_length,warmup,length,draws,thinning,expected_steps', [
    (False, 2, 3, 5, 2, 30),
    (True, 0, 9, 4, 1, 25),  # Halton(.5,.25,.75,.125) gives L=(7,5,8,5).
])
def test_chees_reports_actual_production_gradient_cost(
        adapt_length, warmup, length, draws, thinning, expected_steps):
    import affine_invariant_samplers as samplers
    chees = getattr(samplers, 'sampler_chees', None) or samplers.sampler_chess
    counts = []

    def count_gradient(points):
        counts.append(len(points))

    def gradient(points):
        jax.debug.callback(count_gradient, points, ordered=True)
        return -points

    initial = jax.random.normal(jax.random.key(2), (8, 2))
    samples, info = chees(
        lambda x: -.5 * jnp.sum(x*x, axis=-1), initial, draws,
        warmup=warmup, step_size=.1, L=length, thin_by=thinning,
        grad_log_prob_fn=gradient, adapt_step_size=False, adapt_L=adapt_length,
        find_init_step_size=False, verbose=False)
    samples.block_until_ready()
    jax.effects_barrier()
    production_cost = (expected_steps + draws * thinning) * len(initial)
    warmup_cost = warmup * (length + 1) * len(initial)
    assert sum(counts) == production_cost + warmup_cost
    # Astro's sampler_chess reports costs; main's sampler_chees keeps these optional.
    if chees.__name__ == 'sampler_chess':
        assert info['n_grad_evals'] == production_cost
        assert info['mean_L'] == pytest.approx(expected_steps / (draws * thinning))
