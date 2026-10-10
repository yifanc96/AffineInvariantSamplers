"""ChEES log-time gradients use the time integrated by the trajectory."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import affine_invariant_samplers as samplers

chees_sampler = getattr(samplers, 'sampler_chees', None) or samplers.sampler_chess
MODULES = [importlib.import_module(fn.__module__) for fn in (
    samplers.sampler_peaches, samplers.sampler_peams, samplers.sampler_pickles,
    chees_sampler, samplers.sampler_mams,
)]


def update_arguments(module, positions):
    args = (positions, 2 * positions, positions)
    if module.__name__.endswith(('peaches', 'peams')):
        args += (positions, positions, 'euclidean')
    elif module.__name__.endswith('pickles'):
        args += (positions, positions)
    elif module.__name__.endswith('mams'):
        args += ('euclidean',)
    return args


@pytest.mark.parametrize('module', MODULES, ids=lambda mod: mod.__name__.split('.')[-1])
@pytest.mark.parametrize('jitter', [0., .6, 1.])
def test_chees_time_prefactor_matches_configured_jitter(module, jitter):
    # This centered point cloud has sample covariance I, so every metric agrees.
    positions = jnp.sqrt(1.5) * jnp.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    state = module._chees_init(1., 2.)
    state = state._replace(halton=jnp.asarray(.25))
    log_alpha = jnp.zeros(4)
    reg = 1e-7
    expected_time = ((1 - jitter) + jitter * .25) * 2.
    # diff_sq=4.5 and c_pro @ velocity=3 for each proposal.
    expected_gradient = expected_time * 13.5 * 4 / (4 + reg)
    result = jax.jit(lambda s: module._chees_update(
        s, log_alpha, *update_arguments(module, positions), jitter=jitter))(state)
    np.testing.assert_allclose(result.m, expected_gradient, rtol=2e-6)
    np.testing.assert_allclose(result.v, .05 * expected_gradient**2, rtol=2e-6)


@pytest.mark.parametrize('module', MODULES, ids=lambda mod: mod.__name__.split('.')[-1])
def test_chees_capped_trajectory_time_overrides_unbounded_adapter_time(module):
    positions = jnp.sqrt(1.5) * jnp.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    state = module._chees_init(1., 20.)
    state = state._replace(halton=jnp.asarray(.25))
    actual_time = .3  # eps=.1, realized capped L=3, despite much larger adapter T.
    result = jax.jit(lambda s: module._chees_update(
        s, jnp.zeros(4), *update_arguments(module, positions),
        trajectory_time=actual_time))(state)
    np.testing.assert_allclose(result.m, actual_time * 13.5, rtol=2e-6)


@pytest.mark.parametrize('jitter', [0., 1.])
def test_mams_passes_realized_capped_time_and_configured_jitter(monkeypatch, jitter):
    module = importlib.import_module(samplers.sampler_mams.__module__)
    original_step = module._mams_step
    original_update = module._chees_update
    original_length = module._chees_L
    nominal_calls = []
    integration_times = []
    seen = []

    def record_step(state, potential, gradient, eps, length, *args, **kwargs):
        if not isinstance(length, jax.core.Tracer):
            integration_times.append(float(eps * length))
        return original_step(state, potential, gradient, eps, length, *args, **kwargs)

    def record_update(*args, **kwargs):
        assert kwargs['jitter'] == jitter
        assert float(kwargs['trajectory_time']) == pytest.approx(integration_times[-1])
        seen.append(float(kwargs['trajectory_time']))
        return original_update(*args, **kwargs)

    def record_length(*args, **kwargs):
        if kwargs.get('bar', False):
            assert kwargs.get('jitter') == jitter
            nominal_calls.append(True)
        return original_length(*args, **kwargs)

    monkeypatch.setattr(jax, 'jit', lambda fn, *args, **kwargs: fn)
    monkeypatch.setattr(module, '_chees_L', record_length)
    monkeypatch.setattr(module, '_mams_step', record_step)
    monkeypatch.setattr(module, '_chees_update', record_update)
    initial = jax.random.normal(jax.random.key(4), (8, 2))
    _, info = samplers.sampler_mams(
        lambda x: -.5 * jnp.dot(x, x), initial, 1, warmup=2,
        step_size=.1, max_L=1, jitter=jitter, L='auto',
        adapt_step_size=False, find_init_step_size=False, verbose=False)
    assert len(seen) == 2
    assert len(nominal_calls) == 1
    np.testing.assert_allclose(seen, .1)
    assert info['L'] == 1


@pytest.mark.parametrize('sampler,move_name', [
    (samplers.sampler_peaches, 'h_walk'),
    (samplers.sampler_peams, 'h_walk_mams'),
    (samplers.sampler_pickles, 'pickles_move'),
])
@pytest.mark.parametrize('max_length', [1, 5])
def test_production_acceptance_and_cap_diagnostics_match_actual_proposals(
        monkeypatch, sampler, move_name, max_length):
    module = importlib.import_module(sampler.__module__)
    original_move = getattr(module, move_name)
    probabilities, lengths = [], []

    def record(log_alpha, length):
        probabilities.append(np.clip(np.exp(np.asarray(log_alpha)), 0., 1.))
        lengths.append(int(length))

    def record_move(*args, **kwargs):
        result = original_move(*args, **kwargs)
        jax.debug.callback(record, result[1], args[6], ordered=True)
        return result

    monkeypatch.setattr(module, move_name, record_move)
    initial = jax.random.normal(jax.random.key(9), (8, 2))
    samples, info = sampler(
        lambda x: -.5 * jnp.sum(x*x, axis=-1), initial, 4, warmup=0,
        step_size=.3, L=max_length, max_L=max_length, thin_by=2,
        adapt_step_size=False, find_init_step_size=False, verbose=False)
    samples.block_until_ready()
    jax.effects_barrier()
    arithmetic = np.mean(probabilities)
    harmonic = np.mean([1 / np.mean(1 / np.maximum(p, 1e-10)) for p in probabilities])
    cap_fraction = np.mean(np.asarray(lengths) == max_length)
    assert len(probabilities) == 16  # Two halves for each of eight unthinned draws.
    assert info['mean_acceptance_probability'] == pytest.approx(arithmetic, abs=2e-7)
    assert info['mean_harmonic_acceptance_probability'] == pytest.approx(harmonic, abs=2e-7)
    assert info['max_L_hit_rate'] == pytest.approx(cap_fraction)
    assert 0 <= harmonic <= arithmetic + 2e-7 <= 1 + 2e-7
    if max_length == 1:
        assert info['max_L_hit_rate'] == 1.


@pytest.mark.parametrize('jitter', [-.1, 1.1, np.nan, np.inf])
def test_mams_rejects_invalid_jitter_fraction(jitter):
    with pytest.raises(ValueError, match='jitter'):
        samplers.sampler_mams(lambda x: -.5 * jnp.dot(x, x), jnp.ones((8, 2)), 1,
                              jitter=jitter, warmup=0, find_init_step_size=False,
                              verbose=False)


@pytest.mark.parametrize('module', MODULES, ids=lambda mod: mod.__name__.split('.')[-1])
@pytest.mark.parametrize('velocity_scale', [1e13, 1e20])
def test_finite_large_chess_signals_cannot_overflow_adam_or_poison_lengths(module, velocity_scale):
    positions = jnp.sqrt(1.5) * jnp.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=jnp.float32)
    state = module._chees_init(.1, 3)
    arguments = list(update_arguments(module, positions))
    arguments[1] = 1e6 * positions
    arguments[2] = velocity_scale * positions
    result = jax.jit(lambda s: module._chees_update(
        s, jnp.zeros(4), *arguments, trajectory_time=1.))(state)
    assert np.all(np.isfinite(np.asarray(result)))
    assert result.m > 0  # A representable positive mean still provides its direction.
    assert 1 <= int(module._chees_L(result, .1, max_L=20)) <= 20


@pytest.mark.parametrize('module', MODULES, ids=lambda mod: mod.__name__.split('.')[-1])
def test_chess_recovers_nonfinite_adam_state_and_bounds_integer_length(module):
    positions = jnp.sqrt(1.5) * jnp.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], dtype=jnp.float32)
    state = module._chees_init(.1, 3)._replace(
        log_T=jnp.asarray(np.nan), log_T_bar=jnp.asarray(np.inf),
        m=jnp.asarray(np.inf), v=jnp.asarray(np.inf))
    result = jax.jit(lambda s: module._chees_update(
        s, jnp.zeros(4), *update_arguments(module, positions), trajectory_time=.3))(state)
    assert np.all(np.isfinite(np.asarray(result)))
    for time, expected in [(np.nan, 1), (np.inf, 20), (-np.inf, 1), (-1., 1)]:
        assert int(jax.jit(module._safe_trajectory_L, static_argnums=2)(time, .1, 20)) == expected


@pytest.mark.parametrize('sampler,dimension,seed', [
    (samplers.sampler_peaches, 2, 0),
    (samplers.sampler_pickles, 2, 0),
    (samplers.sampler_peaches, 8, 0),
    (samplers.sampler_peaches, 8, 1),
])
def test_rosenbrock_warmup_cannot_return_zero_or_negative_trajectory_length(
        sampler, dimension, seed):
    # Exact short-budget protocol that previously produced NaN ChEES time,
    # L=0/minint, a frozen PICKLES chain and a growing step size near 2e13.
    log_prob = lambda x: -jnp.sum(
        100. * (x[:, 1::2] - x[:, ::2]**2)**2 + (x[:, ::2] - 1.)**2, axis=-1)
    initial = jax.random.normal(jax.random.key(42 + seed), (64, dimension))
    prepared, _ = samplers.sampler_stretch(log_prob, initial, 100,
                                          warmup=100, seed=seed, verbose=False)
    samples, info = sampler(log_prob, prepared[-1], 4, warmup=750, seed=seed,
                            max_L=20, verbose=False)
    samples.block_until_ready()
    assert np.all(np.isfinite(samples))
    assert np.isfinite(info['final_step_size']) and info['final_step_size'] > 0
    assert 1 <= info['nominal_L'] <= 20
    assert 1 <= info['mean_L'] <= 20
    minimum_cost = 4 * 64 * (2 if sampler is samplers.sampler_peaches else 1)
    assert info['n_grad_evals'] >= minimum_cost
    assert np.any(np.diff(np.asarray(samples), axis=0) != 0.)
