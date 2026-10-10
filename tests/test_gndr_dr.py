"""Deterministic regressions for GN delayed-rejection acceptance."""

from functools import lru_cache
import importlib.util
from itertools import product
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers._dr_utils import (
    _log_accept_ratio,
    _safe_log1m_exp,
)
from affine_invariant_samplers import gndr

if importlib.util.find_spec("affine_invariant_samplers.gndr_full") is not None:
    from affine_invariant_samplers import gndr_full
else:
    gndr_full = None


@pytest.mark.parametrize("log_alpha", [0., -1e-8, -4e-13, -math.log(2), -100., -math.inf])
def test_rejection_log_preserves_float32_probabilities(log_alpha):
    value = np.float32(log_alpha)
    expected = -math.inf if value == 0 else math.log(-math.expm1(float(value)))
    for fn in (_safe_log1m_exp, jax.jit(_safe_log1m_exp)):
        actual = float(fn(jnp.asarray(value)))
        if math.isinf(expected):
            assert actual == expected
        else:
            assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_acceptance_ratio_distinguishes_zero_and_invalid_densities():
    numerator = jnp.array([0., -jnp.inf, 0., -jnp.inf, jnp.inf, jnp.nan, 0., jnp.inf])
    denominator = jnp.array([0., 0., -jnp.inf, -jnp.inf, 0., 0., jnp.nan, jnp.inf])
    expected = np.array([0., -np.inf, 0., 0., 0., -np.inf, -np.inf, -np.inf])
    np.testing.assert_array_equal(jax.jit(_log_accept_ratio)(numerator, denominator), expected)


@pytest.mark.parametrize("small", [1e-8, 1e-13])
def test_stage2_retains_near_certain_rejection_correction(small):
    forward = np.float32(-4 * small)
    reverse = np.float32(-2 * small)
    expected = math.log(math.expm1(float(reverse)) / math.expm1(float(forward)))
    actual = jax.jit(gndr._dr_accept_stage2)(
        0., 0., 0., 0., 0., 0., 0., 0., forward, reverse,
    )
    assert float(actual) == pytest.approx(expected, abs=2e-6)


def test_stage3_retains_two_small_rejection_corrections():
    actual = jax.jit(gndr._dr_accept_stage3)(
        0., 0., 0., 0., 0., 0., 0., 0., -4e-8, -2e-8, -8e-8, -4e-8,
    )
    assert float(actual) == pytest.approx(-math.log(4), abs=3e-6)


@pytest.mark.parametrize("forward,reverse,expected", [(0., -1., 0.), (0., 0., 0.), (-1., 0., -np.inf)])
def test_stage2_exact_zero_path_densities(forward, reverse, expected):
    actual = jax.jit(gndr._dr_accept_stage2)(
        0., 0., 0., 0., 0., 0., 0., 0., forward, reverse,
    )
    assert float(actual) == expected


def test_undefined_reverse_model_outside_support_allows_valid_retry():
    # The rejected first proposal has zero target density and an undefined
    # gradient/metric. Its rejection probability must still be exactly one.
    first = gndr._dr_accept_stage1(0., -jnp.inf, 0., jnp.nan)
    reverse_first = gndr._dr_accept_stage1(0., -jnp.inf, 0., jnp.nan)
    assert float(first) == -np.inf
    assert float(_safe_log1m_exp(first)) == 0.
    retry = jax.jit(gndr._dr_accept_stage2)(
        0., 0., 0., 0., 0., 0., jnp.nan, jnp.nan, first, reverse_first,
    )
    assert float(retry) == 0.


def _finite_state_problem(depth):
    rng = np.random.default_rng(31415)
    pi = rng.dirichlet(np.ones(3))
    kernels = rng.dirichlet(np.ones(3), size=(depth, 3))
    paths = np.array(list(product(range(3), repeat=depth + 1)))
    return pi, kernels, paths


def _reference_acceptance(pi, kernels):
    """Probability-space endpoint-swap reference, independent of log helpers."""
    @lru_cache(None)
    def alpha(path):
        origin, *proposals = path
        endpoint = proposals[-1]
        previous = proposals[:-1]
        stage = len(previous)
        numerator = pi[endpoint] * kernels[stage, endpoint, origin]
        denominator = pi[origin] * kernels[stage, origin, endpoint]
        for i, rejected in enumerate(previous):
            numerator *= kernels[i, endpoint, rejected] * (
                1 - alpha((endpoint, *previous[:i], rejected)))
            denominator *= kernels[i, origin, rejected] * (
                1 - alpha((origin, *previous[:i], rejected)))
        return min(1., numerator / denominator) if denominator else 1.
    return alpha


def _implemented_alphas(log_pi, log_kernels, paths, use_full):
    path_lp = [log_pi[paths[:, i]] for i in range(paths.shape[1])]

    def q(src, dest, stage):
        return log_kernels[stage, paths[:, src], paths[:, dest]]

    if use_full:
        alpha = gndr_full._make_dr_accept(path_lp, q)
    else:
        cache = {}

        def alpha(current, proposal, previous):
            key = (current, proposal, previous)
            if key in cache:
                return cache[key]
            depth = len(previous) + 1
            if depth == 1:
                value = gndr._dr_accept_stage1(
                    path_lp[current], path_lp[proposal],
                    q(current, proposal, 0), q(proposal, current, 0))
            elif depth == 2:
                first = previous[0]
                value = gndr._dr_accept_stage2(
                    path_lp[current], path_lp[proposal],
                    q(current, proposal, 1), q(proposal, current, 1),
                    q(current, first, 0), q(proposal, first, 0),
                    q(first, current, 0), q(first, proposal, 1),
                    alpha(current, first, ()), alpha(proposal, first, ()))
            else:
                first, second = previous
                value = gndr._dr_accept_stage3(
                    path_lp[current], path_lp[proposal],
                    q(current, proposal, 2), q(proposal, current, 2),
                    q(current, first, 0), q(proposal, first, 0),
                    q(current, second, 1), q(proposal, second, 1),
                    alpha(current, first, ()), alpha(proposal, first, ()),
                    alpha(current, second, (first,)),
                    alpha(proposal, second, (first,)))
            cache[key] = value
            return value

    return jnp.stack([alpha(0, i, tuple(range(1, i)))
                      for i in range(1, paths.shape[1])], axis=1)


def _accepted_path_flux(pi, kernels, paths, log_alphas):
    result = pi[paths[:, 0]].copy()
    for i in range(log_alphas.shape[1]):
        result *= kernels[i, paths[:, 0], paths[:, i + 1]]
        result *= (np.exp(log_alphas[:, i]) if i == log_alphas.shape[1] - 1
                   else -np.expm1(log_alphas[:, i]))
    return result


def _assert_stage_flux(depth, use_full):
    pi, kernels, paths = _finite_state_problem(depth)
    reference = _reference_acceptance(pi, kernels)
    expected = np.array([reference(tuple(path)) for path in paths])
    evaluate = jax.jit(lambda log_pi, log_q, p:
                       _implemented_alphas(log_pi, log_q, p, use_full))
    args = jnp.array(np.log(pi)), jnp.array(np.log(kernels))
    forward = np.asarray(evaluate(*args, jnp.array(paths)))
    np.testing.assert_allclose(np.exp(forward[:, -1]), expected, rtol=3e-5, atol=3e-6)

    # Swap endpoints while preserving the intermediate rejection history.
    reverse_paths = paths.copy()
    reverse_paths[:, 0], reverse_paths[:, -1] = paths[:, -1], paths[:, 0]
    reverse = np.asarray(evaluate(*args, jnp.array(reverse_paths)))
    forward_flux = _accepted_path_flux(pi, kernels, paths, forward.astype(float))
    reverse_flux = _accepted_path_flux(pi, kernels, reverse_paths, reverse.astype(float))
    assert np.count_nonzero(forward_flux) > 0
    np.testing.assert_allclose(forward_flux, reverse_flux, rtol=3e-5, atol=1e-9)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_gndr_stagewise_flux_with_asymmetric_kernels(depth):
    _assert_stage_flux(depth, use_full=False)


@pytest.mark.skipif(gndr_full is None, reason="full-depth sampler is available on main")
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_full_gndr_stagewise_flux_with_asymmetric_kernels(depth):
    _assert_stage_flux(depth, use_full=True)


@pytest.mark.skipif(gndr_full is None, reason="full-depth sampler is available on main")
def test_full_recursion_handles_near_zero_and_unreachable_histories():
    def evaluate(log_pi):
        alpha = gndr_full._make_dr_accept(log_pi, lambda src, dest, stage: 0.)
        return alpha(0, 2, (1,))

    # Both first-stage rejections are real, even when exp(log_alpha) rounds 1.
    actual = jax.jit(evaluate)(jnp.array([0., -4e-8, -2e-8]))
    assert float(actual) == pytest.approx(-math.log(2), abs=3e-6)
    assert float(jax.jit(evaluate)(jnp.array([0., 1., 0.]))) == 0.


@pytest.mark.skipif(gndr_full is None, reason="full-depth sampler is available on main")
def test_full_recursion_recovers_from_undefined_model_outside_support():
    def evaluate(log_pi):
        def q(src, dest, stage):
            return jnp.where(jnp.isneginf(log_pi[src]), jnp.nan, 0.)
        alpha = gndr_full._make_dr_accept(log_pi, q)
        return jnp.array([alpha(0, 1, ()), alpha(0, 2, (1,))])

    actual = jax.jit(evaluate)(jnp.array([0., -jnp.inf, 0.]))
    np.testing.assert_array_equal(actual, [-np.inf, 0.])


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_gndr_fixed_step_public_sampler(depth):
    samples, info = gndr.sampler_gndr(
        lambda x: -0.5 * jnp.dot(x, x), jnp.array([[2.], [-1.]]),
        12, warmup=4, step_size=1.1, n_try=depth,
        hessian_fn=lambda x: jnp.eye(x.shape[0]),
        find_init_step_size=False, adapt_step_size=False, verbose=False,
    )
    assert samples.shape == (12, 2, 1)
    assert np.all(np.isfinite(samples))
    assert 0. <= info["acceptance_rate"] <= 1.


@pytest.mark.skipif(gndr_full is None, reason="full-depth sampler is available on main")
def test_full_gndr_fixed_step_public_sampler():
    samples, info = gndr_full.sampler_gndr_full(
        lambda x: -0.5 * jnp.dot(x, x), jnp.array([[2.], [-1.]]),
        12, warmup=4, step_size=1.1, n_try=5,
        hessian_fn=lambda x: jnp.eye(x.shape[0]),
        find_init_step_size=False, adapt_step_size=False, verbose=False,
    )
    assert samples.shape == (12, 2, 1)
    assert np.all(np.isfinite(samples))
    assert 0. <= info["acceptance_rate"] <= 1.
