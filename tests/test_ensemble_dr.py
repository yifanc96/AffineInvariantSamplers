"""Deterministic path-balance regressions for the ensemble DR kernels."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers.ensemble_dr import (
    _dr_side_log_acceptance,
    _dr_stretch_log_acceptance,
    _log_rejection_probability,
    sampler_ensemble_dr_side,
    sampler_ensemble_dr_stretch,
)


def _stretch_density(z, a):
    if not 1. / a <= z <= a:
        return 0.
    return 1. / (2. * (math.sqrt(a) - 1. / math.sqrt(a)) * math.sqrt(z))


def _normal_density(displacement, sigma):
    return math.exp(-0.5 * (displacement / sigma)**2) / (math.sqrt(2. * math.pi) * sigma)


def _accept(log_ratio):
    return math.exp(min(0., float(log_ratio)))


@pytest.mark.parametrize("dim, r, r1, r2, a1, a2", [
    (1, 1., 1.8, 1.2, 2., 1.5),
    (4, 1., 1.8, 1.2, 2., 1.5),
    (4, 1., 1.8, 1.15, 2.5, 1.4),
])
def test_stretch_delayed_path_balance(dim, r, r1, r2, a1, a2):
    # Test the full forward/backward path densities, independently of the
    # simplified stretch exponent in the production acceptance helper.
    target = lambda radius: radius**(dim - 1) * math.exp(-radius**2)
    q1 = lambda dest, source: _stretch_density(dest / source, a1) / source
    q2 = lambda dest, source: _stretch_density(dest / source, a2) / source
    alpha1 = lambda source, dest: min(
        1., target(dest) * q1(source, dest) / (target(source) * q1(dest, source)))

    forward_la2 = _dr_stretch_log_acceptance(
        -r**2, -r1**2, -r2**2, r1 / r, r2 / r, dim, a1)
    reverse_la2 = _dr_stretch_log_acceptance(
        -r2**2, -r1**2, -r**2, r1 / r2, r / r2, dim, a1)
    forward = (target(r) * q1(r1, r) * q2(r2, r)
               * (1. - alpha1(r, r1)) * _accept(forward_la2))
    reverse = (target(r2) * q1(r1, r2) * q2(r, r2)
               * (1. - alpha1(r2, r1)) * _accept(reverse_la2))
    assert forward > 0.
    assert forward == pytest.approx(reverse, rel=2e-6)


@pytest.mark.parametrize("x, y, z, sigma1, sigma2", [
    (0.4, 2.8, 0.9, 1.1, 0.45),
    (-0.7, -2.2, -0.2, 1.4, 0.6),
])
def test_side_delayed_path_balance(x, y, z, sigma1, sigma2):
    target = lambda state: math.exp(-0.5 * state**2)
    alpha1 = lambda source, dest: min(1., target(dest) / target(source))
    forward_la2 = _dr_side_log_acceptance(
        -0.5 * x**2, -0.5 * y**2, -0.5 * z**2,
        (y - x) / sigma1, (z - x) / sigma2, sigma2 / sigma1)
    reverse_la2 = _dr_side_log_acceptance(
        -0.5 * z**2, -0.5 * y**2, -0.5 * x**2,
        (y - z) / sigma1, (x - z) / sigma2, sigma2 / sigma1)
    forward = (target(x) * _normal_density(y - x, sigma1)
               * _normal_density(z - x, sigma2)
               * (1. - alpha1(x, y)) * _accept(forward_la2))
    reverse = (target(z) * _normal_density(y - z, sigma1)
               * _normal_density(x - z, sigma2)
               * (1. - alpha1(z, y)) * _accept(reverse_la2))
    assert forward > 0.
    assert forward == pytest.approx(reverse, rel=2e-6)


def test_log_rejection_probability_preserves_tiny_and_exact_zero():
    logs = _log_rejection_probability(jnp.array([-1e-8, 0., 1., -jnp.inf]))
    assert float(logs[0]) == pytest.approx(math.log(-math.expm1(-1e-8)), rel=1e-6)
    assert np.isneginf(float(logs[1]))
    assert np.isneginf(float(logs[2]))
    assert float(logs[3]) == 0.


def test_stretch_rejects_reverse_path_outside_support():
    # Both forward stretches lie in their respective supports, but the
    # rejected proposal cannot be proposed from the stage-two candidate.
    la2 = _dr_stretch_log_acceptance(0., -5., -1., 1.9, 0.6, 2, 2.)
    assert np.isneginf(float(la2))


@pytest.mark.parametrize("lp1, lp2", [(0., -1.), (-1., -2.)])
def test_stage_two_rejects_zero_forward_or_reverse_rejection(lp1, lp2):
    # With unit stretch factors, the stretch and side MH ratios coincide.
    stretch = _dr_stretch_log_acceptance(0., lp1, lp2, 1., 1., 2, 2.)
    side = _dr_side_log_acceptance(0., lp1, lp2, 1., 0.5, 0.5)
    assert np.isneginf(float(stretch))
    assert np.isneginf(float(side))


def test_stage_two_preserves_positive_infinite_acceptance_ratio():
    stretch = _dr_stretch_log_acceptance(0., -2., jnp.inf, 1., 1., 2, 2.)
    side = _dr_side_log_acceptance(0., -2., jnp.inf, 1., 0.5, 0.5)
    assert np.isposinf(float(stretch))
    assert np.isposinf(float(side))


@pytest.mark.parametrize("sampler, options", [
    # Preserve the existing stretch parameterization.  These settings give
    # a nondegenerate second-stage stretch parameter a2 = a * shrink = 1.5.
    (sampler_ensemble_dr_stretch, {"a": 3., "adapt_a": False}),
    (sampler_ensemble_dr_side,
     {"gamma": 2.5, "adapt_gamma": False, "find_init_gamma": False}),
])
def test_stage_two_moves_and_acceptance_accounting(sampler, options):
    initial = jax.random.normal(jax.random.key(42), (12, 2))
    log_prob = lambda state: -0.5 * jnp.sum(state**2, axis=-1)
    samples, info = sampler(log_prob, initial, 64, warmup=0, seed=7,
                            verbose=False, **options)
    assert samples.shape == (64, 12, 2)
    assert np.isfinite(np.asarray(samples)).all()
    assert info["stage2_rate"] > 0.
    assert info["acceptance_rate"] == pytest.approx(
        info["stage1_rate"] + info["stage2_rate"], abs=1e-7)
    previous = jnp.concatenate([initial[None, ...], samples[:-1]], axis=0)
    observed_move_rate = float(jnp.mean(jnp.any(samples != previous, axis=-1)))
    assert observed_move_rate == pytest.approx(info["acceptance_rate"], abs=1e-7)
