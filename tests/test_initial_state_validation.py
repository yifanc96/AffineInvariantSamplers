"""Reject malformed initial ensembles before executing targets or scale searches."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import affine_invariant_samplers as samplers


PUBLIC_SAMPLERS = [name for name in samplers.__all__ if name.startswith("sampler_")]
ENSEMBLE_SEARCH_SAMPLERS = ["sampler_walk", "sampler_side", "sampler_ensemble_dr_side",
                            "sampler_langevin_walk", "sampler_peaches", "sampler_peams",
                            "sampler_peanuts", "sampler_pickles"]


def _must_not_execute(*args, **kwargs):
    pytest.fail("invalid initial state reached gradient/key/target execution")


def _arguments(name, initial):
    sampler = getattr(samplers, name)
    kwargs = dict(log_prob_fn=_must_not_execute, initial_state=initial,
                  num_samples=1, warmup=0, verbose=False)
    if "forward_fn" in inspect.signature(sampler).parameters:
        kwargs.update(forward_fn=_must_not_execute, M=jnp.eye(initial.shape[-1]))
    return kwargs


@pytest.mark.parametrize("name", PUBLIC_SAMPLERS)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_initial_state_rejected_before_gradient_key_or_target(name, bad_value, monkeypatch):
    initial = np.arange(8, dtype=float).reshape(4, 2)
    initial[3, 1] = bad_value
    monkeypatch.setattr(jax.random, "key", _must_not_execute)
    monkeypatch.setattr(jax, "grad", _must_not_execute)
    with pytest.raises(ValueError, match="initial_state must contain only finite values"):
        getattr(samplers, name)(**_arguments(name, initial))


@pytest.mark.parametrize("name", ENSEMBLE_SEARCH_SAMPLERS)
@pytest.mark.parametrize("half", [0, 1])
def test_search_rejects_either_constant_half_before_key_or_target(name, half, monkeypatch):
    initial = np.arange(8, dtype=float).reshape(4, 2)
    initial[2*half:2*half+2] = initial[2*half]
    monkeypatch.setattr(jax.random, "key", _must_not_execute)
    with pytest.raises(ValueError, match="at least two distinct walkers in each half"):
        getattr(samplers, name)(**_arguments(name, initial))


@pytest.mark.parametrize("name", ["sampler_walk", "sampler_side", "sampler_peaches"])
def test_constant_half_is_allowed_when_initial_search_is_disabled(name):
    sampler = getattr(samplers, name)
    sig = inspect.signature(sampler).parameters
    kwargs = dict(num_samples=2, warmup=0, verbose=False)
    kwargs.update({flag: False for flag in ("find_init_step_size", "find_init_gamma",
                                          "adapt_step_size", "adapt_gamma", "adapt_L") if flag in sig})
    if "step_size" in sig:
        kwargs["step_size"] = .05
    if "L" in sig:
        kwargs["L"] = 1
    initial = jnp.array([[0.], [0.], [1.], [2.]])
    samples, _ = sampler(lambda x: -.5*jnp.sum(x*x, axis=-1), initial, **kwargs)
    assert samples.shape == (2, 4, 1)
    assert np.all(np.isfinite(samples))


def test_search_allows_distinct_rank_deficient_half_ensembles():
    # Both halves have two distinct points in D=3; full rank is deliberately
    # unnecessary for the walk proposal and must not become an entry condition.
    initial = jnp.array([[-1., 0., 0.], [1., 0., 0.], [-2., 0., 0.], [2., 0., 0.]])
    samples, info = samplers.sampler_walk(lambda x: -.5*jnp.sum(x*x, axis=-1), initial,
                                          num_samples=1, warmup=0, verbose=False)
    assert np.all(np.isfinite(samples))
    assert 0 < info["final_step_size"] < np.inf


@pytest.mark.parametrize("name", ["sampler_kalman_move", "sampler_kalman_dr"])
def test_regularized_kalman_search_allows_constant_initial_halves(name):
    # Positive covariance regularization supplies proposal noise even when the
    # ensemble is collapsed; the ensemble-direction guard must exclude this API.
    initial = jnp.zeros((4, 1))
    samples, info = getattr(samplers, name)(
        lambda x: -.5*jnp.sum(x*x, axis=-1), lambda x: x, jnp.eye(1), initial,
        num_samples=1, warmup=0, verbose=False)
    assert np.all(np.isfinite(samples))
    assert 0 < info["final_step_size"] < np.inf
