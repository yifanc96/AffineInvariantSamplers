"""Regressions for MAP restart selection and finite Laplace initialization."""

import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers.init import find_map, find_map_restarts, init_ensemble_from_map


def _lognormal_log_prob(x):
    return -0.5 * jnp.log(x[0])**2 - jnp.log(x[0])


def test_finite_restart_survives_an_invalid_start():
    expected = find_map(_lognormal_log_prob, jnp.array([1.]))
    result = find_map_restarts(_lognormal_log_prob, jnp.array([[-1.], [1.]]))
    assert np.isfinite(result.fun)
    assert np.isfinite(np.asarray(result.x)).all()
    np.testing.assert_allclose(result.x, expected.x)
    assert result.fun == pytest.approx(expected.fun)


def test_valid_converged_restart_is_returned():
    result = find_map_restarts(
        _lognormal_log_prob, jnp.array([[-1.], [np.exp(-1.)]]))
    assert result.success
    assert float(result.x[0]) == pytest.approx(np.exp(-1.), rel=1e-3)


def test_integer_restart_points_are_cast_to_floating_point():
    result = find_map_restarts(lambda x: -0.5 * jnp.dot(x, x), jnp.array([[0], [1]]))
    assert jnp.issubdtype(result.x.dtype, jnp.floating)
    assert float(result.x[0]) == pytest.approx(0., abs=1e-6)


def test_all_invalid_restarts_raise_a_descriptive_error():
    with pytest.raises(ValueError, match="No MAP restart produced a finite"):
        find_map_restarts(_lognormal_log_prob, jnp.array([[-1.], [-2.]]))


def test_laplace_initialization_rejects_nonfinite_map():
    with pytest.raises(ValueError, match="MAP search did not produce a finite"):
        init_ensemble_from_map(_lognormal_log_prob, jnp.array([-1.]), n_chains=4)


def test_laplace_initialization_rejects_nonfinite_hessian():
    # This proper density is differentiable at zero, but has infinite
    # second derivative there and therefore no finite Laplace metric.
    with pytest.raises(ValueError, match="MAP Hessian contains nonfinite"):
        init_ensemble_from_map(lambda x: -jnp.abs(x[0])**1.5,
                               jnp.array([0.]), n_chains=4)


def test_laplace_draw_matches_requested_covariance_scale():
    initial, result = init_ensemble_from_map(
        lambda x: -0.5 * jnp.dot(x, x), jnp.array([0., 0.]),
        n_chains=4000, seed=4, scale=2.)
    assert result.success
    assert np.isfinite(np.asarray(initial)).all()
    np.testing.assert_allclose(np.var(initial, axis=0), [4., 4.], rtol=0.1)
