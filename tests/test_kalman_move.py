"""Proposal covariance and density consistency for the ordinary Kalman move."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers.kalman_move import _ekm_propose


@pytest.mark.parametrize("scale", [0., 1e-4, 1.])
@pytest.mark.parametrize("oblique", [False, True])
def test_sampling_and_density_share_regularized_covariance(scale, oblique):
    # Include collapsed, tiny, and deficient complement covariance matrices.
    active = jnp.array([[0.7, -0.2]])
    second = scale if oblique else 0.
    complement = jnp.array([[-scale, -second], [scale, second]])
    key = jax.random.key(19)
    h = 0.3
    precision = jnp.diag(jnp.array([2., 0.5]))
    proposal, actual_diff = jax.jit(
        lambda x, c, rng: _ekm_propose(x, c, rng, h, lambda y: y, precision, 2)
    )(active, complement, key)

    # Preserve the proposal's dtype when constructing its covariance: an
    # O(eps) basis rounding difference matters in a nearly singular direction.
    c = np.asarray(complement)
    basis = (c - c.mean(axis=0)) / np.sqrt(np.float32(2.))
    covariance = basis.T @ basis
    eps = np.finfo(np.asarray(active).dtype).eps
    ridge = max(1e-8, 4 * eps * np.linalg.norm(covariance, ord=np.inf))
    regularized = covariance + ridge * np.eye(2)
    factor = np.asarray(jnp.linalg.cholesky(jnp.asarray(regularized)), dtype=float)
    assert np.all(np.isfinite(factor))
    represented_covariance = factor @ factor.T
    np.testing.assert_allclose(represented_covariance, regularized, atol=2e-7, rtol=2e-7)
    x = np.asarray(active[0], dtype=float)
    drift = -covariance @ np.asarray(precision) @ x
    _, noise_key = jax.random.split(key)
    z = np.asarray(jax.random.normal(noise_key, (1, 2))[0], dtype=float)
    expected = x + h * drift + np.sqrt(2 * h) * factor @ z
    np.testing.assert_allclose(proposal[0], expected, rtol=1e-6, atol=1e-7)

    y = np.asarray(proposal[0], dtype=float)
    reverse_drift = -covariance @ np.asarray(precision) @ y
    forward_residual = y - x - h * drift
    reverse_residual = x - y - h * reverse_drift
    exact_diff = (reverse_residual @ np.linalg.solve(represented_covariance, reverse_residual)
                  - forward_residual @ np.linalg.solve(represented_covariance, forward_residual)) / (4 * h)
    assert float(actual_diff[0]) == pytest.approx(exact_diff, abs=2e-4, rel=1e-4)
