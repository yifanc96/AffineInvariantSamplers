"""API regressions for unadjusted samplers, without changing adaptive policy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from affine_invariant_samplers import sampler_aldi, sampler_pickles_unadjusted
from affine_invariant_samplers import aldi, pickles_unadjusted


def _log_prob(x):
    return -0.5 * jnp.sum(x * x, axis=1)


@pytest.mark.parametrize("sampler", [sampler_aldi, sampler_pickles_unadjusted])
@pytest.mark.parametrize("adapt", ["maxdrift", "pgn"])
@pytest.mark.parametrize("warmup", [0, 1])
def test_adaptive_info_reports_last_production_step(sampler, adapt, warmup):
    initial = jnp.array([[-8.], [-4.], [4.], [8.]])
    base, target = 0.2, 0.5
    samples, info = sampler(
        _log_prob, initial, num_samples=3, warmup=warmup,
        step_size=base, target_gnorm=target, adapt=adapt,
        verbose=(warmup == 0), seed=73,
    )
    assert samples.shape == (3, 4, 1)
    assert np.all(np.isfinite(samples))
    # The penultimate retained state is the origin of the last production
    # step, whose effective step size is independent of that step's noise.
    previous = np.asarray(samples[-2], dtype=float)
    centered = (previous - previous.mean(axis=0)) / np.sqrt(len(previous))
    pg = previous @ centered.T
    if sampler is sampler_aldi:
        if adapt == "maxdrift":
            drift = -pg @ centered + (2 / len(previous)) * (
                previous - previous.mean(axis=0))
            signal = np.max(np.linalg.norm(drift, axis=1))
        else:
            signal = np.sqrt(np.mean(np.sum(pg * pg, axis=1)))
        expected = base / max(1., signal / target)
    else:
        norms = np.linalg.norm(pg, axis=1)
        signal = np.max(norms) if adapt == "maxdrift" else np.sqrt(np.mean(norms * norms))
        expected = base / max(1., np.sqrt(signal / target))
    assert info["step_size"] == pytest.approx(expected, rel=3e-6)
    assert info["n_grad_evals"] == 12


def test_aldi_accepts_list_initial_state():
    samples, _ = sampler_aldi(
        _log_prob, [[-1.], [-0.5], [0.5], [1.]], num_samples=2,
        warmup=0, adapt=None, step_size=0.01, verbose=False,
    )
    assert samples.shape == (2, 4, 1)
    assert np.all(np.isfinite(samples))


@pytest.mark.parametrize("sampler", [sampler_aldi, sampler_pickles_unadjusted])
@pytest.mark.parametrize("adapt", ["maxdrift", "pgn", "samadams", None])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_zero_warmup_preserves_state_dtype_with_x64_enabled(sampler, adapt, dtype):
    with jax.enable_x64(True):
        initial = jnp.array([[-1.], [-0.5], [0.5], [1.]], dtype=dtype)
        samples, info = sampler(
            _log_prob, initial, num_samples=2, warmup=0, adapt=adapt,
            step_size=0.01, verbose=False,
        )
        assert samples.dtype == initial.dtype
        assert samples.shape == (2, 4, 1)
        assert np.all(np.isfinite(samples))
        assert info["step_size"] > 0


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_nonfinite_guards_restore_only_invalid_rows(bad):
    old = jnp.array([[1.], [2.]])
    proposed = jnp.array([[bad], [3.]])
    for guard in (aldi._nan_guard_state, pickles_unadjusted._nan_guard_state,
                  pickles_unadjusted._nan_guard_mom):
        np.testing.assert_array_equal(guard(proposed, old), [[1.], [3.]])


@pytest.mark.parametrize("variant", ["pgn", "maxdrift", "samadams"])
@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf])
def test_partial_recovery_keeps_gradient_cache_at_returned_positions(monkeypatch, variant, bad):
    def core(*args):
        return (jnp.array([[bad], [3.]]), jnp.zeros((2, 2)),
                jnp.array([[bad], [3.]]))
    monkeypatch.setattr(pickles_unadjusted, "_baoab_core", core)
    position = jnp.array([[1.], [2.]])
    momentum = jnp.zeros((2, 2))
    cached = position
    correction = jnp.zeros((2, 2))
    args = (position, momentum, 0.1, 2., lambda x: -x, correction)
    if variant == "pgn":
        result = pickles_unadjusted._baoab_step_pgn(
            *args, 1., cached, jax.random.key(0))
    elif variant == "maxdrift":
        result = pickles_unadjusted._baoab_step_maxpgn(
            *args, 1., cached, jax.random.key(0))
    else:
        result = pickles_unadjusted._baoab_step_samadams(
            *args, 1., 0.9, 1e-8, cached, jax.random.key(0))
    np.testing.assert_array_equal(result[0], [[1.], [3.]])
    np.testing.assert_array_equal(result[2], result[0])
