"""Numerical and path-balance regressions for the Kalman DR move."""

import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

kalman_dr = importlib.import_module("affine_invariant_samplers.kalman_dr")


def _later_stage(stage, log_alpha_forward, log_alpha_reverse, lp_candidate=0.,
                 q_reverse=0.):
    if stage == 2:
        return kalman_dr._dr_accept_stage2(
            0., lp_candidate, 0., q_reverse, 0., 0.,
            log_alpha_forward, log_alpha_reverse)
    return kalman_dr._dr_accept_stage3(
        0., lp_candidate, 0., q_reverse, 0., 0., 0., 0.,
        log_alpha_forward, log_alpha_reverse, -.5, -.5)


@pytest.mark.parametrize("stage", [2, 3])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("forward,reverse", [(-4e-13, -2e-13), (-1e-8, -1e-7)])
def test_near_unit_acceptance_keeps_rejection_mass(stage, compiled, forward, reverse):
    fn = lambda a, b: _later_stage(stage, a, b)
    if compiled:
        fn = jax.jit(fn)
    actual = fn(jnp.asarray(forward, dtype=jnp.float32),
                jnp.asarray(reverse, dtype=jnp.float32))
    expected = min(0., math.log(-math.expm1(reverse))
                   - math.log(-math.expm1(forward)))
    assert float(actual) == pytest.approx(expected, abs=3e-6)


@pytest.mark.parametrize("stage", [2, 3])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("forward,reverse,expected", [
    (0., -.5, 0.),       # A zero forward mass is an unreachable later stage.
    (-.5, 0., -np.inf),  # A reverse path with zero rejection mass cannot accept.
    (0., 0., 0.),
])
def test_exact_zero_rejection_mass(stage, compiled, forward, reverse, expected):
    fn = lambda a, b: _later_stage(stage, a, b)
    if compiled:
        fn = jax.jit(fn)
    actual = fn(jnp.asarray(forward), jnp.asarray(reverse))
    assert float(actual) == expected


@pytest.mark.parametrize("stage", [2, 3])
def test_positive_infinite_mass_ratio_accepts(stage):
    # Each input is finite, while their sum overflows float32 to positive infinity.
    huge = jnp.asarray(3e38, dtype=jnp.float32)
    assert float(_later_stage(stage, -.5, -.5, huge, huge)) == 0.


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("compiled", [False, True])
def test_zero_density_candidate_rejects_undefined_reverse_drift(stage, compiled):
    if stage == 1:
        fn = lambda lp, q: kalman_dr._dr_accept_stage1(0., lp, 0., q)
    else:
        fn = lambda lp, q: _later_stage(stage, -.5, -.5, lp, q)
    if compiled:
        fn = jax.jit(fn)
    assert float(fn(jnp.asarray(-np.inf), jnp.asarray(np.nan))) == -np.inf


def _kernel_log_density(stage, origin, candidate):
    """Independent scalar Gaussian density for h=(.8,.4,.2), drift=-x/2."""
    h = (.8, .4, .2)[stage]
    return -(candidate - origin + .5 * h * origin) ** 2 / (4. * h) \
        - .5 * math.log(4. * math.pi * h)


def _log_density(x):
    return -.5 * x ** 2


def _path_acceptance(paths):
    """Apply sampler helpers to batched attempted paths of up to three stages."""
    x, candidate = paths[:, 0], paths[:, -1]
    stage = paths.shape[1] - 1
    if stage == 1:
        return np.asarray(kalman_dr._dr_accept_stage1(
            _log_density(x), _log_density(candidate),
            _kernel_log_density(0, x, candidate),
            _kernel_log_density(0, candidate, x)))
    first = paths[:, 1]
    alpha1_forward = _path_acceptance(paths[:, :2])
    alpha1_reverse = _path_acceptance(np.column_stack((candidate, first)))
    if stage == 2:
        return np.asarray(kalman_dr._dr_accept_stage2(
            _log_density(x), _log_density(candidate),
            _kernel_log_density(1, x, candidate),
            _kernel_log_density(1, candidate, x),
            _kernel_log_density(0, x, first),
            _kernel_log_density(0, candidate, first),
            alpha1_forward, alpha1_reverse))
    second = paths[:, 2]
    alpha2_forward = _path_acceptance(paths[:, :3])
    alpha2_reverse = _path_acceptance(np.column_stack((candidate, first, second)))
    return np.asarray(kalman_dr._dr_accept_stage3(
        _log_density(x), _log_density(candidate),
        _kernel_log_density(2, x, candidate),
        _kernel_log_density(2, candidate, x),
        _kernel_log_density(0, x, first),
        _kernel_log_density(0, candidate, first),
        _kernel_log_density(1, x, second),
        _kernel_log_density(1, candidate, second),
        alpha1_forward, alpha1_reverse, alpha2_forward, alpha2_reverse))


def _accepted_path_log_mass(paths):
    """Target × proposal densities × previous rejections × final acceptance."""
    origin = paths[:, 0]
    log_mass = _log_density(origin)
    for stage in range(paths.shape[1] - 1):
        log_mass = log_mass + _kernel_log_density(stage, origin, paths[:, stage + 1])
        if stage > 0:
            with np.errstate(divide="ignore"):
                log_mass = log_mass + np.log(-np.expm1(
                    _path_acceptance(paths[:, :stage + 1])))
    return log_mass + _path_acceptance(paths)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_accepted_path_flux_matches_endpoint_swap(stage):
    paths = np.random.default_rng(2026).uniform(-3., 3., size=(128, stage + 1))
    reverse_paths = paths.copy()
    # The independent-from-origin Kalman proposals retain rejected history order.
    reverse_paths[:, 0], reverse_paths[:, -1] = paths[:, -1], paths[:, 0]
    forward_mass = _accepted_path_log_mass(paths)
    reverse_mass = _accepted_path_log_mass(reverse_paths)
    finite = np.isfinite(forward_mass)
    np.testing.assert_array_equal(finite, np.isfinite(reverse_mass))
    assert int(finite.sum()) >= 5
    np.testing.assert_allclose(forward_mass[finite], reverse_mass[finite],
                               rtol=0., atol=3e-5)


@pytest.mark.parametrize("compiled", [False, True])
def test_invalid_first_proposal_does_not_poison_valid_retry(monkeypatch, compiled):
    def fixed_basis(walkers, complement, key, k, forward_fn):
        return (jnp.zeros((walkers, 1, 1)), jnp.zeros((walkers, 1, 1)),
                jnp.tile(jnp.eye(1), (walkers, 1, 1)))

    def fixed_proposal(x, drift, chol, h, noise):
        return jnp.full_like(x, jnp.where(h > .75, -1., .25))

    monkeypatch.setattr(kalman_dr, "_compute_basis", fixed_basis)
    monkeypatch.setattr(kalman_dr, "_propose", fixed_proposal)

    def log_prob(x):
        return jnp.where(x[:, 0] >= 0., 0., -jnp.inf)

    def forward_fn(x):
        return jnp.sqrt(x)  # Undefined for the deliberately rejected first proposal.

    def step(key):
        return kalman_dr._dr_group_step(
            jnp.asarray([[.5]]), jnp.asarray([[.5], [1.]]), jnp.asarray([0.]),
            key, 1., jnp.asarray([1., .5]), 2,
            forward_fn, jnp.eye(1), log_prob, 1)

    if compiled:
        step = jax.jit(step)
    position, log_prob_new, accepted_first, accepted_any, _ = step(jax.random.key(0))
    np.testing.assert_array_equal(position, [[.25]])
    np.testing.assert_array_equal(log_prob_new, [0.])
    np.testing.assert_array_equal(accepted_first, [False])
    np.testing.assert_array_equal(accepted_any, [True])
