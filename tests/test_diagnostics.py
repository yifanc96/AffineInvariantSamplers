"""Smoke tests for diagnostics and plotting utilities."""

from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def test_autocorrelation_iid():
    """IID samples ⇒ ACF(0)=1, ACF(k>0) ≈ 0 within sampling noise."""
    from affine_invariant_samplers import autocorrelation
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    acf = autocorrelation(x, max_lag=50)
    assert acf.shape == (50, 1)
    assert abs(acf[0, 0] - 1.0) < 1e-6
    # noise tail should be small
    assert np.max(np.abs(acf[5:, 0])) < 0.15


def test_autocorrelation_ar1():
    """AR(1) with φ = 0.9 ⇒ ACF(k) ≈ 0.9^k.  Check first few lags."""
    from affine_invariant_samplers import autocorrelation
    rng = np.random.default_rng(42)
    n, phi = 20_000, 0.9
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    acf = autocorrelation(x, max_lag=20).ravel()
    assert abs(acf[1] - phi) < 0.05
    assert abs(acf[2] - phi ** 2) < 0.05


def test_iat_and_ess_iid():
    """IID ⇒ τ ≈ 1, ESS ≈ N.  ESS is clamped at N (never exceeds it)."""
    from affine_invariant_samplers import (
        integrated_autocorr_time, effective_sample_size,
    )
    rng = np.random.default_rng(7)
    N = 5000
    x = rng.standard_normal(N)
    tau = integrated_autocorr_time(x)
    ess = effective_sample_size(x)
    assert 0.5 < tau < 2.5
    assert 2000 < ess <= N          # capped at N even if τ < 1


def test_ess_clamped_at_n():
    """Even with contrived antithetic samples (τ < 1), ESS must not exceed N."""
    from affine_invariant_samplers import effective_sample_size
    rng = np.random.default_rng(0)
    n = 4000
    # Construct x with strong lag-1 anticorrelation: x[i] ≈ -0.8 x[i-1] + noise
    z = rng.standard_normal(n)
    x = np.zeros(n); x[0] = z[0]
    for i in range(1, n):
        x[i] = -0.8 * x[i - 1] + z[i]
    ess = effective_sample_size(x)
    assert ess <= n


def test_iat_ar1():
    """AR(1) φ=0.9 has τ = (1+φ)/(1-φ) = 19.  Expect same ballpark."""
    from affine_invariant_samplers import integrated_autocorr_time
    rng = np.random.default_rng(42)
    n, phi = 40_000, 0.9
    x = np.zeros(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.standard_normal()
    tau = integrated_autocorr_time(x)
    assert 10 < tau < 40   # exact value is ~19; window estimator is noisy


def test_diagnostics_3d_input():
    """(N, n_chains, D) input works and returns per-dim outputs."""
    from affine_invariant_samplers import (
        autocorrelation, integrated_autocorr_time, effective_sample_size,
    )
    rng = np.random.default_rng(0)
    s = rng.standard_normal((2000, 4, 3))   # N=2000, 4 chains, D=3
    acf = autocorrelation(s, max_lag=30)
    assert acf.shape == (30, 3)
    tau = integrated_autocorr_time(s)
    assert tau.shape == (3,)
    ess = effective_sample_size(s)
    assert ess.shape == (3,)
    # IID ⇒ ESS ≈ N * n_chains  (within factor 2 or so, estimator noisy)
    assert np.all(ess > 2000)


@pytest.mark.parametrize("samples", [np.array([0., 1.]), np.array([1., 1.])])
def test_two_sample_diagnostics_keep_lag_zero(samples):
    from affine_invariant_samplers.diagnostics import (
        autocorrelation, integrated_autocorr_time_1d,
    )
    acf = autocorrelation(samples)
    assert acf.shape == (1, 1)
    assert acf[0, 0] == pytest.approx(1.)
    tau, original_acf, ess = integrated_autocorr_time_1d(samples)
    if np.all(samples == samples[0]):
        assert tau == float("inf") and ess == 0.
    else:
        assert np.isfinite(tau) and np.isfinite(ess)
    assert original_acf.shape == (1,)


def test_frozen_walkers_are_not_reported_as_independent_samples():
    from affine_invariant_samplers.diagnostics import effective_sample_size, integrated_autocorr_time

    # Nonzero pooled variance and correct pooled moments do not establish
    # movement: each walker stays at its initial point for every iteration.
    frozen = np.broadcast_to(np.array([-1., 1.])[None, :, None], (100, 2, 1))
    assert effective_sample_size(frozen)[0] == 0.
    assert np.isinf(integrated_autocorr_time(frozen)[0])


def test_lag_count_is_capped_at_available_samples():
    from affine_invariant_samplers.diagnostics import autocorrelation, autocorrelation_fft
    x = np.arange(10.)
    assert autocorrelation_fft(x, max_lag=20).shape == (10,)
    assert autocorrelation(x, max_lag=20).shape == (10, 1)


@pytest.mark.parametrize("max_lag", [0, -1, 1.5])
def test_invalid_lag_count_is_rejected(max_lag):
    from affine_invariant_samplers.diagnostics import autocorrelation
    with pytest.raises(ValueError, match="max_lag must be a positive integer"):
        autocorrelation(np.arange(10.), max_lag=max_lag)


def test_short_correlated_chain_uses_original_scale_fallback():
    from affine_invariant_samplers.diagnostics import integrated_autocorr_time_1d
    x = np.arange(100.)
    fallback_tau = integrated_autocorr_time_1d(x, max_iterations=0)[0]
    for cap in (1, 2, 3, 10):
        tau, acf, ess = integrated_autocorr_time_1d(x, max_iterations=cap)
        assert tau == pytest.approx(fallback_tau)
        assert acf.shape == (33,)
        assert ess == pytest.approx(len(x) / tau)


def test_converged_reduced_estimate_is_rescaled_once():
    from affine_invariant_samplers.diagnostics import integrated_autocorr_time_1d
    x = np.arange(20.)
    blocked = 0.5 * (x[::2] + x[1::2])
    # At this cutoff the full chain misses the criterion and one pairwise
    # reduction meets it. Its time units are twice the original units.
    local_tau = integrated_autocorr_time_1d(blocked, c=3.06)[0]
    tau = integrated_autocorr_time_1d(x, c=3.06)[0]
    assert tau == pytest.approx(2. * local_tau)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_samples_do_not_report_valid_diagnostics(invalid):
    from affine_invariant_samplers.diagnostics import (
        autocorrelation, effective_sample_size, integrated_autocorr_time,
    )
    x = np.array([0., 1., invalid, 2.])
    for diagnostic in (autocorrelation, integrated_autocorr_time, effective_sample_size):
        with pytest.raises(ValueError, match="finite"):
            diagnostic(x)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting — smoke tests (matplotlib in Agg backend)
# ──────────────────────────────────────────────────────────────────────────────

def test_corner_plot():
    import matplotlib
    matplotlib.use("Agg", force=True)
    from affine_invariant_samplers.plotting import corner_plot
    rng = np.random.default_rng(0)
    s = rng.standard_normal((1000, 4, 3))
    fig = corner_plot(s, labels=["a", "b", "c"], truths=[0.0, 0.0, 0.0])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_trace_and_autocorr_plots():
    import matplotlib
    matplotlib.use("Agg", force=True)
    from affine_invariant_samplers.plotting import (
        trace_plot, autocorrelation_plot,
    )
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    s = rng.standard_normal((800, 3, 2))
    f1 = trace_plot(s, labels=["a", "b"])
    f2 = autocorrelation_plot(s, labels=["a", "b"], max_lag=50)
    assert f1 is not None and f2 is not None
    plt.close(f1); plt.close(f2)


def test_autocorrelation_plot_caps_requested_lags():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from affine_invariant_samplers.plotting import autocorrelation_plot
    fig = autocorrelation_plot(np.arange(10.), max_lag=20)
    assert len(fig.axes[0].lines[0].get_xdata()) == 10
    plt.close(fig)


if __name__ == "__main__":
    test_autocorrelation_iid()
    test_autocorrelation_ar1()
    test_iat_and_ess_iid()
    test_ess_clamped_at_n()
    test_iat_ar1()
    test_diagnostics_3d_input()
    test_corner_plot()
    test_trace_and_autocorr_plots()
    print("All diagnostics / plotting smoke tests passed.")
