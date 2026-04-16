"""Smoke tests for diagnostics and plotting utilities."""

from __future__ import annotations

import numpy as np


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
    """IID ⇒ τ ≈ 1, ESS ≈ N."""
    from affine_invariant_samplers import (
        integrated_autocorr_time, effective_sample_size,
    )
    rng = np.random.default_rng(7)
    x = rng.standard_normal(5000)
    tau = integrated_autocorr_time(x)
    ess = effective_sample_size(x)
    assert 0.5 < tau < 2.5
    assert 2000 < ess < 10_000


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


if __name__ == "__main__":
    test_autocorrelation_iid()
    test_autocorrelation_ar1()
    test_iat_and_ess_iid()
    test_iat_ar1()
    test_diagnostics_3d_input()
    test_corner_plot()
    test_trace_and_autocorr_plots()
    print("All diagnostics / plotting smoke tests passed.")
