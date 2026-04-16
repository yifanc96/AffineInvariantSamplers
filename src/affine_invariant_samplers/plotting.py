"""
affine_invariant_samplers.plotting
==================================

Plotting utilities for MCMC output.  Pure matplotlib, no `corner` dependency.

Functions
---------
corner_plot        : lower-triangular grid with 1D histograms on the diagonal
                     and 2D histograms below.
trace_plot         : per-dimension trace plot across iterations.
autocorrelation_plot : ACF curves per dimension (single chain or averaged).

Requires ``matplotlib``.  Install via ``pip install "affine-invariant-samplers[plot]"``.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure  # noqa: F401
except ImportError as e:                 # pragma: no cover
    raise ImportError(
        "matplotlib is required for affine_invariant_samplers.plotting.  "
        "Install with:  pip install 'affine-invariant-samplers[plot]'"
    ) from e


__all__ = ["corner_plot", "trace_plot", "autocorrelation_plot"]


# ──────────────────────────────────────────────────────────────────────────────
# Shape helper
# ──────────────────────────────────────────────────────────────────────────────

def _flatten_chains(samples):
    """`(N, D)` → unchanged;  `(N, n_chains, D)` → `(N · n_chains, D)`."""
    arr = np.asarray(samples)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        N, C, D = arr.shape
        return arr.reshape(N * C, D)
    raise ValueError(f"samples must have ndim 2 or 3, got {arr.ndim}")


# ──────────────────────────────────────────────────────────────────────────────
# Corner plot
# ──────────────────────────────────────────────────────────────────────────────

def corner_plot(
    samples,
    labels=None,
    truths=None,
    bins=40,
    figsize=None,
    color="C0",
    truth_color="C3",
    hist_kwargs=None,
    hist2d_kwargs=None,
    title=None,
    quantiles=(0.16, 0.5, 0.84),
    show_titles=True,
    fig=None,
):
    """Lower-triangular corner plot for posterior samples.

    Args:
        samples       : (N, D) or (N, n_chains, D).  Chains are flattened.
        labels        : Optional list of D strings for the parameter names.
        truths        : Optional D-vector of "true" values to overlay.
        bins          : Bin count for both 1D and 2D histograms.
        figsize       : Figure size.  Defaults to (2.2 · D, 2.2 · D).
        color         : Colour for histograms.
        truth_color   : Colour for truth lines.
        hist_kwargs   : Extra kwargs for diagonal 1D histograms.
        hist2d_kwargs : Extra kwargs for off-diagonal 2D histograms.
        title         : Overall figure title.
        quantiles     : Tuple of quantiles to mark on diagonals with dashed lines.
                        If `show_titles` is True, the central quantile (median)
                        and its 68% interval are printed as each axis title.
        show_titles   : Show per-parameter median ± CI above each diagonal axis.
        fig           : Optional pre-existing matplotlib Figure to draw on.

    Returns:
        fig : matplotlib.figure.Figure
    """
    data = _flatten_chains(samples)
    N, D = data.shape
    if labels is not None and len(labels) != D:
        raise ValueError(f"labels has length {len(labels)}, expected {D}")
    if truths is not None and len(truths) != D:
        raise ValueError(f"truths has length {len(truths)}, expected {D}")

    figsize = figsize or (max(2.2 * D, 4.0), max(2.2 * D, 4.0))
    if fig is None:
        fig, axes = plt.subplots(D, D, figsize=figsize)
    else:
        axes = np.array(fig.subplots(D, D))
    axes = np.atleast_2d(axes)

    hist_kwargs   = dict(hist_kwargs or {})
    hist2d_kwargs = dict(hist2d_kwargs or {})
    hist_kwargs.setdefault("color", color)
    hist_kwargs.setdefault("histtype", "stepfilled")
    hist_kwargs.setdefault("alpha", 0.5)
    hist_kwargs.setdefault("edgecolor", color)
    hist2d_kwargs.setdefault("cmap", "Blues")

    # axis limits per dim — shared column-wise
    lims = [(float(np.percentile(data[:, d], 0.5)),
             float(np.percentile(data[:, d], 99.5))) for d in range(D)]

    qs = list(quantiles) if quantiles is not None else []

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                # diagonal: 1D histogram
                ax.hist(data[:, i], bins=bins, range=lims[i], density=True,
                        **hist_kwargs)
                # quantile marks
                if qs:
                    qvals = np.quantile(data[:, i], qs)
                    for qv in qvals:
                        ax.axvline(qv, color=color, linestyle="--", alpha=0.6,
                                   linewidth=0.9)
                if truths is not None:
                    ax.axvline(truths[i], color=truth_color, linewidth=1.2)
                if show_titles and len(qs) >= 3:
                    qvals = np.quantile(data[:, i], qs)
                    lo  = qvals[1] - qvals[0]
                    hi  = qvals[-1] - qvals[1]
                    name = labels[i] if labels else f"x_{i}"
                    ax.set_title(f"{name} = {qvals[1]:.2f}$^{{+{hi:.2f}}}_{{-{lo:.2f}}}$",
                                 fontsize=9)
                ax.set_yticks([])
                ax.set_xlim(*lims[i])
            else:
                # lower triangle: 2D histogram
                ax.hist2d(data[:, j], data[:, i], bins=bins,
                          range=[lims[j], lims[i]], **hist2d_kwargs)
                if truths is not None:
                    ax.axvline(truths[j], color=truth_color, linewidth=1.0, alpha=0.8)
                    ax.axhline(truths[i], color=truth_color, linewidth=1.0, alpha=0.8)
                    ax.plot(truths[j], truths[i], "s", color=truth_color,
                            markersize=4)
                ax.set_xlim(*lims[j])
                ax.set_ylim(*lims[i])

            # tick / label cosmetics
            if i < D - 1:
                ax.set_xticklabels([])
            else:
                if labels:
                    ax.set_xlabel(labels[j])
            if j > 0 or i == j:
                ax.set_yticklabels([])
            else:
                if labels:
                    ax.set_ylabel(labels[i])
            ax.tick_params(axis="both", labelsize=8)

    fig.align_labels()
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    if title:
        fig.suptitle(title, y=0.995)
        fig.subplots_adjust(top=0.96)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Trace plot
# ──────────────────────────────────────────────────────────────────────────────

def trace_plot(samples, labels=None, chain_alpha=0.5, figsize=None, fig=None):
    """Per-dimension trace plot across iterations.

    Args:
        samples     : (N, D) or (N, n_chains, D).
        labels      : Optional list of D parameter names.
        chain_alpha : Alpha for individual chain lines (multi-chain input).
        figsize     : Default (9, 1.6 · D).
        fig         : Optional pre-existing Figure.

    Returns:
        fig : matplotlib.figure.Figure
    """
    arr = np.asarray(samples)
    if arr.ndim == 2:
        arr = arr[:, None, :]                  # (N, 1, D)
    elif arr.ndim != 3:
        raise ValueError(f"samples must have ndim 2 or 3, got {arr.ndim}")
    N, C, D = arr.shape

    figsize = figsize or (9.0, max(1.6 * D, 2.0))
    if fig is None:
        fig, axes = plt.subplots(D, 1, figsize=figsize, sharex=True, squeeze=False)
    else:
        axes = np.atleast_2d(fig.subplots(D, 1, sharex=True, squeeze=False))
    axes = axes.ravel()

    for d in range(D):
        ax = axes[d]
        for c in range(C):
            ax.plot(arr[:, c, d], linewidth=0.7, alpha=chain_alpha)
        ax.set_ylabel(labels[d] if labels else f"x_{d}")
        ax.tick_params(axis="both", labelsize=8)
    axes[-1].set_xlabel("iteration")
    fig.subplots_adjust(hspace=0.1)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Autocorrelation plot
# ──────────────────────────────────────────────────────────────────────────────

def autocorrelation_plot(samples, max_lag=None, labels=None, figsize=None, fig=None):
    """Plot the autocorrelation function per dimension.

    Args:
        samples : (N,) or (N, D) or (N, n_chains, D).
        max_lag : Maximum lag to plot (default min(N // 3, 500)).
        labels  : Optional list of parameter names.
        figsize : Default (7, 4).
        fig     : Optional pre-existing Figure.

    Returns:
        fig : matplotlib.figure.Figure
    """
    from .diagnostics import autocorrelation    # local import to avoid cycles

    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr[:, None]                     # treat as (N, D=1)
    if max_lag is None:
        N = arr.shape[0]
        max_lag = min(N // 3, 500)
    acf = autocorrelation(samples, max_lag=max_lag)
    D = acf.shape[1]

    figsize = figsize or (7.0, 4.0)
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = fig.add_subplot(1, 1, 1)

    lags = np.arange(max_lag)
    for d in range(D):
        ax.plot(lags, acf[:, d], linewidth=1.1,
                label=labels[d] if labels else f"x_{d}")
    ax.axhline(0.0, color="k", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.set_xlim(0, max_lag - 1)
    if D <= 20:
        ax.legend(fontsize=8, loc="upper right", frameon=False)
    return fig
