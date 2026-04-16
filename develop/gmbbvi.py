"""
sampler_gmbbvi — Gaussian Mixture Black-Box Variational Inference, JAX only.

Approximates a target density exp(-Phi(x)) with a mixture of K Gaussians
by minimising the (reverse) KL divergence via Monte-Carlo gradient estimates.

For each Gaussian component k with parameters (w_k, m_k, L_k) where
C_k = L_k L_k^T:
  - Draw z ~ N(0,I), form x = m_k + L_k z.
  - Evaluate  f(x) = log rho_GM(x) + (1/T) Phi(x).
  - Gradient w.r.t. m_k:   E_z[ (f - E[f]) * L_k^{-1} z ]
  - Gradient w.r.t. L_k:   eigendecomposition-based multiplicative update
                            L_k <- L_k V diag(exp(-dt/2 * lambda)) V^T
  - Gradient w.r.t. log w_k: -E[f]

Potential annealing (Che et al. 2025):
  - Temperature T decays exponentially from T_start to 1 over anneal_iters.
  - T_start chosen so that the initial potential gradient is alpha times
    the entropy gradient (default alpha=0.1), ensuring exploration first.
  - Schedule: T_n = T_start^{(N_alpha - n) / (N_alpha - 1)}.

Numerical stability:
  - Weights stored in log-space; logsumexp normalisation + clipping.
  - Covariance updated via eigendecomposition to guarantee PD.
  - Learning rate bounded by 1 / max|eigenvalue| of the covariance gradient.
  - Cosine-annealing schedule on top of the adaptive dt.

Reference: Che, Chen, Huan, Huang & Wang,
           "Adaptive Exponential Integration for Stable Gaussian Mixture
            Black-Box Variational Inference"  arXiv:2601.14855
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Learning-rate schedulers
# ──────────────────────────────────────────────────────────────────────────────

def _stable_cos_decay(it, n_iter, eta_min=0.1, eta_max=1.0, decay_frac=0.5):
    n_decay = decay_frac * n_iter
    return jnp.where(
        it <= n_iter - n_decay,
        eta_max,
        eta_min + 0.5 * (eta_max - eta_min) * (1.0 + jnp.cos(jnp.pi * (it - (n_iter - n_decay)) / n_decay)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian mixture helpers  (pure JAX, no loops over modes in hot path)
# ──────────────────────────────────────────────────────────────────────────────

def _log_gaussian_component(x, mean, inv_sqrt_cov):
    """Log N(x | mean, C) up to the (2pi)^{-d/2} constant.

    Args:
        x:             (N, D)
        mean:          (D,)
        inv_sqrt_cov:  (D, D)  such that C^{-1} = inv_sqrt_cov^T inv_sqrt_cov

    Returns: (N,)
    """
    diff = x - mean[None, :]                              # (N, D)
    z = diff @ inv_sqrt_cov.T                             # (N, D)
    maha = jnp.sum(z ** 2, axis=-1)                       # (N,)
    log_det = jnp.sum(jnp.log(jnp.abs(jnp.diag(inv_sqrt_cov))))
    return -0.5 * maha + log_det


def _log_gaussian_mixture(x, log_w, means, inv_sqrt_covs):
    """Log of the Gaussian mixture density at points x.

    Args:
        x:              (N, D)
        log_w:          (K,)
        means:          (K, D)
        inv_sqrt_covs:  (K, D, D)

    Returns: (N,)
    """
    K = log_w.shape[0]

    def _per_mode(im):
        return log_w[im] + _log_gaussian_component(x, means[im], inv_sqrt_covs[im])

    log_components = jax.vmap(_per_mode)(jnp.arange(K))   # (K, N)
    return jax.nn.logsumexp(log_components, axis=0)        # (N,)


# ──────────────────────────────────────────────────────────────────────────────
# One GMBBVI update step
# ──────────────────────────────────────────────────────────────────────────────

def _gmbbvi_step(log_w, means, sqrt_covs, inv_sqrt_covs,
                 phi_fn, n_ens, key, dt_max, it, n_iter, inv_T):
    """Single GMBBVI gradient step with temperature scaling.

    The annealed objective is:  f_k = log rho_GM + (1/T) * Phi.
    inv_T = 1/T is passed in to avoid division.
    """
    K, D = means.shape

    # --- normalise weights for density evaluation ---
    log_w_norm = log_w - jax.nn.logsumexp(log_w)

    # --- draw MC samples for each mode ---
    z_normal = jax.random.normal(key, (K, n_ens, D))
    x_samples = means[:, None, :] + jnp.einsum('kij,knj->kni', sqrt_covs, z_normal)

    # --- evaluate potential Phi on all samples ---
    x_flat = x_samples.reshape(-1, D)
    phi_flat = jax.vmap(phi_fn)(x_flat)
    phi_vals = phi_flat.reshape(K, n_ens)

    # --- evaluate log rho_GM on all samples ---
    log_rho = _log_gaussian_mixture(x_flat, log_w_norm, means, inv_sqrt_covs)
    log_rho = log_rho.reshape(K, n_ens)

    # --- f = log_rho + (1/T)*Phi, centred ---
    f = log_rho + inv_T * phi_vals
    f_mean = jnp.mean(f, axis=1, keepdims=True)
    f_centred = f - f_mean

    # --- gradient for means ---
    g_mean_z = jnp.einsum('kn,knd->kd', f_centred, z_normal) / n_ens

    # --- gradient for covariance (eigendecomposition update) ---
    g_cov_zz = jnp.einsum('kni,knj,kn->kij', z_normal, z_normal, f_centred) / n_ens

    eigvals, eigvecs = jnp.linalg.eigh(0.5 * (g_cov_zz + jnp.swapaxes(g_cov_zz, -2, -1)))
    matrix_norms = jnp.max(jnp.abs(eigvals), axis=1)

    schedule_factor = _stable_cos_decay(it, n_iter)
    dt_per_mode = jnp.minimum(
        schedule_factor * dt_max,
        dt_max / jnp.maximum(matrix_norms, 1e-12),
    )
    dt = jnp.min(dt_per_mode)

    # --- update means ---
    delta_means = -dt * jnp.einsum('kij,kj->ki', sqrt_covs, g_mean_z)
    means_new = means + delta_means

    # --- update covariances via eigendecomposition ---
    scale = jnp.exp(-0.5 * dt * eigvals)
    rot = jnp.einsum('kij,kj,klj->kil', eigvecs, scale, eigvecs)
    sqrt_covs_new = jnp.einsum('kij,kjl->kil', sqrt_covs, rot)

    covs_new = jnp.einsum('kij,klj->kil', sqrt_covs_new, sqrt_covs_new)
    jitter = 1e-8 * jnp.eye(D)[None, :, :]
    sqrt_covs_new = jnp.linalg.cholesky(covs_new + jitter)
    inv_sqrt_covs_new = jnp.linalg.inv(sqrt_covs_new)

    # --- update log-weights ---
    log_w_new = log_w + dt * (-f_mean[:, 0])
    log_w_new = log_w_new - jax.nn.logsumexp(log_w_new)

    return log_w_new, means_new, sqrt_covs_new, inv_sqrt_covs_new


def _clip_weights(log_w, w_min=1e-8):
    """Clip mixture weights from below, renormalise."""
    w = jnp.exp(log_w)
    w = jnp.maximum(w, w_min)
    w = w / jnp.sum(w)
    return jnp.log(w)


# ──────────────────────────────────────────────────────────────────────────────
# T_start auto-detection
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_T_start(log_w, means, sqrt_covs, inv_sqrt_covs,
                      phi_fn, n_ens, key, alpha_ratio):
    """Estimate T_start so that (1/T)*||grad_potential|| ≈ alpha * ||grad_entropy||.

    Runs one step at T=1 to measure the ratio of gradient norms.
    """
    K, D = means.shape
    log_w_norm = log_w - jax.nn.logsumexp(log_w)

    z_normal = jax.random.normal(key, (K, n_ens, D))
    x_samples = means[:, None, :] + jnp.einsum('kij,knj->kni', sqrt_covs, z_normal)

    x_flat = x_samples.reshape(-1, D)
    phi_flat = jax.vmap(phi_fn)(x_flat)
    phi_vals = phi_flat.reshape(K, n_ens)

    log_rho = _log_gaussian_mixture(x_flat, log_w_norm, means, inv_sqrt_covs)
    log_rho = log_rho.reshape(K, n_ens)

    # Entropy gradient norm: E_z[ (log_rho - E[log_rho]) * z ]
    lr_centred = log_rho - jnp.mean(log_rho, axis=1, keepdims=True)
    g_entropy = jnp.einsum('kn,knd->kd', lr_centred, z_normal) / n_ens
    norm_entropy = jnp.sqrt(jnp.mean(jnp.sum(g_entropy ** 2, axis=-1)))

    # Potential gradient norm: E_z[ (Phi - E[Phi]) * z ]
    phi_centred = phi_vals - jnp.mean(phi_vals, axis=1, keepdims=True)
    g_potential = jnp.einsum('kn,knd->kd', phi_centred, z_normal) / n_ens
    norm_potential = jnp.sqrt(jnp.mean(jnp.sum(g_potential ** 2, axis=-1)))

    # Want (1/T_start) * norm_potential = alpha * norm_entropy
    # => T_start = norm_potential / (alpha * norm_entropy)
    T_start = norm_potential / jnp.maximum(alpha_ratio * norm_entropy, 1e-12)
    T_start = jnp.maximum(T_start, 1.0)   # T >= 1 always
    return T_start


# ──────────────────────────────────────────────────────────────────────────────
# sampler_gmbbvi  (main entry point)
# ──────────────────────────────────────────────────────────────────────────────

def sampler_gmbbvi(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup       = 500,
    n_modes      = 4,
    n_ens        = 0,
    dt           = 0.5,
    w_min        = 1e-8,
    anneal       = True,
    anneal_iters = 0,
    anneal_alpha = 0.1,
    thin_by      = 1,
    seed         = 0,
    verbose      = True,
):
    """
    Gaussian Mixture Black-Box Variational Inference.

    Fits a K-component Gaussian mixture to approximate exp(log_prob_fn(x))
    using Monte-Carlo gradient estimates of the ELBO.

    Unlike MCMC samplers, this is an optimisation method: the output
    "samples" are draws from the fitted mixture, not a Markov chain.

    Args:
        log_prob_fn   : (D,) -> scalar.  Log density (up to constant).
        initial_state : (n_chains, D).  Used to initialise mixture centres
                        via K-means-style spread; n_chains >= n_modes.
        num_samples   : Number of samples to draw from the fitted mixture.
        warmup        : Number of VI optimisation iterations.
        n_modes       : Number of Gaussian components K.
        n_ens         : MC samples per mode per iteration (0 = 4*D).
        dt            : Base learning rate.
        w_min         : Minimum mixture weight (clipping threshold).
        anneal        : Enable potential annealing (default True).
        anneal_iters  : Annealing iterations (0 = warmup // 2).
        anneal_alpha  : Gradient ratio for T_start (default 0.1).
        thin_by       : (ignored, kept for interface consistency).
        seed          : Random seed.
        verbose       : Print progress.

    Returns:
        samples : (num_samples, 1, D)   — shape kept for compatibility
        info    : dict with keys:
                    weights  : (K,) final mixture weights
                    means    : (K, D) final means
                    covs     : (K, D, D) final covariances
    """
    init = jnp.asarray(initial_state, dtype=jnp.float64)
    n_chains, D = init.shape

    if n_ens <= 0:
        n_ens = 4 * D

    key = jax.random.key(seed)

    # --- initialise mixture components from initial_state ---
    stride = max(1, n_chains // n_modes)
    means = jnp.stack([jnp.mean(init[i * stride:(i + 1) * stride], axis=0)
                        for i in range(n_modes)])

    cov_init = jnp.cov(init.T) + 1e-6 * jnp.eye(D)
    sqrt_covs = jnp.linalg.cholesky(cov_init)
    sqrt_covs = jnp.tile(sqrt_covs[None, :, :], (n_modes, 1, 1))
    inv_sqrt_covs = jnp.linalg.inv(sqrt_covs)

    log_w = jnp.full(n_modes, -jnp.log(float(n_modes)))

    def phi_single(x):
        return -log_prob_fn(x)

    n_iter = warmup

    # --- annealing schedule ---
    if anneal:
        if anneal_iters <= 0:
            anneal_iters = n_iter // 2

        key, k_est = jax.random.split(key)
        T_start = _estimate_T_start(
            log_w, means, sqrt_covs, inv_sqrt_covs,
            phi_single, n_ens, k_est, anneal_alpha)
        T_start = float(T_start)

        if verbose:
            print(f"GMBBVI annealing:  T_start={T_start:.2f}  "
                  f"anneal_iters={anneal_iters}/{n_iter}")
    else:
        T_start = 1.0
        anneal_iters = 0

    # Build temperature schedule: T_n = T_start^{(N_alpha - n) / (N_alpha - 1)}
    # for n=1..N_alpha, then T=1 for the rest.
    def _get_inv_T(it):
        """Return 1/T for iteration `it` (1-based)."""
        if not anneal or T_start <= 1.0 + 1e-8:
            return 1.0
        T = jnp.where(
            it <= anneal_iters,
            T_start ** ((anneal_iters - it) / jnp.maximum(anneal_iters - 1.0, 1.0)),
            1.0,
        )
        return 1.0 / T

    # --- optimisation loop ---
    @jax.jit
    def _step(carry, it):
        log_w, means, sqrt_covs, inv_sqrt_covs, key = carry
        key, subkey = jax.random.split(key)
        inv_T = _get_inv_T(it)
        log_w, means, sqrt_covs, inv_sqrt_covs = _gmbbvi_step(
            log_w, means, sqrt_covs, inv_sqrt_covs,
            phi_single, n_ens, subkey, dt, it, n_iter, inv_T,
        )
        log_w = _clip_weights(log_w, w_min)
        return (log_w, means, sqrt_covs, inv_sqrt_covs, key), None

    iters = jnp.arange(1, n_iter + 1)
    (log_w, means, sqrt_covs, inv_sqrt_covs, key), _ = jax.lax.scan(
        _step,
        (log_w, means, sqrt_covs, inv_sqrt_covs, key),
        iters,
    )

    # --- final parameters ---
    weights = jnp.exp(log_w - jax.nn.logsumexp(log_w))
    covs = jnp.einsum('kij,klj->kil', sqrt_covs, sqrt_covs)

    if verbose:
        print(f"GMBBVI:  K={n_modes}  D={D}  n_ens={n_ens}  iters={n_iter}"
              f"  anneal={anneal}")
        print(f"  final weights: {weights}")

    # --- draw samples from the fitted mixture ---
    key, k1, k2 = jax.random.split(key, 3)
    mode_idx = jax.random.choice(k1, n_modes, shape=(num_samples,), p=weights)
    z = jax.random.normal(k2, (num_samples, D))
    samples = means[mode_idx] + jnp.einsum('nij,nj->ni', sqrt_covs[mode_idx], z)
    samples = samples[:, None, :]

    info = dict(
        weights=weights,
        means=means,
        covs=covs,
    )
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim = 2
    cov = jnp.array([[1., .95], [.95, 1.]])
    prec = jnp.linalg.inv(cov)

    def log_prob(x):
        return -0.5 * x @ prec @ x

    init = jax.random.normal(jax.random.key(42), (40, dim))
    samples, info = sampler_gmbbvi(
        log_prob, init, num_samples=5000, warmup=500, n_modes=2, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0, 1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
