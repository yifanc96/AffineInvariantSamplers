"""
sampler_dfgmvi — Derivative-Free Gaussian Mixture Variational Inference, JAX only.

Approximates a target density exp(-Phi(x)) with a mixture of K Gaussians
using *only* function evaluations of a forward model F(x), with
Phi(x) = 0.5 * ||F(x)||^2.

Uses sigma-point (unscented-transform) quadrature to approximate the
gradient and Hessian of the expected potential E_k[Phi] without
automatic differentiation.  The entropy term E_k[log rho_GM] and its
derivatives are computed analytically from the Gaussian mixture density.

Update rules (for each mode k, with temperature T):
  Covariance:  C_k^{-1} += dt * (nabla^2 E[log rho] + (1/T) nabla^2 E[Phi])
  Mean:        m_k -= dt * C_k * (nabla E[log rho] + (1/T) nabla E[Phi])
  Weight:      log w_k -= dt * (E[log rho] + (1/T) E[Phi])

Potential annealing (Che et al. 2025):
  - Temperature T decays exponentially from T_start to 1 over anneal_iters.
  - T_start chosen so that the initial potential gradient is alpha times
    the entropy gradient, ensuring exploration before convergence.
  - Schedule: T_n = T_start^{(N_alpha - n) / (N_alpha - 1)}.

Numerical stability:
  - Covariance updated via implicit inverse-Cholesky step (always PD).
  - Hessian of potential approximated from sigma-point finite differences;
    only the guaranteed-SPD part (b b^T + 6 diag(a a^T)) is kept.
  - Weights in log-space with logsumexp normalisation + clipping.

Reference: Che, Chen, Huan, Huang & Wang,
           "Stable Derivative Free Gaussian Mixture Variational Inference
            for Bayesian Inverse Problems"  arXiv:2501.04259
           "Adaptive Exponential Integration for Stable Gaussian Mixture
            Black-Box Variational Inference"  arXiv:2601.14855
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Sigma-point (unscented transform) quadrature
# ──────────────────────────────────────────────────────────────────────────────

def _unscented_sigma_points(mean, sqrt_cov, alpha):
    """Generate 2D+1 sigma points and weights (unscented transform).

    Args:
        mean:      (D,)
        sqrt_cov:  (D, D) lower-triangular Cholesky factor
        alpha:     spread parameter (typically sqrt(3))

    Returns:
        points:  (2D+1, D)
        weights: (2D+1,)  mean weights summing to 1
    """
    D = mean.shape[0]
    N_ens = 2 * D + 1

    scaled = alpha * sqrt_cov

    points = jnp.zeros((N_ens, D))
    points = points.at[0].set(mean)
    points = points.at[1:D + 1].set(mean[None, :] + scaled.T)
    points = points.at[D + 1:].set(mean[None, :] - scaled.T)

    w = jnp.full(N_ens, 1.0 / (2.0 * alpha ** 2))
    w = w.at[0].set(1.0 - D / alpha ** 2)

    return points, w


def _cubature_sigma_points(mean, sqrt_cov):
    """Generate 2D cubature points and weights (3rd-order).

    Args:
        mean:      (D,)
        sqrt_cov:  (D, D) lower-triangular Cholesky factor

    Returns:
        points:  (2D, D)
        weights: (2D,)  equal weights summing to 1
    """
    D = mean.shape[0]
    alpha = jnp.sqrt(jnp.float64(D))
    scaled = alpha * sqrt_cov

    points = jnp.concatenate([
        mean[None, :] + scaled.T,
        mean[None, :] - scaled.T,
    ], axis=0)

    weights = jnp.full(2 * D, 1.0 / (2.0 * D))
    return points, weights


# ──────────────────────────────────────────────────────────────────────────────
# Derivative-free potential expectation from sigma points
# ──────────────────────────────────────────────────────────────────────────────

def _df_potential_expectation(mean, sqrt_cov, inv_sqrt_cov, F_vals, alpha):
    """Compute E[Phi], nabla E[Phi], nabla^2 E[Phi] from sigma-point
    forward-model evaluations  F(x_i), where Phi(x) = 0.5 ||F(x)||^2.

    Uses the unscented transform finite-difference formulas:
        b_i = (F(x_{+i}) - F(x_{-i})) / (2 alpha)
        a_i = (F(x_{+i}) + F(x_{-i}) - 2 F(x_0)) / (2 alpha^2)
        c   = F(x_0)

    Keeping only the SPD part of the Hessian approximation:
        nabla^2 E[Phi] ≈ inv_L^T (6 diag(A A^T) + B B^T) inv_L

    Args:
        mean:          (D,)
        sqrt_cov:      (D, D) Cholesky factor L
        inv_sqrt_cov:  (D, D) L^{-1}
        F_vals:        (2D+1, N_f)  forward model at sigma points
        alpha:         sigma-point spread

    Returns:
        phi_mean:   scalar
        grad_phi:   (D,)
        hess_phi:   (D, D)  SPD approximation
    """
    D = mean.shape[0]
    c = F_vals[0]
    F_plus = F_vals[1:D + 1]
    F_minus = F_vals[D + 1:]

    b = (F_plus - F_minus) / (2.0 * alpha)
    a = (F_plus + F_minus - 2.0 * c[None, :]) / (2.0 * alpha ** 2)

    phi_mean = 0.5 * jnp.sum(c ** 2)
    grad_phi = inv_sqrt_cov.T @ (b @ c)

    ATA = a @ a.T
    BTB = b @ b.T
    hess_phi = inv_sqrt_cov.T @ (6.0 * jnp.diag(jnp.diag(ATA)) + BTB) @ inv_sqrt_cov

    return phi_mean, grad_phi, hess_phi


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian mixture log-density and its derivatives (analytical)
# ──────────────────────────────────────────────────────────────────────────────

def _log_gaussian_single(x, mean, inv_sqrt_cov):
    """Unnormalised log Gaussian at a single point (no (2pi) factor)."""
    z = inv_sqrt_cov @ (x - mean)
    log_det = jnp.sum(jnp.log(jnp.abs(jnp.diag(inv_sqrt_cov))))
    return -0.5 * jnp.dot(z, z) + log_det


def _gm_density_and_derivs(x, log_w, means, inv_sqrt_covs, hessian_correct):
    """Compute rho, nabla rho, nabla^2 rho for a Gaussian mixture at x."""
    K = log_w.shape[0]
    D = means.shape[1]
    w = jnp.exp(log_w)

    rho = 0.0
    grad_rho = jnp.zeros(D)
    hess_rho = jnp.zeros((D, D))

    for im in range(K):
        inv_L = inv_sqrt_covs[im]
        diff = means[im] - x
        prec_diff = inv_L.T @ (inv_L @ diff)
        rho_i = jnp.exp(_log_gaussian_single(x, means[im], inv_L))

        rho = rho + w[im] * rho_i
        grad_rho = grad_rho + w[im] * rho_i * prec_diff
        if hessian_correct:
            hess_rho = hess_rho + w[im] * rho_i * jnp.outer(prec_diff, prec_diff)
        else:
            hess_rho = hess_rho + w[im] * rho_i * (
                jnp.outer(prec_diff, prec_diff) - inv_L.T @ inv_L)

    return rho, grad_rho, hess_rho


def _compute_log_gm_expectations(log_w, means, sqrt_covs, inv_sqrt_covs,
                                  sigma_points, sigma_weights,
                                  hessian_correct=True):
    """Compute E_k[log rho_GM], E_k[nabla log rho_GM], E_k[nabla^2 log rho_GM]
    for each mode k, using sigma-point quadrature."""
    K = log_w.shape[0]
    D = means.shape[1]

    log_w_norm = log_w - jax.nn.logsumexp(log_w)

    log_rho_mean = jnp.zeros(K)
    grad_log_rho = jnp.zeros((K, D))
    hess_log_rho = jnp.zeros((K, D, D))

    for im in range(K):
        pts = sigma_points[im]
        N_pts = pts.shape[0]

        lr_vals = jnp.zeros(N_pts)
        glr_vals = jnp.zeros((N_pts, D))
        hlr_vals = jnp.zeros((N_pts, D, D))

        for ip in range(N_pts):
            rho, g_rho, h_rho = _gm_density_and_derivs(
                pts[ip], log_w_norm, means, inv_sqrt_covs, hessian_correct)
            rho = jnp.maximum(rho, 1e-300)
            d_half = D / 2.0
            lr_vals = lr_vals.at[ip].set(jnp.log(rho) - d_half * jnp.log(2.0 * jnp.pi))
            glr_vals = glr_vals.at[ip].set(g_rho / rho)
            if hessian_correct:
                hlr_vals = hlr_vals.at[ip].set(
                    h_rho / rho - jnp.outer(g_rho, g_rho) / rho ** 2
                    - inv_sqrt_covs[im].T @ inv_sqrt_covs[im])
            else:
                hlr_vals = hlr_vals.at[ip].set(
                    h_rho / rho - jnp.outer(g_rho, g_rho) / rho ** 2)

        log_rho_mean = log_rho_mean.at[im].set(jnp.dot(sigma_weights, lr_vals))
        grad_log_rho = grad_log_rho.at[im].set(sigma_weights @ glr_vals)
        hess_log_rho = hess_log_rho.at[im].set(
            jnp.einsum('n,nij->ij', sigma_weights, hlr_vals))

    return log_rho_mean, grad_log_rho, hess_log_rho


# ──────────────────────────────────────────────────────────────────────────────
# One DF-GMVI update step (with temperature)
# ──────────────────────────────────────────────────────────────────────────────

def _dfgmvi_step(log_w, means, sqrt_covs, inv_sqrt_covs,
                 forward_fn, n_f, alpha_bip, dt, inv_T):
    """Single DF-GMVI update step.  Potential terms scaled by inv_T = 1/T."""
    K = means.shape[0]
    D = means.shape[1]

    # --- 1. Sigma points for potential estimation (unscented, 2D+1) ---
    phi_mean_all = jnp.zeros(K)
    grad_phi_all = jnp.zeros((K, D))
    hess_phi_all = jnp.zeros((K, D, D))

    for im in range(K):
        pts, _ = _unscented_sigma_points(means[im], sqrt_covs[im], alpha_bip)
        F_vals = jax.vmap(forward_fn)(pts)
        phi_m, grad_phi, hess_phi = _df_potential_expectation(
            means[im], sqrt_covs[im], inv_sqrt_covs[im], F_vals, alpha_bip)
        phi_mean_all = phi_mean_all.at[im].set(phi_m)
        grad_phi_all = grad_phi_all.at[im].set(grad_phi)
        hess_phi_all = hess_phi_all.at[im].set(hess_phi)

    # Scale potential by 1/T
    phi_mean_all = inv_T * phi_mean_all
    grad_phi_all = inv_T * grad_phi_all
    hess_phi_all = inv_T * hess_phi_all

    # --- 2. Entropy term: E[log rho_GM] via cubature (2D points per mode) ---
    sigma_points_gm = []
    for im in range(K):
        pts, w_pts = _cubature_sigma_points(means[im], sqrt_covs[im])
        sigma_points_gm.append(pts)
    sigma_weights_gm = w_pts

    log_rho_mean, grad_log_rho, hess_log_rho = _compute_log_gm_expectations(
        log_w, means, sqrt_covs, inv_sqrt_covs,
        sigma_points_gm, sigma_weights_gm, hessian_correct=True)

    # --- 3. Update covariance (implicit, guarantees PD) ---
    sqrt_covs_new = []
    inv_sqrt_covs_new = []
    covs_new = []

    for im in range(K):
        prec_old = inv_sqrt_covs[im].T @ inv_sqrt_covs[im]
        hess_total = hess_log_rho[im] + hess_phi_all[im]
        hess_total = 0.5 * (hess_total + hess_total.T)
        prec_new = prec_old + dt * hess_total
        eigvals, eigvecs = jnp.linalg.eigh(prec_new)
        eigvals = jnp.maximum(eigvals, 1e-8)
        prec_new = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        L_prec = jnp.linalg.cholesky(prec_new + 1e-10 * jnp.eye(D))
        inv_L = jnp.linalg.inv(L_prec)
        cov_new = inv_L.T @ inv_L
        L_cov = jnp.linalg.cholesky(cov_new + 1e-10 * jnp.eye(D))
        inv_L_cov = jnp.linalg.inv(L_cov)
        sqrt_covs_new.append(L_cov)
        inv_sqrt_covs_new.append(inv_L_cov)
        covs_new.append(cov_new)

    # --- 4. Update means ---
    means_new = jnp.zeros_like(means)
    for im in range(K):
        grad_total = grad_log_rho[im] + grad_phi_all[im]
        means_new = means_new.at[im].set(
            means[im] - dt * covs_new[im] @ grad_total)

    # --- 5. Update log-weights ---
    log_w_new = log_w - dt * (log_rho_mean + phi_mean_all)
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

def _estimate_T_start_dfgmvi(log_w, means, sqrt_covs, inv_sqrt_covs,
                              forward_fn, alpha_bip,
                              sigma_weights_gm, alpha_ratio):
    """Estimate T_start so that (1/T)*||grad_potential|| ≈ alpha * ||grad_entropy||.

    Computes gradient norms from one evaluation at the current state.
    """
    K = means.shape[0]
    D = means.shape[1]

    # Potential gradient norms
    grad_phi_norms_sq = 0.0
    for im in range(K):
        pts, _ = _unscented_sigma_points(means[im], sqrt_covs[im], alpha_bip)
        F_vals = jax.vmap(forward_fn)(pts)
        _, grad_phi, _ = _df_potential_expectation(
            means[im], sqrt_covs[im], inv_sqrt_covs[im], F_vals, alpha_bip)
        grad_phi_norms_sq += jnp.sum(grad_phi ** 2)

    # Entropy gradient norms
    sigma_points_gm = []
    for im in range(K):
        pts, w_pts = _cubature_sigma_points(means[im], sqrt_covs[im])
        sigma_points_gm.append(pts)
    sigma_weights_gm = w_pts

    _, grad_log_rho, _ = _compute_log_gm_expectations(
        log_w, means, sqrt_covs, inv_sqrt_covs,
        sigma_points_gm, sigma_weights_gm, hessian_correct=True)

    grad_entropy_norms_sq = jnp.sum(grad_log_rho ** 2)

    norm_potential = jnp.sqrt(grad_phi_norms_sq / K)
    norm_entropy = jnp.sqrt(grad_entropy_norms_sq / K)

    T_start = norm_potential / jnp.maximum(alpha_ratio * norm_entropy, 1e-12)
    T_start = jnp.maximum(T_start, 1.0)
    return T_start


# ──────────────────────────────────────────────────────────────────────────────
# sampler_dfgmvi  (main entry point)
# ──────────────────────────────────────────────────────────────────────────────

def sampler_dfgmvi(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 500,
    n_modes       = 4,
    n_f           = 0,
    forward_fn    = None,
    alpha_bip     = 0.0,
    dt            = 0.5,
    w_min         = 1e-8,
    anneal        = True,
    anneal_iters  = 0,
    anneal_alpha  = 0.1,
    thin_by       = 1,
    seed          = 0,
    verbose       = True,
):
    """
    Derivative-Free Gaussian Mixture Variational Inference.

    Fits a K-component Gaussian mixture to approximate exp(log_prob_fn(x))
    using only evaluations of a forward model F(x) where
    Phi(x) = 0.5 * ||F(x)||^2.

    If no forward_fn is provided, one is constructed from log_prob_fn:
        F(x) = sqrt(2 * max(0, -log_prob_fn(x)))   (scalar, N_f=1)

    Args:
        log_prob_fn   : (D,) -> scalar.  Log density (up to constant).
        initial_state : (n_chains, D).  Used to initialise mixture centres;
                        n_chains >= n_modes.
        num_samples   : Number of samples to draw from the fitted mixture.
        warmup        : Number of VI optimisation iterations.
        n_modes       : Number of Gaussian components K.
        n_f           : Output dimension of forward_fn (auto-detected if 0).
        forward_fn    : (D,) -> (N_f,).  Forward model.  Optional.
        alpha_bip     : Sigma-point spread (0 = sqrt(3)).
        dt            : Time step / learning rate.
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

    if alpha_bip <= 0.0:
        alpha_bip = jnp.sqrt(3.0)

    key = jax.random.key(seed)

    # --- build forward model if not provided ---
    if forward_fn is None:
        def forward_fn(x):
            lp = log_prob_fn(x)
            phi = -lp
            return jnp.array([jnp.sqrt(2.0 * jnp.maximum(phi, 0.0))])
        n_f = 1
    elif n_f <= 0:
        test_out = forward_fn(init[0])
        n_f = test_out.shape[0] if test_out.ndim > 0 else 1

    # --- initialise mixture components ---
    stride = max(1, n_chains // n_modes)
    means = jnp.stack([jnp.mean(init[i * stride:(i + 1) * stride], axis=0)
                        for i in range(n_modes)])

    cov_init = jnp.cov(init.T) + 1e-6 * jnp.eye(D)
    L0 = jnp.linalg.cholesky(cov_init)
    sqrt_covs = [L0.copy() for _ in range(n_modes)]
    inv_sqrt_covs = [jnp.linalg.inv(L0) for _ in range(n_modes)]

    log_w = jnp.full(n_modes, -jnp.log(float(n_modes)))

    n_iter = warmup

    # --- annealing schedule ---
    if anneal:
        if anneal_iters <= 0:
            anneal_iters = n_iter // 2

        T_start = float(_estimate_T_start_dfgmvi(
            log_w, means, sqrt_covs, inv_sqrt_covs,
            forward_fn, alpha_bip, None, anneal_alpha))

        if verbose:
            print(f"DF-GMVI annealing:  T_start={T_start:.2f}  "
                  f"anneal_iters={anneal_iters}/{n_iter}")
    else:
        T_start = 1.0
        anneal_iters = 0

    def _get_inv_T(it):
        if not anneal or T_start <= 1.0 + 1e-8:
            return 1.0
        if it <= anneal_iters:
            T = T_start ** ((anneal_iters - it) / max(anneal_iters - 1.0, 1.0))
        else:
            T = 1.0
        return 1.0 / T

    # --- optimisation loop ---
    for it in range(1, n_iter + 1):
        inv_T = _get_inv_T(it)
        log_w, means, sqrt_covs, inv_sqrt_covs = _dfgmvi_step(
            log_w, means, sqrt_covs, inv_sqrt_covs,
            forward_fn, n_f, alpha_bip, dt, inv_T,
        )
        log_w = _clip_weights(log_w, w_min)

        if verbose and it % max(1, n_iter // 10) == 0:
            w = jnp.exp(log_w - jax.nn.logsumexp(log_w))
            T_cur = 1.0 / _get_inv_T(it) if _get_inv_T(it) > 0 else float('inf')
            print(f"  DF-GMVI iter {it}/{n_iter}  T={T_cur:.2f}  weights={w}")

    # --- final parameters ---
    weights = jnp.exp(log_w - jax.nn.logsumexp(log_w))
    covs = jnp.stack([L @ L.T for L in sqrt_covs])
    sqrt_covs_arr = jnp.stack(sqrt_covs)

    if verbose:
        print(f"DF-GMVI:  K={n_modes}  D={D}  N_f={n_f}  iters={n_iter}"
              f"  anneal={anneal}")

    # --- draw samples from the fitted mixture ---
    key, k1, k2 = jax.random.split(key, 3)
    mode_idx = jax.random.choice(k1, n_modes, shape=(num_samples,), p=weights)
    z = jax.random.normal(k2, (num_samples, D))
    samples = means[mode_idx] + jnp.einsum('nij,nj->ni', sqrt_covs_arr[mode_idx], z)
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

    # Forward model: F(x) = L^T x  where Prec = L L^T,  so  ||F||^2 = x^T L L^T x = x^T Prec x
    L_prec = jnp.linalg.cholesky(prec)
    def forward(x):
        return L_prec.T @ x

    init = jax.random.normal(jax.random.key(42), (40, dim))
    samples, info = sampler_dfgmvi(
        log_prob, init, num_samples=5000, warmup=300, n_modes=2,
        forward_fn=forward, n_f=dim, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0, 1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
