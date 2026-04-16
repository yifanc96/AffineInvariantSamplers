"""
sampler_ig — Iterative Gaussianization, JAX only.

Approximates a target density by constructing a normalizing flow via
repeated application of two simple steps:
  1. Rotation (Score PCA): find directions along which the target
     deviates most from Gaussian, encode as Householder reflections.
  2. Marginal Gaussianization: fit a coordinatewise rational-quadratic
     spline flow to Gaussianize each marginal independently.

The composition of K such layers defines an invertible transport map
from a standard Gaussian base to the target.  Sampling: draw z ~ N(0,I)
and push through the forward map.

Loss: reverse KL  KL(q||p), minimised via Adam with cosine temperature
annealing from beta_0 to 1.

Reference: Chen & Liu, "Rotated Mean-Field Variational Inference and
           Iterative Gaussianization" arXiv:2510.07732
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial


# ══════════════════════════════════════════════════════════════════════════════
# Rational quadratic spline (componentwise, pure JAX)
# ══════════════════════════════════════════════════════════════════════════════

SPLINE_RANGE = 8.0   # default half-width of spline domain

def _rqs_forward_1d(x, widths, heights, derivatives):
    """Forward pass of a single rational-quadratic spline.

    Args:
        x:           scalar input
        widths:      (K,) bin widths (positive, sum to range)
        heights:     (K,) bin heights (positive, sum to range)
        derivatives: (K+1,) knot derivatives (positive)

    Returns:
        y:       scalar output
        logdet:  scalar log |dy/dx|
    """
    K = widths.shape[0]
    range_min, range_max = -SPLINE_RANGE, SPLINE_RANGE
    B = range_max - range_min

    # Cumulative widths/heights -> knot positions
    cum_w = jnp.concatenate([jnp.zeros(1), jnp.cumsum(widths)])
    cum_h = jnp.concatenate([jnp.zeros(1), jnp.cumsum(heights)])
    x_knots = range_min + B * cum_w  # (K+1,)
    y_knots = range_min + B * cum_h  # (K+1,)

    # Find which bin x falls in (clamp to [0, K-1])
    bin_idx = jnp.searchsorted(x_knots[1:], x).clip(0, K - 1)

    # Bin parameters
    x_k = x_knots[bin_idx]
    x_k1 = x_knots[bin_idx + 1]
    y_k = y_knots[bin_idx]
    y_k1 = y_knots[bin_idx + 1]
    d_k = derivatives[bin_idx]
    d_k1 = derivatives[bin_idx + 1]
    w_k = x_k1 - x_k
    h_k = y_k1 - y_k
    s_k = h_k / w_k

    # Normalised position in bin
    xi = (x - x_k) / w_k
    xi = xi.clip(0.0, 1.0)

    # Rational quadratic formula (Durkan et al. 2019)
    num = h_k * (s_k * xi ** 2 + d_k * xi * (1 - xi))
    den = s_k + (d_k + d_k1 - 2.0 * s_k) * xi * (1 - xi)
    y_bin = y_k + num / den

    # Log-derivative
    num_logdet = 2.0 * jnp.log(s_k) + jnp.log(d_k1 * xi ** 2
                 + 2.0 * s_k * xi * (1 - xi) + d_k * (1 - xi) ** 2)
    den_logdet = 2.0 * jnp.log(den)
    logdet_bin = num_logdet - den_logdet + jnp.log(h_k) - jnp.log(w_k)

    # Identity outside range
    inside = (x >= range_min) & (x <= range_max)
    y = jnp.where(inside, y_bin, x)
    logdet = jnp.where(inside, logdet_bin, 0.0)

    return y, logdet


def _rqs_inverse_1d(y, widths, heights, derivatives):
    """Inverse pass of a single rational-quadratic spline."""
    K = widths.shape[0]
    range_min, range_max = -SPLINE_RANGE, SPLINE_RANGE
    B = range_max - range_min

    cum_w = jnp.concatenate([jnp.zeros(1), jnp.cumsum(widths)])
    cum_h = jnp.concatenate([jnp.zeros(1), jnp.cumsum(heights)])
    x_knots = range_min + B * cum_w
    y_knots = range_min + B * cum_h

    bin_idx = jnp.searchsorted(y_knots[1:], y).clip(0, K - 1)

    x_k = x_knots[bin_idx]
    x_k1 = x_knots[bin_idx + 1]
    y_k = y_knots[bin_idx]
    y_k1 = y_knots[bin_idx + 1]
    d_k = derivatives[bin_idx]
    d_k1 = derivatives[bin_idx + 1]
    w_k = x_k1 - x_k
    h_k = y_k1 - y_k
    s_k = h_k / w_k

    # Solve quadratic for xi
    a = h_k * (s_k - d_k) + (y - y_k) * (d_k + d_k1 - 2.0 * s_k)
    b = h_k * d_k - (y - y_k) * (d_k + d_k1 - 2.0 * s_k)
    c = -s_k * (y - y_k)

    disc = b ** 2 - 4.0 * a * c
    disc = jnp.maximum(disc, 0.0)
    xi = (2.0 * c) / (-b - jnp.sqrt(disc))
    xi = xi.clip(0.0, 1.0)

    x_bin = x_k + xi * w_k

    # Log-derivative (same formula as forward, negative sign for inverse)
    num = h_k * (s_k * xi ** 2 + d_k * xi * (1 - xi))
    den = s_k + (d_k + d_k1 - 2.0 * s_k) * xi * (1 - xi)
    num_logdet = 2.0 * jnp.log(s_k) + jnp.log(d_k1 * xi ** 2
                 + 2.0 * s_k * xi * (1 - xi) + d_k * (1 - xi) ** 2)
    den_logdet = 2.0 * jnp.log(den)
    logdet_bin = num_logdet - den_logdet + jnp.log(h_k) - jnp.log(w_k)

    inside = (y >= range_min) & (y <= range_max)
    x = jnp.where(inside, x_bin, y)
    logdet = jnp.where(inside, -logdet_bin, 0.0)

    return x, logdet


def _init_spline_params(dim, num_bins, key):
    """Initialise spline parameters to near-identity.

    Each dimension gets 3*K+1 parameters:
      widths_raw (K), heights_raw (K), derivatives_raw (K+1)
    Initialised to zeros -> softmax gives uniform bins, softplus(0) ~ 0.69 derivatives.
    """
    return jnp.zeros((dim, 3 * num_bins + 1))


_INV_SOFTPLUS_1 = jnp.log(jnp.e - 1.0)  # ≈ 0.5413, so softplus(_INV_SOFTPLUS_1) = 1.0

def _unpack_spline(raw, num_bins):
    """Unpack (3K+1,) raw vector into (widths, heights, derivatives).

    With zero initialisation:
      - widths, heights: uniform (via softmax(0) = 1/K)
      - derivatives: softplus(0 + inv_softplus(1)) = 1.0  -> identity spline
    """
    K = num_bins
    w_raw = raw[:K]
    h_raw = raw[K:2 * K]
    d_raw = raw[2 * K:]

    widths = jax.nn.softmax(w_raw)
    heights = jax.nn.softmax(h_raw)
    derivatives = jax.nn.softplus(d_raw + _INV_SOFTPLUS_1) + 1e-5

    return widths, heights, derivatives


# ══════════════════════════════════════════════════════════════════════════════
# Componentwise flow: apply independent spline to each dimension
# ══════════════════════════════════════════════════════════════════════════════

def _componentwise_forward(params, x, num_bins):
    """Apply componentwise spline flow.

    Args:
        params:   (D, 3K+1) raw spline parameters
        x:        (N, D) input samples
        num_bins: int

    Returns:
        y:      (N, D)
        logdet: (N,)
    """
    D = params.shape[0]

    def apply_one_dim(raw_d, x_d):
        w, h, d = _unpack_spline(raw_d, num_bins)
        return jax.vmap(lambda xi: _rqs_forward_1d(xi, w, h, d))(x_d)

    # vmap over dimensions: (D, N) -> (D, N), (D, N)
    y_t, ld_t = jax.vmap(apply_one_dim)(params, x.T)
    return y_t.T, jnp.sum(ld_t.T, axis=1)


def _componentwise_inverse(params, y, num_bins):
    """Inverse of componentwise spline flow."""
    def apply_one_dim(raw_d, y_d):
        w, h, d = _unpack_spline(raw_d, num_bins)
        return jax.vmap(lambda yi: _rqs_inverse_1d(yi, w, h, d))(y_d)

    x_t, ld_t = jax.vmap(apply_one_dim)(params, y.T)
    return x_t.T, jnp.sum(ld_t.T, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Householder rotations (encode orthogonal matrix as product of reflections)
# ══════════════════════════════════════════════════════════════════════════════

@jax.jit
def _get_householder_vecs(U):
    """Compute Householder vectors W from PCA eigenvectors U.

    Given U (d, r) with orthonormal columns, produce W (d, r) such that
    applying Householder reflections H_1 ... H_r maps e_i -> U[:, i].

    Args:
        U: (d, r) matrix with orthonormal columns

    Returns:
        W: (d, r) Householder vectors
    """
    d, r = U.shape
    W = jnp.zeros_like(U)

    def body(i, carry):
        U, W = carry
        e_i = jax.nn.one_hot(i, d, dtype=U.dtype)
        w = U[:, i] - e_i
        w = w / (jnp.linalg.norm(w) + 1e-12)
        # Apply reflection to remaining columns of U
        proj = w @ U
        mask = (jnp.arange(r) >= i).astype(U.dtype)
        U = U - 2.0 * jnp.outer(w, proj * mask)
        W = W.at[:, i].set(w)
        return U, W

    _, W = jax.lax.fori_loop(0, r, body, (U, W))
    return W


@jax.jit
def _apply_householder(W, x):
    """Apply Householder rotation: x -> H_r ... H_1 x  (reverse order).

    Args:
        W: (d, r) Householder vectors
        x: (d,) input vector

    Returns:
        (d,) rotated vector
    """
    _, r = W.shape

    def body(k, x):
        i = r - 1 - k
        w = W[:, i]
        return x - 2.0 * (w @ x) * w

    return jax.lax.fori_loop(0, r, body, x)


@jax.jit
def _apply_householder_T(W, x):
    """Apply transpose Householder rotation: x -> H_1 ... H_r x  (forward order)."""
    _, r = W.shape

    def body(i, x):
        w = W[:, i]
        return x - 2.0 * (w @ x) * w

    return jax.lax.fori_loop(0, r, body, x)


# ══════════════════════════════════════════════════════════════════════════════
# Score PCA: find rotation directions from score-sample cross-covariance
# ══════════════════════════════════════════════════════════════════════════════

def _score_pca(log_prob_single, dim, n_samples, key, gamma=0.9):
    """Compute principal components of H = Cov(x, score(x) + x).

    The matrix H captures how the target's score deviates from a standard
    Gaussian.  Its top eigenvectors define the rotation that best decorrelates
    the target before coordinatewise Gaussianization.

    Args:
        log_prob_single: (D,) -> scalar  (single-point log density)
        dim:             int
        n_samples:       int
        key:             JAX PRNGKey
        gamma:           fraction of variance to retain (0 = no rotation)

    Returns:
        V: (d, r) top eigenvectors, or (d, d) identity if gamma=0
    """
    if gamma == 0:
        return jnp.eye(dim)

    samples = jax.random.normal(key, (n_samples, dim))
    scores = jax.vmap(jax.grad(log_prob_single))(samples) + samples

    H = scores.T @ samples / n_samples
    H2 = H @ H.T

    eigvals, eigvecs = jnp.linalg.eigh(H2)
    # Reverse to descending order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    total_var = jnp.sum(eigvals)
    if total_var < 1e-6:
        # Target is already (nearly) Gaussian — signal to skip this layer
        return None

    if gamma < 1:
        cumvar = jnp.cumsum(eigvals) / total_var
        indices = jnp.where(cumvar >= gamma)[0]
        rank = int(indices[0]) + 1 if len(indices) > 0 else dim
    else:
        rank = dim

    return eigvecs[:, :rank]


# ══════════════════════════════════════════════════════════════════════════════
# Adam optimiser (pure JAX, no optax dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _adam_init(params):
    return (params, jnp.zeros_like(params), jnp.zeros_like(params), 0)


def _adam_step(state, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
    params, m, v, t = state
    t = t + 1
    m = b1 * m + (1 - b1) * grads
    v = b2 * v + (1 - b2) * grads ** 2
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return (params, m, v, t)


# ══════════════════════════════════════════════════════════════════════════════
# MFVI step: fit componentwise spline flow via reverse KL
# ══════════════════════════════════════════════════════════════════════════════

def _mfvi_step(log_prob_single, dim, num_bins, n_samples, key,
               beta_0=1.0, lr=1e-3, max_iter=1000, T_anneal=None):
    """Fit a componentwise spline flow to Gaussianize the target.

    Minimises KL(q||p) where q = pushforward of N(0,I) through the spline.

    Args:
        log_prob_single: (D,) -> scalar
        dim:             int
        num_bins:        int, spline resolution
        n_samples:       int, training samples from base
        key:             JAX PRNGKey
        beta_0:          initial temperature for cosine annealing
        lr:              Adam learning rate
        max_iter:        optimisation steps
        T_anneal:        steps over which to anneal temperature (default: 80% of max_iter)

    Returns:
        params: (D, 3K+1) optimised spline parameters
        losses: (max_iter,) loss trajectory
    """
    key, subkey = jax.random.split(key)
    params = _init_spline_params(dim, num_bins, subkey)

    key, subkey = jax.random.split(key)
    base_samples = jax.random.normal(subkey, (n_samples, dim))

    if T_anneal is None:
        T_anneal = int(0.8 * max_iter)

    log_base = -0.5 * dim * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(base_samples ** 2, axis=1)

    def loss_fn(params, t):
        X, log_det = _componentwise_forward(params, base_samples, num_bins)
        # Cosine temperature annealing
        t_ = jnp.clip(t, 0, T_anneal)
        beta_t = 1.0 - 0.5 * (1.0 + jnp.cos(jnp.pi * t_ / T_anneal)) * (1.0 - beta_0)
        logp = jax.vmap(log_prob_single)(X) * beta_t
        logp = jnp.where(jnp.abs(logp) < 1e10, logp, jnp.nan)
        return jnp.nanmean(log_base - log_det - logp)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    adam_state = _adam_init(params)
    losses = jnp.zeros(max_iter)

    def scan_step(carry, t):
        adam_state, _ = carry
        params = adam_state[0]
        loss, grads = grad_fn(params, t)
        adam_state = _adam_step(adam_state, grads, lr)
        return (adam_state, None), loss

    (adam_state, _), losses = jax.lax.scan(
        scan_step, (adam_state, None), jnp.arange(max_iter))

    return adam_state[0], losses


# ══════════════════════════════════════════════════════════════════════════════
# Iterative Gaussianization: full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _iterative_gaussianization(log_prob_single, dim, n_samples, key,
                               gamma=0.9, niter=5, num_bins=10,
                               n_pca=None, beta_0=1.0, lr=1e-3,
                               max_iter=1000, verbose=True):
    """Construct a sequence of rotation + spline layers.

    Args:
        log_prob_single: (D,) -> scalar  (unbatched log density)
        dim:             int
        n_samples:       int, samples per MFVI step
        key:             JAX PRNGKey
        gamma:           PCA variance fraction (0 = plain MFVI)
        niter:           number of Gaussianization iterations
        num_bins:        spline bins per dimension
        n_pca:           samples for Score PCA (default: n_samples)
        beta_0:          initial temperature
        lr:              Adam learning rate
        max_iter:        Adam steps per layer
        verbose:         print progress

    Returns:
        transforms: list of (W, params) tuples
        num_bins:   int (needed for forward map)
    """
    if n_pca is None:
        n_pca = n_samples

    logp_k = log_prob_single
    transforms = []

    for i in range(niter):
        if verbose:
            print(f"  IG layer {i + 1}/{niter}", end="", flush=True)

        # 1. Score PCA -> rotation
        key, subkey = jax.random.split(key)
        V = _score_pca(logp_k, dim, n_pca, subkey, gamma)

        if V is None:
            # Target is already Gaussian — no more layers needed
            if verbose:
                print("  already Gaussian, stopping.")
            break

        rank = V.shape[1]
        if verbose:
            print(f"  rank={rank}", end="", flush=True)

        W = _get_householder_vecs(V)

        # Rotate the target
        def _make_rotated(logp, W_local):
            def logp_rot(x):
                return logp(_apply_householder(W_local, x))
            return logp_rot

        logp_k = _make_rotated(logp_k, W)

        # 2. Fit componentwise spline
        key, subkey = jax.random.split(key)
        params, losses = _mfvi_step(
            logp_k, dim, num_bins, n_samples, subkey,
            beta_0=beta_0, lr=lr, max_iter=max_iter)

        if verbose:
            print(f"  loss={float(losses[-1]):.4f}")

        # Pullback target through the fitted flow
        def _make_pullback(logp, params_local, nb):
            def logp_pb(x):
                x_2d = x[None, :]
                y, logdet = _componentwise_forward(params_local, x_2d, nb)
                return logp(y[0]) + logdet[0]
            return logp_pb

        logp_k = _make_pullback(logp_k, params, num_bins)
        transforms.append((W, params))

    return transforms, num_bins


def _forward_map(transforms, num_bins, samples):
    """Push samples from N(0,I) through the full flow.

    Args:
        transforms: list of (W, params), applied in reverse order
        num_bins:   int
        samples:    (N, D)

    Returns:
        samples: (N, D) transformed
        logdet:  (N,) total log-determinant
    """
    logdet = jnp.zeros(samples.shape[0])
    for W, params in reversed(transforms):
        samples, ld = _componentwise_forward(params, samples, num_bins)
        logdet = logdet + ld
        samples = jax.vmap(partial(_apply_householder, W))(samples)
    return samples, logdet


def _inverse_map(transforms, num_bins, samples):
    """Push samples from target space back to N(0,I).

    Args:
        transforms: list of (W, params), applied in forward order
        num_bins:   int
        samples:    (N, D)

    Returns:
        z:       (N, D) base-space
        logdet:  (N,) total log-determinant
    """
    logdet = jnp.zeros(samples.shape[0])
    for W, params in transforms:
        samples = jax.vmap(partial(_apply_householder_T, W))(samples)
        samples, ld = _componentwise_inverse(params, samples, num_bins)
        logdet = logdet + ld
    return samples, logdet


# ══════════════════════════════════════════════════════════════════════════════
# Sampler interface (matches project convention)
# ══════════════════════════════════════════════════════════════════════════════

def _laplace_preprocess(log_prob_single, dim, key, n_opt=2000):
    """Estimate mode and Hessian for pre-standardising the target.

    Uses Adam from random initialisations to find a mode, then estimates
    the covariance from the Hessian at that point.

    Returns:
        shift: (D,) estimated mode
        scale: (D, D) square root of approximate covariance (Cholesky)
    """
    starts = jax.random.normal(key, (8, dim)) * 0.5
    neg_logp = lambda x: -log_prob_single(x)
    val_grad_fn = jax.jit(jax.value_and_grad(neg_logp))

    best_x = starts[0]
    best_val = neg_logp(best_x)

    for i in range(8):
        x = starts[i]
        adam = _adam_init(x)
        for step in range(n_opt):
            _, g = val_grad_fn(x)
            g = jnp.where(jnp.isnan(g), 0.0, g)
            lr = 0.02 * (1.0 / (1.0 + step * 0.0005))
            adam = _adam_step(adam, g, lr)
            x = adam[0]
        val = neg_logp(x)
        cond = val < best_val
        best_x = jnp.where(cond, x, best_x)
        best_val = jnp.where(cond, val, best_val)

    # Polish with Newton steps
    hess_fn = jax.jit(jax.hessian(neg_logp))
    grad_fn = jax.jit(jax.grad(neg_logp))
    for _ in range(20):
        g = grad_fn(best_x)
        H = hess_fn(best_x)
        H = 0.5 * (H + H.T) + 1e-6 * jnp.eye(dim)
        dx = jnp.linalg.solve(H, g)
        best_x = best_x - dx

    shift = best_x

    # Hessian at mode -> covariance approximation
    H = jax.hessian(neg_logp)(shift)
    H = 0.5 * (H + H.T)
    eigvals, eigvecs = jnp.linalg.eigh(H)
    eigvals = jnp.maximum(eigvals, 1e-6)
    scale = eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals))

    return shift, scale


def sampler_ig(log_prob_fn, initial_state, num_samples,
               warmup=1000, seed=0, verbose=True,
               # IG-specific
               gamma=0.9, niter=3, num_bins=10,
               n_train=2000, n_pca=None,
               beta_0=0.5, lr=0.01, max_iter=None,
               preprocess=True):
    """Iterative Gaussianization variational inference.

    Constructs a normalizing flow via alternating Score-PCA rotations and
    coordinatewise rational-quadratic spline fits, then draws iid samples.

    A Laplace-style preprocessing step first standardises the target so
    that the spline range [-5, 5] is appropriate.

    Args:
        log_prob_fn:    (n_chains, D) -> (n_chains,)  batched log density
        initial_state:  (n_chains, D)  starting positions (used for dim only)
        num_samples:    int, number of sample batches to return
        warmup:         int, Adam steps per layer (default 1000)
        seed:           int, random seed
        verbose:        bool, print progress
        gamma:          float, PCA variance fraction (0=no rotation, 0.9-0.99 typical)
        niter:          int, number of Gaussianization layers
        num_bins:       int, spline bins per dimension
        n_train:        int, training samples per MFVI step
        n_pca:          int, samples for Score PCA (default: n_train)
        beta_0:         float, initial temperature for annealing
        lr:             float, Adam learning rate
        max_iter:       int, Adam steps per layer (default: warmup)
        preprocess:     bool, use Laplace preprocessing to standardise target

    Returns:
        samples: (num_samples, n_chains, D)
        info:    dict with diagnostics
    """
    n_chains, dim = initial_state.shape
    key = jax.random.key(seed)

    if max_iter is None:
        max_iter = warmup

    # Build single-point log density from the batched one
    def log_prob_single(x):
        return log_prob_fn(x[None, :])[0]

    # Preprocessing: find a good affine reparametrisation
    shift = jnp.zeros(dim)
    scale = jnp.eye(dim)
    scale_inv = jnp.eye(dim)
    log_det_scale = 0.0

    if preprocess:
        if verbose:
            print("  Laplace preprocessing ...", end=" ", flush=True)
        key, subkey = jax.random.split(key)
        shift, scale = _laplace_preprocess(log_prob_single, dim, subkey)
        # scale_inv = scale^{-1}
        scale_inv = jnp.linalg.inv(scale)
        log_det_scale = jnp.sum(jnp.log(jnp.abs(jnp.diag(
            jnp.linalg.cholesky(scale @ scale.T)))))
        if verbose:
            print("done.")

    # Standardised target: z = scale_inv @ (x - shift)
    # log p_std(z) = log p(shift + scale @ z) + log|det(scale)|
    def log_prob_std(z):
        x = shift + scale @ z
        return log_prob_single(x) + log_det_scale

    # Fit the flow on the standardised target
    if verbose:
        print(f"Iterative Gaussianization: D={dim}, niter={niter}, "
              f"gamma={gamma}, bins={num_bins}, lr={lr}, max_iter={max_iter}")

    key, subkey = jax.random.split(key)
    transforms, nb = _iterative_gaussianization(
        log_prob_std, dim, n_train, subkey,
        gamma=gamma, niter=niter, num_bins=num_bins,
        n_pca=n_pca, beta_0=beta_0, lr=lr, max_iter=max_iter,
        verbose=verbose)

    # Draw iid samples (in standardised space, then un-standardise)
    if verbose:
        print("  Drawing samples ...", end=" ", flush=True)

    all_samples = []
    for t in range(num_samples):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (n_chains, dim))
        z_out, _ = _forward_map(transforms, nb, z)
        # Un-standardise: x = shift + scale @ z_out
        x = z_out @ scale.T + shift[None, :]
        all_samples.append(x)

    samples = jnp.stack(all_samples, axis=0)  # (num_samples, n_chains, D)

    # Compute ELBO estimate
    key, subkey = jax.random.split(key)
    z_eval = jax.random.normal(subkey, (min(n_chains, 500), dim))
    z_out, logdet_eval = _forward_map(transforms, nb, z_eval)
    x_eval = z_out @ scale.T + shift[None, :]
    log_base = -0.5 * dim * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(z_eval ** 2, axis=1)
    logq = log_base - logdet_eval - log_det_scale
    logp_vals = jax.vmap(log_prob_single)(x_eval)
    elbo = float(jnp.mean(logp_vals - logq))

    if verbose:
        print(f"done.  ELBO={elbo:.2f}")

    info = {
        'elbo': elbo,
        'niter': niter,
        'gamma': gamma,
        'num_bins': num_bins,
        'n_layers': len(transforms),
        'ranks': [int(W.shape[1]) for W, _ in transforms],
    }
    return samples, info


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    n_chains = 200
    num_samples = 2000

    # ── Test 1: Ill-conditioned Gaussian (D=10, kappa=1000) ──
    # Laplace preprocessing captures this exactly; flow detects "already
    # Gaussian" and skips.  Demonstrates affine-invariance.
    dim = 10
    print("\n" + "=" * 70)
    print("TEST 1: Ill-conditioned Gaussian  (D=%d, kappa=1000)" % dim)
    print("=" * 70)

    eigvals = jnp.logspace(0, jnp.log10(1000.), dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov = Q @ jnp.diag(eigvals) @ Q.T
    prec = Q @ jnp.diag(1. / eigvals) @ Q.T

    def log_prob_gauss(x):
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)

    init = jax.random.normal(jax.random.key(42), (n_chains, dim))

    s, info = sampler_ig(
        log_prob_gauss, init, num_samples,
        warmup=200, seed=123,
        gamma=0.95, niter=1, num_bins=8,
        n_train=2000, lr=0.005, max_iter=200)

    flat = s.reshape(-1, dim)
    var_est = jnp.var(flat, axis=0)
    var_true = jnp.diag(cov)
    rel_err = jnp.mean(jnp.abs(var_est - var_true) / var_true)
    print(f"  mean_rel_err(var)={rel_err:.4f}"
          f"  var_range=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
          f"  (target: [{jnp.min(var_true):.2f}, {jnp.max(var_true):.2f}])")
    print(f"  info: {info}")

    # ── Test 2: Multivariate Student-t (D=6, nu=3, kappa=100) ──
    # Unimodal, heavy-tailed, ill-conditioned.  IG captures the
    # covariance structure via Laplace + heavy tails via splines.
    dim = 6
    print("\n" + "=" * 70)
    print("TEST 2: Multivariate Student-t  (D=%d, nu=3, kappa=100)" % dim)
    print("=" * 70)

    nu = 3.0
    eigvals_t = jnp.logspace(0, 2, dim)
    Qt, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    Sigma_inv_t = Qt @ jnp.diag(1. / eigvals_t) @ Qt.T
    true_var_t = (nu / (nu - 2)) * eigvals_t

    def log_prob_studentt(x):
        quad = jnp.sum((x @ Sigma_inv_t) * x, axis=-1)
        return -(nu + dim) / 2.0 * jnp.log(1.0 + quad / nu)

    init_t = jax.random.normal(jax.random.key(42), (n_chains, dim))

    s, info = sampler_ig(
        log_prob_studentt, init_t, num_samples,
        warmup=1000, seed=42,
        gamma=0.95, niter=3, num_bins=12,
        n_train=3000, lr=0.005, max_iter=1000, beta_0=0.5)

    flat = s.reshape(-1, dim)
    var_eig = jnp.var(flat @ Qt, axis=0)
    rel_err = jnp.mean(jnp.abs(var_eig - true_var_t) / true_var_t)
    print(f"  eigenbasis var: {var_eig}")
    print(f"  target var:     {true_var_t}")
    print(f"  mean_rel_err(var)={rel_err:.3f}")
    print(f"  info: {info}")

    # ── Test 3: Warped Gaussian (D=8, cubic nonlinearity) ──
    # z ~ N(0, Sigma), x_i = z_i + 0.1*z_i^3.
    # Adds skewness/kurtosis.  IG rotation + spline captures this well.
    dim = 8
    print("\n" + "=" * 70)
    print("TEST 3: Warped Gaussian  (D=%d, alpha=0.1, kappa=3)" % dim)
    print("=" * 70)

    eigvals_w = jnp.linspace(1, 3, dim)
    Qw, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(5), (dim, dim)))
    Sigma_w = Qw @ jnp.diag(eigvals_w) @ Qw.T
    Sigma_inv_w = Qw @ jnp.diag(1. / eigvals_w) @ Qw.T
    alpha_w = 0.1

    def log_prob_warped(x):
        z = x
        for _ in range(10):
            f = z + alpha_w * z ** 3
            fp = 1 + 3 * alpha_w * z ** 2
            z = z - (f - x) / fp
        log_jac = -jnp.sum(jnp.log(1 + 3 * alpha_w * z ** 2), axis=-1)
        return -0.5 * jnp.sum((z @ Sigma_inv_w) * z, axis=-1) + log_jac

    # Reference variance via direct sampling
    z_ref = jax.random.normal(jax.random.key(99), (500000, dim))
    z_ref = z_ref @ jnp.linalg.cholesky(Sigma_w).T
    x_ref = z_ref + alpha_w * z_ref ** 3
    var_true_w = jnp.var(x_ref, axis=0)

    init_w = jax.random.normal(jax.random.key(42), (n_chains, dim))

    s, info = sampler_ig(
        log_prob_warped, init_w, num_samples,
        warmup=1000, seed=42,
        gamma=0.95, niter=3, num_bins=12,
        n_train=3000, lr=0.005, max_iter=1000, beta_0=0.5)

    flat = s.reshape(-1, dim)
    var_est = jnp.var(flat, axis=0)
    rel_err = jnp.mean(jnp.abs(var_est - var_true_w) / var_true_w)
    print(f"  var est:  [{jnp.min(var_est):.3f}, {jnp.max(var_est):.3f}]")
    print(f"  var true: [{jnp.min(var_true_w):.3f}, {jnp.max(var_true_w):.3f}]")
    print(f"  mean_rel_err(var)={rel_err:.3f}")
    print(f"  info: {info}")

    # ── Test 4: Rosenbrock (D=10, a=1, b=100) ──
    # Highly non-Gaussian banana-shaped target.  Componentwise flows
    # cannot capture the x_odd ~ N(x_even^2, ...) coupling, so variance
    # will be underestimated — this shows the method's limitations.
    dim = 10
    print("\n" + "=" * 70)
    print("TEST 4: Rosenbrock  (D=%d, a=1, b=100)" % dim)
    print("=" * 70)

    a, b = 1.0, 100.0
    def log_prob_rosen(x):
        x_even = x[:, ::2]
        x_odd = x[:, 1::2]
        return -(b * jnp.sum((x_odd - x_even ** 2) ** 2, axis=1)
                 + jnp.sum((x_even - a) ** 2, axis=1))

    init_r = jax.random.normal(jax.random.key(42), (n_chains, dim))

    s, info = sampler_ig(
        log_prob_rosen, init_r, num_samples,
        warmup=2000, seed=123,
        gamma=0.95, niter=5, num_bins=12,
        n_train=5000, lr=0.003, max_iter=2000, beta_0=0.1)

    flat = s.reshape(-1, dim)
    me = float(jnp.mean(flat[:, ::2]))
    ve = float(jnp.mean(jnp.var(flat[:, ::2], axis=0)))
    mo = float(jnp.mean(flat[:, 1::2]))
    vo = float(jnp.mean(jnp.var(flat[:, 1::2], axis=0)))
    print(f"  x_even: mean={me:.3f} var={ve:.4f} (target: mean=1, var=0.5)")
    print(f"  x_odd:  mean={mo:.3f} var={vo:.4f} (target: mean=1.5, var~2.505)")
    print(f"  info: {info}")
