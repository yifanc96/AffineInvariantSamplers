"""
sampler_aldi — Affine-invariant Langevin Dynamics (ALDI), unadjusted, single file, JAX only.

Overdamped Langevin with whole-ensemble covariance preconditioning via
the non-symmetric square root B = centered/sqrt(N), avoiding matrix
inversions and square roots.

  dx^(i) = [ -C nabla Phi(x^(i)) + (d+1)/N (x^(i) - m) ] dt + sqrt(2) B dxi^(i)

where C = BB^T,  B = (1/sqrt(N))[x^(1)-m, ..., x^(N)-m],  xi^(i) in R^N.

The correction (d+1)/N (x^(i) - m) = nabla_{x^(i)} . C  ensures the correct
stationary distribution pi(x) propto exp(-Phi(x)).

No Metropolis correction — the discretisation has O(h) bias.

Step-size adaptation options:
  "maxdrift" — per-step max-drift normalization (default, most robust).
               h = h_base / max(1, max_drift / target).
               Normalizes by the worst-case drift across all particles,
               so any stiff chain clamps h.  Works reliably even at
               aggressive h_base on ill-conditioned / curved targets.
  "pgn"      — per-step RMS projected-gradient scaling.
               h = h_base / max(1, rms_pgn / target).
               Uses the ensemble-average projected gradient.  Faster to
               relax when the ensemble has mixed scales, but can blow up
               if a single chain hits a stiff region the RMS doesn't see.
               Single gradient evaluation per step.
  "samadams" — SamAdams (Leimkuhler et al. 2025): EMA of gradient norm,
               h_eff = h_base / max(1, sqrt(zeta + eps)).
               NOT RECOMMENDED for ALDI: designed for kinetic (underdamped)
               integrators; Euler-Maruyama can diverge in a single step
               before the EMA reacts, after which zeta is contaminated and
               h_eff can collapse to 0 or stay too large.  Observed to NaN
               on Rosenbrock at h_base=0.5 and ill-cond. Gaussian at h_base>=0.5.
  None/False — no adaptation.  Fast per step, but safe only if h_base is
               tuned by hand — diverges on stiff problems.

Reference: Garbuno-Inigo, Nusken & Reich, SIADS 2020  (ALDI)
           Leimkuhler, Sherlock & Singh, 2025  (SamAdams)
"""

import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _nan_guard_state(X_new, X_old):
    """Replace NaN particles with old particles (reject divergent step)."""
    ok = ~jnp.any(jnp.isnan(X_new), axis=1, keepdims=True)  # (N, 1)
    return jnp.where(ok, X_new, X_old)


# ──────────────────────────────────────────────────────────────────────────────
# ALDI step variants (Euler-Maruyama), each fused with its adaptation signal
# ──────────────────────────────────────────────────────────────────────────────

def _aldi_step_plain(X, h, grad_lp, corr_coeff, key):
    """Plain EM step, no adaptation info returned."""
    N, d = X.shape
    m = jnp.mean(X, axis=0)
    centered = (X - m) / jnp.sqrt(N)
    g = grad_lp(X)
    drift = (g @ centered.T) @ centered + corr_coeff * (X - m)
    xi = jax.random.normal(key, (N, N))
    noise = jnp.sqrt(2.0 * h) * (xi @ centered)
    return X + h * drift + noise


def _aldi_step_pgn(X, h_base, grad_lp, corr_coeff, target_gnorm, key):
    """EM step with fused per-step PGN adaptation.  Single gradient eval.

    Returns: (X_new, h_eff)
    """
    N, d = X.shape
    m = jnp.mean(X, axis=0)
    centered = (X - m) / jnp.sqrt(N)

    g = grad_lp(X)                                        # nabla log pi

    # PGN: adapt h from projected-gradient RMS
    grad_phi = -g                                          # nabla Phi
    pg = grad_phi @ centered.T                             # (N, N)
    rms = jnp.sqrt(jnp.mean(jnp.sum(pg**2, axis=1)))
    h = h_base / jnp.maximum(1.0, rms / target_gnorm)

    # EM step
    drift = (g @ centered.T) @ centered + corr_coeff * (X - m)
    xi = jax.random.normal(key, (N, N))
    noise = jnp.sqrt(2.0 * h) * (xi @ centered)
    X_new = X + h * drift + noise

    # NaN guard: revert divergent particles
    X_new = _nan_guard_state(X_new, X)
    return X_new, h


def _aldi_step_samadams(X, h_base, grad_lp, corr_coeff, zeta, rho, eps_sa, key):
    """EM step with fused SamAdams adaptation.

    Returns: (X_new, zeta_new, h_eff)
    """
    N, d = X.shape
    m = jnp.mean(X, axis=0)
    centered = (X - m) / jnp.sqrt(N)

    g = grad_lp(X)
    grad_phi = -g
    pg = grad_phi @ centered.T                             # (N, N)

    # SamAdams statistic: mean per-component projected-gradient norm²
    s = jnp.mean(jnp.sum(pg**2, axis=1)) / N

    # h_eff = h_base / max(1, sqrt(zeta))
    # cap at h_base to prevent EM blow-up in flat regions
    h = h_base / jnp.maximum(1.0, jnp.sqrt(zeta + eps_sa))

    # EM step
    drift = (g @ centered.T) @ centered + corr_coeff * (X - m)
    xi = jax.random.normal(key, (N, N))
    noise = jnp.sqrt(2.0 * h) * (xi @ centered)
    X_new = X + h * drift + noise

    # NaN guard: protect both state and EMA
    any_nan = jnp.any(jnp.isnan(X_new))
    X_new = _nan_guard_state(X_new, X)
    s = jnp.where(any_nan | jnp.isnan(s), zeta, s)

    zeta_new = rho * zeta + (1.0 - rho) * s
    return X_new, zeta_new, h


def _aldi_step_maxdrift(X, h_base, grad_lp, corr_coeff, target_gnorm, key):
    """EM step with fused max-drift adaptation.

    Returns: (X_new, h_eff)
    """
    N, d = X.shape
    m = jnp.mean(X, axis=0)
    centered = (X - m) / jnp.sqrt(N)

    g = grad_lp(X)
    drift = (g @ centered.T) @ centered + corr_coeff * (X - m)

    # max-drift normalization: h = h_base / max(1, max_drift / target)
    drift_norms = jnp.sqrt(jnp.sum(drift**2, axis=1))     # (N,)
    max_d = jnp.max(drift_norms)
    max_d = jnp.where(jnp.isnan(max_d), 1e6, max_d)
    h = h_base / jnp.maximum(1.0, max_d / target_gnorm)

    # EM step
    xi = jax.random.normal(key, (N, N))
    noise = jnp.sqrt(2.0 * h) * (xi @ centered)
    X_new = X + h * drift + noise

    # NaN guard
    X_new = _nan_guard_state(X_new, X)
    return X_new, h


# ──────────────────────────────────────────────────────────────────────────────
# sampler_aldi
# ──────────────────────────────────────────────────────────────────────────────

def sampler_aldi(
    log_prob_fn,
    initial_state,
    num_samples,
    step_size        = 0.5,
    warmup           = 1000,
    thin_by          = 1,
    adapt            = "maxdrift",
    target_gnorm     = 1.0,
    rho              = 0.999,
    grad_log_prob_fn = None,
    seed             = 0,
    verbose          = True,
):
    """
    ALDI — unadjusted overdamped Langevin with whole-ensemble covariance
    preconditioning.

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (N, D).  N particles.
        num_samples      : Production samples to return.
        step_size        : Base step size (adapted if adapt is set).
        warmup           : Warmup iterations.
        thin_by          : Keep every thin_by-th sample.
        adapt            : "maxdrift" (default, most robust), "pgn",
                           "samadams" (not recommended — can NaN on stiff
                           problems), or None / False.
                           See module docstring for trade-offs.
        target_gnorm     : Target max drift norm (maxdrift) or RMS
                           projected-gradient norm (pgn).
        rho              : EMA decay for SamAdams (default 0.999).
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
        seed             : Integer random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, N, D)
        info    : dict(step_size, adapt, ...)
    """
    if adapt is True:
        adapt = "maxdrift"
    if adapt is False:
        adapt = None

    X = jnp.asarray(initial_state, dtype=jnp.float64
                     if initial_state.dtype == jnp.float64
                     else jnp.float32)
    N, d = X.shape

    if grad_log_prob_fn is None:
        grad_lp = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_lp = grad_log_prob_fn

    corr_coeff = (d + 1.0) / N
    key = jax.random.key(seed)

    h_base = jnp.float32(step_size)   # keep as JAX scalar for clean tracing

    # ── pgn ─────────────────────────────────────────────────────────────────
    if adapt == "pgn":
        _step = jax.jit(lambda X, key: _aldi_step_pgn(
            X, h_base, grad_lp, corr_coeff, target_gnorm, key))

        # warmup (per-step adaptive)
        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)
        for i in range(warmup):
            X, h = _step(X, wkeys[i])

        if verbose:
            lp = log_prob_fn(X)
            print(f"ALDI [pgn]  N={N}  d={d}  h={float(h):.4f}  "
                  f"mean_lp={float(jnp.mean(lp)):.2f}")

        # production (per-step adaptive via scan)
        @jax.jit
        def _prod(X, key):
            X_new, h = _aldi_step_pgn(
                X, h_base, grad_lp, corr_coeff, target_gnorm, key)
            return X_new, X_new

        key, k = jax.random.split(key)
        skeys = jax.random.split(k, num_samples * thin_by)
        _, all_X = jax.lax.scan(_prod, X, skeys)
        samples = all_X[::thin_by]
        info = dict(step_size=float(h), adapt="pgn")

    # ── samadams ────────────────────────────────────────────────────────────
    elif adapt == "samadams":
        eps_sa = jnp.float32(1e-8)
        rho_j = jnp.float32(rho)

        # initialise zeta from current projected-gradient norm
        m0 = jnp.mean(X, axis=0)
        c0 = (X - m0) / jnp.sqrt(N)
        g0 = -grad_lp(X)
        pg0 = g0 @ c0.T
        zeta = jnp.mean(jnp.sum(pg0**2, axis=1)) / N

        _step = jax.jit(lambda X, zeta, key: _aldi_step_samadams(
            X, h_base, grad_lp, corr_coeff, zeta, rho_j, eps_sa, key))

        # warmup
        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)
        for i in range(warmup):
            X, zeta, h = _step(X, zeta, wkeys[i])

        h_final = h_base / jnp.maximum(1.0, jnp.sqrt(zeta + eps_sa))
        if verbose:
            lp = log_prob_fn(X)
            print(f"ALDI [samadams]  N={N}  d={d}  h_base={step_size}  "
                  f"h_eff={float(h_final):.4f}  zeta={float(zeta):.4f}  "
                  f"mean_lp={float(jnp.mean(lp)):.2f}")

        # production
        @jax.jit
        def _prod(carry, key):
            X, zeta = carry
            X_new, zeta_new, h = _aldi_step_samadams(
                X, h_base, grad_lp, corr_coeff, zeta, rho_j, eps_sa, key)
            return (X_new, zeta_new), X_new

        key, k = jax.random.split(key)
        skeys = jax.random.split(k, num_samples * thin_by)
        (X, zeta), all_X = jax.lax.scan(_prod, (X, zeta), skeys)
        samples = all_X[::thin_by]
        h_out = h_base / jnp.maximum(1.0, jnp.sqrt(zeta + eps_sa))
        info = dict(step_size=float(h_out),
                    adapt="samadams", zeta=float(zeta))

    # ── maxdrift ────────────────────────────────────────────────────────────
    elif adapt == "maxdrift":
        _step = jax.jit(lambda X, key: _aldi_step_maxdrift(
            X, h_base, grad_lp, corr_coeff, target_gnorm, key))

        # warmup
        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)
        for i in range(warmup):
            X, h = _step(X, wkeys[i])

        if verbose:
            lp = log_prob_fn(X)
            print(f"ALDI [maxdrift]  N={N}  d={d}  h={float(h):.4f}  "
                  f"mean_lp={float(jnp.mean(lp)):.2f}")

        # production
        @jax.jit
        def _prod(X, key):
            X_new, h = _aldi_step_maxdrift(
                X, h_base, grad_lp, corr_coeff, target_gnorm, key)
            return X_new, X_new

        key, k = jax.random.split(key)
        skeys = jax.random.split(k, num_samples * thin_by)
        _, all_X = jax.lax.scan(_prod, X, skeys)
        samples = all_X[::thin_by]
        info = dict(step_size=float(h), adapt="maxdrift")

    # ── no adaptation ───────────────────────────────────────────────────────
    else:
        h = jnp.float32(step_size)

        # warmup via scan (was a Python loop; now fully jit'd).
        def _warm(X, key):
            X_new = _aldi_step_plain(X, h, grad_lp, corr_coeff, key)
            return X_new, None

        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)
        X, _ = jax.lax.scan(_warm, X, wkeys)

        if verbose:
            lp = log_prob_fn(X)
            print(f"ALDI [no adapt]  N={N}  d={d}  h={float(h):.4f}  "
                  f"mean_lp={float(jnp.mean(lp)):.2f}")

        @jax.jit
        def _prod(X, key):
            X_new = _aldi_step_plain(X, h, grad_lp, corr_coeff, key)
            return X_new, X_new

        key, k = jax.random.split(key)
        skeys = jax.random.split(k, num_samples * thin_by)
        _, all_X = jax.lax.scan(_prod, X, skeys)
        samples = all_X[::thin_by]
        info = dict(step_size=float(h), adapt=None)

    # Production gradient evals: one per step per particle.
    info["n_grad_evals"] = int(num_samples * thin_by) * int(N)
    if verbose:
        print(f"ALDI done.")
    return samples, info


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # --- shared setup ---
    dim = 20;  kappa = 1000.
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov_gauss = Q @ jnp.diag(eigvals) @ Q.T
    prec_gauss = Q @ jnp.diag(1. / eigvals) @ Q.T
    var_true_gauss = jnp.diag(cov_gauss)
    def log_prob_gauss(x):
        return -0.5 * jnp.sum((x @ prec_gauss) * x, axis=-1)

    a_ros, b_ros = 1.0, 100.0;  dim_ros = 10
    def log_prob_rosen(x):
        x_even = x[:, ::2]; x_odd = x[:, 1::2]
        return -(b_ros * jnp.sum((x_odd - x_even**2)**2, axis=1)
                 + jnp.sum((x_even - a_ros)**2, axis=1))

    init_gauss = jax.random.normal(jax.random.key(42), (100, dim))
    init_rosen = jax.random.normal(jax.random.key(42), (100, dim_ros))

    for adapt_mode in ["pgn", "samadams", "maxdrift"]:
        # --- Gaussian ---
        print("=" * 60)
        print(f"ALDI [{adapt_mode}] — Ill-conditioned Gaussian  (D={dim}, kappa={kappa:.0f})")
        print("=" * 60)
        for hs in [0.1, 0.5, 1.0]:
            samples, info = sampler_aldi(
                log_prob_gauss, init_gauss, num_samples=5000,
                warmup=2000, step_size=hs, adapt=adapt_mode, seed=123)
            flat = samples.reshape(-1, dim)
            var_est = jnp.var(flat, axis=0)
            rel_err = jnp.mean(jnp.abs(var_est - var_true_gauss) / var_true_gauss)
            print(f"  h_base={hs}  h_eff={info['step_size']:.4f}  "
                  f"rel_err(var)={rel_err:.3f}  "
                  f"var_range=[{jnp.min(var_est):.1f}, {jnp.max(var_est):.1f}]  "
                  f"(target: [{jnp.min(var_true_gauss):.1f}, {jnp.max(var_true_gauss):.1f}])")

        # --- Rosenbrock ---
        print()
        print("=" * 60)
        print(f"ALDI [{adapt_mode}] — Rosenbrock  (D={dim_ros}, a=1, b=100)")
        print("=" * 60)
        for hs in [0.01, 0.05, 0.1, 0.2, 0.4]:
            samples, info = sampler_aldi(
                log_prob_rosen, init_rosen, num_samples=50000,
                warmup=10000, step_size=hs, adapt=adapt_mode, seed=123)
            flat = samples.reshape(-1, dim_ros)
            mean_even = jnp.mean(flat[:, ::2])
            mean_odd = jnp.mean(flat[:, 1::2])
            var_even = jnp.mean(jnp.var(flat[:, ::2], axis=0))
            var_odd = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
            print(f"  h_base={hs}  h_eff={info['step_size']:.4f}  "
                  f"x_even: mean={mean_even:.3f} var={var_even:.3f}  "
                  f"x_odd: mean={mean_odd:.3f} var={var_odd:.3f}  "
                  f"(target: even=(1.0, 0.5)  odd=(1.5, 2.505))")
        print()
