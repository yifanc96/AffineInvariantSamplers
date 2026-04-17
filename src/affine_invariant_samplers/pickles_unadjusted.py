"""
sampler_pickles_unadjusted — unadjusted PICKLES, single file, JAX only.

Kinetic (underdamped) Langevin with whole-ensemble covariance preconditioning,
discretised with the BAOAB splitting.  No Metropolis correction.

  dq^(i) = B p^(i) dt
  dp^(i) = [ -B^T nabla Phi(q^(i)) + corr_i  - gamma p^(i) ] dt + sqrt(2 gamma) dW

where B = (1/sqrt(N))[q^(1)-m, ..., q^(N)-m]  in R^{d x N},
      corr_i = (d/sqrt(N))(e_i - 1_N/N)  in R^N   (constant, state-independent),
      p^(i) in R^N   (momentum in ensemble span).

The correction  nabla_{q^(i)} . B^T  ensures the correct stationary distribution
Pi*(Q,P) propto exp( -sum Phi(q^(i)) - (1/2) sum ||p^(i)||^2 ).

The BAOAB splitting gives O(h^2) configurational bias (weak 2nd order).

Gradient caching: the gradient nabla Phi(Q) evaluated for the second B-kick
at the end of step n is reused (re-projected onto the new centered matrix)
for the first B-kick of step n+1.  Thus k steps cost k+1 gradient evaluations.

Step-size adaptation options:
  "maxdrift" — per-step max projected-gradient-norm scaling (default,
               most robust).
               h = h_base / max(1, sqrt(max_pgn / target)).
               Normalizes by the worst-case projected gradient across
               particles; clamps h whenever any chain hits a stiff region.
               Reliable across ill-conditioned and curved targets.
  "pgn"      — per-step RMS projected-gradient norm scaling.
               h = h_base / max(1, sqrt(rms_pgn / target)).
               Uses the ensemble-average; slightly less conservative than
               maxdrift but can blow up on Rosenbrock at large h_base
               (observed variance explosion at h_base=0.5).
  "samadams" — SamAdams (Leimkuhler et al. 2025): EMA of gradient norm,
               h_eff = h_base / max(1, sqrt(zeta + eps)).
               NOT RECOMMENDED in general: the EMA (rho=0.999) takes
               hundreds of steps to react; if the first step already
               diverges on a stiff problem, zeta becomes contaminated and
               h_eff can collapse to 0.  Observed to NaN on Rosenbrock
               at h_base=0.5.  Only safe on well-conditioned targets.
  None/False — no adaptation.  Safe only if h_base is tuned by hand —
               diverges on Rosenbrock at h ≥ 0.1.

Reference: Leimkuhler, Matthews & Weare, Stat. Comput. 2018  (BAOAB)
           Leimkuhler, Sherlock & Singh, 2025  (SamAdams)
"""

import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _nan_guard_state(X_new, X_old):
    """Replace NaN particles with old particles (reject divergent step)."""
    ok = ~jnp.any(jnp.isnan(X_new), axis=1, keepdims=True)
    return jnp.where(ok, X_new, X_old)


def _nan_guard_mom(P_new, P_old):
    """Replace NaN momentum rows with old (reject divergent step)."""
    ok = ~jnp.any(jnp.isnan(P_new), axis=1, keepdims=True)
    return jnp.where(ok, P_new, P_old)


# ──────────────────────────────────────────────────────────────────────────────
# BAOAB core — accepts cached gradient, returns new gradient for reuse
# ──────────────────────────────────────────────────────────────────────────────

def _baoab_core(Q, P, h, gamma, pg_start, centered, corr, grad_lp, key):
    """Core BAOAB step given the first B-kick projected gradient already.

    pg_start = B_n^T nabla Phi(Q_n) projected onto the CURRENT centered.

    Returns: Q_new, P_new, grad_phi_new
      grad_phi_new = nabla Phi(Q_new) in R^{N,d} — cached for the next step.
    """
    N = Q.shape[0]
    alpha = jnp.exp(-gamma * h)
    sigma = jnp.sqrt(1.0 - alpha**2)

    # B: half-kick
    P = P - (h / 2.0) * (pg_start - corr)
    # A: half-drift
    Q = Q + (h / 2.0) * (P @ centered)
    # O: full Ornstein-Uhlenbeck
    xi = jax.random.normal(key, (N, N))
    P = alpha * P + sigma * xi
    # A: half-drift  (same centered, fixed from start of step)
    Q = Q + (h / 2.0) * (P @ centered)
    # B: half-kick at new Q, old centered
    grad_phi_new = -grad_lp(Q)                        # <-- only grad eval
    pg_new = grad_phi_new @ centered.T
    P = P - (h / 2.0) * (pg_new - corr)

    return Q, P, grad_phi_new


# ──────────────────────────────────────────────────────────────────────────────
# BAOAB step variants with gradient caching
# ──────────────────────────────────────────────────────────────────────────────

def _baoab_step_plain(Q, P, h, gamma, grad_lp, corr, grad_phi_cached, key):
    """Plain BAOAB step with gradient caching, no adaptation.

    grad_phi_cached: nabla Phi(Q) from previous step (or initial computation).
    Returns: Q_new, P_new, grad_phi_new
    """
    N, d = Q.shape
    m = jnp.mean(Q, axis=0)
    centered = (Q - m) / jnp.sqrt(N)
    pg = grad_phi_cached @ centered.T                  # re-project onto new B
    return _baoab_core(Q, P, h, gamma, pg, centered, corr, grad_lp, key)


def _baoab_step_pgn(Q, P, h_base, gamma, grad_lp, corr,
                    target_gnorm, grad_phi_cached, key):
    """BAOAB step with PGN adaptation and gradient caching.

    Returns: (Q_new, P_new, grad_phi_new, h_eff)
    """
    N, d = Q.shape
    m = jnp.mean(Q, axis=0)
    centered = (Q - m) / jnp.sqrt(N)

    # re-project cached gradient onto current centered
    pg = grad_phi_cached @ centered.T                  # (N, N)

    # PGN: RMS projected-gradient norm, sqrt scaling for BAOAB
    rms = jnp.sqrt(jnp.mean(jnp.sum(pg**2, axis=1)))
    h = h_base / jnp.maximum(1.0, jnp.sqrt(rms / target_gnorm))

    Q_new, P_new, grad_phi_new = _baoab_core(
        Q, P, h, gamma, pg, centered, corr, grad_lp, key)

    # NaN guard
    any_nan = jnp.any(jnp.isnan(Q_new))
    Q_new = _nan_guard_state(Q_new, Q)
    P_new = _nan_guard_mom(P_new, P)
    grad_phi_new = jnp.where(any_nan, grad_phi_cached, grad_phi_new)

    return Q_new, P_new, grad_phi_new, h


def _baoab_step_samadams(Q, P, h_base, gamma, grad_lp, corr,
                         zeta, rho, eps_sa, grad_phi_cached, key):
    """BAOAB step with SamAdams adaptation and gradient caching.

    Returns: (Q_new, P_new, grad_phi_new, zeta_new, h_eff)
    """
    N, d = Q.shape
    m = jnp.mean(Q, axis=0)
    centered = (Q - m) / jnp.sqrt(N)

    pg = grad_phi_cached @ centered.T                  # (N, N)

    # SamAdams statistic
    s = jnp.mean(jnp.sum(pg**2, axis=1)) / N

    h = h_base / jnp.maximum(1.0, jnp.sqrt(zeta + eps_sa))

    Q_new, P_new, grad_phi_new = _baoab_core(
        Q, P, h, gamma, pg, centered, corr, grad_lp, key)

    # NaN guard: protect state, gradient cache, and EMA
    any_nan = jnp.any(jnp.isnan(Q_new))
    Q_new = _nan_guard_state(Q_new, Q)
    P_new = _nan_guard_mom(P_new, P)
    grad_phi_new = jnp.where(any_nan, grad_phi_cached, grad_phi_new)
    s = jnp.where(any_nan | jnp.isnan(s), zeta, s)

    zeta_new = rho * zeta + (1.0 - rho) * s
    return Q_new, P_new, grad_phi_new, zeta_new, h


def _baoab_step_maxpgn(Q, P, h_base, gamma, grad_lp, corr,
                       target_gnorm, grad_phi_cached, key):
    """BAOAB step with max-PGN adaptation and gradient caching.

    Returns: (Q_new, P_new, grad_phi_new, h_eff)
    """
    N, d = Q.shape
    m = jnp.mean(Q, axis=0)
    centered = (Q - m) / jnp.sqrt(N)

    pg = grad_phi_cached @ centered.T                  # (N, N)

    max_pgn = jnp.max(jnp.sqrt(jnp.sum(pg**2, axis=1)))
    max_pgn = jnp.where(jnp.isnan(max_pgn), 1e6, max_pgn)
    h = h_base / jnp.maximum(1.0, jnp.sqrt(max_pgn / target_gnorm))

    Q_new, P_new, grad_phi_new = _baoab_core(
        Q, P, h, gamma, pg, centered, corr, grad_lp, key)

    # NaN guard
    any_nan = jnp.any(jnp.isnan(Q_new))
    Q_new = _nan_guard_state(Q_new, Q)
    P_new = _nan_guard_mom(P_new, P)
    grad_phi_new = jnp.where(any_nan, grad_phi_cached, grad_phi_new)

    return Q_new, P_new, grad_phi_new, h


# ──────────────────────────────────────────────────────────────────────────────
# sampler_pickles_unadjusted
# ──────────────────────────────────────────────────────────────────────────────

def sampler_pickles_unadjusted(
    log_prob_fn,
    initial_state,
    num_samples,
    step_size        = 0.5,
    gamma            = 2.0,
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
    Unadjusted PICKLES — kinetic Langevin with whole-ensemble preconditioning.

    Each BAOAB step requires only ONE gradient evaluation (the gradient from
    the previous step's second B-kick is cached and re-projected).  Thus k
    iterations cost k+1 gradient evaluations total.

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (N, D).  N particles.
        num_samples      : Production samples to return.
        step_size        : Base step size (adapted if adapt is set).
        gamma            : Friction coefficient (> 0).
        warmup           : Warmup iterations.
        thin_by          : Keep every thin_by-th sample.
        adapt            : "maxdrift" (default, most robust), "pgn",
                           "samadams" (not recommended — can NaN on stiff
                           problems), or None / False.
                           See module docstring for trade-offs.
        target_gnorm     : Target projected-gradient norm for maxdrift/pgn.
        rho              : EMA decay for SamAdams (default 0.999).
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
        seed             : Integer random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, N, D)
        info    : dict(step_size, gamma, adapt, ...)
    """
    if adapt is True:
        adapt = "maxdrift"
    if adapt is False:
        adapt = None

    Q = jnp.asarray(initial_state)
    N, d = Q.shape

    if grad_log_prob_fn is None:
        grad_lp = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_lp = grad_log_prob_fn

    # constant correction matrix: (d/sqrt(N)) * (I - 11^T/N)
    corr = (d / jnp.sqrt(N)) * (jnp.eye(N) - jnp.ones((N, N)) / N)

    key = jax.random.key(seed)
    key, k = jax.random.split(key)
    P = jax.random.normal(k, (N, N))

    h_base = jnp.float32(step_size)

    # initial gradient (the only "extra" evaluation; all subsequent steps reuse)
    gp = -grad_lp(Q)                                  # nabla Phi(Q_0)

    # ── pgn ─────────────────────────────────────────────────────────────────
    if adapt == "pgn":
        _step = jax.jit(lambda Q, P, gp, key: _baoab_step_pgn(
            Q, P, h_base, gamma, grad_lp, corr, target_gnorm, gp, key))

        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)
        for i in range(warmup):
            Q, P, gp, h = _step(Q, P, gp, wkeys[i])

        if verbose:
            lp = log_prob_fn(Q)
            print(f"Unadj. PICKLES [pgn]  N={N} d={d} h={float(h):.4f} "
                  f"gamma={gamma} mean_lp={float(jnp.mean(lp)):.2f}")

        @jax.jit
        def _prod(carry, key):
            Q, P, gp = carry
            Q_new, P_new, gp_new, h = _baoab_step_pgn(
                Q, P, h_base, gamma, grad_lp, corr, target_gnorm, gp, key)
            return (Q_new, P_new, gp_new), Q_new

        key, k = jax.random.split(key)
        skeys = jax.random.split(k, num_samples * thin_by)
        _, all_Q = jax.lax.scan(_prod, (Q, P, gp), skeys)
        samples = all_Q[::thin_by]
        info = dict(step_size=float(h), gamma=gamma, adapt="pgn")

    # ── samadams ────────────────────────────────────────────────────────────
    elif adapt == "samadams":
        eps_sa = jnp.float32(1e-8)
        rho_j = jnp.float32(rho)

        # initialise zeta
        m0 = jnp.mean(Q, axis=0)
        c0 = (Q - m0) / jnp.sqrt(N)
        pg0 = gp @ c0.T
        zeta = jnp.mean(jnp.sum(pg0**2, axis=1)) / N

        _step = jax.jit(lambda Q, P, gp, zeta, key: _baoab_step_samadams(
            Q, P, h_base, gamma, grad_lp, corr, zeta, rho_j, eps_sa, gp, key))

        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)
        for i in range(warmup):
            Q, P, gp, zeta, h = _step(Q, P, gp, zeta, wkeys[i])

        h_final = h_base / jnp.maximum(1.0, jnp.sqrt(zeta + eps_sa))
        if verbose:
            lp = log_prob_fn(Q)
            print(f"Unadj. PICKLES [samadams]  N={N} d={d} h_base={step_size} "
                  f"h_eff={float(h_final):.4f} zeta={float(zeta):.4f} "
                  f"gamma={gamma} mean_lp={float(jnp.mean(lp)):.2f}")

        @jax.jit
        def _prod(carry, key):
            Q, P, gp, zeta = carry
            Q_new, P_new, gp_new, zeta_new, h = _baoab_step_samadams(
                Q, P, h_base, gamma, grad_lp, corr,
                zeta, rho_j, eps_sa, gp, key)
            return (Q_new, P_new, gp_new, zeta_new), Q_new

        key, k = jax.random.split(key)
        skeys = jax.random.split(k, num_samples * thin_by)
        (Q, P, gp, zeta), all_Q = jax.lax.scan(
            _prod, (Q, P, gp, zeta), skeys)
        samples = all_Q[::thin_by]
        h_out = h_base / jnp.maximum(1.0, jnp.sqrt(zeta + eps_sa))
        info = dict(step_size=float(h_out),
                    gamma=gamma, adapt="samadams", zeta=float(zeta))

    # ── maxdrift ────────────────────────────────────────────────────────────
    elif adapt == "maxdrift":
        _step = jax.jit(lambda Q, P, gp, key: _baoab_step_maxpgn(
            Q, P, h_base, gamma, grad_lp, corr, target_gnorm, gp, key))

        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)
        for i in range(warmup):
            Q, P, gp, h = _step(Q, P, gp, wkeys[i])

        if verbose:
            lp = log_prob_fn(Q)
            print(f"Unadj. PICKLES [maxdrift]  N={N} d={d} h={float(h):.4f} "
                  f"gamma={gamma} mean_lp={float(jnp.mean(lp)):.2f}")

        @jax.jit
        def _prod(carry, key):
            Q, P, gp = carry
            Q_new, P_new, gp_new, h = _baoab_step_maxpgn(
                Q, P, h_base, gamma, grad_lp, corr, target_gnorm, gp, key)
            return (Q_new, P_new, gp_new), Q_new

        key, k = jax.random.split(key)
        skeys = jax.random.split(k, num_samples * thin_by)
        _, all_Q = jax.lax.scan(_prod, (Q, P, gp), skeys)
        samples = all_Q[::thin_by]
        info = dict(step_size=float(h), gamma=gamma, adapt="maxdrift")

    # ── no adaptation ───────────────────────────────────────────────────────
    else:
        h = jnp.float32(step_size)

        # warmup via scan (was a Python loop; now fully jit'd).
        def _warm(carry, key):
            Q, P, gp = carry
            Q_new, P_new, gp_new = _baoab_step_plain(
                Q, P, h, gamma, grad_lp, corr, gp, key)
            return (Q_new, P_new, gp_new), None

        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)
        (Q, P, gp), _ = jax.lax.scan(_warm, (Q, P, gp), wkeys)

        if verbose:
            lp = log_prob_fn(Q)
            print(f"Unadj. PICKLES [no adapt]  N={N} d={d} h={float(h):.4f} "
                  f"gamma={gamma} mean_lp={float(jnp.mean(lp)):.2f}")

        @jax.jit
        def _prod(carry, key):
            Q, P, gp = carry
            Q_new, P_new, gp_new = _baoab_step_plain(
                Q, P, h, gamma, grad_lp, corr, gp, key)
            return (Q_new, P_new, gp_new), Q_new

        key, k = jax.random.split(key)
        skeys = jax.random.split(k, num_samples * thin_by)
        _, all_Q = jax.lax.scan(_prod, (Q, P, gp), skeys)
        samples = all_Q[::thin_by]
        info = dict(step_size=float(h), gamma=gamma, adapt=None)

    # One gradient evaluation per BAOAB step per particle (cached across
    # iterations for the second B-kick; the off-by-one is negligible).
    info["n_grad_evals"] = int(num_samples * thin_by) * int(N)
    if verbose:
        print(f"Unadj. PICKLES done.")
    return samples, info


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # --- shared setup ---
    dim = 20;  kappa = 1000.
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    QQ, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov_gauss = QQ @ jnp.diag(eigvals) @ QQ.T
    prec_gauss = QQ @ jnp.diag(1. / eigvals) @ QQ.T
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
        print(f"Unadj. PICKLES [{adapt_mode}] — Ill-cond. Gaussian (D={dim}, kappa={kappa:.0f})")
        print("=" * 60)
        for hs in [0.5, 1.0, 2.0]:
            samples, info = sampler_pickles_unadjusted(
                log_prob_gauss, init_gauss, num_samples=5000,
                warmup=2000, step_size=hs, gamma=2.0,
                adapt=adapt_mode, seed=123)
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
        print(f"Unadj. PICKLES [{adapt_mode}] — Rosenbrock (D={dim_ros}, a=1, b=100)")
        print("=" * 60)
        for hs in [0.01, 0.05, 0.1, 0.2]:
            samples, info = sampler_pickles_unadjusted(
                log_prob_rosen, init_rosen, num_samples=10000,
                warmup=2000, step_size=hs, gamma=2.0,
                adapt=adapt_mode, seed=123)
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
