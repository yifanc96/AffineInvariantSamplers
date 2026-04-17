"""
sampler_gvi — Single-Gaussian Variational Inference, JAX only.

Approximates a target density exp(-Phi(x)) with a single Gaussian N(m, C)
by minimising KL[N(m,C) || exp(-Phi)] under three different gradient flows:

  "fisher-rao"  (natural gradient / information geometry):
      dm/dt = -C  E[nabla Phi]
      Implicit precision: prec_new = (1-dt)*prec + dt*E[H]
    Affine-invariant; convergence rate independent of conditioning.

  "wasserstein"  (optimal-transport / Bures-Wasserstein gradient):
      dm/dt = -E[nabla Phi]
      Matrix-exponential Cholesky: L_new = expm(-dt*A)*L
    Preconditioned by position; good for well-separated modes.

  "wasserstein-fb"  (forward-backward splitting in BW space):
      Forward on potential (matrix-exponential), backward on entropy (BW proximal).
      From Diao & Balasubramanian (arXiv:2304.05398).

  "wfr"  (Wasserstein-Fisher-Rao / Hellinger-Kantorovich):
      dm/dt = -(alpha*I + beta*C) E[nabla Phi]
      dC/dt = alpha*(2I - HC - CH) + beta*(C - CHC)
      Strang splitting of W and FR steps (arXiv:2504.20400).
    Interpolates between W (alpha=1,beta=0) and FR (alpha=0,beta=1).

  "gradient-descent"  (Euclidean / parameter-space gradient):
      dm/dt = -E[nabla Phi]
      Cholesky eigendecomposition update preserving PD.
    Simplest; convergence rate depends on condition number.

Expectations E[nabla Phi] and E[nabla^2 Phi] are computed via sigma-point
(unscented transform) quadrature, so no auto-differentiation is needed
beyond the scalar Phi(x).

Reference: arXiv:2310.03597
           PKU-CMEGroup/InverseProblems.jl  (NGD.jl)
"""

from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# Sigma-point quadrature (unscented transform)
# ──────────────────────────────────────────────────────────────────────────────

@jax.jit
def _ut_sigma_points(mean, sqrt_cov, alpha):
    """Generate 2D+1 sigma points and mean/cov weights."""
    D = mean.shape[0]
    N = 2 * D + 1
    kappa = 0.0
    lam = alpha ** 2 * (D + kappa) - D

    scaled = jnp.sqrt(D + lam) * sqrt_cov   # (D, D)

    pts = jnp.zeros((N, D))
    pts = pts.at[0].set(mean)
    pts = pts.at[1:D + 1].set(mean[None, :] + scaled.T)
    pts = pts.at[D + 1:].set(mean[None, :] - scaled.T)

    w_m = jnp.full(N, 1.0 / (2.0 * (D + lam)))
    w_m = w_m.at[0].set(lam / (D + lam))

    beta = 2.0
    w_c = jnp.full(N, 1.0 / (2.0 * (D + lam)))
    w_c = w_c.at[0].set(lam / (D + lam) + 1.0 - alpha ** 2 + beta)

    return pts, w_m, w_c


# ──────────────────────────────────────────────────────────────────────────────
# Compute E[Phi], E[nabla Phi], E[nabla^2 Phi] via auto-diff + quadrature
# ──────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('phi_fn',))
def _compute_expectations(phi_fn, pts, w_m):
    """Compute expected value, gradient, and Hessian of Phi under sigma-point
    quadrature.

    Args:
        phi_fn:  (D,) -> scalar
        pts:     (N_ens, D)
        w_m:     (N_ens,)   mean weights

    Returns:
        E_phi:     scalar
        E_grad:    (D,)
        E_hess:    (D, D)
    """
    grad_fn = jax.grad(phi_fn)
    hess_fn = jax.hessian(phi_fn)

    phi_vals = jax.vmap(phi_fn)(pts)            # (N,)
    grad_vals = jax.vmap(grad_fn)(pts)          # (N, D)
    hess_vals = jax.vmap(hess_fn)(pts)          # (N, D, D)

    E_phi = jnp.dot(w_m, phi_vals)
    E_grad = w_m @ grad_vals
    E_hess = jnp.einsum('n,nij->ij', w_m, hess_vals)

    return E_phi, E_grad, E_hess


# ──────────────────────────────────────────────────────────────────────────────
# Update steps for the three gradient flows
# ──────────────────────────────────────────────────────────────────────────────

@jax.jit
def _step_fisher_rao(mean, cov, sqrt_cov, E_grad, E_hess, dt):
    """Fisher-Rao (natural gradient) — implicit precision update.

    Continuous:  dm/dt = -C E[nabla Phi]
                 dC/dt = C - C E[H] C
    Discretised: C^{-1}_new = (1-dt) C^{-1} + dt E[H]
                 Unconditionally PD when E[H] is PD and 0 < dt <= 1.
    """
    D = mean.shape[0]
    mean_new = mean - dt * cov @ E_grad

    prec = jnp.linalg.inv(cov)
    prec_new = (1.0 - dt) * prec + dt * E_hess
    # Eigenclamp for safety (e.g. when E[H] is indefinite)
    eigvals, eigvecs = jnp.linalg.eigh(prec_new)
    eigvals = jnp.maximum(eigvals, 1e-10)
    prec_new = eigvecs @ jnp.diag(eigvals) @ eigvecs.T
    cov_new = jnp.linalg.inv(prec_new)
    cov_new = 0.5 * (cov_new + cov_new.T)
    sqrt_cov_new = jnp.linalg.cholesky(cov_new + 1e-12 * jnp.eye(D))

    return mean_new, cov_new, sqrt_cov_new


@jax.jit
def _step_wasserstein(mean, cov, sqrt_cov, E_grad, E_hess, dt):
    """Wasserstein (W2) gradient — matrix-exponential Cholesky update.

    Continuous:  dm/dt = -E[nabla Phi]
                 dC/dt = 2I - E[H] C - C E[H]
    Discretised via Cholesky exponential integrator:
        A = E[H] - C^{-1}   (symmetrised)
        L_new = expm(-dt A) L
        C_new = L_new L_new^T
    Always PD since expm(-dt A) is always invertible.
    """
    D = mean.shape[0]
    mean_new = mean - dt * E_grad

    prec = jnp.linalg.inv(cov)
    A = 0.5 * ((E_hess - prec) + (E_hess - prec).T)
    # Matrix exponential via eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(A)
    exp_neg_dtA = eigvecs @ jnp.diag(jnp.exp(-dt * eigvals)) @ eigvecs.T

    sqrt_cov_new = exp_neg_dtA @ sqrt_cov
    cov_new = sqrt_cov_new @ sqrt_cov_new.T
    cov_new = 0.5 * (cov_new + cov_new.T)
    # Re-Cholesky for hygiene
    sqrt_cov_new = jnp.linalg.cholesky(cov_new + 1e-12 * jnp.eye(D))

    return mean_new, cov_new, sqrt_cov_new


@jax.jit
def _step_gradient_descent(mean, cov, sqrt_cov, E_grad, E_hess, dt):
    """Euclidean (parameter-space) GD — Cholesky exponential integrator.

    Continuous:  dm/dt = -E[nabla Phi]
                 dC/dt = (1/2)(C^{-1} - E[H])
    Discretised via Cholesky factor:
        G = (1/2)(C^{-1} - E[H])          (gradient in cov-space)
        Q = L^{-1} G L^{-T}               (transform to identity frame)
        L_new = L V diag(sqrt(max(1 + 2 dt d_i, eps))) V^T
    where Q = V diag(d_i) V^T.  PD is preserved by clamping.
    """
    D = mean.shape[0]
    mean_new = mean - dt * E_grad

    prec = jnp.linalg.inv(cov)
    G = 0.5 * (prec - E_hess)
    G = 0.5 * (G + G.T)

    inv_sqrt = jnp.linalg.inv(sqrt_cov)
    Q = inv_sqrt @ G @ inv_sqrt.T
    Q = 0.5 * (Q + Q.T)
    eigvals, eigvecs = jnp.linalg.eigh(Q)
    # C_new = L(I + 2*dt*Q)L^T;  need I + 2*dt*Q PD
    scales = jnp.sqrt(jnp.maximum(1.0 + 2.0 * dt * eigvals, 1e-10))
    sqrt_cov_new = sqrt_cov @ (eigvecs * scales[None, :]) @ eigvecs.T
    cov_new = sqrt_cov_new @ sqrt_cov_new.T
    cov_new = 0.5 * (cov_new + cov_new.T)
    sqrt_cov_new = jnp.linalg.cholesky(cov_new + 1e-12 * jnp.eye(D))

    return mean_new, cov_new, sqrt_cov_new


@jax.jit
def _bw_entropy_proximal(cov_half, dt):
    """BW proximal of the negative entropy (Diao et al. 2023, arXiv:2304.05398).

    Solves:  argmin_Σ { H(Σ) + (1/(2η)) W_2²(Σ, Σ_half) }
    where H(Σ) = -½ log det Σ  (neg entropy).

    Closed form:
        Σ_new = ½ (Σ_half + 2η I + [Σ_half (Σ_half + 4η I)]^{1/2})

    PD-preserving: even if Σ_half has small eigenvalues, the 4η I term
    inside the square root guarantees the result is well-conditioned.
    """
    D = cov_half.shape[0]
    # Compute [Σ_half (Σ_half + 4η I)]^{1/2} via eigendecomposition of Σ_half
    eigvals, eigvecs = jnp.linalg.eigh(cov_half)
    eigvals = jnp.maximum(eigvals, 0.0)   # clamp any tiny negatives
    # Σ_half (Σ_half + 4η I) has eigenvalues d_i * (d_i + 4η)  in eigvecs basis
    inner_eigs = jnp.sqrt(eigvals * (eigvals + 4.0 * dt))
    sqrtm = eigvecs @ jnp.diag(inner_eigs) @ eigvecs.T

    cov_new = 0.5 * (cov_half + 2.0 * dt * jnp.eye(D) + sqrtm)
    cov_new = 0.5 * (cov_new + cov_new.T)
    sqrt_cov_new = jnp.linalg.cholesky(cov_new + 1e-12 * jnp.eye(D))
    return cov_new, sqrt_cov_new


@jax.jit
def _step_wasserstein_fb(mean, cov, sqrt_cov, E_grad, E_hess, dt):
    """Wasserstein with forward-backward splitting (Diao et al. 2023).

    Forward step on potential E_q[Phi]  (pushforward):
        M = I - dt * E[H]
        Σ_half = M Σ M^T
    Backward step on negative entropy  (BW proximal):
        Σ_new = ½ (Σ_half + 2dt I + [Σ_half (Σ_half + 4dt I)]^{1/2})
    Always PD by construction — the proximal heals any PD loss from forward step.
    """
    D = mean.shape[0]
    mean_new = mean - dt * E_grad

    # Forward step: pushforward by affine map  x -> x - dt * E[H](x - m)
    # Σ_half = M Σ M^T  where  M = I - dt * E[H]
    H_sym = 0.5 * (E_hess + E_hess.T)
    M = jnp.eye(D) - dt * H_sym
    cov_half = M @ cov @ M.T
    cov_half = 0.5 * (cov_half + cov_half.T)

    # Backward step: BW proximal of negative entropy
    cov_new, sqrt_cov_new = _bw_entropy_proximal(cov_half, dt)

    return mean_new, cov_new, sqrt_cov_new


@jax.jit
def _step_wfr(mean, cov, sqrt_cov, E_grad, E_hess, dt,
              alpha=1.0, beta=1.0):
    """Wasserstein-Fisher-Rao (Hellinger-Kantorovich) flow.

    From arXiv:2504.20400.
    Mean ODE:   dm/dt = -(alpha I + beta C) E[∇Φ]
    Cov ODE:    dC/dt = alpha(2I - HC - CH) + beta(C - CHC)

    Discretisation: combined implicit-exponential.
      1. Mean: combined Wasserstein + Fisher-Rao mean update
      2. Cov:  implicit precision for the FR part,
               matrix-exponential for the W part,
               applied as a single step using the combined operator.
    """
    D = mean.shape[0]

    # Mean: dm = -(alpha*I + beta*C) * E[grad Phi] * dt
    mean_new = mean - dt * (alpha * E_grad + beta * cov @ E_grad)

    # Covariance: use implicit precision approach on the combined flow.
    # The continuous flow is:
    #   dC/dt = alpha*(2I - HC - CH) + beta*(C - CHC)
    # In precision P = C^{-1}:
    #   dP/dt = -alpha*(2P - H - H) + beta*(H - P)    [linearised]
    #         = -(2*alpha + beta)*P + (2*alpha + beta)*H
    # Implicit: P_new = (1 - (2*alpha+beta)*dt)*P + (2*alpha+beta)*dt*H
    # But this is a simplification. More carefully:
    #
    # FR part: P_new = (1-beta*dt)*P + beta*dt*H  (implicit, PD for dt<=1/beta)
    # W part:  L_new = expm(-alpha*dt*A)*L  where A = H - P
    #
    # Apply sequentially: first FR (implicit precision), then W (exponential).
    prec = jnp.linalg.inv(cov)
    H_sym = 0.5 * (E_hess + E_hess.T)

    # Fisher-Rao sub-step (implicit precision)
    beta_dt = jnp.minimum(beta * dt, 0.95)   # clamp for safety
    prec_fr = (1.0 - beta_dt) * prec + beta_dt * H_sym
    eig_p, V_p = jnp.linalg.eigh(prec_fr)
    eig_p = jnp.maximum(eig_p, 1e-10)
    prec_fr = V_p @ jnp.diag(eig_p) @ V_p.T
    cov_fr = jnp.linalg.inv(prec_fr)
    cov_fr = 0.5 * (cov_fr + cov_fr.T)
    sqrt_fr = jnp.linalg.cholesky(cov_fr + 1e-12 * jnp.eye(D))

    # Wasserstein sub-step (matrix exponential)
    A_w = 0.5 * ((H_sym - prec_fr) + (H_sym - prec_fr).T)
    eig_w, V_w = jnp.linalg.eigh(A_w)
    exp_w = V_w @ jnp.diag(jnp.exp(-alpha * dt * eig_w)) @ V_w.T
    sqrt_new = exp_w @ sqrt_fr
    cov_new = sqrt_new @ sqrt_new.T
    cov_new = 0.5 * (cov_new + cov_new.T)
    sqrt_new = jnp.linalg.cholesky(cov_new + 1e-12 * jnp.eye(D))

    return mean_new, cov_new, sqrt_new


_FLOW_DISPATCH = {
    "fisher-rao": _step_fisher_rao,
    "wasserstein": _step_wasserstein,
    "wasserstein-fb": _step_wasserstein_fb,
    "gradient-descent": _step_gradient_descent,
    "wfr": _step_wfr,
}


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive dt based on spectral norm
# ──────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('flow',))
def _adaptive_dt(cov, sqrt_cov, E_hess, dt_max, flow):
    """Bound dt to keep the covariance update stable."""
    D = cov.shape[0]
    prec = jnp.linalg.inv(cov)
    A = 0.5 * ((E_hess - prec) + (E_hess - prec).T)

    if flow == "fisher-rao":
        # Implicit update is unconditionally stable for dt in (0,1].
        # For dt > 1, bound by spectral radius of C*(H - prec).
        M = cov @ A
        spec = jnp.max(jnp.abs(jnp.linalg.eigvalsh(0.5 * (M + M.T))))
        return jnp.minimum(dt_max, jnp.minimum(1.0, 0.99 / jnp.maximum(spec, 1e-12)))
    elif flow in ("wasserstein", "wasserstein-fb"):
        # Exponential integrator is unconditionally stable, but large dt
        # can overshoot. Bound by spectral norm of A.
        spec = jnp.max(jnp.abs(jnp.linalg.eigvalsh(A)))
        return jnp.minimum(dt_max, 2.0 / jnp.maximum(spec, 1e-12))
    elif flow == "wfr":
        # Splitting: bound by both W and FR constraints
        spec_w = jnp.max(jnp.abs(jnp.linalg.eigvalsh(A)))
        dt_w = 2.0 / jnp.maximum(spec_w, 1e-12)
        M = cov @ A
        spec_fr = jnp.max(jnp.abs(jnp.linalg.eigvalsh(0.5 * (M + M.T))))
        dt_fr = jnp.minimum(1.0, 0.99 / jnp.maximum(spec_fr, 1e-12))
        return jnp.minimum(dt_max, jnp.minimum(dt_w, dt_fr))
    else:  # gradient-descent
        # Need 1 + 2*dt*min_eig(Q) > 0 where Q = L^{-1} G L^{-T}
        inv_sqrt = jnp.linalg.inv(sqrt_cov)
        G = 0.5 * (prec - E_hess)
        Q = inv_sqrt @ (0.5 * (G + G.T)) @ inv_sqrt.T
        min_eig = jnp.min(jnp.linalg.eigvalsh(Q))
        # If min_eig < 0, need dt < -1/(2*min_eig); else no constraint from PD
        dt_pd = jnp.where(min_eig < 0, -0.45 / min_eig, dt_max)
        spec = jnp.max(jnp.abs(jnp.linalg.eigvalsh(Q)))
        dt_speed = 0.9 / jnp.maximum(spec, 1e-12)
        return jnp.minimum(dt_max, jnp.minimum(dt_pd, dt_speed))


# ──────────────────────────────────────────────────────────────────────────────
# sampler_gvi  (main entry point)
# ──────────────────────────────────────────────────────────────────────────────

def sampler_gvi(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup       = 500,
    flow         = "fisher-rao",
    dt           = 0.5,
    alpha_ut     = 0.0,
    adaptive     = True,
    wfr_alpha    = 1.0,
    wfr_beta     = 1.0,
    thin_by      = 1,
    seed         = 0,
    verbose      = True,
):
    """
    Single-Gaussian Variational Inference under five gradient flows.

    Args:
        log_prob_fn   : (D,) -> scalar.  Log density (up to constant).
        initial_state : (n_chains, D).  Empirical mean/cov used for init.
        num_samples   : Number of samples to draw from the fitted Gaussian.
        warmup        : Number of VI optimisation iterations.
        flow          : "fisher-rao", "wasserstein", "wasserstein-fb", "wfr",
                        or "gradient-descent".
        dt            : Base learning rate / time step.
        alpha_ut      : Unscented transform spread (0 = min(sqrt(4/(D+kappa)), 1)).
        adaptive      : Adapt dt per step for stability (default True).
        wfr_alpha     : Transport weight for WFR flow (default 1.0).
        wfr_beta      : Reaction weight for WFR flow (default 1.0).
        thin_by       : (ignored, kept for interface consistency).
        seed          : Random seed.
        verbose       : Print progress.

    Returns:
        samples : (num_samples, 1, D)
        info    : dict with keys:
                    mean          : (D,) final mean
                    cov           : (D, D) final covariance
                    mean_history  : (n_iter+1, D)
                    cov_history   : (n_iter+1, D, D)
                    phi_history   : (n_iter,) E[Phi] per iteration
    """
    init = jnp.asarray(initial_state, dtype=jnp.float64)
    n_chains, D = init.shape

    assert flow in _FLOW_DISPATCH, f"Unknown flow '{flow}'. Choose from {list(_FLOW_DISPATCH)}"
    if flow == "wfr":
        step_fn = lambda m, c, sc, eg, eh, dt_: _step_wfr(m, c, sc, eg, eh, dt_,
                                                           alpha=wfr_alpha, beta=wfr_beta)
    else:
        step_fn = _FLOW_DISPATCH[flow]

    if alpha_ut <= 0.0:
        kappa = 0.0
        alpha_ut = min(jnp.sqrt(4.0 / (D + kappa)), 1.0)

    key = jax.random.key(seed)

    # --- initialise from empirical mean/cov ---
    mean = jnp.mean(init, axis=0)
    cov = jnp.cov(init.T) + 1e-6 * jnp.eye(D)

    # Potential
    def phi_fn(x):
        return -log_prob_fn(x)

    n_iter = warmup
    dt_max = dt

    # Store history for convergence plots
    mean_hist = [mean]
    cov_hist = [cov]
    phi_hist = []
    dt_hist = []

    # Initialise Cholesky factor
    sqrt_cov = jnp.linalg.cholesky(cov + 1e-10 * jnp.eye(D))

    for it in range(1, n_iter + 1):
        pts, w_m, w_c = _ut_sigma_points(mean, sqrt_cov, alpha_ut)

        E_phi, E_grad, E_hess = _compute_expectations(phi_fn, pts, w_m)

        if adaptive:
            dt_eff = _adaptive_dt(cov, sqrt_cov, E_hess, dt_max, flow)
        else:
            dt_eff = dt_max

        mean, cov, sqrt_cov = step_fn(mean, cov, sqrt_cov, E_grad, E_hess, dt_eff)

        mean_hist.append(mean)
        cov_hist.append(cov)
        phi_hist.append(float(E_phi))
        dt_hist.append(float(dt_eff))

        if verbose and it % max(1, n_iter // 10) == 0:
            print(f"  GVI({flow}) iter {it}/{n_iter}  dt={float(dt_eff):.4f}"
                  f"  E[Phi]={float(E_phi):.4f}")

    if verbose:
        print(f"GVI({flow}):  D={D}  iters={n_iter}")

    # --- draw samples ---
    key, k = jax.random.split(key)
    sqrt_cov = jnp.linalg.cholesky(cov + 1e-10 * jnp.eye(D))
    z = jax.random.normal(k, (num_samples, D))
    samples = mean[None, :] + z @ sqrt_cov.T
    samples = samples[:, None, :]

    info = dict(
        mean=mean,
        cov=cov,
        mean_history=jnp.stack(mean_hist),
        cov_history=jnp.stack(cov_hist),
        phi_history=jnp.array(phi_hist),
        dt_history=jnp.array(dt_hist),
    )
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# Comparison demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ══════════════════════════════════════════════════════════════════════
    # Reproduce the W-vs-FR crossover from arXiv:2504.20400
    #
    # Theory (Gaussian target with cov eigenvalues lambda_1 >= ... >= lambda_D):
    #   W rate per eigendirection i:  2*alpha / lambda_i
    #     => fast modes (small lambda_i) decay quickly, slow modes (large lambda_i) linger
    #     => initial rate ~ 2*alpha/lambda_min (fast), asymptotic ~ 2*alpha/lambda_max (slow)
    #   FR rate:  beta  (uniform across all directions)
    #   WFR optimal rate:  beta + 2*alpha*lambda_min(Gamma^{-1})
    #
    # Prediction:
    #   - W starts faster (rate 2*alpha/lambda_min >> beta when lambda_min small)
    #   - FR overtakes W when only the slow modes remain (rate beta > 2*alpha/lambda_max)
    #   - WFR interpolates: fast start from W + steady tail from FR
    #
    # Key setup: init cov >> target cov in all directions, so initial error
    # is dominated by well-conditioned directions where W is fast.
    # ══════════════════════════════════════════════════════════════════════

    dim = 10
    kappa = 1000.
    cov_eigvals = jnp.logspace(0, jnp.log10(kappa), dim)   # 1 to 1000
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    true_cov = Q @ jnp.diag(cov_eigvals) @ Q.T
    prec = Q @ jnp.diag(1. / cov_eigvals) @ Q.T

    def log_prob(x):
        return -0.5 * jnp.sum((x @ prec) * x)

    # Initialise far from target in ALL directions:
    # init cov ~ 10000*I, so error dominated by well-conditioned dirs (lambda=1)
    # where W is fastest (rate = 2/1 = 2 >> FR rate = 1).
    init = jax.random.normal(jax.random.key(42), (100, dim)) * 100.0
    n_samples = 100
    n_iter = 300

    print("=" * 70)
    print(f"W-vs-FR crossover analysis  D={dim}  kappa={kappa}")
    print("=" * 70)

    # --- Run W, FR, and WFR with various alpha/beta ---
    configs = [
        ("wasserstein",  "W2 (alpha=1,beta=0)", dict(flow="wasserstein")),
        ("fisher-rao",   "FR (alpha=0,beta=1)", dict(flow="fisher-rao")),
        ("wfr-balanced", "WFR (alpha=1,beta=1)", dict(flow="wfr", wfr_alpha=1.0, wfr_beta=1.0)),
        ("wfr-w-heavy",  "WFR (alpha=2,beta=0.5)", dict(flow="wfr", wfr_alpha=2.0, wfr_beta=0.5)),
        ("wfr-fr-heavy", "WFR (alpha=0.5,beta=2)", dict(flow="wfr", wfr_alpha=0.5, wfr_beta=2.0)),
    ]

    results = {}
    for key, label, kwargs in configs:
        s, info = sampler_gvi(
            log_prob, init, n_samples, warmup=n_iter,
            dt=0.5, seed=123, verbose=False, **kwargs)

        c_hist = info["cov_history"]   # (n_iter+1, D, D)

        # Total Frobenius error
        cov_errs = jnp.array([
            float(jnp.linalg.norm(c_hist[i] - true_cov) / jnp.linalg.norm(true_cov))
            for i in range(len(c_hist))
        ])

        # Per-eigendirection error:  project C(t) - C* onto eigenvectors of C*
        # error_i(t) = |v_i^T (C(t) - C*) v_i| / lambda_i
        per_eig_errs = []
        for i in range(len(c_hist)):
            diff = c_hist[i] - true_cov
            proj = jnp.diag(Q.T @ diff @ Q)          # project onto eigenbasis
            per_eig_errs.append(jnp.abs(proj) / cov_eigvals)
        per_eig_errs = jnp.stack(per_eig_errs)       # (n_iter+1, D)

        dt_h = info["dt_history"]
        cum_time = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dt_h)])

        results[key] = dict(
            label=label, cov_errs=cov_errs, per_eig_errs=per_eig_errs,
            phi_hist=info["phi_history"], dt_hist=dt_h, cum_time=cum_time,
        )
        print(f"  {label:30s}  final_cov_err={float(cov_errs[-1]):.6f}"
              f"  mean_dt={float(jnp.mean(dt_h)):.4f}"
              f"  cum_time={float(cum_time[-1]):.1f}")

    # ── Plotting ─────────────────────────────────────────────────────────
    colors = {
        "wasserstein": "C0", "fisher-rao": "C2",
        "wfr-balanced": "C4", "wfr-w-heavy": "C1", "wfr-fr-heavy": "C5",
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    main_keys = ["wasserstein", "fisher-rao", "wfr-balanced"]

    # (0,0) Total cov error vs ITERATION (discrete steps)
    ax = axes[0, 0]
    for key in main_keys:
        r = results[key]
        ax.semilogy(r["cov_errs"], label=r["label"], color=colors[key], linewidth=2)
    ax.set_xlabel("iteration (discrete steps)")
    ax.set_ylabel("||C - C*||_F / ||C*||_F")
    ax.set_title("Cov error vs discrete step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,1) Total cov error vs CUMULATIVE TIME (sum of dt_eff)
    ax = axes[0, 1]
    for key in main_keys:
        r = results[key]
        ax.semilogy(r["cum_time"], r["cov_errs"], label=r["label"],
                    color=colors[key], linewidth=2)
    ax.set_xlabel("cumulative time (sum dt_eff)")
    ax.set_ylabel("||C - C*||_F / ||C*||_F")
    ax.set_title("Cov error vs continuous time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Adaptive dt per iteration
    ax = axes[0, 2]
    for key in main_keys:
        r = results[key]
        ax.plot(r["dt_hist"], label=r["label"], color=colors[key],
                linewidth=1, alpha=0.7)
    ax.set_xlabel("iteration")
    ax.set_ylabel("dt_eff")
    ax.set_title("Adaptive step size per iteration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,3) Per-eigdir errors for WFR — combined rates
    ax = axes[0, 3]
    pe = results["wfr-balanced"]["per_eig_errs"]
    sel = [0, dim // 4, dim // 2, 3 * dim // 4, dim - 1]
    for i in sel:
        ax.semilogy(pe[:, i], label=f"lambda={float(cov_eigvals[i]):.0f}",
                    linewidth=1.5, alpha=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-eigdir relative error")
    ax.set_title("WFR(1,1): per-eigendirection")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,0) Per-eigdir errors for WASSERSTEIN — multirate
    ax = axes[1, 0]
    pe = results["wasserstein"]["per_eig_errs"]
    for i in sel:
        ax.semilogy(pe[:, i], label=f"lambda={float(cov_eigvals[i]):.0f}",
                    linewidth=1.5, alpha=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-eigdir relative error")
    ax.set_title("W2: per-eigendirection (multirate)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) Per-eigdir errors for FISHER-RAO — uniform rate
    ax = axes[1, 1]
    pe = results["fisher-rao"]["per_eig_errs"]
    for i in sel:
        ax.semilogy(pe[:, i], label=f"lambda={float(cov_eigvals[i]):.0f}",
                    linewidth=1.5, alpha=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-eigdir relative error")
    ax.set_title("FR: per-eigendirection (uniform rate)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,2) WFR alpha/beta sweep
    ax = axes[1, 2]
    for key in ["wasserstein", "fisher-rao", "wfr-balanced", "wfr-w-heavy", "wfr-fr-heavy"]:
        r = results[key]
        ax.semilogy(r["cum_time"], r["cov_errs"], label=r["label"],
                    color=colors[key], linewidth=1.5)
    ax.set_xlabel("cumulative time")
    ax.set_ylabel("||C - C*||_F / ||C*||_F")
    ax.set_title("WFR interpolation vs continuous time")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # (1,3) Theory vs measured rates per eigendirection
    ax = axes[1, 3]
    prec_eigvals = 1.0 / cov_eigvals
    w_rates = 2.0 * prec_eigvals
    fr_rate = 1.0
    wfr_rates = 2.0 * prec_eigvals + 1.0
    ax.semilogy(cov_eigvals, w_rates, 'o-', color=colors["wasserstein"],
                label="W2 theory: 2/lambda_i", markersize=5)
    ax.axhline(fr_rate, color=colors["fisher-rao"], linestyle="--",
               linewidth=2, label="FR theory: beta=1")
    ax.semilogy(cov_eigvals, wfr_rates, 's-', color=colors["wfr-balanced"],
                label="WFR theory: 2/lambda_i + 1", markersize=5)
    # Measured rates: use cumulative time for proper continuous-time rate
    for key, marker in [("wasserstein", "x"), ("fisher-rao", "+"), ("wfr-balanced", "D")]:
        pe = results[key]["per_eig_errs"]
        ct = results[key]["cum_time"]
        n = pe.shape[0]
        mid = n // 2
        measured = []
        for i in range(dim):
            vals = jnp.maximum(pe[mid:, i], 1e-16)
            # Rate in continuous time: -log(err_end/err_mid) / (t_end - t_mid)
            dt_total = float(ct[-1] - ct[mid])
            if dt_total > 0:
                slope = -(jnp.log(vals[-1]) - jnp.log(vals[0])) / dt_total
            else:
                slope = 0.0
            measured.append(float(slope))
        ax.scatter(cov_eigvals, jnp.maximum(jnp.array(measured), 1e-4),
                   marker=marker, s=40, color=colors[key], zorder=5,
                   label=f"{results[key]['label'][:3]} measured")
    ax.set_xlabel("cov eigenvalue lambda_i")
    ax.set_ylabel("convergence rate (continuous time)")
    ax.set_title("Theory vs measured rate")
    ax.set_xscale("log")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"W vs FR Crossover (D={dim}, kappa={int(kappa)}): "
                 "discrete step vs continuous time",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig("gvi_crossover.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to gvi_crossover.png")
