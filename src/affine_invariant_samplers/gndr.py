"""
sampler_gndr — Gauss-Newton Proposal Langevin with multi-stage Delayed Rejection.

Single-chain (batched) sampler with Gauss-Newton-preconditioned Langevin proposals:
  H_GN(x) = J_r(x)^T J_r(x)    (Gauss-Newton approximation, no 2nd derivatives)
  drift    = h * H_GN(x)^{-1} @ grad log pi(x)
  noise    = sqrt(2h) * L(x)^{-T} @ z,   L L^T = H_GN(x)

where r(x) is a residual such that log pi(x) ≈ -0.5 * ||r(x)||^2, and J_r = dr/dx.

Multi-stage DR: on rejection, retry with shrunk step size h * shrink^stage.
Full DR acceptance correction ensures detailed balance.

Adaptation (warmup only, toggleable):
  Heuristic initial step-size search   (find_init_step_size)
  Dual averaging → step size h         (adapt_step_size)

Reference: https://arxiv.org/abs/2505.02987
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial


# ──────────────────────────────────────────────────────────────────────────────
# Dual averaging
# ──────────────────────────────────────────────────────────────────────────────

class DAState(NamedTuple):
    iteration: int
    log_h: float
    log_h_bar: float
    H_bar: float

def _da_init(log_h0):
    return DAState(0, log_h0, log_h0, 0.)

def _da_update(state, accept_rate, log_h0, target, t0=10., gamma=0.05, kappa=0.75):
    it    = state.iteration + 1
    eta   = 1. / (it + t0)
    H_bar = (1. - eta) * state.H_bar + eta * (target - accept_rate)
    log_h = log_h0 - jnp.sqrt(it) / ((it + t0) * gamma) * H_bar
    log_hb = it**(-kappa) * log_h + (1. - it**(-kappa)) * state.log_h_bar
    return DAState(it, log_h, log_hb, H_bar)


# ──────────────────────────────────────────────────────────────────────────────
# Hessian regularization and Cholesky
# ──────────────────────────────────────────────────────────────────────────────

def _safe_cholesky(H, reg_small=1e-6, reg_large=1e-3):
    """Regularise and Cholesky-decompose H (per-chain batched)."""
    D = H.shape[-1]
    eye = jnp.eye(D)
    H_sym = 0.5 * (H + jnp.swapaxes(H, -2, -1)) + reg_small * eye
    L = jnp.linalg.cholesky(H_sym)
    # fallback: if any NaN, add larger regularization
    bad = jnp.any(jnp.isnan(L), axis=(-2, -1), keepdims=True)
    H_safe = H_sym + reg_large * eye
    L_safe = jnp.linalg.cholesky(H_safe)
    L = jnp.where(bad, L_safe, L)
    return L


# ──────────────────────────────────────────────────────────────────────────────
# Transition log-probability  q(y | x)
# ──────────────────────────────────────────────────────────────────────────────

def _transition_logp(x, y, grad_x, L_x, h):
    """
    log q(y | x) for Langevin proposal with Hessian preconditioning.
    drift = h * H^{-1} grad = h * L^{-T} L^{-1} grad
    Σ = 2h * H^{-1} = 2h * L^{-T} L^{-1}
    """
    # drift = h * solve(L L^T, grad) = h * L^{-T} L^{-1} grad
    v = jax.scipy.linalg.solve_triangular(L_x, grad_x, lower=True)  # L^{-1} grad
    drift = h * jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L_x, -2, -1), v, lower=False)                  # H^{-1} grad

    diff = y - x - drift
    # quadratic: diff^T (2h H^{-1})^{-1} diff = diff^T H diff / (2h)
    Lt_diff = jnp.einsum('...ij,...j->...i',
                         jnp.swapaxes(L_x, -2, -1), diff)            # L^T diff
    quad = jnp.sum(Lt_diff ** 2, axis=-1) / (2. * h)

    D = x.shape[-1]
    log_det = D * jnp.log(4. * jnp.pi * h) - 2. * jnp.sum(
        jnp.log(jnp.abs(jnp.diagonal(L_x, axis1=-2, axis2=-1))), axis=-1)

    return -0.5 * (quad + log_det)


# ──────────────────────────────────────────────────────────────────────────────
# Generate a Langevin proposal
# ──────────────────────────────────────────────────────────────────────────────

def _propose(x, grad_x, L_x, h, z):
    """
    x' = x + h * H^{-1} grad + sqrt(2h) * L^{-T} z
    """
    v = jax.scipy.linalg.solve_triangular(L_x, grad_x, lower=True)
    drift = h * jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L_x, -2, -1), v, lower=False)
    noise = jnp.sqrt(2. * h) * jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L_x, -2, -1), z, lower=False)
    return x + drift + noise


# ──────────────────────────────────────────────────────────────────────────────
# Multi-stage DR acceptance (unrolled for n_try ≤ 4)
# ──────────────────────────────────────────────────────────────────────────────

def _dr_accept_stage1(lp_x, lp_y1, q_xy1, q_y1x):
    """Standard MH log-acceptance for stage 1."""
    return jnp.minimum(0., lp_y1 - lp_x + q_y1x - q_xy1)


def _dr_accept_stage2(lp_x, lp_y2, q_xy2, q_y2x, q_xy1, q_y2y1, q_y1x, q_y1y2,
                      alpha1_x_y1, alpha1_y2_y1):
    """DR log-acceptance for stage 2, with stage-1 rejection correction."""
    log_num  = lp_y2 + q_y2x + q_y2y1
    log_den  = lp_x  + q_xy2 + q_xy1

    # (1 - alpha1) terms
    safe_a1_x  = jnp.where(alpha1_x_y1 > -1e-12, -jnp.inf,
                            jnp.log1p(-jnp.exp(alpha1_x_y1)))
    safe_a1_y2 = jnp.where(alpha1_y2_y1 > -1e-12, -jnp.inf,
                            jnp.log1p(-jnp.exp(alpha1_y2_y1)))

    la = log_num - log_den + safe_a1_y2 - safe_a1_x
    return jnp.where(jnp.isfinite(la), jnp.minimum(0., la), -jnp.inf)


def _dr_accept_stage3(lp_x, lp_y3,
                      q_xy3, q_y3x, q_xy1, q_y3y1, q_xy2, q_y3y2,
                      alpha1_x_y1, alpha1_y3_y1,
                      alpha2_x_y2, alpha2_y3_y2):
    """DR log-acceptance for stage 3."""
    log_num  = lp_y3 + q_y3x + q_y3y1 + q_y3y2
    log_den  = lp_x  + q_xy3 + q_xy1  + q_xy2

    def safe_log1m(a):
        return jnp.where(a > -1e-12, -jnp.inf, jnp.log1p(-jnp.exp(a)))

    la = (log_num - log_den
          + safe_log1m(alpha1_y3_y1) - safe_log1m(alpha1_x_y1)
          + safe_log1m(alpha2_y3_y2) - safe_log1m(alpha2_x_y2))
    return jnp.where(jnp.isfinite(la), jnp.minimum(0., la), -jnp.inf)


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (stage-1 accept ≈ target_accept at init positions)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, x, lp, grad_x, L_x, eps0, v_lp, target_accept):
    fi = jnp.finfo(jnp.result_type(eps0))
    N, D = x.shape

    def body(s):
        eps, _, d, rk = s
        rk, kz, ku = jax.random.split(rk, 3)
        eps = (2.**d) * eps
        z = jax.random.normal(kz, (N, D))
        prop = _propose(x, grad_x, L_x, eps, z)
        lp_prop = v_lp(prop)
        q_x_y = _transition_logp(x, prop, grad_x, L_x, eps)
        # reverse q needs grad at proposal; use current L_x as cheap proxy
        # (sign-correct; the search doesn't need exact detailed balance).
        q_y_x = _transition_logp(prop, x, grad_x, L_x, eps)
        la = jnp.minimum(0., lp_prop - lp + q_y_x - q_x_y)
        u = jnp.log(jax.random.uniform(ku, (N,), minval=1e-10))
        acc = u < la
        avg = jnp.mean(acc.astype(float))
        return eps, d, jnp.where(avg > target_accept, 1, -1), rk

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps


# ──────────────────────────────────────────────────────────────────────────────
# sampler_gndr
# ──────────────────────────────────────────────────────────────────────────────

def sampler_gndr(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    step_size     = 0.1,
    n_try         = 3,
    shrink        = 0.5,
    target_accept = 0.574,
    thin_by       = 1,
    residual_fn   = None,
    hessian_fn    = None,
    grad_fn       = None,
    seed          = 0,
    verbose       = True,
    find_init_step_size = True,
    adapt_step_size     = True,
):
    """
    Gauss-Newton Proposal Langevin with multi-stage Delayed Rejection.

    The proposal Hessian uses the Gauss-Newton approximation:
        H_GN(x) = J_r(x)^T J_r(x)
    where r(x) is a residual vector such that log pi(x) ≈ -0.5 * ||r(x)||^2,
    and J_r = dr/dx is the Jacobian.  This avoids second derivatives of r.

    Args:
        log_prob_fn   : (D,) -> scalar.  Single-point log density (auto-batched).
        initial_state : (n_chains, D).
        num_samples   : Post-warmup samples to return.
        warmup        : Warmup iterations for step-size adaptation.
        step_size     : Initial step size h.
        n_try         : Number of DR stages (1-3, default 3).
        shrink        : Shrink factor per stage (default 0.5).
        target_accept : Target stage-1 acceptance for DA.
        thin_by       : Thinning factor.
        residual_fn   : (D,) -> (N_res,).  Residual vector r(x) for GN Hessian.
                        H_GN(x) = J_r(x)^T J_r(x) where J_r = dr/dx.
        hessian_fn    : (D,) -> (D,D).  Override: explicit GN Hessian.
                        If None and residual_fn is given, auto-derived.
                        If both None, falls back to -jax.hessian(log_prob).
        grad_fn       : (D,) -> (D,).   If None, auto-derived via jax.grad.
        seed          : Random seed.
        verbose       : Print progress.
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` so that
                              stage-1 acceptance ≈ `target_accept`.
                              If False, use `step_size` as-is.
        adapt_step_size : If True (default), dual-averaging adapts step size during warmup.
                          If False, uses `step_size` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, stage_rates, final_step_size)
    """
    assert 1 <= n_try <= 3, "n_try must be 1, 2, or 3"
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape

    # auto-derive gradient
    if grad_fn is None:
        grad_fn = jax.grad(log_prob_fn)

    # build Hessian function (Gauss-Newton preferred)
    if hessian_fn is None:
        if residual_fn is not None:
            # Gauss-Newton: H_GN(x) = J_r(x)^T @ J_r(x)
            def hessian_fn(x):
                J = jax.jacobian(residual_fn)(x)    # (N_res, D)
                return J.T @ J                       # (D, D)
        else:
            # fallback: full Hessian (negated to get precision)
            _full_hess = jax.hessian(log_prob_fn)
            def hessian_fn(x):
                return -_full_hess(x)

    # vectorise: (n_chains, D) -> (n_chains,) / (n_chains, D) / (n_chains, D, D)
    v_lp   = jax.vmap(log_prob_fn)
    v_grad = jax.vmap(grad_fn)
    v_hess = jax.vmap(hessian_fn)

    # step sizes per stage
    shrink_vec = jnp.array([shrink ** s for s in range(n_try)])  # [1, shrink, shrink^2, ...]

    key = jax.random.key(seed)
    step_size = jnp.asarray(step_size, jnp.float32)

    x = state
    lp = v_lp(x)

    if find_init_step_size:
        _user_h = float(step_size)
        key, k_ = jax.random.split(key)
        grad_x0 = v_grad(x)
        L_x0 = _safe_cholesky(v_hess(x))
        step_size = _find_init_eps(k_, x, lp, grad_x0, L_x0, step_size, v_lp,
                                    target_accept)
        if verbose:
            print(f"[gndr] find_init_step_size: step_size {_user_h:.4g} → "
                  f"{float(step_size):.4g}\n"
                  f"   (if the chain later stalls, set find_init_step_size=False "
                  f"and pass your own step_size — the heuristic can overshoot "
                  f"on targets the Gauss–Newton approximation misjudges.)")

    log_h0 = jnp.log(step_size)
    da = _da_init(log_h0)

    # --- warmup ---
    @jax.jit
    def _warmup_step(x, lp, da, rng):
        h_base = jnp.exp(da.log_h) if adapt_step_size else step_size
        N = x.shape[0]
        D = x.shape[1]

        grad_x = v_grad(x)                                     # (N, D)
        H_x = v_hess(x)                                        # (N, D, D)  GN precision
        L_x = _safe_cholesky(H_x)                              # (N, D, D)

        # generate noise for all stages at once
        rng, *stage_keys = jax.random.split(rng, n_try + 1)
        zs = jnp.stack([jax.random.normal(sk, (N, D)) for sk in stage_keys])  # (n_try, N, D)

        # proposals for each stage
        hs = h_base * shrink_vec                                # (n_try,)
        props = jnp.stack([_propose(x, grad_x, L_x, hs[s], zs[s]) for s in range(n_try)])
        lps   = jnp.stack([v_lp(props[s]) for s in range(n_try)])
        grads = jnp.stack([v_grad(props[s]) for s in range(n_try)])
        Hs    = jnp.stack([v_hess(props[s]) for s in range(n_try)])
        Ls    = jnp.stack([_safe_cholesky(Hs[s]) for s in range(n_try)])

        # Stage 1
        q_x_y1  = _transition_logp(x, props[0], grad_x, L_x, hs[0])
        q_y1_x  = _transition_logp(props[0], x, grads[0], Ls[0], hs[0])
        a1_x_y1 = _dr_accept_stage1(lp, lps[0], q_x_y1, q_y1_x)
        rng, uk = jax.random.split(rng)
        u = jnp.log(jax.random.uniform(uk, (N,), minval=1e-10))
        acc1 = u < a1_x_y1

        if n_try >= 2:
            # Stage 2 (only matters for !acc1, but computed for all)
            q_x_y2  = _transition_logp(x, props[1], grad_x, L_x, hs[1])
            q_y2_x  = _transition_logp(props[1], x, grads[1], Ls[1], hs[1])
            q_y2_y1 = _transition_logp(props[1], props[0], grads[1], Ls[1], hs[0])
            q_x_y1_s0 = q_x_y1  # same kernel 0
            a1_y2_y1 = _dr_accept_stage1(lps[1], lps[0],
                         _transition_logp(props[1], props[0], grads[1], Ls[1], hs[0]),
                         _transition_logp(props[0], props[1], grads[0], Ls[0], hs[0]))

            a2_x_y2 = _dr_accept_stage2(
                lp, lps[1], q_x_y2, q_y2_x, q_x_y1_s0, q_y2_y1,
                q_y1_x, _transition_logp(props[0], props[1], grads[0], Ls[0], hs[1]),
                a1_x_y1, a1_y2_y1)
            rng, uk2 = jax.random.split(rng)
            u2 = jnp.log(jax.random.uniform(uk2, (N,), minval=1e-10))
            acc2 = (~acc1) & (u2 < a2_x_y2)
        else:
            acc2 = jnp.zeros(N, dtype=bool)

        if n_try >= 3:
            # Stage 3
            q_x_y3  = _transition_logp(x, props[2], grad_x, L_x, hs[2])
            q_y3_x  = _transition_logp(props[2], x, grads[2], Ls[2], hs[2])
            q_y3_y1 = _transition_logp(props[2], props[0], grads[2], Ls[2], hs[0])
            q_y3_y2 = _transition_logp(props[2], props[1], grads[2], Ls[2], hs[1])

            a1_y3_y1 = _dr_accept_stage1(lps[2], lps[0],
                         _transition_logp(props[2], props[0], grads[2], Ls[2], hs[0]),
                         _transition_logp(props[0], props[2], grads[0], Ls[0], hs[0]))

            # a2(y3, y2): DR stage-2 acceptance from y3 to y2 given y1 was rejected
            q_y3_y2_s1 = _transition_logp(props[2], props[1], grads[2], Ls[2], hs[1])
            q_y2_y3_s1 = _transition_logp(props[1], props[2], grads[1], Ls[1], hs[1])
            a1_y3_y1_for_s2 = a1_y3_y1
            a1_y2_y1_for_s2 = _dr_accept_stage1(lps[1], lps[0],
                _transition_logp(props[1], props[0], grads[1], Ls[1], hs[0]),
                _transition_logp(props[0], props[1], grads[0], Ls[0], hs[0]))

            a2_y3_y2 = _dr_accept_stage2(
                lps[2], lps[1], q_y3_y2_s1, q_y2_y3_s1,
                _transition_logp(props[2], props[0], grads[2], Ls[2], hs[0]),
                _transition_logp(props[1], props[0], grads[1], Ls[1], hs[0]),
                _transition_logp(props[0], props[2], grads[0], Ls[0], hs[0]),
                _transition_logp(props[0], props[1], grads[0], Ls[0], hs[1]),
                a1_y3_y1_for_s2, a1_y2_y1_for_s2)

            a3 = _dr_accept_stage3(
                lp, lps[2], q_x_y3, q_y3_x,
                q_x_y1, q_y3_y1, q_x_y2, q_y3_y2,
                a1_x_y1, a1_y3_y1, a2_x_y2, a2_y3_y2)
            rng, uk3 = jax.random.split(rng)
            u3 = jnp.log(jax.random.uniform(uk3, (N,), minval=1e-10))
            acc3 = (~acc1) & (~acc2) & (u3 < a3)
        else:
            acc3 = jnp.zeros(N, dtype=bool)

        # apply accepted proposals
        new_x = x
        new_lp = lp
        for s, (acc_s, p_s, lp_s) in enumerate(
            [(acc1, props[0], lps[0]),
             (acc2, props[1] if n_try >= 2 else x, lps[1] if n_try >= 2 else lp),
             (acc3, props[2] if n_try >= 3 else x, lps[2] if n_try >= 3 else lp)]):
            if s < n_try:
                new_x  = jnp.where(acc_s[:, None], p_s, new_x)
                new_lp = jnp.where(acc_s, lp_s, new_lp)

        s1_rate = jnp.mean(acc1.astype(float))
        overall = jnp.mean((acc1 | acc2 | acc3).astype(float))

        if adapt_step_size:
            da_new = _da_update(da, s1_rate, log_h0, target_accept)
        else:
            da_new = da
        return new_x, new_lp, da_new, overall, rng

    key, k_ = jax.random.split(key)
    total_acc = 0.
    rng = k_
    for i in range(warmup):
        x, lp, da, rate, rng = _warmup_step(x, lp, da, rng)
        total_acc += rate

    final_h = jnp.exp(da.log_h_bar) if adapt_step_size else step_size
    if verbose:
        print(f"GN-DR:  n_try={n_try}  h={float(final_h):.4f}"
              f"  warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    # --- production ---
    final_hs = final_h * shrink_vec

    @jax.jit
    def _step(carry, rng):
        x, lp = carry
        N, D = x.shape

        grad_x = v_grad(x)
        H_x = v_hess(x)                                        # (N, D, D)  GN precision
        L_x = _safe_cholesky(H_x)

        rng, *stage_keys = jax.random.split(rng, n_try + 1)
        zs = jnp.stack([jax.random.normal(sk, (N, D)) for sk in stage_keys])

        props = jnp.stack([_propose(x, grad_x, L_x, final_hs[s], zs[s]) for s in range(n_try)])
        lps   = jnp.stack([v_lp(props[s]) for s in range(n_try)])
        grads = jnp.stack([v_grad(props[s]) for s in range(n_try)])
        Hs    = jnp.stack([v_hess(props[s]) for s in range(n_try)])
        Ls    = jnp.stack([_safe_cholesky(Hs[s]) for s in range(n_try)])

        # Stage 1
        q_x_y1 = _transition_logp(x, props[0], grad_x, L_x, final_hs[0])
        q_y1_x = _transition_logp(props[0], x, grads[0], Ls[0], final_hs[0])
        a1_x_y1 = _dr_accept_stage1(lp, lps[0], q_x_y1, q_y1_x)
        rng, uk = jax.random.split(rng)
        u = jnp.log(jax.random.uniform(uk, (N,), minval=1e-10))
        acc1 = u < a1_x_y1

        if n_try >= 2:
            q_x_y2  = _transition_logp(x, props[1], grad_x, L_x, final_hs[1])
            q_y2_x  = _transition_logp(props[1], x, grads[1], Ls[1], final_hs[1])
            q_y2_y1 = _transition_logp(props[1], props[0], grads[1], Ls[1], final_hs[0])
            a1_y2_y1 = _dr_accept_stage1(lps[1], lps[0],
                _transition_logp(props[1], props[0], grads[1], Ls[1], final_hs[0]),
                _transition_logp(props[0], props[1], grads[0], Ls[0], final_hs[0]))
            a2_x_y2 = _dr_accept_stage2(
                lp, lps[1], q_x_y2, q_y2_x, q_x_y1, q_y2_y1,
                q_y1_x, _transition_logp(props[0], props[1], grads[0], Ls[0], final_hs[1]),
                a1_x_y1, a1_y2_y1)
            rng, uk2 = jax.random.split(rng)
            u2 = jnp.log(jax.random.uniform(uk2, (N,), minval=1e-10))
            acc2 = (~acc1) & (u2 < a2_x_y2)
        else:
            acc2 = jnp.zeros(N, dtype=bool)

        if n_try >= 3:
            q_x_y3 = _transition_logp(x, props[2], grad_x, L_x, final_hs[2])
            q_y3_x = _transition_logp(props[2], x, grads[2], Ls[2], final_hs[2])
            q_y3_y1 = _transition_logp(props[2], props[0], grads[2], Ls[2], final_hs[0])
            q_y3_y2 = _transition_logp(props[2], props[1], grads[2], Ls[2], final_hs[1])

            a1_y3_y1 = _dr_accept_stage1(lps[2], lps[0],
                _transition_logp(props[2], props[0], grads[2], Ls[2], final_hs[0]),
                _transition_logp(props[0], props[2], grads[0], Ls[0], final_hs[0]))
            a1_y2_y1_s2 = _dr_accept_stage1(lps[1], lps[0],
                _transition_logp(props[1], props[0], grads[1], Ls[1], final_hs[0]),
                _transition_logp(props[0], props[1], grads[0], Ls[0], final_hs[0]))
            a2_y3_y2 = _dr_accept_stage2(
                lps[2], lps[1],
                _transition_logp(props[2], props[1], grads[2], Ls[2], final_hs[1]),
                _transition_logp(props[1], props[2], grads[1], Ls[1], final_hs[1]),
                _transition_logp(props[2], props[0], grads[2], Ls[2], final_hs[0]),
                _transition_logp(props[1], props[0], grads[1], Ls[1], final_hs[0]),
                _transition_logp(props[0], props[2], grads[0], Ls[0], final_hs[0]),
                _transition_logp(props[0], props[1], grads[0], Ls[0], final_hs[1]),
                a1_y3_y1, a1_y2_y1_s2)
            a3 = _dr_accept_stage3(
                lp, lps[2], q_x_y3, q_y3_x,
                q_x_y1, q_y3_y1, q_x_y2, q_y3_y2,
                a1_x_y1, a1_y3_y1, a2_x_y2, a2_y3_y2)
            rng, uk3 = jax.random.split(rng)
            u3 = jnp.log(jax.random.uniform(uk3, (N,), minval=1e-10))
            acc3 = (~acc1) & (~acc2) & (u3 < a3)
        else:
            acc3 = jnp.zeros(N, dtype=bool)

        new_x, new_lp = x, lp
        for s in range(n_try):
            acc_s = [acc1, acc2, acc3][s]
            new_x  = jnp.where(acc_s[:, None], props[s], new_x)
            new_lp = jnp.where(acc_s, lps[s], new_lp)

        acc_all = (acc1 | acc2 | acc3).astype(float)
        s1 = acc1.astype(float)
        return (new_x, new_lp), (new_x, acc_all, s1)

    key, k = jax.random.split(key)
    skeys = jax.random.split(k, num_samples * thin_by)

    (x, lp), (all_states, all_acc, all_s1) = jax.lax.scan(
        _step, (x, lp), skeys)

    samples = all_states[::thin_by][:num_samples]

    # Per DR iteration per walker: 1 grad at the current position  x
    # (for all stage proposals, shared) + n_try grads at the proposals.
    # Hessians are separate and not counted.
    n_grad_evals = (int(num_samples * thin_by)
                     * (int(n_try) + 1)
                     * int(n_chains))
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                stage1_rate=float(jnp.mean(all_s1)),
                final_step_size=float(final_h),
                n_grad_evals=n_grad_evals)
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}"
              f"  stage1={info['stage1_rate']:.3f}")
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim = 2
    cov = jnp.array([[1., .95], [.95, 1.]])
    prec = jnp.linalg.inv(cov)
    L_prec = jnp.linalg.cholesky(prec)

    # log prob = -0.5 x^T prec x = -0.5 ||L^T x||^2
    # residual r(x) = L^T x  =>  H_GN = J^T J = L L^T = prec
    def log_prob(x):
        return -0.5 * jnp.sum((x @ prec) * x)
    def residual(x):
        return L_prec.T @ x

    init = jax.random.normal(jax.random.key(42), (10, dim))
    samples, info = sampler_gndr(log_prob, init, num_samples=5000, warmup=1000,
                                  step_size=0.5, n_try=3,
                                  residual_fn=residual, seed=123, find_init_step_size = False)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
