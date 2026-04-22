"""
sampler_kalman_dr — Ensemble Kalman Move with multi-stage Delayed Rejection.

Ensemble-preconditioned Langevin proposal using parameter-space and data-space
ensemble statistics (derivative-free — no gradients of log_prob needed):
  d(x)    = -B_param @ B_data^T @ M @ G(x)       (h-independent drift direction)
  drift   = h * d(x)
  noise   = sqrt(2h) * L_C @ z,   L_C L_C^T = C_reg = B_param^T B_param + eps*I

where B_param, B_data are centered subsets of the complement ensemble in
parameter/data space, M is the data-space precision, and G is the forward model.

On rejection, retry with shrunk step h * shrink^stage, up to n_try stages.
Full DR acceptance correction (Green-Mira) ensures detailed balance.

Adaptation (warmup only, toggleable):
  Heuristic initial step-size search   (find_init_step_size)
  Dual averaging → step size h         (adapt_step_size)

"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


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
# Safe Cholesky
# ──────────────────────────────────────────────────────────────────────────────

def _safe_cholesky(H, reg_small=1e-6, reg_large=1e-3):
    D = H.shape[-1]
    eye = jnp.eye(D)
    H_sym = 0.5 * (H + jnp.swapaxes(H, -2, -1)) + reg_small * eye
    L = jnp.linalg.cholesky(H_sym)
    bad = jnp.any(jnp.isnan(L), axis=(-2, -1), keepdims=True)
    L_safe = jnp.linalg.cholesky(H_sym + reg_large * eye)
    return jnp.where(bad, L_safe, L)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble basis:  B_param, B_data, L_C  (from complement subset)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_basis(W, complement, key, k, forward_fn):
    """
    Select a random k-subset of the complement for each of W active walkers.
    Compute centered B_param, B_data, and Cholesky L_C of C_reg = B^T B + eps*I.

    Returns:
        B_param : (W, k, D)
        B_data  : (W, k, D_data)
        L_C     : (W, D, D)     Cholesky of C_reg
    """
    Wc, D = complement.shape
    idx = jnp.arange(Wc)
    subsets = jax.vmap(lambda rk: jax.random.choice(rk, idx, (k,), replace=False))(
        jax.random.split(key, W))                                    # (W, k)
    selected = complement[subsets]                                    # (W, k, D)
    means = jnp.mean(selected, axis=1, keepdims=True)
    B_param = (selected - means) / jnp.sqrt(k)                       # (W, k, D)

    sel_flat = selected.reshape(-1, D)
    F_flat = forward_fn(sel_flat)
    D_data = F_flat.shape[-1]
    F_sel = F_flat.reshape(W, k, D_data)
    F_means = jnp.mean(F_sel, axis=1, keepdims=True)
    B_data = (F_sel - F_means) / jnp.sqrt(k)                         # (W, k, D_data)

    C = jnp.einsum('wkd,wke->wde', B_param, B_param)                 # (W, D, D)
    L_C = _safe_cholesky(C)
    return B_param, B_data, L_C


# ──────────────────────────────────────────────────────────────────────────────
# Drift direction (h-independent)
# ──────────────────────────────────────────────────────────────────────────────

def _drift_direction(positions, B_param, B_data, M, forward_fn):
    """
    d(x) = -B_param @ B_data^T @ M @ G(x)      (W, D)
    """
    G = forward_fn(positions)                                         # (W, D_data)
    MG = G @ M                                                        # (W, D_data)
    bt_MG = jnp.einsum('wkd,wd->wk', B_data, MG)                    # (W, k)
    return -jnp.einsum('wkd,wk->wd', B_param, bt_MG)                 # (W, D)


# ──────────────────────────────────────────────────────────────────────────────
# Transition log-probability  q(y | x)
# ──────────────────────────────────────────────────────────────────────────────

def _transition_logp(x, y, d_x, L_C, h):
    """
    log q(y | x) for the Kalman proposal.

    Proposal:  y = x + h*d_x + sqrt(2h)*L_C @ z,  z ~ N(0, I)
    Covariance Sigma = 2h * C_reg,  L_C L_C^T = C_reg

    log q = -D/2 log(4 pi h) - sum log diag(L_C) - ||L_C^{-1}(y - x - h d_x)||^2 / (4h)
    """
    D = x.shape[-1]
    diff = y - x - h * d_x                                           # (..., D)
    v = jax.scipy.linalg.solve_triangular(L_C, diff, lower=True)     # L^{-1} diff
    quad = jnp.sum(v ** 2, axis=-1) / (4. * h)

    log_det_LC = jnp.sum(jnp.log(jnp.abs(
        jnp.diagonal(L_C, axis1=-2, axis2=-1))), axis=-1)
    log_norm = 0.5 * D * jnp.log(4. * jnp.pi * h) + log_det_LC

    return -quad - log_norm


# ──────────────────────────────────────────────────────────────────────────────
# Generate a Kalman proposal
# ──────────────────────────────────────────────────────────────────────────────

def _propose(x, d_x, L_C, h, z):
    """x' = x + h * d(x) + sqrt(2h) * L_C @ z"""
    noise = jnp.sqrt(2. * h) * jnp.einsum('...ij,...j->...i', L_C, z)
    return x + h * d_x + noise


# ──────────────────────────────────────────────────────────────────────────────
# Multi-stage DR acceptance (unrolled for n_try <= 3)
# ──────────────────────────────────────────────────────────────────────────────

def _dr_accept_stage1(lp_x, lp_y1, q_xy1, q_y1x):
    """Standard MH log-acceptance for stage 1."""
    return jnp.minimum(0., lp_y1 - lp_x + q_y1x - q_xy1)


def _dr_accept_stage2(lp_x, lp_y2, q_xy2, q_y2x, q_xy1, q_y2y1,
                      alpha1_x_y1, alpha1_y2_y1):
    """DR log-acceptance for stage 2, with stage-1 rejection correction."""
    log_num = lp_y2 + q_y2x + q_y2y1
    log_den = lp_x  + q_xy2 + q_xy1

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
    log_num = lp_y3 + q_y3x + q_y3y1 + q_y3y2
    log_den = lp_x  + q_xy3 + q_xy1  + q_xy2

    def safe_log1m(a):
        return jnp.where(a > -1e-12, -jnp.inf, jnp.log1p(-jnp.exp(a)))

    la = (log_num - log_den
          + safe_log1m(alpha1_y3_y1) - safe_log1m(alpha1_x_y1)
          + safe_log1m(alpha2_y3_y2) - safe_log1m(alpha2_x_y2))
    return jnp.where(jnp.isfinite(la), jnp.minimum(0., la), -jnp.inf)


# ──────────────────────────────────────────────────────────────────────────────
# Full DR step for one group
# ──────────────────────────────────────────────────────────────────────────────

def _dr_group_step(active, complement, lp_active, key, h_base, shrink_vec, n_try,
                   forward_fn, M, log_prob_fn, k):
    """
    One DR step for the active group using complement for ensemble basis.

    Returns: (new_active, new_lp, acc1, acc_any)
    """
    W, D = active.shape

    # 1) compute ensemble basis (shared across all DR stages)
    key, kb = jax.random.split(key)
    B_param, B_data, L_C = _compute_basis(W, complement, kb, k, forward_fn)

    # 2) step sizes per stage
    hs = h_base * shrink_vec                                          # (n_try,)

    # 3) drift direction at current positions (h-independent)
    d_x = _drift_direction(active, B_param, B_data, M, forward_fn)   # (W, D)

    # 4) generate proposals for all stages
    key, *zkeys = jax.random.split(key, n_try + 1)
    zs = jnp.stack([jax.random.normal(zk, (W, D)) for zk in zkeys])  # (n_try, W, D)
    props = jnp.stack([_propose(active, d_x, L_C, hs[s], zs[s])
                       for s in range(n_try)])                        # (n_try, W, D)

    # 5) evaluate log probs and drift directions at proposals
    lps = jnp.stack([log_prob_fn(props[s]) for s in range(n_try)])    # (n_try, W)
    ds  = jnp.stack([_drift_direction(props[s], B_param, B_data, M, forward_fn)
                     for s in range(n_try)])                          # (n_try, W, D)

    # ── Stage 1 ──
    q_x_y0  = _transition_logp(active, props[0], d_x, L_C, hs[0])
    q_y0_x  = _transition_logp(props[0], active, ds[0], L_C, hs[0])
    a1_x_y0 = _dr_accept_stage1(lp_active, lps[0], q_x_y0, q_y0_x)
    key, uk = jax.random.split(key)
    u = jnp.log(jax.random.uniform(uk, (W,), minval=1e-10))
    acc1 = u < a1_x_y0

    if n_try >= 2:
        # ── Stage 2 ──
        q_x_y1  = _transition_logp(active, props[1], d_x, L_C, hs[1])
        q_y1_x  = _transition_logp(props[1], active, ds[1], L_C, hs[1])
        # cross-terms using stage-0 kernel (h0)
        q_y1_y0 = _transition_logp(props[1], props[0], ds[1], L_C, hs[0])  # q_0(y0|y1)
        # alpha_1(y1, y0): stage-1 acceptance from y1 to y0
        a1_y1_y0 = _dr_accept_stage1(
            lps[1], lps[0],
            _transition_logp(props[1], props[0], ds[1], L_C, hs[0]),
            _transition_logp(props[0], props[1], ds[0], L_C, hs[0]))

        a2_x_y1 = _dr_accept_stage2(
            lp_active, lps[1], q_x_y1, q_y1_x, q_x_y0, q_y1_y0,
            a1_x_y0, a1_y1_y0)

        key, uk2 = jax.random.split(key)
        u2 = jnp.log(jax.random.uniform(uk2, (W,), minval=1e-10))
        acc2 = (~acc1) & (u2 < a2_x_y1)
    else:
        acc2 = jnp.zeros(W, dtype=bool)
        a2_x_y1 = jnp.full(W, -jnp.inf)

    if n_try >= 3:
        # ── Stage 3 ──
        q_x_y2  = _transition_logp(active, props[2], d_x, L_C, hs[2])
        q_y2_x  = _transition_logp(props[2], active, ds[2], L_C, hs[2])
        # cross-terms
        q_y2_y0 = _transition_logp(props[2], props[0], ds[2], L_C, hs[0])  # q_0(y0|y2)
        q_y2_y1 = _transition_logp(props[2], props[1], ds[2], L_C, hs[1])  # q_1(y1|y2)

        # alpha_1(y2, y0)
        a1_y2_y0 = _dr_accept_stage1(
            lps[2], lps[0],
            _transition_logp(props[2], props[0], ds[2], L_C, hs[0]),
            _transition_logp(props[0], props[2], ds[0], L_C, hs[0]))

        # alpha_2(y2, y1): DR stage-2 from y2 to y1
        a1_y1_y0_s2 = _dr_accept_stage1(
            lps[1], lps[0],
            _transition_logp(props[1], props[0], ds[1], L_C, hs[0]),
            _transition_logp(props[0], props[1], ds[0], L_C, hs[0]))

        a2_y2_y1 = _dr_accept_stage2(
            lps[2], lps[1],
            _transition_logp(props[2], props[1], ds[2], L_C, hs[1]),
            _transition_logp(props[1], props[2], ds[1], L_C, hs[1]),
            _transition_logp(props[2], props[0], ds[2], L_C, hs[0]),
            _transition_logp(props[1], props[0], ds[1], L_C, hs[0]),
            a1_y2_y0, a1_y1_y0_s2)

        a3 = _dr_accept_stage3(
            lp_active, lps[2], q_x_y2, q_y2_x,
            q_x_y0, q_y2_y0, q_x_y1, q_y2_y1,
            a1_x_y0, a1_y2_y0, a2_x_y1, a2_y2_y1)

        key, uk3 = jax.random.split(key)
        u3 = jnp.log(jax.random.uniform(uk3, (W,), minval=1e-10))
        acc3 = (~acc1) & (~acc2) & (u3 < a3)
    else:
        acc3 = jnp.zeros(W, dtype=bool)

    # apply accepted proposals
    new_x  = active
    new_lp = lp_active
    for s in range(n_try):
        acc_s = [acc1, acc2, acc3][s]
        new_x  = jnp.where(acc_s[:, None], props[s], new_x)
        new_lp = jnp.where(acc_s, lps[s], new_lp)

    acc_any = acc1 | acc2 | acc3
    return new_x, new_lp, acc1, acc_any, key


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (stage-1 accept ≈ target_accept at init positions)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, g1, g2, lp1, lp2, eps0, forward_fn, M, log_prob_fn,
                   W, k, n_try, target_accept):
    fi = jnp.finfo(jnp.result_type(eps0))
    shrink_one = jnp.ones(n_try)  # all stages use the same eps during search

    def body(s):
        eps, _, d, rk = s
        rk, k1, k2 = jax.random.split(rk, 3)
        eps = (2.**d) * eps
        _, _, a1, _, _ = _dr_group_step(g1, g2, lp1, k1, eps, shrink_one, n_try,
                                        forward_fn, M, log_prob_fn, k)
        _, _, a2, _, _ = _dr_group_step(g2, g1, lp2, k2, eps, shrink_one, n_try,
                                        forward_fn, M, log_prob_fn, k)
        avg = 0.5 * (jnp.mean(a1.astype(float)) + jnp.mean(a2.astype(float)))
        return eps, d, jnp.where(avg > target_accept, 1, -1), rk

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps


# ──────────────────────────────────────────────────────────────────────────────
# sampler_kalman_dr
# ──────────────────────────────────────────────────────────────────────────────

def sampler_kalman_dr(
    log_prob_fn,
    forward_fn,
    M,
    initial_state,
    num_samples,
    warmup        = 1000,
    step_size     = 0.05,
    n_try         = 3,
    shrink        = 0.5,
    subset_size   = None,
    target_accept = 0.574,
    thin_by       = 1,
    seed          = 0,
    verbose       = True,
    find_init_step_size = False,
    adapt_step_size     = True,
):
    """
    Ensemble Kalman Move with multi-stage Delayed Rejection.

    Derivative-free: only needs forward model G(x) and log probability,
    no gradients required.

    Args:
        log_prob_fn   : (batch, D) -> (batch,).  Vectorised log density.
        forward_fn    : (batch, D) -> (batch, D_data).  Forward model.
        M             : (D_data, D_data).  Data-space precision matrix.
        initial_state : (n_chains, D).  n_chains must be even and >= 4.
        num_samples   : Post-warmup samples to return.
        warmup        : Warmup iterations for step-size adaptation.
        step_size     : Initial step size h (adapted during warmup).
        n_try         : DR stages (1-3, default 3).
        shrink        : Shrink factor per stage (default 0.5).
        subset_size   : k-subset size (default: n_chains//2).
        target_accept : Target stage-1 acceptance for DA (default 0.574).
        thin_by       : Thinning factor.
        seed          : Random seed.
        verbose       : Print progress.
        find_init_step_size : If True, run a short heuristic search at
                              the initial positions to scale `step_size` so that
                              stage-1 acceptance ≈ `target_accept`.
                              If False, use `step_size` as-is.
        adapt_step_size : If True (default), dual-averaging adapts step size during warmup.
                          If False, uses `step_size` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, stage1_rate, final_step_size)
    """
    assert 1 <= n_try <= 3, "n_try must be 1, 2, or 3"
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0
    M = jnp.asarray(M)

    W = n_chains // 2
    k = subset_size if subset_size is not None else W
    assert 2 <= k <= W

    shrink_vec = jnp.array([shrink ** s for s in range(n_try)])

    g1, g2 = state[:W], state[W:]
    lp1, lp2 = log_prob_fn(g1), log_prob_fn(g2)
    key = jax.random.key(seed)

    step_size = jnp.asarray(step_size, jnp.float32)
    if find_init_step_size:
        key, k_ = jax.random.split(key)
        step_size = _find_init_eps(k_, g1, g2, lp1, lp2, step_size,
                                    forward_fn, M, log_prob_fn, W, k, n_try,
                                    target_accept)
        if verbose:
            print(f"Kalman-DR:  init_eps={float(step_size):.4f}")
    log_h0 = jnp.log(step_size)
    da = _da_init(log_h0)

    # --- warmup ---
    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, rng):
        h = jnp.exp(da.log_h) if adapt_step_size else step_size

        g1, lp1, a1_1, any1, rng = _dr_group_step(
            g1, g2, lp1, rng, h, shrink_vec, n_try, forward_fn, M, log_prob_fn, k)

        g2, lp2, a1_2, any2, rng = _dr_group_step(
            g2, g1, lp2, rng, h, shrink_vec, n_try, forward_fn, M, log_prob_fn, k)

        s1_rate = (jnp.mean(a1_1.astype(float)) + jnp.mean(a1_2.astype(float))) / 2.
        overall = (jnp.mean(any1.astype(float)) + jnp.mean(any2.astype(float))) / 2.
        if adapt_step_size:
            da = _da_update(da, s1_rate, log_h0, target_accept)
        return g1, g2, lp1, lp2, da, overall, rng

    rng = key
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, rate, rng = _warmup_step(g1, g2, lp1, lp2, da, rng)
        total_acc += rate

    final_h = jnp.exp(da.log_h_bar) if adapt_step_size else step_size
    if verbose:
        print(f"Kalman-DR:  k={k}  n_try={n_try}  h={float(final_h):.4f}"
              f"  warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    # --- production ---
    final_hs_vec = final_h * shrink_vec

    @jax.jit
    def _step(carry, rng):
        g1, g2, lp1, lp2 = carry

        g1, lp1, a1_1, any1, rng = _dr_group_step(
            g1, g2, lp1, rng, final_h, shrink_vec, n_try,
            forward_fn, M, log_prob_fn, k)

        g2, lp2, a1_2, any2, rng = _dr_group_step(
            g2, g1, lp2, rng, final_h, shrink_vec, n_try,
            forward_fn, M, log_prob_fn, k)

        st = jnp.concatenate([g1, g2])
        acc_any = jnp.concatenate([any1, any2]).astype(float)
        acc_s1  = jnp.concatenate([a1_1, a1_2]).astype(float)
        return (g1, g2, lp1, lp2), (st, acc_any, acc_s1)

    skeys = jax.random.split(rng, num_samples * thin_by)
    (g1, g2, lp1, lp2), (all_states, all_acc, all_s1) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2), skeys)

    samples = all_states[::thin_by][:num_samples]

    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                stage1_rate=float(jnp.mean(all_s1)),
                final_step_size=float(final_h))
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}"
              f"  stage1={info['stage1_rate']:.3f}")
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# Demo:  Gaussian with G(x) = x, M = precision
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim = 2
    cov = jnp.array([[1., .95], [.95, 1.]])
    prec = jnp.linalg.inv(cov)

    def log_prob(x):
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)
    def forward(x):
        return x
    M_mat = prec

    init = jax.random.normal(jax.random.key(42), (40, dim))
    samples, info = sampler_kalman_dr(log_prob, forward, M_mat, init,
                                       num_samples=5000, warmup=1000,
                                       step_size=0.1, n_try=3, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
