"""
sampler_gndr_full — Gauss-Newton Proposal Langevin with **arbitrary-depth**
                    Delayed Rejection.

Same proposal as ``sampler_gndr``:
  H_GN(x) = J_r(x)^T J_r(x)
  drift   = h * H_GN(x)^{-1} @ grad log pi(x)
  noise   = sqrt(2h) * L(x)^{-T} @ z,   L L^T = H_GN(x)

Difference: the existing ``sampler_gndr`` hard-codes the DR acceptance up to
3 stages.  This version implements the same standard multi-stage DR
acceptance (Mira 1998 Sec. 5.2; see also Modi/Barnett/Carpenter,
arXiv:2110.00610) for **any** ``n_try``, via a recursive evaluation that
is unrolled at JIT trace time (so each extra stage adds work).

Why bother
----------
Light-tailed targets (e.g. ``log pi(x) ~ -|x|^4``) are notorious failure
cases for vanilla MALA: the Langevin drift ``-grad U`` grows faster than
the noise, so a too-large step size causes systematic overshoot and the
chain loses geometric ergodicity.  Multi-stage DR sidesteps this — when
the first proposal is rejected at the tail, retry with a shrunk step.
Multi-stage DR is expected to help restore geometric ergodicity in
practice if enough retries are allowed.  Three stages is sometimes not
enough; this sampler lets you use as many as you need.

Reference
---------
Standard multi-stage DR acceptance formula (Mira 1998, Sec. 5.2;
see also Modi/Barnett/Carpenter, arXiv:2110.00610).
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Dual averaging (identical to gndr.py)
# ──────────────────────────────────────────────────────────────────────────────

class DAState(NamedTuple):
    iteration: int
    log_h: float
    log_h_bar: float
    H_bar: float

def _da_init(log_h0):
    return DAState(0, log_h0, log_h0, 0.)

def _da_update(state, accept_rate, log_h0, target,
                t0=10., gamma=0.05, kappa=0.75):
    it    = state.iteration + 1
    eta   = 1. / (it + t0)
    H_bar = (1. - eta) * state.H_bar + eta * (target - accept_rate)
    log_h = log_h0 - jnp.sqrt(it) / ((it + t0) * gamma) * H_bar
    log_hb = it**(-kappa) * log_h + (1. - it**(-kappa)) * state.log_h_bar
    return DAState(it, log_h, log_hb, H_bar)


# ──────────────────────────────────────────────────────────────────────────────
# Hessian regularization and Cholesky (identical to gndr.py)
# ──────────────────────────────────────────────────────────────────────────────

def _safe_cholesky(H, reg_small=1e-6, reg_large=1e-3):
    D = H.shape[-1]
    eye = jnp.eye(D)
    H_sym = 0.5 * (H + jnp.swapaxes(H, -2, -1)) + reg_small * eye
    L = jnp.linalg.cholesky(H_sym)
    bad = jnp.any(jnp.isnan(L), axis=(-2, -1), keepdims=True)
    L_safe = jnp.linalg.cholesky(H_sym + (reg_large - reg_small) * eye)
    return jnp.where(bad, L_safe, L)


# ──────────────────────────────────────────────────────────────────────────────
# Langevin proposal kernel and its log-density (identical to gndr.py)
# ──────────────────────────────────────────────────────────────────────────────

def _transition_logp(x, y, grad_x, L_x, h):
    """log q(y | x) for h-Langevin proposal with metric H_GN(x) = L L^T."""
    v = jax.scipy.linalg.solve_triangular(L_x, grad_x, lower=True)
    drift = h * jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L_x, -2, -1), v, lower=False)
    diff = y - x - drift
    Lt_diff = jnp.einsum('...ij,...j->...i',
                         jnp.swapaxes(L_x, -2, -1), diff)
    quad = jnp.sum(Lt_diff ** 2, axis=-1) / (2. * h)
    D = x.shape[-1]
    log_det = D * jnp.log(4. * jnp.pi * h) - 2. * jnp.sum(
        jnp.log(jnp.abs(jnp.diagonal(L_x, axis1=-2, axis2=-1))), axis=-1)
    return -0.5 * (quad + log_det)


def _propose(x, grad_x, L_x, h, z):
    """Sample x' = x + h * H^{-1} grad + sqrt(2h) * L^{-T} z."""
    v = jax.scipy.linalg.solve_triangular(L_x, grad_x, lower=True)
    drift = h * jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L_x, -2, -1), v, lower=False)
    noise = jnp.sqrt(2. * h) * jax.scipy.linalg.solve_triangular(
        jnp.swapaxes(L_x, -2, -1), z, lower=False)
    return x + drift + noise


def _safe_log1m_exp(la):
    """Numerically safe log(1 - exp(la)) for la <= 0.

    The naive ``log1p(-exp(0)) = -inf`` is mathematically right (α = 1
    means rejection has zero probability), but in Mira's DR formula it
    appears in BOTH the numerator (via reverse inner α) and the
    denominator (via forward inner α).  When both saturate, ``-inf -
    (-inf) = NaN`` silently kills the outer acceptance.

    Concrete failure mode on light-tailed targets: when the deep-stage
    proposal ``y_k ≈ y_{k-1}``, the inner sub-path α(y_k, y_{k-1}) hits
    1 exactly, every outer α becomes NaN, and stages ≥ 4 *never accept*
    even though the underlying recursion is algebraically correct.

    We replace ``-inf`` with a large but finite floor so that matching
    saturated terms cancel in the num/den difference; only genuinely
    *unbalanced* saturations contribute (correctly) ~ -∞ to log α.
    """
    LOG1M_FLOOR = -1e8   # exp(-1e8) is well below any realistic α-bias
    return jnp.where(la > -1e-30,
                      LOG1M_FLOOR,
                      jnp.log1p(-jnp.exp(jnp.minimum(la, -1e-30))))


# ──────────────────────────────────────────────────────────────────────────────
# Recursive DR acceptance — arbitrary depth, Mira (1998) convention
# ──────────────────────────────────────────────────────────────────────────────
#
# Path P = (P_0, P_1, ..., P_k) with all proposals generated from P_0 = x
# at steps h_0, h_1, ..., h_{k-1}.  Every q-evaluation involving P_j uses
# step h_{j-1} (the one that originally proposed P_j), regardless of source.
# As deep-stage proposals converge to x, α → 1 monotonically — the
# "eventual acceptance" property useful for geometric ergodicity on
# light-tailed targets.
#
# The same convention is used by the hand-coded sampler_gndr (n_try ≤ 3)
# and by the original notebook in the initial-samplers branch.
#
# The recursion is parameterised by (current_idx, proposal_idx, prev_tuple)
# where prev_tuple is the forward-index-ordered subset of intermediate
# rejected proposals.  At each call:
#
#   α(current, proposal | prev) = min(1, N / D),
#
#   N = π(P_proposal) · q(P_proposal → P_current; h_{|prev|})
#                     · Π_{j ∈ prev} q(P_proposal → P_j; h_{j-1})
#                     · Π_{i,j enumerated from prev}
#                         (1 − α(P_proposal, P_j | prev[:i]))
#
#   D = π(P_current)  · q(P_current → P_proposal; h_{|prev|})
#                     · Π_{j ∈ prev} q(P_current → P_j; h_{j-1})
#                     · Π_{i,j enumerated from prev}
#                         (1 − α(P_current, P_j | prev[:i]))
#
# The "main" kernel uses h_{|prev|} (its position in the DR ladder); each
# prev-term kernel uses h_{j-1} (the step that originally proposed P_j).
# At the top level current=0, proposal=k, prev=(1, 2, ..., k-1).
#
# Implementation: Python recursion + Python-level memoization, fully
# unrolled at JIT trace time since n_try is static.


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (identical idea to gndr.py)
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
# Build a single DR step (warmup or production) given step sizes
# ──────────────────────────────────────────────────────────────────────────────

def _make_dr_step(n_try, v_lp, v_grad, v_hess, n_chains, dim):
    """Returns step(x, lp, hs, rng) -> (new_x, new_lp, accepted_any, alpha1, rng).

    Performs one full multi-stage DR transition under the proposal-binding
    convention.  The per-stage step sizes hs[0], hs[1], ..., hs[n_try-1]
    correspond to the steps used at DR ladder positions 1, 2, ..., n_try
    (i.e. hs[j-1] is the step that originally proposed P_j from x).
    """

    def step(x, lp, hs, rng):
        N, D = x.shape

        # Gradient + Hessian + Cholesky at x = P_0.
        grad_x = v_grad(x)
        L_x    = _safe_cholesky(v_hess(x))

        # Generate all k proposals from x with shrunk steps.
        rng, *zkeys = jax.random.split(rng, n_try + 1)
        path_pts  = [x]
        path_lp   = [lp]
        path_grad = [grad_x]
        path_L    = [L_x]
        for s in range(n_try):
            z = jax.random.normal(zkeys[s], (N, D))
            y = _propose(x, grad_x, L_x, hs[s], z)
            path_pts.append(y)
            path_lp.append(v_lp(y))
            path_grad.append(v_grad(y))
            path_L.append(_safe_cholesky(v_hess(y)))

        # Memoize log q(P_src → P_dest; hs[step_idx]) by (src, dest, step_idx).
        q_cache = {}
        def get_q(src, dest, step_idx):
            key = (src, dest, step_idx)
            if key not in q_cache:
                q_cache[key] = _transition_logp(
                    path_pts[src], path_pts[dest],
                    path_grad[src], path_L[src], hs[step_idx])
            return q_cache[key]

        # Memoized α with proposal-binding pairing.
        alpha_cache = {}
        def alpha(current, proposal, prev_tuple):
            key = (current, proposal, prev_tuple)
            if key in alpha_cache:
                return alpha_cache[key]
            num_rej = len(prev_tuple)
            cur_idx = num_rej             # main step uses h_{num_rej}
            qf = get_q(current, proposal, cur_idx)
            qb = get_q(proposal, current, cur_idx)
            log_num = path_lp[proposal] + qb
            log_den = path_lp[current]  + qf
            for j in prev_tuple:
                step_idx = j - 1          # prev term uses h_{j-1}
                log_num = log_num + get_q(proposal, j, step_idx)
                log_den = log_den + get_q(current,  j, step_idx)
            for i, j in enumerate(prev_tuple):
                af = alpha(current,  j, prev_tuple[:i])
                ar = alpha(proposal, j, prev_tuple[:i])
                log_num = log_num + _safe_log1m_exp(ar)
                log_den = log_den + _safe_log1m_exp(af)
            res = jnp.minimum(0., log_num - log_den)
            alpha_cache[key] = res
            return res

        # Stage-by-stage acceptance (sequential).
        accepted_any = jnp.zeros(N, dtype=bool)
        new_x  = x
        new_lp = lp
        alpha1_log = None
        for j in range(1, n_try + 1):
            prev = tuple(range(1, j))
            la = alpha(0, j, prev)
            if j == 1:
                alpha1_log = la
            rng, uk = jax.random.split(rng)
            u = jnp.log(jax.random.uniform(uk, (N,), minval=1e-10))
            this_acc = (~accepted_any) & (u < la)
            new_x  = jnp.where(this_acc[:, None], path_pts[j], new_x)
            new_lp = jnp.where(this_acc, path_lp[j], new_lp)
            accepted_any = accepted_any | this_acc

        return new_x, new_lp, accepted_any, alpha1_log, rng

    return step


# ──────────────────────────────────────────────────────────────────────────────
# Public sampler
# ──────────────────────────────────────────────────────────────────────────────

def sampler_gndr_full(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    step_size     = 0.1,
    n_try         = 5,
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
    """Gauss-Newton Proposal Langevin with arbitrary-depth Delayed Rejection.

    Parameters mirror :func:`sampler_gndr`, with ``n_try`` now unbounded.
    Trace-time cost grows ~O(n_try³) (recursion size); JIT compile time
    grows accordingly.  Use ``n_try`` up to ~20 in practice.

    Notes
    -----
    Acceptance follows Mira (2001).  Stage-j acceptance probability is

        α_j(P_0, ..., P_j) = min(1,  N_j / D_j)

    with the standard reverse-path / inner-rejection corrections.  At trace
    time we recursively compute every α(s, e) needed for α(0, n_try) and
    memoize.  The forward-path proposals are generated all at once from
    P_0 = x with shrunk steps h * shrink^stage.

    Returns
    -------
    samples : (num_samples, n_chains, D)
    info    : dict with acceptance rates per stage, final step size, and
              n_grad_evals.
    """
    assert n_try >= 1, "n_try must be ≥ 1"
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape

    if grad_fn is None:
        grad_fn = jax.grad(log_prob_fn)

    if hessian_fn is None:
        if residual_fn is not None:
            # Gauss-Newton: H_GN(x) = J_r(x)^T @ J_r(x), PSD by construction.
            def hessian_fn(x):
                J = jax.jacobian(residual_fn)(x)
                return J.T @ J
        else:
            raise ValueError(
                "sampler_gndr_full requires a state-dependent PSD metric H(x).  "
                "Pass one of:\n"
                "  residual_fn=r   with -log π(x) = ½ ‖r(x)‖²  (preferred — "
                "gives the Gauss-Newton metric J_r^T J_r, PSD by construction).\n"
                "  hessian_fn=H    with H(x) PSD by construction (e.g. via "
                "eigenvalue clipping of -∇²log π).\n"
                "There is no auto-fallback: -∇²log π is not PSD when log π "
                "is locally non-log-concave (singular peaks, inflection "
                "points, multimodal targets), and using it silently corrupts "
                "the proposal.")

    v_lp   = jax.vmap(log_prob_fn)
    v_grad = jax.vmap(grad_fn)
    v_hess = jax.vmap(hessian_fn)

    shrink_vec = jnp.array([shrink ** s for s in range(n_try)])

    key = jax.random.key(seed)
    step_size = jnp.asarray(step_size, jnp.float32)

    x  = state
    lp = v_lp(x)

    if find_init_step_size:
        _user_h = float(step_size)
        key, k_ = jax.random.split(key)
        grad_x0 = v_grad(x)
        L_x0    = _safe_cholesky(v_hess(x))
        step_size = _find_init_eps(k_, x, lp, grad_x0, L_x0, step_size,
                                    v_lp, target_accept)
        if verbose:
            print(f"[gndr_full] find_init_step_size: step_size {_user_h:.4g} "
                  f"→ {float(step_size):.4g}\n"
                  f"   (if the chain later stalls, set find_init_step_size=False "
                  f"and pass your own step_size.)")

    log_h0 = jnp.log(step_size)
    da     = _da_init(log_h0)

    dr_step = _make_dr_step(n_try, v_lp, v_grad, v_hess, n_chains, dim)

    # ── warmup ───────────────────────────────────────────────────────────
    @jax.jit
    def _warmup_step(x, lp, da, rng):
        h_base = jnp.exp(da.log_h) if adapt_step_size else step_size
        hs = h_base * shrink_vec
        x, lp, acc_any, alpha1_log, rng = dr_step(x, lp, hs, rng)
        s1_rate = jnp.mean(jnp.exp(jnp.minimum(0., alpha1_log)))
        if adapt_step_size:
            da = _da_update(da, s1_rate, log_h0, target_accept)
        return x, lp, da, jnp.mean(acc_any.astype(float)), rng

    key, k_ = jax.random.split(key)
    rng = k_
    total_acc = 0.
    for _ in range(warmup):
        x, lp, da, rate, rng = _warmup_step(x, lp, da, rng)
        total_acc += rate

    final_h = jnp.exp(da.log_h_bar) if adapt_step_size else step_size
    if verbose:
        print(f"GN-DR-full:  n_try={n_try}  h={float(final_h):.4f}  "
              f"warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    # ── production ───────────────────────────────────────────────────────
    final_hs = final_h * shrink_vec

    @jax.jit
    def _step(carry, rng):
        x, lp = carry
        x, lp, acc_any, _, rng = dr_step(x, lp, final_hs, rng)
        return (x, lp), (x, acc_any.astype(float))

    key, k = jax.random.split(key)
    skeys = jax.random.split(k, num_samples * thin_by)
    (x, lp), (all_states, all_acc) = jax.lax.scan(
        _step, (x, lp), skeys)

    samples = all_states[::thin_by][:num_samples]

    # 1 grad at x + n_try grads at the proposals per iteration per walker.
    n_grad_evals = (int(num_samples * thin_by)
                    * (int(n_try) + 1)
                    * int(n_chains))
    info = dict(
        acceptance_rate = float(jnp.mean(all_acc)),
        final_step_size = float(final_h),
        n_grad_evals    = n_grad_evals,
        n_try           = int(n_try),
    )
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}")
    return samples, info
