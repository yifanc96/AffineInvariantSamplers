"""
sampler_pickles — Parallel Interacting Covariance-preconditioned Kinetic Langevin
Ensemble Sampler (PICKLES), single file, JAX only.

Ensemble-preconditioned MALT using the complementary ensemble's non-symmetric
square root B = centered/√W to precondition both position and momentum updates.
Momentum p ∈ R^W lives in the complement span; B maps it to position space R^d.

The integrator is BABO+O (full momentum refresh between trajectories, BAB-O
within), with the Metropolis acceptance from Proposition 4.3 of the paper.

Gradient caching: the raw gradient ∇U(q) is cached across trajectories.
After accept/reject, the appropriate gradient (at the accepted or current
position) is carried forward and re-projected onto the new centered matrix.
A trajectory of L BAB-O steps thus costs L gradient evaluations (not L+1).

Special cases:
  gamma = 0        →  ensemble-preconditioned HMC (peaches h-walk)
  L = 1            →  ensemble-preconditioned MALA
  gamma > 0, L > 1 →  full PICKLES

Adaptation (warmup only, each independently toggleable):
  Dual averaging       → step size             (adapt_step_size)
  ChEES criterion      → integration length    (adapt_L)
  Heuristic line search→ initial step size     (find_init_step_size)

Setting all three flags to False reduces warmup to a plain burn-in at the
user-supplied (step_size, L), leaving only NaN guards in effect.

Reference: Tan, Osher & Chen, "Ensemble preconditioning kinetic Langevin"
           Chen, arXiv:2505.02987  (peaches / peanuts framework)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# BAB-O integrator in complement span
# ──────────────────────────────────────────────────────────────────────────────

def _leapfrog_walk_pickles(q, p, gB, grad_U, eps, L, eta_ou, zeta, centered, noise):
    """L steps of BAB-O in the complement span.

    Args:
        q        : (W, D)  positions
        p        : (W, W)  momentum in complement span
        gB       : (W, W)  projected gradient B^T @ grad_U(q) at initial q
        grad_U   : (W, D) -> (W, D)  gradient of potential
        eps      : step size
        L        : number of steps (may be traced)
        eta_ou   : exp(-gamma * eps)   momentum decay per O-step
        zeta     : sqrt(1 - eta_ou^2)  noise scale per O-step
        centered : (W, D)  complement centered / sqrt(W)  = B^T
        noise    : (max_L, W, W)  pre-drawn normals (only first L used)

    Returns:
        q, p, gB, Delta, vel, g_raw
        vel   : (W, D)  position-space velocity from last BAB (before final O)
        g_raw : (W, D)  raw gradient grad_U(q) at final position (for caching)
    """
    half = eps / 2.0
    W, D = q.shape
    dummy_vel = jnp.zeros_like(q)
    dummy_g   = jnp.zeros_like(q)

    def step(j, carry):
        q, p, gB, Delta, _, _ = carry
        xi = noise[j]
        # B: half kick in momentum space
        p = p - half * gB
        # A: full drift in position space
        q = q + eps * (p @ centered)
        # gradient at new position
        g_new = grad_U(q)
        gB_new = g_new @ centered.T                     # B^T @ grad
        # accumulate midpoint energy error per walker
        Delta = Delta - half * jnp.sum(p * (gB + gB_new), axis=1)
        # B: half kick
        p = p - half * gB_new
        # velocity before O-step (for ChEES)
        vel = p @ centered                               # (W, D)
        # O: Ornstein-Uhlenbeck momentum refresh
        p = eta_ou * p + zeta * xi
        gB = gB_new
        return (q, p, gB, Delta, vel, g_new)

    init = (q, p, gB, jnp.zeros(W), dummy_vel, dummy_g)
    q, p, gB, Delta, vel, g_raw = jax.lax.fori_loop(0, L, step, init)
    return q, p, gB, Delta, vel, g_raw


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble move with gradient caching
# ──────────────────────────────────────────────────────────────────────────────

def pickles_move(group, complement, eps, key, log_prob, grad_U, L, gamma,
                 lp_group, gu_cached, max_L=100):
    """Ensemble-preconditioned MALT move (PICKLES) for one group.

    Each walker's momentum lives in R^W (complement span).
    B = centered.T ∈ R^{d×W} maps momentum to position space.

    Args:
        gu_cached : (W, D) cached raw gradient grad_U(group) from a previous
                    call.  Re-projected onto the current centered matrix.

    Returns:
        q_new, log_alpha, vel, lp_new, centered, gu_new
        gu_new : (W, D) raw gradient grad_U(q_new) at the proposed position.
    """
    W, D = group.shape
    centered = (complement - jnp.mean(complement, axis=0)) / jnp.sqrt(W)

    # precompute O-U parameters
    eta_ou = jnp.exp(-gamma * eps)
    zeta   = jnp.sqrt(1.0 - eta_ou**2)

    key_p, key_noise = jax.random.split(key)

    # Step 1: full momentum refresh  p ~ N(0, I_W) per walker
    p0 = jax.random.normal(key_p, (W, W))

    # Step 2: O-step noise  (max_L, W, W) — only first L used by fori_loop
    noise = jax.random.normal(key_noise, (max_L, W, W))

    # Re-project cached gradient onto current centered matrix
    # (no new gradient evaluation needed!)
    gB0 = gu_cached @ centered.T                     # (W, W)
    gBsq0 = jnp.sum(gB0**2, axis=1)                 # (W,)

    # Step 3: run L steps of BAB-O
    q_new, _, gB_new, mid_Delta, vel, gu_new = _leapfrog_walk_pickles(
        group, p0, gB0, grad_U, eps, L, eta_ou, zeta, centered, noise)

    # Step 4: full acceptance ratio  (Prop 4.3, eq 4.4, summed over steps)
    gBsq_new = jnp.sum(gB_new**2, axis=1)            # (W,)
    lp_new   = log_prob(q_new)                         # (W,)

    # Delta = [U_new - U_old] + midpoint_accum + (eps^2/8)(||gB_new||^2 - ||gB_old||^2)
    Delta = (-lp_new + lp_group) + mid_Delta + (eps**2 / 8.0) * (gBsq_new - gBsq0)
    Delta = jnp.where(jnp.isnan(Delta), jnp.inf, Delta)
    log_alpha = jnp.minimum(0.0, -Delta)

    return q_new, log_alpha, vel, lp_new, centered, gu_new


# ──────────────────────────────────────────────────────────────────────────────
# Metropolis accept / reject (also selects cached gradient)
# ──────────────────────────────────────────────────────────────────────────────

def _mh(current, proposed, log_alpha, key):
    accept = jnp.log(jax.random.uniform(key, log_alpha.shape, minval=1e-10)) < log_alpha
    return jnp.where(accept[:, None], proposed, current), accept


def _mh_with_grad(current, proposed, log_alpha, key, gu_cur, gu_pro):
    """MH accept/reject that also selects the cached gradient."""
    accept = jnp.log(jax.random.uniform(key, log_alpha.shape, minval=1e-10)) < log_alpha
    pos = jnp.where(accept[:, None], proposed, current)
    gu  = jnp.where(accept[:, None], gu_pro, gu_cur)
    return pos, accept, gu


# ──────────────────────────────────────────────────────────────────────────────
# Dual averaging — step size
# ──────────────────────────────────────────────────────────────────────────────

class DAState(NamedTuple):
    iteration: int
    log_eps: float
    log_eps_bar: float
    H_bar: float

def _da_init(log_eps0):
    return DAState(0, log_eps0, log_eps0, 0.)

def _da_update(state, log_alpha, log_eps0, target, t0=10., gamma=0.05, kappa=0.75):
    it     = state.iteration + 1
    accept = log_alpha.size / jnp.sum(1. / jnp.clip(jnp.exp(log_alpha), 1e-10, 1.))
    eta    = 1. / (it + t0)
    H_bar  = (1. - eta) * state.H_bar + eta * (target - accept)
    log_e  = log_eps0 - jnp.sqrt(it) / ((it + t0) * gamma) * H_bar
    log_eb = it**(-kappa) * log_e + (1. - it**(-kappa)) * state.log_eps_bar
    return DAState(it, log_e, log_eb, H_bar)


# ──────────────────────────────────────────────────────────────────────────────
# ChEES — integration length via ADAM ascent on trajectory quality
# ──────────────────────────────────────────────────────────────────────────────

class ChEESState(NamedTuple):
    log_T: float;  log_T_bar: float
    m: float;      v: float
    iteration: int; halton: float

@jax.jit
def _halton(n, base=2):
    i, b = jnp.asarray(n, jnp.int32), jnp.asarray(base, jnp.int32)
    def body(s):
        i, f, r = s; f = f / jnp.float32(b)
        return i // b, f, r + f * jnp.mod(i, b)
    _, _, r = jax.lax.while_loop(lambda s: s[0] > 0, body, (i, 1., 0.))
    return r

def _chees_init(eps, L):
    T = eps * L
    return ChEESState(jnp.log(T), jnp.log(T), 0., 0., 1, _halton(1))

def _chees_update(state, log_alpha, pos_cur, pos_pro, vel, complement, centered,
                  lr=0.025, beta1=0., beta2=0.95, reg=1e-7,
                  T_min=0.25, T_max=10., T_interp=0.9):
    """ChEES gradient with affine-invariant metric (full Sigma^{-1})."""
    alpha = jnp.clip(jnp.exp(log_alpha), 0., 1.)
    c_cur = pos_cur - jnp.mean(pos_cur, axis=0)
    c_pro = pos_pro - jnp.mean(pos_pro, axis=0)

    # affine-invariant metric
    cov  = jnp.atleast_2d(jnp.cov(complement, rowvar=False))
    W    = c_cur.shape[0]
    sol  = jnp.linalg.solve(cov, jnp.concatenate([c_cur.T, c_pro.T], axis=1))
    Sc   = sol[:, :W].T;   Sp = sol[:, W:].T
    diff_sq = jnp.sum(c_pro * Sp, 1) - jnp.sum(c_cur * Sc, 1)
    inner   = jnp.sum(Sp * vel, 1)

    g_m = state.halton * jnp.exp(state.log_T) * diff_sq * inner
    g_m = jnp.where((alpha > 1e-4) & jnp.isfinite(g_m), g_m, 0.)
    g   = jnp.sum(alpha * g_m) / (jnp.sum(alpha) + reg)

    it = state.iteration + 1
    m  = beta1 * state.m + (1 - beta1) * g
    v  = beta2 * state.v + (1 - beta2) * g**2
    delta = lr * (m / (1 - beta1**it)) / jnp.sqrt(v / (1 - beta2**it) + reg)
    delta = jnp.clip(delta, -0.35, 0.35)
    log_T = jnp.clip(state.log_T + delta, jnp.log(T_min), jnp.log(T_max))
    log_Tb = jnp.logaddexp(jnp.log(T_interp) + state.log_T_bar,
                            jnp.log(1 - T_interp) + log_T)
    log_Tb = jnp.clip(log_Tb, jnp.log(T_min), jnp.log(T_max))
    return ChEESState(log_T, log_Tb, m, v, it, _halton(it))

def _chees_L(state, eps, jitter=0.6, bar=False, max_L=100):
    T = jnp.exp(state.log_T_bar if bar else state.log_T)
    T = (1 - jitter) * T + jitter * state.halton * T
    return jnp.clip(jnp.ceil(T / eps), 1, max_L).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (~80% acceptance with L=1)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, g1, g2, log_prob, grad_U, gamma, eps0):
    lp1, lp2 = log_prob(g1), log_prob(g2)
    gu1, gu2 = grad_U(g1), grad_U(g2)    # initial gradient evals
    fi = jnp.finfo(jnp.result_type(eps0))

    def body(s):
        eps, _, d, k = s
        k, k1, k2 = jax.random.split(k, 3)
        eps = (2.**d) * eps
        _, la1, _, _, _, _ = pickles_move(
            g1, g2, eps, k1, log_prob, grad_U, 1, gamma, lp1, gu1, max_L=1)
        _, la2, _, _, _, _ = pickles_move(
            g2, g1, eps, k2, log_prob, grad_U, 1, gamma, lp2, gu2, max_L=1)
        la  = jnp.concatenate([la1, la2])
        avg = jnp.log(la.shape[0]) - jax.scipy.special.logsumexp(-la)
        return eps, d, jnp.where(jnp.log(.8) < avg, 1, -1), k

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps / 2.


# ──────────────────────────────────────────────────────────────────────────────
# sampler_pickles
# ──────────────────────────────────────────────────────────────────────────────

def sampler_pickles(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup              = 1000,
    step_size           = 0.05,
    L                   = 5,
    gamma               = 2.0,
    max_L               = 20,
    thin_by             = 1,
    target_accept       = 0.651,
    grad_log_prob_fn    = None,
    seed                = 0,
    verbose             = True,
    adapt_step_size     = True,
    adapt_L             = True,
    find_init_step_size = True,
):
    """
    Parallel Interacting Covariance-preconditioned Kinetic Langevin Ensemble
    Sampler (PICKLES) with MALT-style Metropolization.

    Gradient caching: the raw gradient grad_U(q) is cached across trajectories.
    After accept/reject, the appropriate gradient (accepted or current) is
    carried forward.  A trajectory of L BAB-O steps thus costs L gradient
    evaluations, not L+1.

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).  n_chains must be even and >= 4.
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations.
        step_size        : Initial step size (adapted during warmup).
        L                : Initial leapfrog steps (adapted during warmup via ChEES).
        gamma            : Friction coefficient (>= 0).
                           0 → ensemble HMC (peaches h-walk);
                           large → more damping, closer to ensemble MALA.
        max_L            : Maximum leapfrog steps.
        thin_by          : Keep every thin_by-th sample.
        target_accept    : Target acceptance rate for dual averaging.
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
                           If None, uses jax.vmap(jax.grad(log_prob_fn)).
        seed             : Integer random seed.
        verbose          : Print progress.
        adapt_step_size  : If True, tune step size by dual averaging during
                           warmup.  If False, use `step_size` as given.
        adapt_L          : If True, tune integration length by ChEES during
                           warmup.  If False, use `L` as given.
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` to
                              ~80% acceptance before warmup.
                              If False, use `step_size` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_step_size, nominal_L, gamma)
    """
    state    = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0, "Need >= 4 even chains"
    assert 1 <= L <= max_L, f"Need 1 <= L <= max_L (got L={L}, max_L={max_L})"

    if grad_log_prob_fn is None:
        grad_U = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_U = grad_log_prob_fn
    _grad_U = lambda x: -grad_U(x)            # gradient of potential U = -log_prob

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    key = jax.random.key(seed)

    if find_init_step_size:
        key, k = jax.random.split(key)
        step_size = _find_init_eps(k, g1, g2, log_prob_fn, _grad_U, gamma,
                                   step_size)
    step_size = jnp.asarray(step_size)
    if verbose:
        src = "search" if find_init_step_size else "user"
        print(f"PICKLES  gamma={gamma}  init_eps={float(step_size):.4f} ({src})"
              f"  adapt_step_size={adapt_step_size}  adapt_L={adapt_L}")

    log_eps0 = jnp.log(step_size)
    da  = _da_init(log_eps0)
    ch  = _chees_init(step_size, L)
    lp1 = log_prob_fn(g1);  lp2 = log_prob_fn(g2)

    # Initial gradient evaluations (the only "extra" evals)
    gu1 = _grad_U(g1)     # (W, D)  grad_U at g1 positions
    gu2 = _grad_U(g2)     # (W, D)  grad_U at g2 positions

    # ── warmup ──────────────────────────────────────────────────────────────
    # Closure-captured Python bools resolve at trace time, so the unused
    # adaptation branches are removed from the compiled graph entirely.
    fixed_L = jnp.int32(L)

    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, gu1, gu2, da, ch, keys):
        k1, k2, ka1, ka2 = keys
        eps   = jnp.exp(da.log_eps) if adapt_step_size else step_size
        cur_L = _chees_L(ch, eps, max_L=max_L) if adapt_L else fixed_L

        p1, la1, v1, lp1n, cen1, gu1n = pickles_move(
            g1, g2, eps, k1, log_prob_fn, _grad_U, cur_L, gamma, lp1, gu1, max_L)
        if adapt_step_size:
            da = _da_update(da, la1, log_eps0, target_accept)
        if adapt_L:
            ch = _chees_update(ch, la1, g1, p1, v1, g2, cen1)
        g1, a1, gu1 = _mh_with_grad(g1, p1, la1, ka1, gu1, gu1n)
        lp1 = jnp.where(a1, lp1n, lp1)

        eps   = jnp.exp(da.log_eps) if adapt_step_size else step_size
        cur_L = _chees_L(ch, eps, max_L=max_L) if adapt_L else fixed_L

        p2, la2, v2, lp2n, cen2, gu2n = pickles_move(
            g2, g1, eps, k2, log_prob_fn, _grad_U, cur_L, gamma, lp2, gu2, max_L)
        if adapt_step_size:
            da = _da_update(da, la2, log_eps0, target_accept)
        if adapt_L:
            ch = _chees_update(ch, la2, g2, p2, v2, g1, cen2)
        g2, a2, gu2 = _mh_with_grad(g2, p2, la2, ka2, gu2, gu2n)
        lp2 = jnp.where(a2, lp2n, lp2)

        acc = (jnp.mean(a1.astype(float)) + jnp.mean(a2.astype(float))) / 2
        return g1, g2, lp1, lp2, gu1, gu2, da, ch, acc

    key, k = jax.random.split(key)
    flat  = jax.random.split(k, warmup * 4)
    wkeys = flat.reshape(warmup, 4, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, gu1, gu2, da, ch, acc = _warmup_step(
            g1, g2, lp1, lp2, gu1, gu2, da, ch, wkeys[i])
        total_acc += acc

    final_eps   = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
    if adapt_L:
        final_log_T = ch.log_T_bar
        nominal_L   = _chees_L(ch, final_eps, bar=True, max_L=max_L)
    else:
        final_log_T = jnp.log(final_eps * L)
        nominal_L   = fixed_L
    if verbose:
        print(f"Warmup done.  eps={float(final_eps):.4f}  L={int(nominal_L)}"
              f"  accept={float(total_acc)/max(warmup,1):.3f}")

    # ── production ──────────────────────────────────────────────────────────
    jitter = 0.6
    halton_offset = ch.iteration

    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2, gu1, gu2, step_i = carry
        k1, k2, ka1, ka2 = keys
        if adapt_L:
            h = _halton(halton_offset + step_i)
            T = jnp.exp(final_log_T)
            T_jit = (1 - jitter) * T + jitter * h * T
            cur_L = jnp.clip(jnp.ceil(T_jit / final_eps), 1, max_L).astype(int)
        else:
            cur_L = fixed_L

        p1, la1, _, lp1n, _, gu1n = pickles_move(
            g1, g2, final_eps, k1, log_prob_fn, _grad_U, cur_L, gamma, lp1,
            gu1, max_L)
        g1, a1, gu1 = _mh_with_grad(g1, p1, la1, ka1, gu1, gu1n)
        lp1 = jnp.where(a1, lp1n, lp1)

        p2, la2, _, lp2n, _, gu2n = pickles_move(
            g2, g1, final_eps, k2, log_prob_fn, _grad_U, cur_L, gamma, lp2,
            gu2, max_L)
        g2, a2, gu2 = _mh_with_grad(g2, p2, la2, ka2, gu2, gu2n)
        lp2 = jnp.where(a2, lp2n, lp2)

        state  = jnp.concatenate([g1, g2])
        accept = jnp.concatenate([a1, a2]).astype(float)
        return (g1, g2, lp1, lp2, gu1, gu2, step_i + 1), (state, accept)

    key, k  = jax.random.split(key)
    flat    = jax.random.split(k, num_samples * thin_by * 4)
    skeys   = flat.reshape(num_samples * thin_by, 4, *flat.shape[1:])
    (g1, g2, lp1, lp2, gu1, gu2, _), (all_states, all_acc) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2, gu1, gu2, jnp.int32(0)), skeys)

    samples = all_states[::thin_by]
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_step_size=float(final_eps),
                nominal_L=int(nominal_L),
                gamma=float(gamma))
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}")
    return samples, info


if __name__ == "__main__":

    # --- Test 1: Ill-conditioned Gaussian (20D, kappa=1000) ---
    print("=" * 60)
    print("Test 1: Ill-conditioned Gaussian  (D=20, kappa=1000)")
    print("=" * 60)
    dim = 20
    kappa = 1000.
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov_gauss = Q @ jnp.diag(eigvals) @ Q.T
    prec_gauss = Q @ jnp.diag(1. / eigvals) @ Q.T
    def log_prob_gauss(x):
        return -0.5 * jnp.sum((x @ prec_gauss) * x, axis=-1)

    init = jax.random.normal(jax.random.key(42), (100, dim))

    for g in [2.0]:
        print(f"\n  gamma={g}:")
        samples, info = sampler_pickles(log_prob_gauss, init, num_samples=5000,
                                      warmup=1000, seed=123, gamma=g, step_size=0.01)
        flat = samples.reshape(-1, dim)
        var_est = jnp.var(flat, axis=0)
        var_true = jnp.diag(cov_gauss)
        rel_err = jnp.mean(jnp.abs(var_est - var_true) / var_true)
        print(f"    mean_rel_err(var)={rel_err:.3f}"
              f"  var_range=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
              f"  (target: [{jnp.min(var_true):.2f}, {jnp.max(var_true):.2f}])")
        print(f"    info: {info}")

    # --- Test 2: Rosenbrock (10D) ---
    print()
    print("=" * 60)
    print("Test 2: Rosenbrock  (D=10, a=1, b=100)")
    print("=" * 60)
    a_ros, b_ros = 1.0, 100.0
    dim_ros = 10
    def log_prob_rosen(x):
        x_even = x[:, ::2]
        x_odd = x[:, 1::2]
        return -(b_ros * jnp.sum((x_odd - x_even**2)**2, axis=1)
                 + jnp.sum((x_even - a_ros)**2, axis=1))

    init_r = jax.random.normal(jax.random.key(42), (100, dim_ros))
    samples, info = sampler_pickles(log_prob_rosen, init_r, num_samples=2000,
                                  warmup=500, seed=123, step_size=0.01, gamma=2.0)
    flat = samples.reshape(-1, dim_ros)
    mean_even = jnp.mean(flat[:, ::2])
    mean_odd = jnp.mean(flat[:, 1::2])
    var_even = jnp.mean(jnp.var(flat[:, ::2], axis=0))
    var_odd = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
    print(f"  x_even: mean={mean_even:.3f}  var={var_even:.4f}"
          f"  (target: mean={a_ros}, var=0.5)")
    print(f"  x_odd:  mean={mean_odd:.3f}  var={var_odd:.4f}"
          f"  (target: mean=1.5, var~2.505)")
    print(f"  info: {info}")

    # --- Test 3: Plain (no adaptation) on Gaussian -----------------------
    # Run with user-supplied (step_size, L), no DA / ChEES / init search.
    print()
    print("=" * 60)
    print("Test 3: Gaussian, plain mode (all adaptation off)")
    print("=" * 60)
    samples, info = sampler_pickles(
        log_prob_gauss, init, num_samples=5000, warmup=1000, seed=123,
        gamma=2.0, step_size=0.05, L=5,
        adapt_step_size=False, adapt_L=False, find_init_step_size=False)
    flat = samples.reshape(-1, dim)
    var_est = jnp.var(flat, axis=0)
    var_true = jnp.diag(cov_gauss)
    rel_err = jnp.mean(jnp.abs(var_est - var_true) / var_true)
    print(f"  mean_rel_err(var)={rel_err:.3f}  info: {info}")
