"""
sampler_peams — ensemble-preconditioned microcanonical HMC (PEAMS), single file, JAX only.

Combines MAMS dynamics (velocity on the unit sphere, Metropolis correction)
with the ensemble walk move from PEACHES.  The velocity lives on S^{W-1}
(not S^{D-1}): it indexes directions in the complement-walker subspace,
so the sampler is automatically affine-invariant.

The key idea: the complement ensemble defines a W-dimensional coordinate
system.  Microcanonical dynamics run in this W-space; position updates are
mapped back to D-space via  x → x + eps * (u @ centered).

Supports two integrator modes:
  deterministic (default) — standard BAB leapfrog
  langevin                — OBABO scheme with partial velocity refreshment
                            at every leapfrog step (Algorithm 1 of MAMS paper)

Adaptation (warmup only, toggleable):
  Heuristic initial step-size search   (find_init_step_size)
  Dual averaging  → step size          (adapt_step_size)
  ChEES criterion → integration length

"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Microcanonical B step (velocity update on S^{W-1})
# ──────────────────────────────────────────────────────────────────────────────

def _b_step(u, g, dt, W):
    """Solve du/dt = -(I - uu^T) g / (W-1) analytically, then renormalise."""
    g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-30)
    e = -g / g_norm
    ue = jnp.dot(u, e)
    delta = dt * g_norm / (W - 1)
    zeta = jnp.exp(-delta)
    uu = e * (1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta * u
    dk = delta - jnp.log(2.) + jnp.log(
        jnp.maximum(1 + ue + (1 - ue) * zeta**2, 1e-30))
    uu = uu / jnp.sqrt(jnp.sum(uu**2) + 1e-30)
    return uu, dk


# ──────────────────────────────────────────────────────────────────────────────
# O step — partial velocity refreshment on S^{W-1}  (Langevin noise)
# ──────────────────────────────────────────────────────────────────────────────

def _o_step(u, key, W, c1):
    """O_eps(u) = (c1*u + c2*Z/sqrt(W)) / ||...||  where c2 = sqrt(1-c1^2).
    u : (W,) unit velocity on S^{W-1}."""
    c2 = jnp.sqrt(1. - c1**2)
    z  = jax.random.normal(key, u.shape)
    w  = c1 * u + c2 * z / jnp.sqrt(jnp.float32(W))
    return w / jnp.sqrt(jnp.sum(w**2) + 1e-30)


# ──────────────────────────────────────────────────────────────────────────────
# Microcanonical walk-move OBABO in W-dimensional ensemble subspace
# ──────────────────────────────────────────────────────────────────────────────

def _leapfrog_walk_mams(q, u, grad_U, eps, L, centered, W, c1, key):
    """OBABO integrator in the ensemble subspace (reduces to BAB when c1=1.0).

    q        : (W, D) positions
    u        : (W, W) unit velocities — each row ∈ S^{W-1}
    grad_U   : (W, D) -> (W, D)  gradient of U = -log p
    centered : (W, D) centered complement / sqrt(W)
    W        : int, number of walkers per group
    c1       : Langevin damping.  c1=1.0 → deterministic BAB.
    key      : PRNG key for O steps.

    Returns: (q', u', velocity_D, W_k)
        velocity_D : (W, D)  velocity in D-space = u @ centered
        W_k        : (W,)    accumulated kinetic energy change per walker
    """
    half = eps / 2.0

    def one_step(_, carry):
        q, u, g_w, W_k, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        # O — vmap over walkers
        o_keys1 = jax.random.split(k1, W)
        u = jax.vmap(lambda ui, ki: _o_step(ui, ki, W, c1))(u, o_keys1)
        # B_{h}
        u, dk = jax.vmap(lambda ui, gi: _b_step(ui, gi, half, W))(u, g_w)
        W_k = W_k + (W - 1) * dk
        # A
        q = q + eps * (u @ centered)
        # B_{h}
        g_w = grad_U(q) @ centered.T
        u, dk = jax.vmap(lambda ui, gi: _b_step(ui, gi, half, W))(u, g_w)
        W_k = W_k + (W - 1) * dk
        # O
        o_keys2 = jax.random.split(k2, W)
        u = jax.vmap(lambda ui, ki: _o_step(ui, ki, W, c1))(u, o_keys2)
        return (q, u, g_w, W_k, key)

    g_w0 = grad_U(q) @ centered.T
    W_k = jnp.zeros(W)
    q, u, _, W_k, _ = jax.lax.fori_loop(
        0, L, one_step, (q, u, g_w0, W_k, key))

    return q, u, u @ centered, W_k


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble move — returns (proposed, log_alpha, velocity, proposed_lp, aux)
# ──────────────────────────────────────────────────────────────────────────────

def h_walk_mams(group, complement, eps, key, log_prob, grad_U, L, lp_group,
                c1=1.0):
    """Microcanonical walk move: MAMS dynamics in complement subspace."""
    W = group.shape[0]
    centered = (complement - jnp.mean(complement, axis=0)) / jnp.sqrt(W)

    key, k_u, k_traj = jax.random.split(key, 3)

    # Fresh velocity on S^{W-1}
    z = jax.random.normal(k_u, (W, W))
    u0 = z / jnp.sqrt(jnp.sum(z**2, axis=1, keepdims=True) + 1e-30)

    proposed, _, vel, W_k = _leapfrog_walk_mams(
        group, u0, grad_U, eps, L, centered, W, c1, k_traj)
    lp1 = log_prob(proposed)

    # Energy error: potential change + kinetic contribution
    dV = (-lp1) - (-lp_group)
    W_total = dV + W_k
    W_total = jnp.where(jnp.isnan(W_total), jnp.inf, W_total)

    return proposed, jnp.minimum(0., -W_total), vel, lp1, centered


# ──────────────────────────────────────────────────────────────────────────────
# Metropolis accept / reject
# ──────────────────────────────────────────────────────────────────────────────

def _mh(current, proposed, log_alpha, key):
    accept = jnp.log(jax.random.uniform(key, log_alpha.shape, minval=1e-10)) < log_alpha
    return jnp.where(accept[:, None], proposed, current), accept


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

def _chees_update(state, log_alpha, pos_cur, pos_pro, vel, complement, aux, metric,
                  lr=0.025, beta1=0., beta2=0.95, reg=1e-7,
                  T_min=0.25, T_max=10., T_interp=0.9):
    alpha = jnp.clip(jnp.exp(log_alpha), 0., 1.)
    c_cur = pos_cur - jnp.mean(pos_cur, axis=0)
    c_pro = pos_pro - jnp.mean(pos_pro, axis=0)

    if metric == "affine-invariant":
        cov  = jnp.atleast_2d(jnp.cov(complement, rowvar=False))
        W    = c_cur.shape[0]
        sol  = jnp.linalg.solve(cov, jnp.concatenate([c_cur.T, c_pro.T], axis=1))
        Sc   = sol[:, :W].T;   Sp = sol[:, W:].T
        diff_sq = jnp.sum(c_pro * Sp, 1) - jnp.sum(c_cur * Sc, 1)
        inner   = jnp.sum(Sp * vel, 1)
    else:
        diff_sq = jnp.sum(c_pro**2, 1) - jnp.sum(c_cur**2, 1)
        inner   = jnp.sum(c_pro * vel, 1)

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
# Initial step-size search
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, g1, g2, log_prob, grad_U, eps0, move_fn, L):
    lp1, lp2 = log_prob(g1), log_prob(g2)
    fi = jnp.finfo(jnp.result_type(eps0))

    def body(s):
        eps, _, d, k = s
        k, k1, k2 = jax.random.split(k, 3)
        eps = (2.**d) * eps
        _, la1, _, _, _ = move_fn(g1, g2, eps, k1, log_prob, grad_U, L, lp1)
        _, la2, _, _, _ = move_fn(g2, g1, eps, k2, log_prob, grad_U, L, lp2)
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
# sampler_peams
# ──────────────────────────────────────────────────────────────────────────────

def sampler_peams(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup           = 1000,
    step_size        = 0.05,
    L                = 5,
    max_L            = 20,
    thin_by          = 1,
    target_accept    = 0.651,
    chees_metric     = "affine-invariant",
    langevin         = None,
    grad_log_prob_fn = None,
    seed             = 0,
    verbose          = True,
    find_init_step_size = True,
    adapt_step_size     = True,
):
    """
    Ensemble-preconditioned microcanonical HMC (walk move).

    MAMS dynamics in the W-dimensional complement-walker subspace.
    Velocity lives on S^{W-1}; position updates via  x + eps * (u @ centered).
    Affine-invariant by construction.

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).  n_chains must be even and >= 4.
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations.
        step_size        : Initial step size (adapted during warmup).
        L                : Initial leapfrog steps (adapted during warmup).
        max_L            : Maximum leapfrog steps.
        thin_by          : Keep every thin_by-th sample.
        target_accept    : Target acceptance rate for dual averaging.
        chees_metric     : "affine-invariant" or "euclidean".
        langevin         : Ratio L_partial / (L * eps) for OBABO partial refreshment.
                           None (default) → deterministic BAB integrator.
                           A positive float → OBABO with c1 = exp(-1 / (langevin * L)).
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
        seed             : Integer random seed.
        verbose          : Print progress.
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` to
                              ~80% acceptance before warmup.
                              If False, use `step_size` as-is.
        adapt_step_size  : If True (default), dual-averaging adapts step size
                           during warmup. If False, uses `step_size` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_step_size, nominal_L)
    """
    state    = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0, "Need >= 4 even chains"

    _metric = chees_metric

    if grad_log_prob_fn is None:
        grad_U = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_U = grad_log_prob_fn
    _grad_U = lambda x: -grad_U(x)

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    key = jax.random.key(seed)

    _use_langevin = langevin is not None
    def _c1_from_L(cur_L):
        if _use_langevin:
            return jnp.exp(-1. / (langevin * cur_L))
        return 1.0

    step_size = jnp.asarray(step_size, jnp.float32)
    if find_init_step_size:
        key, k = jax.random.split(key)
        step_size = _find_init_eps(k, g1, g2, log_prob_fn, _grad_U, step_size, h_walk_mams, L)
    if verbose:
        print(f"peams  metric={_metric}"
              + (f"  langevin={langevin}" if _use_langevin else "")
              + f"  init_eps={float(step_size):.4f}")

    log_eps0 = jnp.log(step_size)
    da  = _da_init(log_eps0)
    ch  = _chees_init(step_size, L)
    lp1 = log_prob_fn(g1);  lp2 = log_prob_fn(g2)

    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, ch, keys):
        k1, k2, ka1, ka2 = keys
        eps = jnp.exp(da.log_eps) if adapt_step_size else step_size
        cur_L = _chees_L(ch, eps, max_L=max_L)
        cur_c1 = _c1_from_L(cur_L)

        p1, la1, v1, lp1n, aux1 = h_walk_mams(g1, g2, eps, k1, log_prob_fn, _grad_U, cur_L, lp1, cur_c1)
        if adapt_step_size:
            da = _da_update(da, la1, log_eps0, target_accept)
        ch  = _chees_update(ch, la1, g1, p1, v1, g2, aux1, _metric)
        g1, a1 = _mh(g1, p1, la1, ka1);   lp1 = jnp.where(a1, lp1n, lp1)

        eps = jnp.exp(da.log_eps) if adapt_step_size else step_size
        cur_L = _chees_L(ch, eps, max_L=max_L)
        cur_c1 = _c1_from_L(cur_L)

        p2, la2, v2, lp2n, aux2 = h_walk_mams(g2, g1, eps, k2, log_prob_fn, _grad_U, cur_L, lp2, cur_c1)
        if adapt_step_size:
            da = _da_update(da, la2, log_eps0, target_accept)
        ch  = _chees_update(ch, la2, g2, p2, v2, g1, aux2, _metric)
        g2, a2 = _mh(g2, p2, la2, ka2);   lp2 = jnp.where(a2, lp2n, lp2)

        acc = (jnp.mean(a1.astype(float)) + jnp.mean(a2.astype(float))) / 2
        return g1, g2, lp1, lp2, da, ch, acc

    key, k = jax.random.split(key)
    flat  = jax.random.split(k, warmup * 4)
    wkeys = flat.reshape(warmup, 4, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, ch, acc = _warmup_step(g1, g2, lp1, lp2, da, ch, wkeys[i])
        total_acc += acc

    final_eps   = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
    final_log_T = ch.log_T_bar
    nominal_L   = _chees_L(ch, final_eps, bar=True, max_L=max_L)
    if verbose:
        print(f"Warmup done.  eps={float(final_eps):.4f}  L={int(nominal_L)}"
              f"  accept={float(total_acc)/max(warmup,1):.3f}")

    jitter = 0.6
    halton_offset = ch.iteration

    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2, step_i = carry
        k1, k2, ka1, ka2 = keys
        h = _halton(halton_offset + step_i)
        T = jnp.exp(final_log_T)
        T_jit = (1 - jitter) * T + jitter * h * T
        cur_L = jnp.clip(jnp.ceil(T_jit / final_eps), 1, max_L).astype(int)

        cur_c1 = _c1_from_L(cur_L)
        p1, la1, _, lp1n, _ = h_walk_mams(g1, g2, final_eps, k1, log_prob_fn, _grad_U, cur_L, lp1, cur_c1)
        g1, a1 = _mh(g1, p1, la1, ka1);   lp1 = jnp.where(a1, lp1n, lp1)
        p2, la2, _, lp2n, _ = h_walk_mams(g2, g1, final_eps, k2, log_prob_fn, _grad_U, cur_L, lp2, cur_c1)
        g2, a2 = _mh(g2, p2, la2, ka2);   lp2 = jnp.where(a2, lp2n, lp2)

        state  = jnp.concatenate([g1, g2])
        accept = jnp.concatenate([a1, a2]).astype(float)
        return (g1, g2, lp1, lp2, step_i + 1), (state, accept)

    key, k  = jax.random.split(key)
    flat    = jax.random.split(k, num_samples * thin_by * 4)
    skeys   = flat.reshape(num_samples * thin_by, 4, *flat.shape[1:])
    (g1, g2, lp1, lp2, _), (all_states, all_acc) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2, jnp.int32(0)), skeys)

    samples = all_states[::thin_by]
    n_grad_evals = int(num_samples * thin_by) * int(nominal_L) * int(n_chains)
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_step_size=float(final_eps),
                nominal_L=int(nominal_L),
                n_grad_evals=n_grad_evals)
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
    for ut in ["affine-invariant", "euclidean"]:
        for _lang in [None, 1.25]:
            label = f"{ut}" + (f", langevin={_lang}" if _lang else ", deterministic")
            samples, info = sampler_peams(log_prob_gauss, init, num_samples=5000,
                                          warmup=1000, seed=123, step_size=0.01,
                                          chees_metric=ut, langevin=_lang)
            flat = samples.reshape(-1, dim)
            var_est = jnp.var(flat, axis=0)
            var_true = jnp.diag(cov_gauss)
            rel_err = jnp.mean(jnp.abs(var_est - var_true) / var_true)
            print(f"  {label}:  mean_rel_err(var)={rel_err:.3f}"
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
    for _lang in [None, 1.25]:
        label = "deterministic" if _lang is None else f"langevin={_lang}"
        print(f"\n  {label}:")
        samples, info = sampler_peams(log_prob_rosen, init_r, num_samples=2000,
                                      warmup=400, seed=123, step_size=0.01,
                                      langevin=_lang)
        flat = samples.reshape(-1, dim_ros)
        mean_even = jnp.mean(flat[:, ::2])
        mean_odd = jnp.mean(flat[:, 1::2])
        var_even = jnp.mean(jnp.var(flat[:, ::2], axis=0))
        var_odd = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
        print(f"    x_even: mean={mean_even:.3f}  var={var_even:.4f}"
              f"  (target: mean={a_ros}, var=0.5)")
        print(f"    x_odd:  mean={mean_odd:.3f}  var={var_odd:.4f}"
              f"  (target: mean=1.5, var~2.505)")
        print(f"    info: {info}")
