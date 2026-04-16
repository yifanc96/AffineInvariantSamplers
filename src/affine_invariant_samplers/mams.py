"""
sampler_mams — Metropolis Adjusted Microcanonical Sampler, single file, JAX only.

MAMS uses microcanonical (unit-velocity) dynamics with a Metropolis correction.
The velocity lives on the unit sphere S^{d-1}; the potential gradient drives
a geodesic-like flow that approximately preserves the target energy surface.
A single Metropolis-Hastings step per trajectory makes the chain exact.

Supports two integrator modes:
  "deterministic" (default) — standard BAB leapfrog
  "langevin"                — OBABO scheme with partial velocity refreshment
                              at every leapfrog step (Algorithm 1 of the paper)

Adaptation (warmup only, toggleable):
  Heuristic initial step-size search    (find_init_step_size)
  Dual averaging → step size            (adapt_step_size)
  ChEES or autocorrelation → integration length
    (set L to an explicit int to skip length adaptation)

Reference: Robnik, Cohn-Gordon & Seljak, arXiv:2503.01707
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Microcanonical B step (velocity update on the unit sphere)
# ──────────────────────────────────────────────────────────────────────────────

def _b_step(u, g, dt, d):
    """Solve du/dt = -(I - uu^T) g / (d-1) analytically, then renormalise.

    Args:
        u  : (D,) unit velocity
        g  : (D,) grad L(x)  where L = -log p
        dt : integration time (eps or eps/2)
        d  : dimension
    Returns:
        u_new : (D,) renormalised velocity
        dk    : scalar, kinetic energy change / (d-1)
    """
    g_norm = jnp.sqrt(jnp.sum(g**2) + 1e-30)
    e = -g / g_norm
    ue = jnp.dot(u, e)
    delta = dt * g_norm / (d - 1)
    zeta = jnp.exp(-delta)
    uu = e * (1 - zeta) * (1 + zeta + ue * (1 - zeta)) + 2 * zeta * u
    dk = delta - jnp.log(2.) + jnp.log(
        jnp.maximum(1 + ue + (1 - ue) * zeta**2, 1e-30))
    uu = uu / jnp.sqrt(jnp.sum(uu**2) + 1e-30)
    return uu, dk


# ──────────────────────────────────────────────────────────────────────────────
# O step — partial velocity refreshment on S^{d-1}  (Langevin noise)
# ──────────────────────────────────────────────────────────────────────────────

def _o_step(u, key, d, c1):
    """O_eps(u) = (c1*u + c2*Z/sqrt(d)) / ||...||  where c2 = sqrt(1-c1^2)."""
    c2 = jnp.sqrt(1. - c1**2)
    z  = jax.random.normal(key, u.shape)
    w  = c1 * u + c2 * z / jnp.sqrt(jnp.float32(d))
    return w / jnp.sqrt(jnp.sum(w**2) + 1e-30)


# ──────────────────────────────────────────────────────────────────────────────
# Microcanonical trajectory (single chain)
# ──────────────────────────────────────────────────────────────────────────────

def _mams_trajectory(x, u, grad_L, L_x, L_fn, grad_L_fn, eps, n_steps, d,
                     c1, key):
    """OBABO integrator (reduces to BAB when c1=1.0).

    c1 = 1.0 → deterministic BAB (O steps are identity)
    c1 < 1.0 → Langevin MAMS / OBABO (Algorithm 1 of the paper)
               c1 = exp(-eps / L_partial)

    Each of the L steps is:  O → B_{h} → A → B_{h} → O
    Energy error W only accumulates from the BAB part (Theorem 5.1).
    """
    half = eps / 2.0

    def one_step(_, carry):
        x, u, g, W_k, key = carry
        key, k1, k2 = jax.random.split(key, 3)
        # O
        u = _o_step(u, k1, d, c1)
        # B_{h}
        u, dk = _b_step(u, g, half, d)
        W_k = W_k + (d - 1) * dk
        # A
        x = x + eps * u
        # B_{h}
        g = grad_L_fn(x)
        u, dk = _b_step(u, g, half, d)
        W_k = W_k + (d - 1) * dk
        # O
        u = _o_step(u, k2, d, c1)
        return (x, u, g, W_k, key)

    W_k = jnp.float32(0.)
    x, u, grad_new, W_k, _ = jax.lax.fori_loop(
        0, n_steps, one_step, (x, u, grad_L, W_k, key))

    L_new = L_fn(x)
    W = (L_new - L_x) + W_k
    return x, u, grad_new, L_new, W


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised MAMS kernel (one step over all chains)
# ──────────────────────────────────────────────────────────────────────────────

def _mams_step(state, L_fn, grad_L_fn, eps, n_steps, d, key, c1=1.0):
    """One MAMS iteration for all chains in parallel.

    state : (x, grad_L, L_x)
        x      : (n_chains, D)
        grad_L : (n_chains, D)
        L_x    : (n_chains,)
    c1    : Langevin damping.  c1=1.0 → deterministic, c1<1.0 → OBABO.
            c1 = exp(-eps / L_partial).
    """
    x, grad_L, L_x = state
    N = x.shape[0]
    key_u, key_a, key_traj = jax.random.split(key, 3)

    # Fresh velocity on S^{d-1} (always full refreshment between trajectories)
    z = jax.random.normal(key_u, x.shape)
    u = z / jnp.sqrt(jnp.sum(z**2, axis=1, keepdims=True) + 1e-30)

    # Per-chain PRNG keys for O steps inside trajectory
    traj_keys = jax.random.split(key_traj, N)

    # Trajectory (vmapped over chains)
    x_new, u_end, grad_new, L_new, W = jax.vmap(
        lambda xi, ui, gi, Li, ki: _mams_trajectory(
            xi, ui, gi, Li, L_fn, grad_L_fn, eps, n_steps, d, c1, ki)
    )(x, u, grad_L, L_x, traj_keys)

    # Accept / reject
    W = jnp.where(jnp.isnan(W), jnp.inf, W)
    log_alpha = -W
    log_u = jnp.log(jax.random.uniform(key_a, (N,), minval=1e-30))
    accept = log_u < log_alpha

    x_out    = jnp.where(accept[:, None], x_new, x)
    grad_out = jnp.where(accept[:, None], grad_new, grad_L)
    L_out    = jnp.where(accept, L_new, L_x)

    return (x_out, grad_out, L_out), accept, x_new, u, log_alpha


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

def _da_update(state, accept_rate, log_eps0, target, t0=10., gamma_da=0.05, kappa=0.75):
    it    = state.iteration + 1
    eta   = 1. / (it + t0)
    H_bar = (1. - eta) * state.H_bar + eta * (target - accept_rate)
    log_e = log_eps0 - jnp.sqrt(it) / ((it + t0) * gamma_da) * H_bar
    log_eb = it**(-kappa) * log_e + (1. - it**(-kappa)) * state.log_eps_bar
    return DAState(it, log_e, log_eb, H_bar)


# ──────────────────────────────────────────────────────────────────────────────
# ChEES — integration length via ADAM ascent on trajectory quality
# ──────────────────────────────────────────────────────────────────────────────

class ChEESState(NamedTuple):
    log_T: float;  log_T_bar: float
    m: float;      v: float
    iteration: int; halton: float

def _chees_init(eps, L):
    T = eps * L
    return ChEESState(jnp.log(T), jnp.log(T), 0., 0., 1, _halton(1))

def _chees_update(state, log_alpha, pos_cur, pos_pro, vel, metric="euclidean",
                  lr=0.025, beta1=0., beta2=0.95, reg=1e-7,
                  T_min=0.25, T_max=10., T_interp=0.9):
    """ChEES gradient for vanilla (non-ensemble) MAMS.

    pos_cur, pos_pro : (n_chains, D)
    vel              : (n_chains, D)   — unit-sphere velocity at proposal
    log_alpha        : (n_chains,)     — log acceptance probability
    metric           : "affine-invariant" or "euclidean"
    """
    alpha = jnp.clip(jnp.exp(log_alpha), 0., 1.)
    c_cur = pos_cur - jnp.mean(pos_cur, axis=0)
    c_pro = pos_pro - jnp.mean(pos_pro, axis=0)

    if metric == "affine-invariant":
        cov  = jnp.atleast_2d(jnp.cov(pos_cur, rowvar=False))
        N    = c_cur.shape[0]
        sol  = jnp.linalg.solve(cov, jnp.concatenate([c_cur.T, c_pro.T], axis=1))
        Sc   = sol[:, :N].T;   Sp = sol[:, N:].T
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
# Integrated autocorrelation time (batch-means estimator)
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_tau(samples, n_batches=20):
    """Batch-means estimator of integrated autocorrelation time.

    Args:
        samples : (n_steps, n_chains, D)
    Returns:
        scalar — harmonic average of tau across dimensions,
                 averaged across chains.
    """
    n, C, D = samples.shape
    bs = n // n_batches
    if bs < 2:
        return 1.
    samples = samples[:bs * n_batches]
    batches = samples.reshape(n_batches, bs, C, D)
    var_batch = jnp.var(jnp.mean(batches, axis=1), axis=0, ddof=1)  # (C, D)
    var_total = jnp.var(samples, axis=0)                              # (C, D)
    tau = bs * var_batch / (var_total + 1e-30)                        # (C, D)
    tau = jnp.clip(tau, 1., None)
    tau_chain = D / jnp.sum(1. / tau, axis=1)       # harmonic avg over dims → (C,)
    return float(jnp.mean(tau_chain))                # mean over chains → scalar


# ──────────────────────────────────────────────────────────────────────────────
# Halton sequence (trajectory-length jitter)
# ──────────────────────────────────────────────────────────────────────────────

@jax.jit
def _halton(n, base=2):
    i, b = jnp.asarray(n, jnp.int32), jnp.asarray(base, jnp.int32)
    def body(s):
        i, f, r = s
        f = f / jnp.float32(b)
        return i // b, f, r + f * jnp.mod(i, b)
    _, _, r = jax.lax.while_loop(lambda s: s[0] > 0, body, (i, 1., 0.))
    return r


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (~80% acceptance with n_steps=1)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, x, L_fn, grad_L_fn, eps0, d, n_steps):
    fi = jnp.finfo(jnp.result_type(eps0))

    grad_L = jax.vmap(grad_L_fn)(x)
    L_x    = jax.vmap(L_fn)(x)
    state0 = (x, grad_L, L_x)

    def body(s):
        eps, _, direction, k = s
        k, ks = jax.random.split(k)
        eps = (2.**direction) * eps
        _, accept, _, _, _ = _mams_step(state0, L_fn, grad_L_fn, eps, n_steps, d, ks)
        avg = jnp.mean(accept.astype(float))
        return eps, direction, jnp.where(avg > 0.8, 1, -1), k

    def cond(s):
        eps, ld, direction, _ = s
        return (((eps > fi.tiny) | (direction >= 0))
                & ((eps < fi.max) | (direction <= 0))
                & ((ld == 0) | (direction == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps / 4.


# ──────────────────────────────────────────────────────────────────────────────
# sampler_mams
# ──────────────────────────────────────────────────────────────────────────────

def sampler_mams(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup           = 1000,
    step_size        = 0.05,
    L                = "auto",
    tune             = "chees",
    thin_by          = 1,
    target_accept    = 0.65,
    jitter           = 0.6,
    max_L            = 20,
    chees_metric     = "affine-invariant",
    langevin         = None,
    grad_log_prob_fn = None,
    seed             = 0,
    verbose          = True,
    find_init_step_size = True,
    adapt_step_size     = True,
):
    """
    Metropolis Adjusted Microcanonical Sampler (MAMS).

    Args:
        log_prob_fn      : (D,) -> scalar.  Log density (single point).
        initial_state    : (n_chains, D).
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations (dual averaging for step size).
        step_size        : Initial step size (adapted during warmup).
        L                : Number of leapfrog steps per trajectory.
                           "auto" (default) → initialise at ceil(sqrt(D)),
                           then tune via ChEES or autocorrelation.
        tune             : "chees" (default) or "tau".
                           "chees" — joint DA + ChEES (eps and L adapted together).
                           "tau"   — 3-stage: DA → autocorrelation L → re-DA.
        thin_by          : Keep every thin_by-th sample.
        target_accept    : Target acceptance rate (default 0.65).
        jitter           : Halton jitter fraction for trajectory length (default 0.6).
        max_L            : Maximum trajectory length (default 100).
        chees_metric     : "affine-invariant" (default) or "euclidean".
        langevin         : Ratio L_partial / (L * eps) for OBABO partial refreshment.
                           None (default) → deterministic BAB integrator.
                           A positive float → OBABO with c1 = exp(-1 / (langevin * L)).
                           Paper recommends ~1.25.
        grad_log_prob_fn : (D,) -> (D,).  Gradient of log density.
                           If None, uses jax.grad(log_prob_fn).
        seed             : Integer random seed.
        verbose          : Print progress.
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` to
                              ~80% acceptance before warmup.
                              If False, use `step_size` as-is.
        adapt_step_size  : If True (default), dual-averaging adapts step size
                           during warmup (across all tune stages).
                           If False, uses `step_size` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_step_size, L)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert dim >= 2, "MAMS requires dimension >= 2"
    _metric = chees_metric

    tune_L = (L == "auto")
    if tune_L:
        L = max(1, int(jnp.ceil(jnp.sqrt(dim))))

    L_fn = lambda x: -log_prob_fn(x)
    if grad_log_prob_fn is None:
        grad_L_fn = jax.grad(lambda x: -log_prob_fn(x))
    else:
        grad_L_fn = lambda x: -grad_log_prob_fn(x)

    key = jax.random.key(seed)

    # --- initialise ---
    grad_L = jax.vmap(grad_L_fn)(state)
    L_x    = jax.vmap(L_fn)(state)
    mams_state = (state, grad_L, L_x)

    # --- initial step size ---
    step_size = jnp.asarray(step_size, jnp.float32)
    if find_init_step_size:
        key, k = jax.random.split(key)
        step_size = _find_init_eps(k, state, L_fn, grad_L_fn, step_size, dim, L)

    # --- compute c1 for OBABO ---
    # c1 = exp(-eps / L_partial) = exp(-eps / (langevin * L * eps)) = exp(-1 / (langevin * L))
    _use_langevin = langevin is not None
    def _c1_from_L(cur_L):
        if _use_langevin:
            return jnp.exp(-1. / (langevin * cur_L))
        return 1.0

    if verbose:
        print(f"MAMS  tune={tune}"
              + (f"  metric={_metric}" if tune == 'chees' and tune_L else "")
              + (f"  langevin={langevin}" if _use_langevin else "")
              + f"  L={L}{'(auto)' if tune_L else ''}"
              f"  init_eps={float(step_size):.4f}")

    log_eps0 = jnp.log(step_size)
    da = _da_init(log_eps0)

    use_chees = (tune == "chees") and tune_L

    if use_chees:
        # ── Joint DA + ChEES warmup (two phases: explore, then refine) ────
        ch = _chees_init(step_size, L)
        n_phase1 = warmup // 2
        n_phase2 = warmup - n_phase1

        @jax.jit
        def _warmup_chees(mams_state, da, ch, key, log_eps_anchor):
            eps = jnp.exp(da.log_eps) if adapt_step_size else step_size
            cur_L = _chees_L(ch, eps, jitter=jitter, max_L=max_L)
            cur_c1 = _c1_from_L(cur_L)
            new_state, accept, x_pro, vel, log_alpha = _mams_step(
                mams_state, L_fn, grad_L_fn, eps, cur_L, dim, key, cur_c1)
            acc_rate = jnp.mean(accept.astype(float))
            if adapt_step_size:
                da = _da_update(da, acc_rate, log_eps_anchor, target_accept)
            ch  = _chees_update(ch, log_alpha, mams_state[0], x_pro, vel, _metric)
            return new_state, da, ch, acc_rate

        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, warmup)

        # Phase 1: burn in with init eps anchor
        total_acc = 0.
        for i in range(n_phase1):
            mams_state, da, ch, acc = _warmup_chees(
                mams_state, da, ch, wkeys[i], log_eps0)
            total_acc += acc

        # Reset DA anchor to current estimate (chains now near typical set)
        if adapt_step_size:
            log_eps_mid = da.log_eps_bar
            da = _da_init(log_eps_mid)
        else:
            log_eps_mid = log_eps0

        # Phase 2: refine from updated anchor
        for i in range(n_phase1, warmup):
            mams_state, da, ch, acc = _warmup_chees(
                mams_state, da, ch, wkeys[i], log_eps_mid)
            total_acc += acc

        final_eps = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
        final_L   = int(_chees_L(ch, final_eps, bar=True, max_L=max_L))
        if verbose:
            print(f"Warmup done.  eps={float(final_eps):.4f}  L={final_L}"
                  f"  accept={float(total_acc)/max(warmup,1):.3f}")

        halton_offset = ch.iteration
        final_log_T   = ch.log_T_bar

    else:
        # ── 3-stage tau warmup (original) ─────────────────────────────────
        if tune_L:
            n_eps1    = warmup // 2
            n_L_tune  = warmup // 4
            n_eps2    = warmup - n_eps1 - n_L_tune
        else:
            n_eps1, n_L_tune, n_eps2 = warmup, 0, 0

        @jax.jit
        def _warmup_step(mams_state, da, key, n_steps):
            eps = jnp.exp(da.log_eps) if adapt_step_size else step_size
            cur_c1 = _c1_from_L(n_steps)
            new_state, accept, _, _, _ = _mams_step(
                mams_state, L_fn, grad_L_fn, eps, n_steps, dim, key, cur_c1)
            acc_rate = jnp.mean(accept.astype(float))
            if adapt_step_size:
                da = _da_update(da, acc_rate, log_eps0, target_accept)
            return new_state, da, acc_rate

        key, k = jax.random.split(key)
        wkeys = jax.random.split(k, n_eps1)
        total_acc = 0.
        for i in range(n_eps1):
            mams_state, da, acc = _warmup_step(mams_state, da, wkeys[i], L)
            total_acc += acc

        final_eps = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
        if verbose:
            print(f"Stage 1 done.  eps={float(final_eps):.4f}"
                  f"  accept={float(total_acc)/max(n_eps1,1):.3f}")

        # Stage 2: trajectory-length tuning via autocorrelation
        if tune_L and n_L_tune >= 40:
            @jax.jit
            def _tune_step(mams_state, key):
                new_state, accept, _, _, _ = _mams_step(
                    mams_state, L_fn, grad_L_fn, final_eps, L, dim, key, _c1_from_L(L))
                q = new_state[0]
                return new_state, (q, accept.astype(float))

            key, k = jax.random.split(key)
            tkeys = jax.random.split(k, n_L_tune)
            mams_state, (tune_q, tune_acc) = jax.lax.scan(
                _tune_step, mams_state, tkeys)

            tau = _estimate_tau(tune_q)
            L_new = max(1, int(round(0.3 * L * tau)))
            if verbose:
                print(f"Stage 2 done.  tau_int={tau:.2f}"
                      f"  L: {L} -> {L_new}"
                      f"  accept={float(jnp.mean(tune_acc)):.3f}")
            L = L_new

        # Stage 3: re-tune eps for final L
        if n_eps2 > 0:
            if adapt_step_size:
                log_eps0 = jnp.log(final_eps)
                da = _da_init(log_eps0)

            key, k = jax.random.split(key)
            wkeys2 = jax.random.split(k, n_eps2)
            total_acc2 = 0.
            for i in range(n_eps2):
                mams_state, da, acc = _warmup_step(mams_state, da, wkeys2[i], L)
                total_acc2 += acc

            final_eps = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
            if verbose:
                print(f"Stage 3 done.  eps={float(final_eps):.4f}"
                      f"  accept={float(total_acc2)/max(n_eps2,1):.3f}")

        final_L = L

    # ── production ──────────────────────────────────────────────────────────
    if use_chees:
        @jax.jit
        def _step(carry, key):
            mams_state, step_i = carry
            h = _halton(halton_offset + step_i)
            T = jnp.exp(final_log_T)
            T_jit = (1 - jitter) * T + jitter * h * T
            cur_L = jnp.clip(jnp.ceil(T_jit / final_eps), 1, max_L).astype(int)
            cur_c1 = _c1_from_L(cur_L)
            new_state, accept, _, _, _ = _mams_step(
                mams_state, L_fn, grad_L_fn, final_eps, cur_L, dim, key, cur_c1)
            q = new_state[0]
            return (new_state, step_i + 1), (q, accept.astype(float))
    else:
        @jax.jit
        def _step(carry, key):
            mams_state, step_i = carry
            h = _halton(step_i + 1)
            L_jit = jnp.clip(
                jnp.ceil(((1 - jitter) + jitter * h) * final_L), 1, 2 * final_L
            ).astype(int)
            cur_c1 = _c1_from_L(L_jit)
            new_state, accept, _, _, _ = _mams_step(
                mams_state, L_fn, grad_L_fn, final_eps, L_jit, dim, key, cur_c1)
            q = new_state[0]
            return (new_state, step_i + 1), (q, accept.astype(float))

    key, k = jax.random.split(key)
    skeys = jax.random.split(k, num_samples * thin_by)
    (mams_state, _), (all_q, all_acc) = jax.lax.scan(
        _step, (mams_state, jnp.int32(0)), skeys)

    samples = all_q[::thin_by]
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_step_size=float(final_eps),
                L=final_L)
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}")
    return samples, info


if __name__ == "__main__":

    def _run_test(name, log_prob, init, num_samples, warmup, seed, tune_modes,
                  true_var=None, rosen=False, a_ros=None, langevin_list=None):
        print("=" * 60)
        print(name)
        print("=" * 60)
        if langevin_list is None:
            langevin_list = [None]
        for tm in tune_modes:
            for cm in (["affine-invariant", "euclidean"] if tm == "chees" else [None]):
                for _lang in langevin_list:
                    label = f"tune={tm}" + (f", metric={cm}" if cm else "")
                    if _lang is not None:
                        label += f", langevin={_lang}"
                    else:
                        label += ", deterministic"
                    print(f"\n  {label}:")
                    kw = dict(tune=tm, langevin=_lang)
                    if cm is not None:
                        kw["chees_metric"] = cm
                    samples, info = sampler_mams(log_prob, init, num_samples=num_samples,
                                                 warmup=warmup, seed=seed, **kw)
                    flat = samples.reshape(-1, init.shape[1])
                    if rosen:
                        me = jnp.mean(flat[:, ::2]);  mo = jnp.mean(flat[:, 1::2])
                        ve = jnp.mean(jnp.var(flat[:, ::2], axis=0))
                        vo = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
                        print(f"    x_even: mean={me:.3f}  var={ve:.4f}  (target: mean={a_ros}, var=0.5)")
                        print(f"    x_odd:  mean={mo:.3f}  var={vo:.4f}  (target: mean=1.5, var~2.505)")
                    elif true_var is not None:
                        var_est = jnp.var(flat, axis=0)
                        rel_err = jnp.mean(jnp.abs(var_est - true_var) / true_var)
                        print(f"    mean_rel_err(var)={rel_err:.3f}"
                              f"  var_range=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
                              f"  (target: [{jnp.min(true_var):.2f}, {jnp.max(true_var):.2f}])")
                    else:
                        print(f"    mean={jnp.mean(flat):.3f}  var={jnp.mean(jnp.var(flat, axis=0)):.3f}"
                              f"  (target: mean=0, var=1)")
                    print(f"    info: {info}")

    # --- Test 1: Ill-conditioned Gaussian (20D, kappa=1000) ---
    dim = 20;  kappa = 1000.
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov_gauss = Q @ jnp.diag(eigvals) @ Q.T
    prec_gauss = Q @ jnp.diag(1. / eigvals) @ Q.T
    def log_prob_gauss(x):
        return -0.5 * jnp.dot(x, prec_gauss @ x)
    init = jax.random.normal(jax.random.key(42), (50, dim))
    _run_test("Test 1: Ill-conditioned Gaussian  (D=20, kappa=1000)",
              log_prob_gauss, init, 5000, 1000, 123, ["chees", "tau"],
              true_var=jnp.diag(cov_gauss))

    # --- Test 2: Rosenbrock (10D) ---
    print()
    a_ros, b_ros = 1.0, 100.0;  dim_ros = 20
    def log_prob_rosen(x):
        return -(b_ros * jnp.sum((x[1::2] - x[::2]**2)**2) + jnp.sum((x[::2] - a_ros)**2))
    init_r = jax.random.normal(jax.random.key(42), (50, dim_ros))
    _run_test("Test 2: Rosenbrock  (D=20, a=1, b=100)",
              log_prob_rosen, init_r, 5000, 1000, 123, ["chees", "tau"],
              rosen=True, a_ros=a_ros, langevin_list=[None, 1.0, 1.25])

    # --- Test 3: Standard Gaussian (10D) ---
    print()
    dim_s = 10
    def log_prob_simple(x):
        return -0.5 * jnp.sum(x**2)
    init_s = jax.random.normal(jax.random.key(42), (30, dim_s))
    _run_test("Test 3: Standard Gaussian  (D=10)",
              log_prob_simple, init_s, 2000, 500, 42, ["chees", "tau"])
