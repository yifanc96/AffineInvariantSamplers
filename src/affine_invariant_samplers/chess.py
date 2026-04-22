"""
sampler_chess — standard ChEES HMC (no ensemble preconditioning), single file, JAX only.

Standard HMC with identity mass matrix.
Adaptation (warmup only, each independently toggleable):
  Dual averaging       → step size             (adapt_step_size)
  ChEES criterion      → integration length    (adapt_L)
  Heuristic line search→ initial step size     (find_init_step_size)

Setting all three flags to False reduces warmup to a plain burn-in at the
user-supplied (step_size, L).

Reference: https://proceedings.mlr.press/v130/hoffman21a.html
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Leapfrog integrator  (standard, identity mass matrix)
# ──────────────────────────────────────────────────────────────────────────────

def _leapfrog(q, p, grad_U, eps, L):
    """q:(N,D)  p:(N,D) — returns (q', p', velocity).  velocity=p for M=I."""
    g = grad_U(q)
    p = p - 0.5 * eps * g
    def step(_, s):
        q, p = s
        q = q + eps * p
        p = p - eps * grad_U(q)
        return q, p
    q, p = jax.lax.fori_loop(0, L - 1, step, (q, p))
    q = q + eps * p
    p = p - 0.5 * eps * grad_U(q)
    return q, p, p              # velocity = p for identity mass


# ──────────────────────────────────────────────────────────────────────────────
# HMC move — return (proposed, log_alpha, velocity, proposed_lp)
# ──────────────────────────────────────────────────────────────────────────────

def _hmc_move(positions, eps, key, log_prob, grad_U, L, lp_cur):
    """Standard HMC move for all chains.  p ~ N(0, I)."""
    N, D = positions.shape
    p0 = jax.random.normal(key, (N, D))
    proposed, p1, vel = _leapfrog(positions, p0, grad_U, eps, L)
    lp1 = log_prob(proposed)
    dH  = (-lp1 + 0.5 * jnp.sum(p1**2, axis=1)) \
        - (-lp_cur + 0.5 * jnp.sum(p0**2, axis=1))
    dH  = jnp.where(jnp.isnan(dH), jnp.inf, dH)
    return proposed, jnp.minimum(0., -dH), vel, lp1


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

def _chees_update(state, log_alpha, pos_cur, pos_pro, vel,
                  lr=0.025, beta1=0., beta2=0.95, reg=1e-7,
                  T_min=0.25, T_max=10., T_interp=0.9):
    """
    ChEES gradient signal (euclidean metric):
      g = t_n * diff_sq * inner
      diff_sq = ||c_pro||^2 - ||c_cur||^2     (change in centered second moment)
      inner   = c_pro^T vel                    (alignment with velocity)
    """
    alpha = jnp.clip(jnp.exp(log_alpha), 0., 1.)
    c_cur = pos_cur - jnp.mean(pos_cur, axis=0)
    c_pro = pos_pro - jnp.mean(pos_pro, axis=0)

    diff_sq = jnp.sum(c_pro**2, 1) - jnp.sum(c_cur**2, 1)
    inner   = jnp.sum(c_pro * vel, 1)

    g_m = state.halton * jnp.exp(state.log_T) * diff_sq * inner
    g_m = jnp.where((alpha > 1e-4) & jnp.isfinite(g_m), g_m, 0.)
    g   = jnp.sum(alpha * g_m) / (jnp.sum(alpha) + reg)

    it = state.iteration + 1
    m  = beta1 * state.m + (1 - beta1) * g
    v  = beta2 * state.v + (1 - beta2) * g**2
    delta = lr * (m / (1 - beta1**it)) / jnp.sqrt(v / (1 - beta2**it) + reg)
    delta = jnp.clip(delta, -0.35, 0.35)       # per-step clip (TFP/BlackJAX)
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

def _find_init_eps(key, positions, log_prob, grad_U, eps0):
    lp = log_prob(positions)
    fi = jnp.finfo(jnp.result_type(eps0))

    def body(s):
        eps, _, d, k = s
        k, k1 = jax.random.split(k)
        eps = (2.**d) * eps
        _, la, _, _ = _hmc_move(positions, eps, k1, log_prob, grad_U, 1, lp)
        avg = jnp.log(la.shape[0]) - jax.scipy.special.logsumexp(-la)
        return eps, d, jnp.where(jnp.log(.8) < avg, 1, -1), k

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps / 2.


# ──────────────────────────────────────────────────────────────────────────────
# sampler_chess
# ──────────────────────────────────────────────────────────────────────────────

def sampler_chess(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup           = 1000,
    step_size        = 0.1,
    L                = 5,
    max_L            = 20,
    thin_by          = 1,
    target_accept    = 0.651,
    grad_log_prob_fn    = None,
    find_init_step_size = True,
    adapt_step_size     = True,
    adapt_L             = True,
    seed                = 0,
    verbose             = True,
):
    """
    Standard ChEES HMC with automatic step-size and integration-length tuning.
    No ensemble preconditioning — identity mass matrix, p ~ N(0, I).

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations.
        step_size        : Initial step size (adapted during warmup).
        L                : Initial leapfrog steps (adapted during warmup).
        max_L            : Maximum leapfrog steps (default 100).
        thin_by          : Keep every thin_by-th sample.
        target_accept    : Target acceptance rate for dual averaging.
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
                           If None, uses jax.vmap(jax.grad(log_prob_fn)).
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` to
                              ~80% acceptance before warmup.
                              If False, use `step_size` as-is.
        adapt_step_size  : If True, tune step size by dual averaging during
                           warmup.  If False, use `step_size` as given.
        adapt_L          : If True, tune integration length by ChEES during
                           warmup.  If False, use `L` as given.
        seed             : Integer random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_step_size, final_L)
    """
    state    = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert 1 <= L <= max_L, f"Need 1 <= L <= max_L (got L={L}, max_L={max_L})"

    if grad_log_prob_fn is None:
        grad_U = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_U = grad_log_prob_fn
    _grad_U = lambda x: -grad_U(x)

    key = jax.random.key(seed)

    # --- initial step size ---
    if find_init_step_size:
        _user_h = float(step_size)
        key, k = jax.random.split(key)
        step_size = _find_init_eps(k, state, log_prob_fn, _grad_U, step_size)
        if verbose:
            print(f"[chess] find_init_step_size: step_size {_user_h:.4g} → "
                  f"{float(step_size):.4g}\n"
                  f"   (if the chain later stalls, set find_init_step_size=False "
                  f"and pass your own step_size — the heuristic can overshoot "
                  f"when the initial positions are under-dispersed vs the target.)")
    step_size = jnp.asarray(step_size)
    if verbose:
        print(f"metric=euclidean  init_eps={float(step_size):.4f}"
              f"  find_init_step_size={find_init_step_size}"
              f"  adapt_step_size={adapt_step_size}  adapt_L={adapt_L}")

    log_eps0 = jnp.log(step_size)
    da  = _da_init(log_eps0)
    ch  = _chees_init(step_size, L)
    lp  = log_prob_fn(state)

    # Closure-captured Python bools resolve at trace time, so unused
    # adaptation branches are removed from the compiled graph entirely.
    fixed_L = jnp.int32(L)

    # --- warmup ---
    @jax.jit
    def _warmup_step(positions, lp, da, ch, keys):
        k1, ka = keys
        eps   = jnp.exp(da.log_eps) if adapt_step_size else step_size
        cur_L = _chees_L(ch, eps, max_L=max_L) if adapt_L else fixed_L

        proposed, la, vel, lp_new = _hmc_move(
            positions, eps, k1, log_prob_fn, _grad_U, cur_L, lp)
        if adapt_step_size:
            da = _da_update(da, la, log_eps0, target_accept)
        if adapt_L:
            ch = _chees_update(ch, la, positions, proposed, vel)
        positions, accept = _mh(positions, proposed, la, ka)
        lp = jnp.where(accept, lp_new, lp)

        acc = jnp.mean(accept.astype(float))
        return positions, lp, da, ch, acc

    key, k = jax.random.split(key)
    flat  = jax.random.split(k, warmup * 2)
    wkeys = flat.reshape(warmup, 2, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        state, lp, da, ch, acc = _warmup_step(state, lp, da, ch, wkeys[i])
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

    # --- main sampling (jitter L each step via Halton, standard ChEES) ---
    jitter = 0.6
    halton_offset = ch.iteration

    @jax.jit
    def _step(carry, keys):
        positions, lp, step_i = carry
        k1, ka = keys
        if adapt_L:
            h = _halton(halton_offset + step_i)
            T = jnp.exp(final_log_T)
            T_jit = (1 - jitter) * T + jitter * h * T
            Lc = jnp.clip(jnp.ceil(T_jit / final_eps), 1, max_L).astype(int)
        else:
            Lc = fixed_L
        proposed, la, _, lp_new = _hmc_move(
            positions, final_eps, k1, log_prob_fn, _grad_U, Lc, lp)
        positions, accept = _mh(positions, proposed, la, ka)
        lp = jnp.where(accept, lp_new, lp)
        return (positions, lp, step_i + 1), (positions, accept.astype(float))

    key, k  = jax.random.split(key)
    flat    = jax.random.split(k, num_samples * thin_by * 2)
    skeys   = flat.reshape(num_samples * thin_by, 2, *flat.shape[1:])
    (state, lp, _), (all_states, all_acc) = jax.lax.scan(
        _step, (state, lp, jnp.int32(0)), skeys)

    samples = all_states[::thin_by]
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_step_size=float(final_eps),
                nominal_L=int(nominal_L))
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}")
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim  = 2
    cov  = jnp.array([[1., .95], [.95, 1.]])
    prec = jnp.linalg.inv(cov)
    def log_prob(x):
        return -0.5 * jnp.sum(x @ prec * x, axis=-1)
    chains = 20
    init = jax.random.normal(jax.random.key(42), (chains, dim))

    print("=" * 60)
    print("ChEES HMC")
    print("=" * 60)
    samples, info = sampler_chess(log_prob, init, num_samples=2000, warmup=500, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
    print(f"info : {info}")
