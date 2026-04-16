"""
sampler_walk — affine-invariant walk move with k-subset, JAX only.

Each walker proposes by a random linear combination of centered complement walkers.
Proposal:  x' = x + stepsize * z^T @ centered_complement,  z ~ N(0, I_k)

Adaptation (warmup only, toggleable):
  Heuristic initial step-size search   (find_init_step_size)
  Dual averaging → step size           (adapt_step_size)

Reference: https://arxiv.org/abs/2505.02987
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
# Walk proposal with k-subset
# ──────────────────────────────────────────────────────────────────────────────

def _walk_proposal(active, complement, key, stepsize, k):
    """Propose by random linear combination of k centered complement walkers."""
    W, D = active.shape
    Wc = complement.shape[0]
    k1, k2 = jax.random.split(key)

    # each active walker gets its own random k-subset
    idx = jnp.arange(Wc)
    subsets = jax.vmap(lambda rk: jax.random.choice(rk, idx, (k,), replace=False))(
        jax.random.split(k1, W))                                  # (W, k)
    selected = complement[subsets]                                 # (W, k, D)
    means = jnp.mean(selected, axis=1, keepdims=True)             # (W, 1, D)
    centered = (selected - means) / jnp.sqrt(k)                   # (W, k, D)

    z = jax.random.normal(k2, (W, k))                             # (W, k)
    proposal = active + stepsize * jnp.einsum('wk,wkd->wd', z, centered)
    return proposal


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (accept ≈ target_accept at initial walker positions)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, g1, g2, lp1, lp2, eps0, log_prob_fn, W, k, target_accept):
    fi = jnp.finfo(jnp.result_type(eps0))

    def body(s):
        eps, _, d, rk = s
        rk, k1, k2, ka1, ka2 = jax.random.split(rk, 5)
        eps = (2.**d) * eps

        prop1 = _walk_proposal(g1, g2, k1, eps, k)
        la1 = log_prob_fn(prop1) - lp1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1

        prop2 = _walk_proposal(g2, g1, k2, eps, k)
        la2 = log_prob_fn(prop2) - lp2
        acc2 = jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2

        avg = 0.5 * (jnp.mean(acc1.astype(float)) + jnp.mean(acc2.astype(float)))
        return eps, d, jnp.where(avg > target_accept, 1, -1), rk

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps


# ──────────────────────────────────────────────────────────────────────────────
# sampler_walk
# ──────────────────────────────────────────────────────────────────────────────

def sampler_walk(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    step_size     = 1.0,
    subset_size   = None,
    target_accept   = 0.50,
    thin_by         = 1,
    seed            = 0,
    verbose         = True,
    find_init_step_size = True,
    adapt_step_size     = True,
):
    """
    Affine-invariant walk move with k-subset and dual averaging step-size adaptation.

    Args:
        log_prob_fn   : (batch, D) -> (batch,).  Vectorised log density.
        initial_state : (n_chains, D).  n_chains must be even and >= 4.
        num_samples   : Post-warmup samples to return.
        warmup        : Warmup iterations for step-size adaptation.
        step_size     : Initial step size (adapted during warmup).
        subset_size   : Number of complement walkers to use (default: n_chains//2).
        target_accept : Target acceptance for dual averaging (default 0.50).
        thin_by       : Thinning factor.
        seed          : Random seed.
        verbose       : Print progress.
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` so that
                              mean acceptance ≈ `target_accept`.
                              If False, use `step_size` as-is.
        adapt_step_size : If True (default), dual-averaging adapts step size during warmup.
                          If False, uses `step_size` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_step_size)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0

    W = n_chains // 2
    k = subset_size if subset_size is not None else W
    assert 2 <= k <= W, f"subset_size must be in [2, {W}]"

    g1, g2 = state[:W], state[W:]
    lp1, lp2 = log_prob_fn(g1), log_prob_fn(g2)
    key = jax.random.key(seed)

    step_size = jnp.asarray(step_size, jnp.float32)
    if find_init_step_size:
        key, k_ = jax.random.split(key)
        step_size = _find_init_eps(k_, g1, g2, lp1, lp2, step_size,
                                    log_prob_fn, W, k, target_accept)
        if verbose:
            print(f"Walk move:  init_stepsize={float(step_size):.4f}")
    log_h0 = jnp.log(step_size)
    da = _da_init(log_h0)

    # --- warmup ---
    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, keys):
        k1, k2, ka1, ka2 = keys
        h = jnp.exp(da.log_h) if adapt_step_size else step_size

        prop1 = _walk_proposal(g1, g2, k1, h, k)
        lp_p1 = log_prob_fn(prop1)
        la1 = lp_p1 - lp1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1
        g1 = jnp.where(acc1[:, None], prop1, g1)
        lp1 = jnp.where(acc1, lp_p1, lp1)

        prop2 = _walk_proposal(g2, g1, k2, h, k)
        lp_p2 = log_prob_fn(prop2)
        la2 = lp_p2 - lp2
        acc2 = jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2
        g2 = jnp.where(acc2[:, None], prop2, g2)
        lp2 = jnp.where(acc2, lp_p2, lp2)

        rate = (jnp.mean(acc1.astype(float)) + jnp.mean(acc2.astype(float))) / 2
        if adapt_step_size:
            da = _da_update(da, rate, log_h0, target_accept)
        return g1, g2, lp1, lp2, da, rate

    key, k_ = jax.random.split(key)
    flat = jax.random.split(k_, warmup * 4)
    wkeys = flat.reshape(warmup, 4, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, rate = _warmup_step(g1, g2, lp1, lp2, da, wkeys[i])
        total_acc += rate

    final_h = jnp.exp(da.log_h_bar) if adapt_step_size else step_size
    if verbose:
        print(f"Walk move:  k={k}  stepsize={float(final_h):.4f}"
              f"  warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    # --- production ---
    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2 = carry
        k1, k2, ka1, ka2 = keys

        prop1 = _walk_proposal(g1, g2, k1, final_h, k)
        lp_p1 = log_prob_fn(prop1)
        la1 = lp_p1 - lp1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1
        g1 = jnp.where(acc1[:, None], prop1, g1)
        lp1 = jnp.where(acc1, lp_p1, lp1)

        prop2 = _walk_proposal(g2, g1, k2, final_h, k)
        lp_p2 = log_prob_fn(prop2)
        la2 = lp_p2 - lp2
        acc2 = jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2
        g2 = jnp.where(acc2[:, None], prop2, g2)
        lp2 = jnp.where(acc2, lp_p2, lp2)

        st = jnp.concatenate([g1, g2])
        acc = jnp.concatenate([acc1, acc2]).astype(float)
        return (g1, g2, lp1, lp2), (st, acc)

    key, k_ = jax.random.split(key)
    flat = jax.random.split(k_, num_samples * thin_by * 4)
    skeys = flat.reshape(num_samples * thin_by, 4, *flat.shape[1:])
    (g1, g2, lp1, lp2), (all_states, all_acc) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2), skeys)

    samples = all_states[::thin_by][:num_samples]
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_step_size=float(final_h))
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}")
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim = 2
    cov = jnp.array([[1., .95], [.95, 1.]])
    prec = jnp.linalg.inv(cov)
    def log_prob(x):
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)

    init = jax.random.normal(jax.random.key(42), (40, dim))
    samples, info = sampler_walk(log_prob, init, num_samples=5000, warmup=500, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
