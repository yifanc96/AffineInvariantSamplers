"""
sampler_side — affine-invariant side move (derivative-free), JAX only.

Each walker moves along a random difference direction from the complement.
Proposal:  x' = x + gamma/sqrt(D) * (x_r1 - x_r2) * N(0,1)

Adaptation (warmup only, toggleable):
  Heuristic initial scale search           (find_init_gamma)
  Dual averaging → scale parameter gamma   (adapt_gamma)

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
# Side proposal
# ──────────────────────────────────────────────────────────────────────────────

def _side_proposal(active, complement, key, gamma_scale):
    """Propose by moving along a random complement difference direction."""
    W, D = active.shape
    Wc = complement.shape[0]
    k1, k2 = jax.random.split(key)

    # pick two distinct complement walkers per active walker
    idx = jnp.arange(Wc)
    pairs = jax.vmap(lambda k: jax.random.choice(k, idx, (2,), replace=False))(
        jax.random.split(k1, W))
    diff = complement[pairs[:, 0]] - complement[pairs[:, 1]]   # (W, D)

    noise = jax.random.normal(k2, (W,))                        # scalar per walker
    proposal = active + gamma_scale * diff * noise[:, None]
    return proposal


# ──────────────────────────────────────────────────────────────────────────────
# Initial gamma-scale search (accept ≈ target_accept at initial walker positions)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_gs(key, g1, g2, lp1, lp2, gs0, log_prob_fn, W, target_accept):
    fi = jnp.finfo(jnp.result_type(gs0))

    def body(s):
        gs, _, d, rk = s
        rk, k1, k2, ka1, ka2 = jax.random.split(rk, 5)
        gs = (2.**d) * gs

        prop1 = _side_proposal(g1, g2, k1, gs)
        la1 = log_prob_fn(prop1) - lp1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1

        prop2 = _side_proposal(g2, g1, k2, gs)
        la2 = log_prob_fn(prop2) - lp2
        acc2 = jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2

        avg = 0.5 * (jnp.mean(acc1.astype(float)) + jnp.mean(acc2.astype(float)))
        return gs, d, jnp.where(avg > target_accept, 1, -1), rk

    def cond(s):
        gs, ld, d, _ = s
        return (((gs > fi.tiny) | (d >= 0)) & ((gs < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    gs, *_ = jax.lax.while_loop(cond, body, (gs0, 0, 0, key))
    return gs


# ──────────────────────────────────────────────────────────────────────────────
# sampler_side
# ──────────────────────────────────────────────────────────────────────────────

def sampler_side(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    gamma         = 1.6869,
    target_accept = 0.50,
    thin_by       = 1,
    seed          = 0,
    verbose       = True,
    find_init_gamma = True,
    adapt_gamma     = True,
):
    """
    Affine-invariant side move ensemble sampler with dual averaging.

    Args:
        log_prob_fn   : (batch, D) -> (batch,).  Vectorised log density.
        initial_state : (n_chains, D).  n_chains must be even and >= 4.
        num_samples   : Samples to collect.
        warmup        : Warmup iterations for parameter adaptation.
        gamma         : Initial scale parameter (adapted during warmup).
        target_accept : Target acceptance for DA (default 0.50).
        thin_by       : Thinning factor.
        seed          : Random seed.
        verbose       : Print progress.
        find_init_gamma : If True (default), run a short heuristic search at the
                          initial positions to scale `gamma` so that mean
                          acceptance ≈ `target_accept`.
                          If False, use `gamma` as-is.
        adapt_gamma   : If True (default), dual-averaging adapts scale parameter
                        `gamma` during warmup. If False, uses `gamma` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_gamma)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    lp1, lp2 = log_prob_fn(g1), log_prob_fn(g2)
    key = jax.random.key(seed)

    # DA tunes log(gamma_scale) where gamma_scale = gamma / sqrt(D)
    gamma_scale0 = jnp.asarray(gamma / jnp.sqrt(dim), jnp.float32)
    if find_init_gamma:
        key, k_ = jax.random.split(key)
        gamma_scale0 = _find_init_gs(k_, g1, g2, lp1, lp2, gamma_scale0,
                                      log_prob_fn, W, target_accept)
        if verbose:
            print(f"Side move:  init_gamma={float(gamma_scale0 * jnp.sqrt(dim)):.4f}")
    log_gs0 = jnp.log(gamma_scale0)
    da = _da_init(log_gs0)

    # --- warmup ---
    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, keys):
        k1, k2, ka1, ka2 = keys
        gs = jnp.exp(da.log_h) if adapt_gamma else gamma_scale0

        prop1 = _side_proposal(g1, g2, k1, gs)
        lp_p1 = log_prob_fn(prop1)
        la1 = lp_p1 - lp1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1
        g1 = jnp.where(acc1[:, None], prop1, g1)
        lp1 = jnp.where(acc1, lp_p1, lp1)

        prop2 = _side_proposal(g2, g1, k2, gs)
        lp_p2 = log_prob_fn(prop2)
        la2 = lp_p2 - lp2
        acc2 = jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2
        g2 = jnp.where(acc2[:, None], prop2, g2)
        lp2 = jnp.where(acc2, lp_p2, lp2)

        rate = (jnp.mean(acc1.astype(float)) + jnp.mean(acc2.astype(float))) / 2
        if adapt_gamma:
            da = _da_update(da, rate, log_gs0, target_accept)
        return g1, g2, lp1, lp2, da, rate

    key, k_ = jax.random.split(key)
    flat = jax.random.split(k_, warmup * 4)
    wkeys = flat.reshape(warmup, 4, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, rate = _warmup_step(g1, g2, lp1, lp2, da, wkeys[i])
        total_acc += rate

    final_gs = jnp.exp(da.log_h_bar) if adapt_gamma else gamma_scale0
    final_gamma = float(final_gs * jnp.sqrt(dim))
    if verbose:
        print(f"Side move:  gamma={final_gamma:.4f}"
              f"  warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    # --- production ---
    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2 = carry
        k1, k2, ka1, ka2 = keys

        prop1 = _side_proposal(g1, g2, k1, final_gs)
        lp_p1 = log_prob_fn(prop1)
        la1 = lp_p1 - lp1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1
        g1 = jnp.where(acc1[:, None], prop1, g1)
        lp1 = jnp.where(acc1, lp_p1, lp1)

        prop2 = _side_proposal(g2, g1, k2, final_gs)
        lp_p2 = log_prob_fn(prop2)
        la2 = lp_p2 - lp2
        acc2 = jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2
        g2 = jnp.where(acc2[:, None], prop2, g2)
        lp2 = jnp.where(acc2, lp_p2, lp2)

        state = jnp.concatenate([g1, g2])
        acc = jnp.concatenate([acc1, acc2]).astype(float)
        return (g1, g2, lp1, lp2), (state, acc)

    key, k = jax.random.split(key)
    flat = jax.random.split(k, num_samples * thin_by * 4)
    skeys = flat.reshape(num_samples * thin_by, 4, *flat.shape[1:])

    (g1, g2, lp1, lp2), (all_states, all_acc) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2), skeys)

    samples = all_states[::thin_by][:num_samples]

    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_gamma=final_gamma)
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
    samples, info = sampler_side(log_prob, init, num_samples=5000, warmup=500, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
