"""
sampler_stretch — affine-invariant stretch move (Goodman & Weare 2010), JAX only.

Each walker is updated by stretching toward a random complementary walker.
Proposal:  x' = x_j + z * (x_i - x_j),  z ~ g(z) on [1/a, a].

Adaptation (warmup only, toggleable):
  Dual averaging → stretch parameter a    (adapt_a)

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
# Stretch proposal:  z ~ ((a-1)*U + 1)^2 / a,  U ~ Uniform(0,1)
# ──────────────────────────────────────────────────────────────────────────────

def _stretch_proposal(active, complement, key, a):
    """Propose for all active walkers by stretching toward a random complement walker."""
    W, D = active.shape
    Wc = complement.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)

    # pick one random complement walker per active walker
    idx = jax.random.randint(k1, (W,), 0, Wc)
    partner = complement[idx]

    # stretch factor z ~ g(z) = 1/(2(a-1)) * 1/sqrt(z) on [1/a, a]
    u = jax.random.uniform(k2, (W,))
    z = ((a - 1.) * u + 1.) ** 2 / a          # (W,)

    proposal = partner + z[:, None] * (active - partner)
    return proposal, z


# ──────────────────────────────────────────────────────────────────────────────
# sampler_stretch
# ──────────────────────────────────────────────────────────────────────────────

def sampler_stretch(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    a             = 2.0,
    target_accept = 0.40,
    thin_by       = 1,
    seed          = 0,
    verbose       = True,
    adapt_a       = True,
):
    """
    Affine-invariant stretch move ensemble sampler with dual averaging.

    Args:
        log_prob_fn   : (batch, D) -> (batch,).  Vectorised log density.
        initial_state : (n_chains, D).  n_chains must be even and >= 4.
        num_samples   : Samples to collect.
        warmup        : Warmup iterations for parameter adaptation.
        a             : Initial stretch parameter (adapted during warmup).
        target_accept : Target acceptance for DA (default 0.40).
        thin_by       : Thinning factor.
        seed          : Random seed.
        verbose       : Print progress.
        adapt_a       : If True (default), dual-averaging adapts stretch parameter `a`
                        during warmup. If False, uses `a` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_a)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    lp1, lp2 = log_prob_fn(g1), log_prob_fn(g2)
    key = jax.random.key(seed)

    a = jnp.asarray(a, jnp.float32)
    log_a0 = jnp.log(a)
    da = _da_init(log_a0)

    # --- warmup ---
    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, keys):
        k1, k2, ka1, ka2 = keys
        a_cur = jnp.exp(da.log_h) if adapt_a else a

        prop1, z1 = _stretch_proposal(g1, g2, k1, a_cur)
        lp_prop1 = log_prob_fn(prop1)
        la1 = (dim - 1) * jnp.log(z1) + lp_prop1 - lp1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1
        g1 = jnp.where(acc1[:, None], prop1, g1)
        lp1 = jnp.where(acc1, lp_prop1, lp1)

        prop2, z2 = _stretch_proposal(g2, g1, k2, a_cur)
        lp_prop2 = log_prob_fn(prop2)
        la2 = (dim - 1) * jnp.log(z2) + lp_prop2 - lp2
        acc2 = jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2
        g2 = jnp.where(acc2[:, None], prop2, g2)
        lp2 = jnp.where(acc2, lp_prop2, lp2)

        rate = (jnp.mean(acc1.astype(float)) + jnp.mean(acc2.astype(float))) / 2
        if adapt_a:
            da = _da_update(da, rate, log_a0, target_accept)
        return g1, g2, lp1, lp2, da, rate

    key, k_ = jax.random.split(key)
    flat = jax.random.split(k_, warmup * 4)
    wkeys = flat.reshape(warmup, 4, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, rate = _warmup_step(g1, g2, lp1, lp2, da, wkeys[i])
        total_acc += rate

    final_a = jnp.exp(da.log_h_bar) if adapt_a else a
    if verbose:
        print(f"Stretch move:  a={float(final_a):.3f}"
              f"  warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    # --- production ---
    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2 = carry
        k1, k2, ka1, ka2 = keys

        prop1, z1 = _stretch_proposal(g1, g2, k1, final_a)
        lp_prop1 = log_prob_fn(prop1)
        la1 = (dim - 1) * jnp.log(z1) + lp_prop1 - lp1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1
        g1 = jnp.where(acc1[:, None], prop1, g1)
        lp1 = jnp.where(acc1, lp_prop1, lp1)

        prop2, z2 = _stretch_proposal(g2, g1, k2, final_a)
        lp_prop2 = log_prob_fn(prop2)
        la2 = (dim - 1) * jnp.log(z2) + lp_prop2 - lp2
        acc2 = jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2
        g2 = jnp.where(acc2[:, None], prop2, g2)
        lp2 = jnp.where(acc2, lp_prop2, lp2)

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
                final_a=float(final_a))
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
    samples, info = sampler_stretch(log_prob, init, num_samples=5000, warmup=500, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
