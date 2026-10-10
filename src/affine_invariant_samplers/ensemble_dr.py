"""
sampler_ensemble_dr — delayed rejection variants for stretch and side moves, JAX only.

Two-stage delayed rejection: on first rejection, retry with a shrunk scale.
The second-stage acceptance includes a correction term ensuring detailed balance.

Adaptation (warmup only, toggleable):
  Dual averaging → base parameter a (stretch variant; adapt_a)
  Heuristic initial scale search    (side variant; find_init_gamma)
  Dual averaging → base parameter gamma (side variant; adapt_gamma)

"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from ._dr_utils import _safe_log1m_exp


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
    log_h = log_h0 - jnp.sqrt(it) / gamma * H_bar
    log_hb = it**(-kappa) * log_h + (1. - it**(-kappa)) * state.log_h_bar
    return DAState(it, log_h, log_hb, H_bar)


# ──────────────────────────────────────────────────────────────────────────────
# Stretch helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stretch_z(key, a, W):
    """Sample stretch factor z ~ g(z) with parameter a."""
    u = jax.random.uniform(key, (W,))
    return ((a - 1.) * u + 1.) ** 2 / a


def _log_rejection_probability(log_acceptance):
    """Return log(1 - min(1, exp(log_acceptance))) without cancellation."""
    return _safe_log1m_exp(jnp.minimum(0., log_acceptance))


def _dr_stretch_log_acceptance(lp_active, lp1, lp2, z1, z2, dim, a):
    """Second-stage stretch ratio, conditional on the same partner in both stages.

    The ray's target density is r**(dim - 1) * pi(x).  The first-stage
    proposal density ratio q1(prop1 | prop2) / q1(prop1 | active) is
    z2**(-1/2), while the second-stage radial proposal is symmetric.
    """
    la1 = (dim - 1) * jnp.log(z1) + lp1 - lp_active
    z_rev = z1 / z2
    in_support = (z_rev >= 1. / a) & (z_rev <= a)
    la1_rev = (dim - 1) * jnp.log(z_rev) + lp1 - lp2
    la2 = ((dim - 1.5) * jnp.log(z2) + lp2 - lp_active
           + _log_rejection_probability(la1_rev)
           - _log_rejection_probability(la1))
    # Stage two is unreachable if stage one accepts with probability one.
    # A zero reverse proposal density forbids the move.  Positive infinity,
    # however, is a valid acceptance ratio and must remain accepted.
    valid = (la1 < 0.) & in_support & ~jnp.isnan(la2)
    return jnp.where(valid, la2, -jnp.inf)


def _dr_side_log_acceptance(lp_active, lp1, lp2, noise1, noise2, scale_ratio):
    """Second-stage side ratio, conditional on the shared difference direction."""
    la1 = lp1 - lp_active
    # Each first-stage Gaussian random walk is symmetric, so its MH ratio
    # contains only target densities.  Its density along the rejected path
    # still changes when the current position changes from active to prop2.
    la1_rev = lp1 - lp2
    noise_rev = noise1 - scale_ratio * noise2
    log_q1_ratio = -0.5 * (noise_rev**2 - noise1**2)
    la2 = (lp2 - lp_active + log_q1_ratio
           + _log_rejection_probability(la1_rev)
           - _log_rejection_probability(la1))
    valid = (la1 < 0.) & ~jnp.isnan(la2)
    return jnp.where(valid, la2, -jnp.inf)


# ──────────────────────────────────────────────────────────────────────────────
# DR Stretch move
# ──────────────────────────────────────────────────────────────────────────────

def sampler_ensemble_dr_stretch(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    a             = 2.0,
    shrink        = 0.5,
    target_accept = 0.40,
    thin_by       = 1,
    seed          = 0,
    verbose       = True,
    adapt_a       = True,
):
    """
    Two-stage delayed rejection stretch move with dual averaging.

    Args:
        log_prob_fn   : (batch, D) -> (batch,).  Vectorised log density.
        initial_state : (n_chains, D).  n_chains must be even and >= 4.
        num_samples   : Samples to collect.
        warmup        : Warmup iterations for parameter adaptation.
        a             : Initial stretch parameter. Must exceed 1 when adapt_a=True.
                        Fixed positive reciprocal parameters remain supported.
        shrink        : Multiplier for the second-stage stretch parameter:
                        a2 = a * shrink (default 0.5). a2 = 1 gives identity
                        proposals; a2 < 1 has the same width as 1 / a2.
        target_accept : Target stage-1 acceptance for DA (default 0.40).
        thin_by       : Thinning factor.
        seed          : Random seed.
        verbose       : Print progress.
        adapt_a       : If True (default), dual-averaging adapts stretch parameter `a`
                        during warmup. If False, uses `a` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, stage1_rate, stage2_rate, final_a)
    """
    # Match momentum/noise precision, including with integer or float32 starts.
    state = jnp.asarray(initial_state, dtype=jnp.float64
                        if jax.config.jax_enable_x64 else jnp.float32)
    if not bool(jnp.all(jnp.isfinite(state))):
        raise ValueError("initial_state must contain only finite values")
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    lp1, lp2 = log_prob_fn(g1), log_prob_fn(g2)
    key = jax.random.key(seed)

    a = jnp.asarray(a, jnp.float32)
    if not bool(jnp.isfinite(a)) or float(a) <= 0.:
        raise ValueError("a must be finite and positive")
    if adapt_a and float(a) <= 1.:
        raise ValueError("adapt_a=True requires a > 1; use adapt_a=False for a fixed reciprocal stretch parameter")
    # Adapt the positive width rather than a itself. DA can then shrink the
    # proposal toward identity without crossing a=1 and reversing feedback.
    log_a0 = jnp.log(a - 1.) if adapt_a else jnp.log(a)
    da = _da_init(log_a0)

    def _one_group(active, comp, lp_active, keys, a_cur, a2_cur):
        k_s1, k_s2, ka1, ka2 = keys

        # --- Stage 1: standard stretch ---
        k1a, k1b = jax.random.split(k_s1)
        idx = jax.random.randint(k1a, (W,), 0, W)
        partner = comp[idx]
        z1 = _stretch_z(k1b, a_cur, W)
        prop1 = partner + z1[:, None] * (active - partner)
        lp1 = log_prob_fn(prop1)
        la1 = (dim - 1) * jnp.log(z1) + lp1 - lp_active
        accept1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1

        # --- Stage 2: shrunk stretch (only matters for rejected walkers) ---
        k2a, k2b = jax.random.split(k_s2)
        z2 = _stretch_z(k2b, a2_cur, W)
        prop2 = partner + z2[:, None] * (active - partner)
        lp2 = log_prob_fn(prop2)

        la2 = _dr_stretch_log_acceptance(
            lp_active, lp1, lp2, z1, z2, dim, a_cur)
        accept2 = (~accept1) & (jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2)

        new_pos = jnp.where(accept1[:, None], prop1,
                  jnp.where(accept2[:, None], prop2, active))
        new_lp = jnp.where(accept1, lp1,
                 jnp.where(accept2, lp2, lp_active))
        return new_pos, new_lp, accept1, accept2

    # --- warmup ---
    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, keys):
        a_cur = (1. + jnp.maximum(jnp.exp(da.log_h), jnp.finfo(a.dtype).eps)
                 if adapt_a else a)
        a2_cur = a_cur * shrink

        g1, lp1, a1_s1, a1_s2 = _one_group(g1, g2, lp1, keys[:4], a_cur, a2_cur)
        g2, lp2, a2_s1, a2_s2 = _one_group(g2, g1, lp2, keys[4:], a_cur, a2_cur)

        # adapt based on stage-1 acceptance
        s1_rate = (jnp.mean(a1_s1.astype(float)) + jnp.mean(a2_s1.astype(float))) / 2
        if adapt_a:
            da = _da_update(da, s1_rate, log_a0, target_accept)
        overall = (jnp.mean((a1_s1 | a1_s2).astype(float)) +
                   jnp.mean((a2_s1 | a2_s2).astype(float))) / 2
        return g1, g2, lp1, lp2, da, overall

    key, k_ = jax.random.split(key)
    flat = jax.random.split(k_, warmup * 8)
    wkeys = flat.reshape(warmup, 8, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, rate = _warmup_step(g1, g2, lp1, lp2, da, wkeys[i])
        total_acc += rate

    final_a = (1. + jnp.maximum(jnp.exp(da.log_h_bar), jnp.finfo(a.dtype).eps)
               if adapt_a else a)
    final_a2 = final_a * shrink
    if verbose:
        print(f"DR Stretch:  a={float(final_a):.3f}  shrink={shrink}"
              f"  warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    # --- production ---
    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2 = carry

        g1, lp1, a1_s1, a1_s2 = _one_group(g1, g2, lp1, keys[:4], final_a, final_a2)
        g2, lp2, a2_s1, a2_s2 = _one_group(g2, g1, lp2, keys[4:], final_a, final_a2)

        st = jnp.concatenate([g1, g2])
        acc = jnp.concatenate([a1_s1 | a1_s2, a2_s1 | a2_s2]).astype(float)
        s1 = jnp.concatenate([a1_s1, a2_s1]).astype(float)
        s2 = jnp.concatenate([a1_s2, a2_s2]).astype(float)
        return (g1, g2, lp1, lp2), (st, acc, s1, s2)

    key, k = jax.random.split(key)
    flat = jax.random.split(k, num_samples * thin_by * 8)
    skeys = flat.reshape(num_samples * thin_by, 8, *flat.shape[1:])

    (g1, g2, lp1, lp2), (all_states, all_acc, all_s1, all_s2) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2), skeys)

    samples = all_states[::thin_by][:num_samples]

    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                stage1_rate=float(jnp.mean(all_s1)),
                stage2_rate=float(jnp.mean(all_s2)),
                final_a=float(final_a))
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}"
              f"  (stage1={info['stage1_rate']:.3f}, stage2={info['stage2_rate']:.3f})")
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# DR Side move
# ──────────────────────────────────────────────────────────────────────────────

def sampler_ensemble_dr_side(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    gamma         = 1.6869,
    shrink        = 0.5,
    target_accept = 0.50,
    thin_by       = 1,
    seed          = 0,
    verbose       = True,
    find_init_gamma = True,
    adapt_gamma     = True,
):
    """
    Two-stage delayed rejection side move with dual averaging.

    Args:
        log_prob_fn   : (batch, D) -> (batch,).  Vectorised log density.
        initial_state : (n_chains, D).  n_chains must be even and >= 4.
        num_samples   : Samples to collect.
        warmup        : Warmup iterations for parameter adaptation.
        gamma         : Initial scale parameter (adapted during warmup).
        shrink        : Shrink factor for second stage (default 0.5).
        target_accept : Target stage-1 acceptance for DA (default 0.50).
        thin_by       : Thinning factor.
        seed          : Random seed.
        verbose       : Print progress.
        find_init_gamma : If True (default), run a short heuristic search at the
                          initial positions to scale `gamma` so that stage-1
                          acceptance ≈ `target_accept`.
                          If False, use `gamma` as-is.
        adapt_gamma   : If True (default), dual-averaging adapts scale parameter
                        `gamma` during warmup. If False, uses `gamma` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, stage1_rate, stage2_rate, final_gamma)
    """
    if not (0.0 < float(gamma) < float("inf")):
        raise ValueError("gamma must be finite and strictly positive")

    # Match momentum/noise precision, including with integer or float32 starts.
    state = jnp.asarray(initial_state, dtype=jnp.float64
                        if jax.config.jax_enable_x64 else jnp.float32)
    if not bool(jnp.all(jnp.isfinite(state))):
        raise ValueError("initial_state must contain only finite values")
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    if find_init_gamma and (bool(jnp.all(g1 == g1[0])) or
                                 bool(jnp.all(g2 == g2[0]))):
        raise ValueError("initial scale search requires at least two distinct walkers in each half; "
                         "spread the initial ensemble or set find_init_gamma=False "
                         "and supply a finite positive gamma")
    lp1, lp2 = log_prob_fn(g1), log_prob_fn(g2)
    key = jax.random.key(seed)

    gamma_scale0 = jnp.asarray(gamma / jnp.sqrt(dim), jnp.float32)

    def _one_group(active, comp, lp_active, keys, gs_cur, gs2_cur):
        k_s1, k_s2, ka1, ka2 = keys
        k1a, k1b = jax.random.split(k_s1)

        # pick diff direction (shared between stages)
        idx = jnp.arange(W)
        pairs = jax.vmap(lambda rk: jax.random.choice(rk, idx, (2,), replace=False))(
            jax.random.split(k1a, W))
        diff = comp[pairs[:, 0]] - comp[pairs[:, 1]]

        # --- Stage 1 ---
        noise1 = jax.random.normal(k1b, (W,))
        prop1 = active + gs_cur * diff * noise1[:, None]
        lp1 = log_prob_fn(prop1)
        la1 = lp1 - lp_active
        accept1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1

        # --- Stage 2: shrunk scale ---
        noise2 = jax.random.normal(k_s2, (W,))
        prop2 = active + gs2_cur * diff * noise2[:, None]
        lp2 = log_prob_fn(prop2)

        la2 = _dr_side_log_acceptance(
            lp_active, lp1, lp2, noise1, noise2, gs2_cur / gs_cur)
        accept2 = (~accept1) & (jnp.log(jax.random.uniform(ka2, (W,), minval=1e-10)) < la2)

        new_pos = jnp.where(accept1[:, None], prop1,
                  jnp.where(accept2[:, None], prop2, active))
        new_lp = jnp.where(accept1, lp1,
                 jnp.where(accept2, lp2, lp_active))
        return new_pos, new_lp, accept1, accept2

    # --- initial gamma-scale search (stage-1 accept ≈ target_accept) ---
    if find_init_gamma:
        fi = jnp.finfo(jnp.result_type(gamma_scale0))
        def _find_body(s):
            gs, _, d, rk = s
            rk, k1, k2 = jax.random.split(rk, 3)
            gs = (2.**d) * gs
            ks1 = jax.random.split(k1, 4)
            ks2 = jax.random.split(k2, 4)
            _, _, a1, _ = _one_group(g1, g2, lp1, ks1, gs, gs * shrink)
            _, _, a2, _ = _one_group(g2, g1, lp2, ks2, gs, gs * shrink)
            avg = 0.5 * (jnp.mean(a1.astype(float)) + jnp.mean(a2.astype(float)))
            return gs, d, jnp.where(avg > target_accept, 1, -1), rk
        def _find_cond(s):
            gs, ld, d, _ = s
            return (((gs > fi.tiny) | (d >= 0)) & ((gs < fi.max) | (d <= 0))
                    & ((ld == 0) | (d == ld)))
        key, k_ = jax.random.split(key)
        gamma_scale0, *_ = jax.lax.while_loop(_find_cond, _find_body,
                                               (gamma_scale0, 0, 0, k_))
        if not (0.0 < float(gamma_scale0) < float("inf")):
            raise ValueError("initial scale search failed: check that the initial ensemble and target are finite and nondegenerate, or disable the search and supply a finite positive scale")
        if verbose:
            print(f"DR Side:  init_gamma={float(gamma_scale0 * jnp.sqrt(dim)):.4f}")

    log_gs0 = jnp.log(gamma_scale0)
    da = _da_init(log_gs0)

    # --- warmup ---
    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, keys):
        gs_cur = jnp.exp(da.log_h) if adapt_gamma else gamma_scale0
        gs2_cur = gs_cur * shrink

        g1, lp1, a1_s1, a1_s2 = _one_group(g1, g2, lp1, keys[:4], gs_cur, gs2_cur)
        g2, lp2, a2_s1, a2_s2 = _one_group(g2, g1, lp2, keys[4:], gs_cur, gs2_cur)

        s1_rate = (jnp.mean(a1_s1.astype(float)) + jnp.mean(a2_s1.astype(float))) / 2
        if adapt_gamma:
            da = _da_update(da, s1_rate, log_gs0, target_accept)
        overall = (jnp.mean((a1_s1 | a1_s2).astype(float)) +
                   jnp.mean((a2_s1 | a2_s2).astype(float))) / 2
        return g1, g2, lp1, lp2, da, overall

    key, k_ = jax.random.split(key)
    flat = jax.random.split(k_, warmup * 8)
    wkeys = flat.reshape(warmup, 8, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, rate = _warmup_step(g1, g2, lp1, lp2, da, wkeys[i])
        total_acc += rate

    final_gs = jnp.exp(da.log_h_bar) if adapt_gamma else gamma_scale0
    final_gamma = float(final_gs * jnp.sqrt(dim))
    if verbose:
        print(f"DR Side:  gamma={final_gamma:.4f}  shrink={shrink}"
              f"  warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    final_gs2 = final_gs * shrink

    # --- production ---
    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2 = carry

        g1, lp1, a1_s1, a1_s2 = _one_group(g1, g2, lp1, keys[:4], final_gs, final_gs2)
        g2, lp2, a2_s1, a2_s2 = _one_group(g2, g1, lp2, keys[4:], final_gs, final_gs2)

        st = jnp.concatenate([g1, g2])
        acc = jnp.concatenate([a1_s1 | a1_s2, a2_s1 | a2_s2]).astype(float)
        s1 = jnp.concatenate([a1_s1, a2_s1]).astype(float)
        s2 = jnp.concatenate([a1_s2, a2_s2]).astype(float)
        return (g1, g2, lp1, lp2), (st, acc, s1, s2)

    key, k = jax.random.split(key)
    flat = jax.random.split(k, num_samples * thin_by * 8)
    skeys = flat.reshape(num_samples * thin_by, 8, *flat.shape[1:])

    (g1, g2, lp1, lp2), (all_states, all_acc, all_s1, all_s2) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2), skeys)

    samples = all_states[::thin_by][:num_samples]

    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                stage1_rate=float(jnp.mean(all_s1)),
                stage2_rate=float(jnp.mean(all_s2)),
                final_gamma=final_gamma)
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}"
              f"  (stage1={info['stage1_rate']:.3f}, stage2={info['stage2_rate']:.3f})")
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

    print("=" * 50)
    print("DR Stretch move")
    print("=" * 50)
    samples, info = sampler_ensemble_dr_stretch(log_prob, init, num_samples=5000, warmup=500, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")

    print("\n" + "=" * 50)
    print("DR Side move")
    print("=" * 50)
    samples, info = sampler_ensemble_dr_side(log_prob, init, num_samples=5000, warmup=500, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
