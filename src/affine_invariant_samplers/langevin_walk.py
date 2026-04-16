"""
sampler_langevin_walk_walk — affine-invariant Langevin walk move (LWM), JAX only.

Overdamped Langevin with ensemble-preconditioned covariance:
  Proposal:  x' = x - h * C @ grad_U(x) + sqrt(2h) * C @ z,   z ~ N(0, I_k)
  where C = centered_complement / sqrt(k),  so C @ C^T ≈ Sigma.

MH correction for the asymmetric proposal (forward ≠ reverse).

Adaptation (warmup only, toggleable):
  Heuristic initial step-size search   (find_init_step_size)
  Dual averaging → step size h         (adapt_step_size)

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
# Langevin walk proposal + MH log-ratio
# ──────────────────────────────────────────────────────────────────────────────

def _lwm_propose(active, complement, key, h, grad_U, k):
    """
    Langevin walk move proposal for one group.

    Returns: (proposal, log_q_forward - log_q_reverse) per walker.
    The MH ratio is: log_alpha = log_prob(prop) - log_prob(cur) + log_q_rev - log_q_fwd
    """
    W, D = active.shape
    Wc = complement.shape[0]
    k1, k2 = jax.random.split(key)

    # k-subset per walker
    idx = jnp.arange(Wc)
    subsets = jax.vmap(lambda rk: jax.random.choice(rk, idx, (k,), replace=False))(
        jax.random.split(k1, W))                                  # (W, k)
    selected = complement[subsets]                                 # (W, k, D)
    means = jnp.mean(selected, axis=1, keepdims=True)             # (W, 1, D)
    B = (selected - means) / jnp.sqrt(k)                          # (W, k, D)

    # grad at current position
    grad_cur = grad_U(active)                                     # (W, D)

    # project gradient into k-subspace: bt_g = B @ grad^T  per walker
    bt_g = jnp.einsum('wkd,wd->wk', B, grad_cur)                 # (W, k)

    # drift = -h * B^T @ (B @ grad) = -h * C C^T grad
    drift = -h * jnp.einsum('wkd,wk->wd', B, bt_g)               # (W, D)

    # noise = sqrt(2h) * B^T @ z
    z = jax.random.normal(k2, (W, k))
    noise = jnp.sqrt(2. * h) * jnp.einsum('wkd,wk->wd', B, z)   # (W, D)

    proposal = active + drift + noise

    # --- MH correction (algebraic simplification, no matrix inverse needed) ---
    # Since drift and noise both lie in range(B^T), the null-space projection
    # terms in the quadratic forms cancel exactly.  Defining g_sum = bt_g + bt_gp
    # (k-dimensional), the correction reduces to:
    #   log_q_diff = h/4 * ||g_sum||^2 - sqrt(2h)/2 * (g_sum · z)
    grad_prop = grad_U(proposal)
    bt_gp = jnp.einsum('wkd,wd->wk', B, grad_prop)               # (W, k)
    g_sum = bt_g + bt_gp                                           # (W, k)
    log_q_diff = (h / 4.) * jnp.sum(g_sum ** 2, axis=1) \
               - (jnp.sqrt(2. * h) / 2.) * jnp.sum(g_sum * z, axis=1)

    return proposal, log_q_diff


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (accept ≈ target_accept at initial walker positions)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, g1, g2, lp1, lp2, eps0, log_prob_fn, grad_U, W, k,
                   target_accept):
    fi = jnp.finfo(jnp.result_type(eps0))

    def body(s):
        eps, _, d, rk = s
        rk, k1, k2, ka1, ka2 = jax.random.split(rk, 5)
        eps = (2.**d) * eps

        prop1, lqd1 = _lwm_propose(g1, g2, k1, eps, grad_U, k)
        la1 = log_prob_fn(prop1) - lp1 - lqd1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1

        prop2, lqd2 = _lwm_propose(g2, g1, k2, eps, grad_U, k)
        la2 = log_prob_fn(prop2) - lp2 - lqd2
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
# sampler_langevin_walk
# ──────────────────────────────────────────────────────────────────────────────

def sampler_langevin_walk(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup           = 1000,
    step_size        = 0.01,
    subset_size      = None,
    target_accept    = 0.574,
    thin_by          = 1,
    grad_log_prob_fn = None,
    seed             = 0,
    verbose          = True,
    find_init_step_size = True,
    adapt_step_size     = True,
):
    """
    Affine-invariant Langevin walk move with k-subset and dual averaging.

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).  n_chains must be even and >= 4.
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations for step-size adaptation.
        step_size        : Initial step size h (adapted during warmup).
        subset_size      : k-subset size (default: n_chains//2).
        target_accept    : Target acceptance for DA (default 0.574, optimal for MALA).
        thin_by          : Thinning factor.
        grad_log_prob_fn : Vectorised gradient.  If None, auto-derived.
        seed             : Random seed.
        verbose          : Print progress.
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` so that
                              mean acceptance ≈ `target_accept`.
                              If False, use `step_size` as-is.
        adapt_step_size  : If True (default), dual-averaging adapts step size during warmup.
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
    assert 2 <= k <= W

    if grad_log_prob_fn is None:
        grad_log_prob_fn = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    _grad_U = lambda x: -grad_log_prob_fn(x)

    g1, g2 = state[:W], state[W:]
    lp1, lp2 = log_prob_fn(g1), log_prob_fn(g2)
    key = jax.random.key(seed)

    step_size = jnp.asarray(step_size, jnp.float32)
    if find_init_step_size:
        key, k_ = jax.random.split(key)
        step_size = _find_init_eps(k_, g1, g2, lp1, lp2, step_size,
                                    log_prob_fn, _grad_U, W, k, target_accept)
        if verbose:
            print(f"Langevin walk:  init_eps={float(step_size):.4f}")
    log_h0 = jnp.log(step_size)
    da = _da_init(log_h0)

    # --- warmup ---
    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, keys):
        k1, k2, ka1, ka2 = keys
        h = jnp.exp(da.log_h) if adapt_step_size else step_size

        prop1, lqd1 = _lwm_propose(g1, g2, k1, h, _grad_U, k)
        lp_p1 = log_prob_fn(prop1)
        la1 = lp_p1 - lp1 - lqd1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1
        g1 = jnp.where(acc1[:, None], prop1, g1)
        lp1 = jnp.where(acc1, lp_p1, lp1)

        prop2, lqd2 = _lwm_propose(g2, g1, k2, h, _grad_U, k)
        lp_p2 = log_prob_fn(prop2)
        la2 = lp_p2 - lp2 - lqd2
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
        print(f"Langevin walk:  k={k}  h={float(final_h):.4f}"
              f"  warmup_accept={float(total_acc)/max(warmup,1):.3f}")

    # --- production ---
    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2 = carry
        k1, k2, ka1, ka2 = keys

        prop1, lqd1 = _lwm_propose(g1, g2, k1, final_h, _grad_U, k)
        lp_p1 = log_prob_fn(prop1)
        la1 = lp_p1 - lp1 - lqd1
        acc1 = jnp.log(jax.random.uniform(ka1, (W,), minval=1e-10)) < la1
        g1 = jnp.where(acc1[:, None], prop1, g1)
        lp1 = jnp.where(acc1, lp_p1, lp1)

        prop2, lqd2 = _lwm_propose(g2, g1, k2, final_h, _grad_U, k)
        lp_p2 = log_prob_fn(prop2)
        la2 = lp_p2 - lp2 - lqd2
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
    # Per iteration: 2 gradient calls per walker (forward + reverse in the MH
    # correction), × 2 groups combined into `n_chains` total evaluations.
    n_grad_evals = int(num_samples * thin_by) * 2 * int(n_chains)
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_step_size=float(final_h),
                n_grad_evals=n_grad_evals)
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
    samples, info = sampler_langevin_walk(log_prob, init, num_samples=5000, warmup=500, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
