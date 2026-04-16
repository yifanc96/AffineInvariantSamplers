"""
sampler_zigzag — Zig-Zag Sampler, single file, JAX only.

PDMP sampler: straight-line motion x(t) = x₀ + t·v between events.
Velocity v ∈ {−1, +1}^D — each component flips independently.
Per-component flip rate: λⱼ(x,v) = max(vⱼ · ∂ⱼU(x), 0).
At flip: vⱼ ← −vⱼ  (negate the selected component).

Uses Poisson thinning with local bounds + safety excess.
No discretisation bias; no Metropolis correction.

Reference: Bierkens, Fearnhead & Roberts, arXiv:1607.03188
"""

import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# Zig-Zag event loop (single chain)
# ──────────────────────────────────────────────────────────────────────────────

def _zz_events(x, v, grad_U, key, n_events, excess, dt_max):
    """Simulate n_events of Zig-Zag from (x, v).  Returns (x', v')."""
    D = x.shape[0]

    def step(carry, _):
        x, v, key = carry
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Per-component rates and bounds
        g = grad_U(x)
        rates = jnp.maximum(v * g, 0.)          # (D,)
        bounds = rates + excess                  # (D,)
        total = jnp.sum(bounds)

        # Propose event time
        dt = -jnp.log(jax.random.uniform(k1, minval=1e-30)) / total
        dt = jnp.minimum(dt, dt_max)
        x = x + dt * v

        # Select component proportional to bounds
        cum = jnp.cumsum(bounds)
        u_comp = jax.random.uniform(k2) * total
        j = jnp.searchsorted(cum, u_comp, side='right')
        j = jnp.clip(j, 0, D - 1)

        # Thinning: accept flip with prob actual_rate[j] / bound[j]
        g_new = grad_U(x)
        actual_j = jnp.maximum(v[j] * g_new[j], 0.)
        accept = jax.random.uniform(k3) * bounds[j] < actual_j

        v = v.at[j].set(jnp.where(accept, -v[j], v[j]))
        return (x, v, key), None

    (x, v, _), _ = jax.lax.scan(step, (x, v, key), None, length=n_events)
    return x, v


# ──────────────────────────────────────────────────────────────────────────────
# sampler_zigzag
# ──────────────────────────────────────────────────────────────────────────────

def sampler_zigzag(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    n_events      = 20,
    excess_rate   = 1.0,
    dt_max        = 1.0,
    thin_by       = 1,
    grad_log_prob_fn = None,
    seed          = 0,
    verbose       = True,
):
    """
    Zig-Zag Sampler.

    Args:
        log_prob_fn      : (n_chains, D) -> (n_chains,).  Batched log density.
        initial_state    : (n_chains, D).
        num_samples      : Post-warmup samples.
        warmup           : Burn-in events per chain.
        n_events         : Events per chain between recorded samples.
        excess_rate      : Safety margin for Poisson thinning bound.
        dt_max           : Maximum travel time per event (prevents overshooting).
        thin_by          : Thinning factor.
        grad_log_prob_fn : Batched gradient (n_chains,D)->(n_chains,D).
                           If None, derived via jax.grad.
        seed             : Random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(n_events, excess_rate)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape

    # Per-chain grad U = -grad log prob
    if grad_log_prob_fn is None:
        _grad_lp = jax.grad(lambda x: log_prob_fn(x[None])[0])
    else:
        _grad_lp = lambda x: grad_log_prob_fn(x[None])[0]
    grad_U = lambda x: -_grad_lp(x)

    key = jax.random.key(seed)

    # Initialise velocities ∈ {−1, +1}^D uniformly
    key, vk = jax.random.split(key)
    vel = 2. * jax.random.bernoulli(vk, shape=state.shape).astype(float) - 1.

    if verbose:
        print(f"ZigZag  n_events={n_events}  excess={excess_rate}"
              f"  dt_max={dt_max}  warmup={warmup}")

    # ── warmup (burn-in) ──────────────────────────────────────────────────────
    key, wk = jax.random.split(key)
    wkeys = jax.random.split(wk, n_chains)
    state, vel = jax.jit(jax.vmap(
        lambda x, v, k: _zz_events(x, v, grad_U, k, warmup, excess_rate, dt_max)
    ))(state, vel, wkeys)
    if verbose:
        print("Warmup done.")

    # ── production ────────────────────────────────────────────────────────────
    total_steps = num_samples * thin_by

    @jax.jit
    def _step(carry, key):
        pos, vel = carry
        keys = jax.random.split(key, n_chains)
        pos, vel = jax.vmap(
            lambda x, v, k: _zz_events(x, v, grad_U, k, n_events, excess_rate, dt_max)
        )(pos, vel, keys)
        return (pos, vel), pos

    key, sk = jax.random.split(key)
    skeys = jax.random.split(sk, total_steps)
    (state, vel), all_pos = jax.lax.scan(_step, (state, vel), skeys)

    samples = all_pos[::thin_by]
    info = dict(n_events=n_events, excess_rate=excess_rate, dt_max=dt_max)
    if verbose:
        print("Done.")
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test 1: Ill-conditioned Gaussian ──────────────────────────────────────
    print("=" * 60)
    print("Test 1: Ill-conditioned Gaussian  (D=10, kappa=100)")
    print("=" * 60)
    dim = 10
    eigvals = jnp.logspace(0, 2, dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov = Q @ jnp.diag(eigvals) @ Q.T
    prec = Q @ jnp.diag(1. / eigvals) @ Q.T
    def log_prob_gauss(x):
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)

    n_chains = 50
    init = jax.random.normal(jax.random.key(42), (n_chains, dim))
    samples, info = sampler_zigzag(log_prob_gauss, init, num_samples=2000,
                                   warmup=500, n_events=50, excess_rate=2.0,
                                   seed=123)
    flat = samples.reshape(-1, dim)
    var_est = jnp.var(flat, axis=0)
    var_true = jnp.diag(cov)
    rel_err = jnp.mean(jnp.abs(var_est - var_true) / var_true)
    print(f"  mean_rel_err(var)={rel_err:.3f}"
          f"  var_range=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
          f"  (target: [{jnp.min(var_true):.2f}, {jnp.max(var_true):.2f}])")

    # ── Test 2: Rosenbrock ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Test 2: Rosenbrock  (D=10, a=1, b=100)")
    print("=" * 60)
    a_ros, b_ros = 1.0, 100.0
    dim_ros = 10
    def log_prob_rosen(x):
        xe, xo = x[..., ::2], x[..., 1::2]
        return -(b_ros * jnp.sum((xo - xe**2)**2, axis=-1)
                 + jnp.sum((xe - a_ros)**2, axis=-1))

    init_r = jax.random.normal(jax.random.key(42), (n_chains, dim_ros)) * 0.5
    samples, info = sampler_zigzag(log_prob_rosen, init_r, num_samples=2000,
                                   warmup=500, n_events=100, excess_rate=5.0,
                                   seed=123)
    flat = samples.reshape(-1, dim_ros)
    me = jnp.mean(flat[:, ::2]);  ve = jnp.mean(jnp.var(flat[:, ::2], axis=0))
    mo = jnp.mean(flat[:, 1::2]); vo = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
    print(f"  x_even: mean={me:.3f} var={ve:.4f} (target: mean=1, var=0.5)")
    print(f"  x_odd:  mean={mo:.3f} var={vo:.4f} (target: mean=1.5, var~2.505)")

    # ── Test 3: Neal's Funnel ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Test 3: Neal's Funnel  (D=3)")
    print("=" * 60)
    funnel_dim = 3
    def log_prob_funnel(x):
        v  = x[..., 0]
        xs = x[..., 1:]
        lp_v = -0.5 * v**2 / 9.
        lp_x = (-0.5 * jnp.sum(xs**2 * jnp.exp(-v)[..., None], axis=-1)
                 - 0.5 * (funnel_dim - 1) * v)
        return lp_v + lp_x

    init_f = jax.random.normal(jax.random.key(99), (n_chains, funnel_dim)) * 0.5
    samples, info = sampler_zigzag(log_prob_funnel, init_f, num_samples=2000,
                                   warmup=500, n_events=50, excess_rate=2.0,
                                   seed=123)
    flat = samples.reshape(-1, funnel_dim)
    v = flat[:, 0]; xs = flat[:, 1:]
    print(f"  v:   mean={jnp.mean(v):.3f}  var={jnp.var(v):.2f}  (target: 0, 9)")
    print(f"  x_i: mean={jnp.mean(xs):.3f}  var={jnp.mean(jnp.var(xs, axis=0)):.1f}"
          f"  (target: 0, ~90)")
