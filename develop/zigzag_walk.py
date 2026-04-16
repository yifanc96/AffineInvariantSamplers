"""
sampler_zigzag_walk — ensemble-preconditioned Zig-Zag, walk move, single file, JAX only.

Zig-Zag sampler in the W-dimensional complement-walker subspace.
Velocity w ∈ {−1,+1}^W; position updates via  x → x + dt · (w @ centered).
Component-wise flips operate in W-space → affine-invariant.

Uses Poisson thinning with local bounds in W-space.
Fresh velocity drawn at the start of each group update.

Reference (Zig-Zag): Bierkens, Fearnhead & Roberts, arXiv:1607.03188
Reference (PEACHES): https://arxiv.org/abs/2505.02987
"""

import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# Zig-Zag walk-move dynamics in W-space (single walker, Poisson thinning)
# ──────────────────────────────────────────────────────────────────────────────

def _zz_walk_events(q, w, grad_U_single, key, n_events, excess, dt_max, centered):
    """Poisson-thinning Zig-Zag in ensemble subspace (one walker).

    q         : (D,)  position
    w         : (W,)  velocity in W-space, entries ∈ {−1, +1}
    grad_U_single : (D,) -> (D,)  gradient of U = -log p  (single point)
    centered  : (W, D) centered complement / sqrt(W)

    Returns: (q', w')
    """
    W = centered.shape[0]

    def step(carry, _):
        q, w, key = carry
        key, k1, k2, k3 = jax.random.split(key, 4)

        v_D = w @ centered                                      # (D,)
        g_W = grad_U_single(q) @ centered.T                     # (W,)

        # Per-component rates and bounds in W-space
        rates  = jnp.maximum(w * g_W, 0.)                       # (W,)
        bounds = rates + excess
        total  = jnp.sum(bounds)

        # Propose event time
        dt = jnp.minimum(
            -jnp.log(jax.random.uniform(k1, minval=1e-30)) / total, dt_max)
        q = q + dt * v_D

        # Select component proportional to bounds
        j = jnp.searchsorted(jnp.cumsum(bounds),
                              jax.random.uniform(k2) * total, side='right')
        j = jnp.clip(j, 0, W - 1)

        # Thinning: accept flip?
        g_W_new = grad_U_single(q) @ centered.T
        actual_j = jnp.maximum(w[j] * g_W_new[j], 0.)
        accept = jax.random.uniform(k3) * bounds[j] < actual_j

        w = w.at[j].set(jnp.where(accept, -w[j], w[j]))
        return (q, w, key), None

    (q, w, _), _ = jax.lax.scan(step, (q, w, key), None, length=n_events)
    return q, w


# ──────────────────────────────────────────────────────────────────────────────
# sampler_zigzag_walk
# ──────────────────────────────────────────────────────────────────────────────

def sampler_zigzag_walk(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    n_events      = 50,
    excess_rate   = 1.0,
    dt_max        = 1.0,
    thin_by       = 1,
    grad_log_prob_fn = None,
    seed          = 0,
    verbose       = True,
):
    """
    Ensemble-preconditioned Zig-Zag Sampler (walk move).

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).  n_chains must be even and >= 4.
        num_samples      : Post-warmup samples.
        warmup           : Burn-in iterations.
        n_events         : Zig-Zag events per walker per group update.
        excess_rate      : Safety margin for Poisson thinning bound.
        dt_max           : Maximum travel time per event.
        thin_by          : Thinning factor.
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
                           If None, derived via jax.grad.
        seed             : Random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(n_events, excess_rate, dt_max)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0, "Need >= 4 even chains"

    # Single-chain grad U
    if grad_log_prob_fn is None:
        _grad_lp = jax.grad(lambda x: log_prob_fn(x[None])[0])
    else:
        _grad_lp = lambda x: grad_log_prob_fn(x[None])[0]
    grad_U_single = lambda x: -_grad_lp(x)

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    key = jax.random.key(seed)

    if verbose:
        print(f"ZigZag-walk  n_events={n_events}  excess={excess_rate}"
              f"  dt_max={dt_max}  warmup={warmup}")

    # ── one iteration: update g1 then g2 ──────────────────────────────────────

    def _one_iter(carry, keys):
        g1, g2 = carry
        k1, kw1, k2, kw2 = keys

        # Update g1 using g2 as complement
        centered1 = (g2 - jnp.mean(g2, axis=0)) / jnp.sqrt(W)
        w1 = 2. * jax.random.bernoulli(kw1, shape=(W, W)).astype(float) - 1.
        keys1 = jax.random.split(k1, W)
        g1, _ = jax.vmap(
            lambda q, w, k: _zz_walk_events(
                q, w, grad_U_single, k, n_events, excess_rate, dt_max, centered1)
        )(g1, w1, keys1)

        # Update g2 using updated g1 as complement
        centered2 = (g1 - jnp.mean(g1, axis=0)) / jnp.sqrt(W)
        w2 = 2. * jax.random.bernoulli(kw2, shape=(W, W)).astype(float) - 1.
        keys2 = jax.random.split(k2, W)
        g2, _ = jax.vmap(
            lambda q, w, k: _zz_walk_events(
                q, w, grad_U_single, k, n_events, excess_rate, dt_max, centered2)
        )(g2, w2, keys2)

        return (g1, g2), jnp.concatenate([g1, g2])

    # ── warmup ────────────────────────────────────────────────────────────────
    key, wk = jax.random.split(key)
    flat = jax.random.split(wk, warmup * 4)
    wkeys = flat.reshape(warmup, 4, *flat.shape[1:])

    (g1, g2), _ = jax.lax.scan(
        jax.jit(_one_iter), (g1, g2), wkeys)
    if verbose:
        print("Warmup done.")

    # ── production ────────────────────────────────────────────────────────────
    total_iters = num_samples * thin_by

    key, sk = jax.random.split(key)
    flat = jax.random.split(sk, total_iters * 4)
    skeys = flat.reshape(total_iters, 4, *flat.shape[1:])

    (g1, g2), all_states = jax.lax.scan(
        jax.jit(_one_iter), (g1, g2), skeys)

    samples = all_states[::thin_by]
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
    print("Test 1: Ill-conditioned Gaussian  (D=10, kappa=1000)")
    print("=" * 60)
    dim = 10
    eigvals = jnp.logspace(0, 3, dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov = Q @ jnp.diag(eigvals) @ Q.T
    prec = Q @ jnp.diag(1. / eigvals) @ Q.T
    def log_prob_gauss(x):
        return -0.5 * jnp.sum((x @ prec) * x, axis=-1)

    n_chains = 100
    init = jax.random.normal(jax.random.key(42), (n_chains, dim))
    samples, info = sampler_zigzag_walk(log_prob_gauss, init, num_samples=2000,
                                        warmup=500, n_events=50, excess_rate=1.0,
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
    samples, info = sampler_zigzag_walk(log_prob_rosen, init_r, num_samples=2000,
                                        warmup=500, n_events=100, excess_rate=2.0,
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
    samples, info = sampler_zigzag_walk(log_prob_funnel, init_f, num_samples=2000,
                                        warmup=500, n_events=50, excess_rate=1.0,
                                        seed=123)
    flat = samples.reshape(-1, funnel_dim)
    v = flat[:, 0]; xs = flat[:, 1:]
    print(f"  v:   mean={jnp.mean(v):.3f}  var={jnp.var(v):.2f}  (target: 0, 9)")
    print(f"  x_i: mean={jnp.mean(xs):.3f}  var={jnp.mean(jnp.var(xs, axis=0)):.1f}"
          f"  (target: 0, ~90)")
