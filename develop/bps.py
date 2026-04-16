"""
sampler_bps — Bouncy Particle Sampler (BPS), single file, JAX only.

PDMP sampler: straight-line motion x(t) = x₀ + t·v between events.
Events: specular bounces (gradient-driven) or velocity refreshments.
Bounce rate: λ(x,v) = max(⟨∇U(x), v⟩, 0).
At bounce: v ← v − 2⟨g,v⟩/‖g‖² · g  (specular reflection).
At refresh: v ~ N(0, I).

Uses time-stepping discretisation with Poisson event probabilities.
Step size controls accuracy; small enough → negligible bias.

Reference: Bouchard-Côté, Vollmer & Doucet, arXiv:1510.02451
"""

import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# BPS event loop (single chain, time-stepping)
# ──────────────────────────────────────────────────────────────────────────────

def _bps_steps(x, v, grad_U, key, n_steps, refresh_rate, step_size):
    """Run n_steps of time-discretised BPS from (x, v).  Returns (x', v')."""
    D = x.shape[0]

    def step(carry, _):
        x, v, key = carry
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Evaluate rate at midpoint (second-order accuracy)
        g = grad_U(x + 0.5 * step_size * v)
        lam = jnp.maximum(jnp.dot(g, v), 0.)
        total = lam + refresh_rate

        # Poisson event probability over [0, step_size]
        p_event = 1. - jnp.exp(-total * step_size)
        event = jax.random.uniform(k1) < p_event

        # Event type: bounce vs refresh
        u = jax.random.uniform(k2)
        bounce  = event & (u < lam / (total + 1e-30))
        refresh = event & ~bounce

        # Move
        x = x + step_size * v

        # Specular reflection (use gradient at new position)
        g_new = grad_U(x)
        gn2 = jnp.dot(g_new, g_new) + 1e-30
        v_b = v - 2. * jnp.dot(g_new, v) / gn2 * g_new

        # Refreshment
        v_r = jax.random.normal(k3, (D,))

        v = jnp.where(bounce, v_b, jnp.where(refresh, v_r, v))
        return (x, v, key), None

    (x, v, _), _ = jax.lax.scan(step, (x, v, key), None, length=n_steps)
    return x, v


# ──────────────────────────────────────────────────────────────────────────────
# sampler_bps
# ──────────────────────────────────────────────────────────────────────────────

def sampler_bps(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    n_steps       = 50,
    step_size     = 0.05,
    refresh_rate  = 0.5,
    thin_by       = 1,
    grad_log_prob_fn = None,
    seed          = 0,
    verbose       = True,
):
    """
    Bouncy Particle Sampler.

    Args:
        log_prob_fn      : (n_chains, D) -> (n_chains,).  Batched log density.
        initial_state    : (n_chains, D).
        num_samples      : Post-warmup samples.
        warmup           : Burn-in steps per chain.
        n_steps          : Steps per chain between recorded samples.
        step_size        : Time-step size (controls accuracy).
        refresh_rate     : Poisson rate of velocity refreshment.
        thin_by          : Thinning factor.
        grad_log_prob_fn : Batched gradient (n_chains,D)->(n_chains,D).
                           If None, derived via jax.grad.
        seed             : Random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(step_size, refresh_rate, n_steps)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape

    if grad_log_prob_fn is None:
        _grad_lp = jax.grad(lambda x: log_prob_fn(x[None])[0])
    else:
        _grad_lp = lambda x: grad_log_prob_fn(x[None])[0]
    grad_U = lambda x: -_grad_lp(x)

    key = jax.random.key(seed)

    # Initialise velocities ~ N(0, I)
    key, vk = jax.random.split(key)
    vel = jax.random.normal(vk, state.shape)

    if verbose:
        print(f"BPS  n_steps={n_steps}  step_size={step_size}"
              f"  refresh={refresh_rate}  warmup={warmup}")

    # ── warmup (burn-in) ──────────────────────────────────────────────────────
    key, wk = jax.random.split(key)
    wkeys = jax.random.split(wk, n_chains)
    state, vel = jax.jit(jax.vmap(
        lambda x, v, k: _bps_steps(x, v, grad_U, k, warmup,
                                    refresh_rate, step_size)
    ))(state, vel, wkeys)
    if verbose:
        print("Warmup done.")

    # ── production ────────────────────────────────────────────────────────────
    total_iters = num_samples * thin_by

    @jax.jit
    def _step(carry, key):
        pos, vel = carry
        keys = jax.random.split(key, n_chains)
        pos, vel = jax.vmap(
            lambda x, v, k: _bps_steps(x, v, grad_U, k, n_steps,
                                        refresh_rate, step_size)
        )(pos, vel, keys)
        return (pos, vel), pos

    key, sk = jax.random.split(key)
    skeys = jax.random.split(sk, total_iters)
    (state, vel), all_pos = jax.lax.scan(_step, (state, vel), skeys)

    samples = all_pos[::thin_by]
    info = dict(step_size=float(step_size), refresh_rate=refresh_rate,
                n_steps=n_steps)
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
    samples, info = sampler_bps(log_prob_gauss, init, num_samples=2000,
                                warmup=2000, n_steps=100, step_size=0.05,
                                refresh_rate=0.5, seed=123)
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
    samples, info = sampler_bps(log_prob_rosen, init_r, num_samples=2000,
                                warmup=2000, n_steps=200, step_size=0.01,
                                refresh_rate=1.0, seed=123)
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
    samples, info = sampler_bps(log_prob_funnel, init_f, num_samples=2000,
                                warmup=2000, n_steps=100, step_size=0.05,
                                refresh_rate=1.0, seed=123)
    flat = samples.reshape(-1, funnel_dim)
    v = flat[:, 0]; xs = flat[:, 1:]
    print(f"  v:   mean={jnp.mean(v):.3f}  var={jnp.var(v):.2f}  (target: 0, 9)")
    print(f"  x_i: mean={jnp.mean(xs):.3f}  var={jnp.mean(jnp.var(xs, axis=0)):.1f}"
          f"  (target: 0, ~90)")
