"""
sampler_bps_walk — ensemble-preconditioned BPS, walk move, single file, JAX only.

Bouncy Particle Sampler in the W-dimensional complement-walker subspace.
Velocity w ∈ R^W; position updates via  x → x + dt · (w @ centered).
Bounce and refresh events operate in W-space → affine-invariant.

Time-stepping discretisation with midpoint rate evaluation.
Fresh velocity drawn at the start of each group update.

Reference (BPS):     Bouchard-Côté, Vollmer & Doucet, arXiv:1510.02451
"""

import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# BPS walk-move dynamics in W-space (all W walkers, time-stepping)
# ──────────────────────────────────────────────────────────────────────────────

def _bps_walk_steps(q, w, grad_U, key, n_steps, refresh_rate, step_size, centered):
    """Time-stepping BPS in ensemble subspace.

    q        : (W, D) positions
    w        : (W, W) velocities in W-space
    grad_U   : (W, D) -> (W, D)  gradient of U = -log p  (vectorised)
    centered : (W, D) centered complement / sqrt(W)

    Returns: (q', w')
    """
    W = centered.shape[0]

    def step(carry, _):
        q, w, key = carry
        key, k1, k2, k3 = jax.random.split(key, 4)

        v_D = w @ centered                                          # (W, D)

        # Midpoint gradient projected to W-space
        g_W = grad_U(q + 0.5 * step_size * v_D) @ centered.T       # (W, W)

        # Per-walker bounce rate
        lam = jnp.maximum(jnp.sum(g_W * w, axis=1), 0.)            # (W,)
        total = lam + refresh_rate

        # Poisson event probability
        p_ev = 1. - jnp.exp(-total * step_size)
        ev = jax.random.uniform(k1, (W,)) < p_ev
        u  = jax.random.uniform(k2, (W,))
        bounce  = ev & (u < lam / (total + 1e-30))
        refresh = ev & ~bounce

        # Advance positions
        q = q + step_size * v_D

        # Specular reflection in W-space (gradient at new position)
        g_W_new = grad_U(q) @ centered.T                            # (W, W)
        gn2 = jnp.sum(g_W_new**2, axis=1, keepdims=True) + 1e-30   # (W, 1)
        gw  = jnp.sum(g_W_new * w, axis=1, keepdims=True)           # (W, 1)
        w_b = w - 2. * gw / gn2 * g_W_new

        # Refreshment
        w_r = jax.random.normal(k3, w.shape)

        w = jnp.where(bounce[:, None], w_b,
                jnp.where(refresh[:, None], w_r, w))
        return (q, w, key), None

    (q, w, _), _ = jax.lax.scan(step, (q, w, key), None, length=n_steps)
    return q, w


# ──────────────────────────────────────────────────────────────────────────────
# sampler_bps_walk
# ──────────────────────────────────────────────────────────────────────────────

def sampler_bps_walk(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup        = 1000,
    n_steps       = 50,
    step_size     = 0.1,
    refresh_rate  = 0.5,
    thin_by       = 1,
    grad_log_prob_fn = None,
    seed          = 0,
    verbose       = True,
):
    """
    Ensemble-preconditioned Bouncy Particle Sampler (walk move).

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).  n_chains must be even and >= 4.
        num_samples      : Post-warmup samples.
        warmup           : Burn-in iterations (alternating group updates).
        n_steps          : BPS steps per group update.
        step_size        : Time-step size.
        refresh_rate     : Poisson rate of velocity refreshment.
        thin_by          : Thinning factor.
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
                           If None, derived via jax.vmap(jax.grad).
        seed             : Random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(step_size, refresh_rate, n_steps)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0, "Need >= 4 even chains"

    if grad_log_prob_fn is None:
        grad_U = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_U = grad_log_prob_fn
    _grad_U = lambda x: -grad_U(x)

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    key = jax.random.key(seed)

    if verbose:
        print(f"BPS-walk  n_steps={n_steps}  step_size={step_size}"
              f"  refresh={refresh_rate}  warmup={warmup}")

    # ── one iteration: update g1 then g2 ──────────────────────────────────────

    def _one_iter(carry, keys):
        g1, g2 = carry
        k1, kw1, k2, kw2 = keys

        # Update g1 using g2 as complement
        centered1 = (g2 - jnp.mean(g2, axis=0)) / jnp.sqrt(W)
        w1 = jax.random.normal(kw1, (W, W))
        g1, _ = _bps_walk_steps(g1, w1, _grad_U, k1, n_steps,
                                 refresh_rate, step_size, centered1)

        # Update g2 using updated g1 as complement
        centered2 = (g1 - jnp.mean(g1, axis=0)) / jnp.sqrt(W)
        w2 = jax.random.normal(kw2, (W, W))
        g2, _ = _bps_walk_steps(g2, w2, _grad_U, k2, n_steps,
                                 refresh_rate, step_size, centered2)

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
    samples, info = sampler_bps_walk(log_prob_gauss, init, num_samples=2000,
                                     warmup=500, n_steps=50, step_size=0.1,
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
    samples, info = sampler_bps_walk(log_prob_rosen, init_r, num_samples=2000,
                                     warmup=500, n_steps=200, step_size=0.005,
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
    samples, info = sampler_bps_walk(log_prob_funnel, init_f, num_samples=2000,
                                     warmup=1000, n_steps=100, step_size=0.01,
                                     refresh_rate=1.0, seed=123)
    flat = samples.reshape(-1, funnel_dim)
    v = flat[:, 0]; xs = flat[:, 1:]
    print(f"  v:   mean={jnp.mean(v):.3f}  var={jnp.var(v):.2f}  (target: 0, 9)")
    print(f"  x_i: mean={jnp.mean(xs):.3f}  var={jnp.mean(jnp.var(xs, axis=0)):.1f}"
          f"  (target: 0, ~90)")
