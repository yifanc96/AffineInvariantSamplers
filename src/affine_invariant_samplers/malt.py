"""
sampler_malt — Metropolis Adjusted Langevin Trajectories, single file, JAX only.

MALT discretises the underdamped Langevin diffusion with a BABO+O splitting
and applies a single Metropolis correction to the whole L-step trajectory.

Special cases:
  gamma = 0        →  standard HMC
  L = 1            →  MALA
  gamma > 0, L > 1 →  full MALT

Adaptation (warmup only, toggleable):
  Heuristic initial step-size search   (find_init_step_size)
  Dual averaging → step size           (adapt_step_size)

Reference: Riou-Durand & Vogrinc, arXiv:2202.13230
           https://github.com/lrioudurand/malt
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# BABO+O integrator  (one trajectory of L steps)
# ──────────────────────────────────────────────────────────────────────────────

def _malt_trajectory(q, v, grad_q, U_fn, grad_U_fn, eps, L, eta, zeta, noise):
    """Run L steps of BAB-O and accumulate energy error Delta.

    Args:
        q       : (D,)  position
        v       : (D,)  velocity (freshly drawn)
        grad_q  : (D,)  grad U at q  (pre-computed)
        U_fn    : (D,) -> scalar   potential energy
        grad_U_fn: (D,) -> (D,)    gradient of U
        eps     : scalar step size
        L       : int   leapfrog steps
        eta     : scalar  exp(-gamma * eps)
        zeta    : scalar  sqrt(1 - eta^2)
        noise   : (L, D)  pre-drawn standard normals for O-steps

    Returns:
        q, v, grad_q, Delta
    """
    half = eps / 2.0

    def step(carry, xi):
        q, v, grad_q, Delta = carry
        # B: half kick
        v = v - half * grad_q
        # A: full drift
        q = q + eps * v
        # evaluate gradient at new position
        grad_new = grad_U_fn(q)
        # accumulate midpoint energy error
        Delta = Delta - half * jnp.dot(v, grad_q + grad_new)
        # B: half kick
        v = v - half * grad_new
        # O: Ornstein-Uhlenbeck momentum refresh
        v = eta * v + zeta * xi
        # update gradient reference
        grad_q = grad_new
        return (q, v, grad_q, Delta), None

    (q, v, grad_q, Delta), _ = jax.lax.scan(step, (q, v, grad_q, 0.0), noise)
    return q, v, grad_q, Delta


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised MALT kernel (one step over all chains)
# ──────────────────────────────────────────────────────────────────────────────

def _malt_step(state, U_fn, grad_U_fn, eps, L, gamma, key):
    """One MALT iteration for all chains in parallel.

    state : (q, grad_q, gradsq, U_q)
            q:       (n_chains, D)
            grad_q:  (n_chains, D)
            gradsq:  (n_chains,)
            U_q:     (n_chains,)

    Returns:
        new_state, accept  (accept: (n_chains,) bool)
    """
    q, grad_q, gradsq, U_q = state
    n_chains, dim = q.shape

    eta  = jnp.exp(-gamma * eps)
    zeta = jnp.sqrt(1.0 - eta**2)
    small = eps**2 / 8.0

    key_v, key_noise, key_u = jax.random.split(key, 3)

    # Step 1: fresh momentum
    v = jax.random.normal(key_v, q.shape)

    # Step 2: pre-generate O-step noise  (n_chains, L, D)
    noise = jax.random.normal(key_noise, (n_chains, L, dim))

    # Step 3: run trajectory (vectorised over chains)
    q_new, _, grad_new, Delta = jax.vmap(
        lambda qi, vi, gi, ni: _malt_trajectory(
            qi, vi, gi, U_fn, grad_U_fn, eps, L, eta, zeta, ni)
    )(q, v, grad_q, noise)

    # Step 4: full acceptance log-ratio
    gradsq_new = jnp.sum(grad_new**2, axis=1)
    U_new      = jax.vmap(U_fn)(q_new)
    Delta      = Delta + small * (gradsq_new - gradsq) + U_new - U_q

    # Step 5: accept / reject
    log_u  = jnp.log(jax.random.uniform(key_u, (n_chains,), minval=1e-30))
    accept = log_u < -Delta

    q_out      = jnp.where(accept[:, None], q_new, q)
    grad_out   = jnp.where(accept[:, None], grad_new, grad_q)
    gradsq_out = jnp.where(accept, gradsq_new, gradsq)
    U_out      = jnp.where(accept, U_new, U_q)

    return (q_out, grad_out, gradsq_out, U_out), accept


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

def _da_update(state, accept_rate, log_eps0, target, t0=10., gamma_da=0.05, kappa=0.75):
    it    = state.iteration + 1
    eta   = 1. / (it + t0)
    H_bar = (1. - eta) * state.H_bar + eta * (target - accept_rate)
    log_e = log_eps0 - jnp.sqrt(it) / ((it + t0) * gamma_da) * H_bar
    log_eb = it**(-kappa) * log_e + (1. - it**(-kappa)) * state.log_eps_bar
    return DAState(it, log_e, log_eb, H_bar)


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (~80% acceptance with L=1)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, q, U_fn, grad_U_fn, gamma, eps0):
    fi = jnp.finfo(jnp.result_type(eps0))
    n_chains, dim = q.shape

    grad_q  = jax.vmap(grad_U_fn)(q)
    gradsq  = jnp.sum(grad_q**2, axis=1)
    U_q     = jax.vmap(U_fn)(q)
    state0  = (q, grad_q, gradsq, U_q)

    def body(s):
        eps, _, d, k = s
        k, ks = jax.random.split(k)
        eps = (2.**d) * eps
        _, accept = _malt_step(state0, U_fn, grad_U_fn, eps, 1, gamma, ks)
        avg = jnp.mean(accept.astype(float))
        return eps, d, jnp.where(avg > 0.8, 1, -1), k

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps / 2.


# ──────────────────────────────────────────────────────────────────────────────
# sampler_malt
# ──────────────────────────────────────────────────────────────────────────────

def sampler_malt(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup           = 1000,
    step_size        = 0.05,
    L                = 10,
    gamma            = 1.0,
    thin_by          = 1,
    target_accept    = 0.651,
    grad_log_prob_fn = None,
    seed             = 0,
    verbose          = True,
    find_init_step_size = True,
    adapt_step_size     = True,
):
    """
    Metropolis Adjusted Langevin Trajectories (MALT).

    Args:
        log_prob_fn      : (D,) -> scalar.  Log density (single point).
        initial_state    : (n_chains, D).
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations.
        step_size        : Initial step size (adapted during warmup).
        L                : Number of leapfrog steps per trajectory.
        gamma            : Friction coefficient (>= 0).
                           0 → HMC;  large → more damping / closer to MALA.
        thin_by          : Keep every thin_by-th sample.
        target_accept    : Target acceptance rate for dual averaging.
        grad_log_prob_fn : (D,) -> (D,).  Gradient of log density (single point).
                           If None, uses jax.grad(log_prob_fn).
        seed             : Integer random seed.
        verbose          : Print progress.
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` to
                              ~80% acceptance before warmup.
                              If False, use `step_size` as-is.
        adapt_step_size  : If True (default), dual-averaging adapts step size
                           during warmup. If False, uses `step_size` as-is.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_step_size, gamma, L)
    """
    state = jnp.asarray(initial_state)
    n_chains, dim = state.shape

    # potential U = -log_prob  and its gradient
    U_fn = lambda x: -log_prob_fn(x)
    if grad_log_prob_fn is None:
        grad_U_fn = jax.grad(lambda x: -log_prob_fn(x))
    else:
        grad_U_fn = lambda x: -grad_log_prob_fn(x)

    key = jax.random.key(seed)

    # --- initialise state ---
    grad_q  = jax.vmap(grad_U_fn)(state)
    gradsq  = jnp.sum(grad_q**2, axis=1)
    U_q     = jax.vmap(U_fn)(state)
    malt_state = (state, grad_q, gradsq, U_q)

    # --- find initial step size ---
    step_size = jnp.asarray(step_size, jnp.float32)
    if find_init_step_size:
        _user_h = float(step_size)
        key, k = jax.random.split(key)
        step_size = _find_init_eps(k, state, U_fn, grad_U_fn, gamma, step_size)
        if verbose:
            print(f"[malt] find_init_step_size: step_size {_user_h:.4g} → "
                  f"{float(step_size):.4g}\n"
                  f"   (if the chain later stalls, set find_init_step_size=False "
                  f"and pass your own step_size — the heuristic can overshoot "
                  f"when the initial positions are under-dispersed vs the target.)")
    if verbose:
        print(f"MALT  gamma={gamma}  L={L}  init_eps={float(step_size):.4f}")

    log_eps0 = jnp.log(step_size)
    da = _da_init(log_eps0)

    # ── warmup ──────────────────────────────────────────────────────────────
    @jax.jit
    def _warmup_step(malt_state, da, key):
        eps = jnp.exp(da.log_eps) if adapt_step_size else step_size
        new_state, accept = _malt_step(malt_state, U_fn, grad_U_fn, eps, L, gamma, key)
        acc_rate = jnp.mean(accept.astype(float))
        if adapt_step_size:
            da = _da_update(da, acc_rate, log_eps0, target_accept)
        return new_state, da, acc_rate

    key, k = jax.random.split(key)
    wkeys = jax.random.split(k, warmup)
    total_acc = 0.
    for i in range(warmup):
        malt_state, da, acc = _warmup_step(malt_state, da, wkeys[i])
        total_acc += acc

    final_eps = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
    if verbose:
        print(f"Warmup done.  eps={float(final_eps):.4f}"
              f"  accept={float(total_acc)/max(warmup,1):.3f}")

    # ── production ──────────────────────────────────────────────────────────
    @jax.jit
    def _step(carry, key):
        malt_state = carry
        new_state, accept = _malt_step(malt_state, U_fn, grad_U_fn, final_eps, L, gamma, key)
        q = new_state[0]  # positions
        return new_state, (q, accept.astype(float))

    key, k = jax.random.split(key)
    skeys = jax.random.split(k, num_samples * thin_by)
    malt_state, (all_q, all_acc) = jax.lax.scan(_step, malt_state, skeys)

    samples = all_q[::thin_by]
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_step_size=float(final_eps),
                gamma=float(gamma),
                L=L)
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}")
    return samples, info


if __name__ == "__main__":

    # --- Test 1: Ill-conditioned Gaussian (20D, kappa=1000) ---
    print("=" * 60)
    print("Test 1: Ill-conditioned Gaussian  (D=20, kappa=1000)")
    print("=" * 60)
    dim = 20
    kappa = 1000.
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov_gauss = Q @ jnp.diag(eigvals) @ Q.T
    prec_gauss = Q @ jnp.diag(1. / eigvals) @ Q.T
    def log_prob_gauss(x):
        return -0.5 * jnp.dot(x, prec_gauss @ x)

    init = jax.random.normal(jax.random.key(42), (50, dim))

    for g in [0.0, 1.0, 2.0]:
        print(f"\n  gamma={g}:")
        samples, info = sampler_malt(log_prob_gauss, init, num_samples=5000,
                                     warmup=1000, seed=123, gamma=g, L=10)
        flat = samples.reshape(-1, dim)
        var_est = jnp.var(flat, axis=0)
        var_true = jnp.diag(cov_gauss)
        rel_err = jnp.mean(jnp.abs(var_est - var_true) / var_true)
        print(f"    mean_rel_err(var)={rel_err:.3f}"
              f"  var_range=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
              f"  (target: [{jnp.min(var_true):.2f}, {jnp.max(var_true):.2f}])")
        print(f"    info: {info}")

    # --- Test 2: Rosenbrock (10D) ---
    print()
    print("=" * 60)
    print("Test 2: Rosenbrock  (D=10, a=1, b=100)")
    print("=" * 60)
    a_ros, b_ros = 1.0, 100.0
    dim_ros = 10
    def log_prob_rosen(x):
        x_even = x[::2]
        x_odd = x[1::2]
        return -(b_ros * jnp.sum((x_odd - x_even**2)**2)
                 + jnp.sum((x_even - a_ros)**2))

    init_r = jax.random.normal(jax.random.key(42), (50, dim_ros))
    samples, info = sampler_malt(log_prob_rosen, init_r, num_samples=5000,
                                 warmup=1000, seed=123, gamma=1.0, L=10)
    flat = samples.reshape(-1, dim_ros)
    mean_even = jnp.mean(flat[:, ::2])
    mean_odd = jnp.mean(flat[:, 1::2])
    var_even = jnp.mean(jnp.var(flat[:, ::2], axis=0))
    var_odd = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
    print(f"  x_even: mean={mean_even:.3f}  var={var_even:.4f}"
          f"  (target: mean={a_ros}, var=0.5)")
    print(f"  x_odd:  mean={mean_odd:.3f}  var={var_odd:.4f}"
          f"  (target: mean=1.5, var~2.505)")
    print(f"  info: {info}")

    # --- Test 3: MALT as HMC (gamma=0) vs MALA (L=1) ---
    print()
    print("=" * 60)
    print("Test 3: Special cases — HMC (gamma=0) and MALA (L=1)")
    print("=" * 60)
    dim_s = 10
    def log_prob_simple(x):
        return -0.5 * jnp.sum(x**2)

    init_s = jax.random.normal(jax.random.key(42), (30, dim_s))

    print("\n  HMC (gamma=0, L=10):")
    samples, info = sampler_malt(log_prob_simple, init_s, num_samples=2000,
                                 warmup=500, seed=42, gamma=0.0, L=10)
    flat = samples.reshape(-1, dim_s)
    print(f"    mean={jnp.mean(flat):.3f}  var={jnp.mean(jnp.var(flat, axis=0)):.3f}"
          f"  (target: mean=0, var=1)  info: {info}")

    print("\n  MALA (gamma=2, L=1):")
    samples, info = sampler_malt(log_prob_simple, init_s, num_samples=2000,
                                 warmup=500, seed=42, gamma=2.0, L=1)
    flat = samples.reshape(-1, dim_s)
    print(f"    mean={jnp.mean(flat):.3f}  var={jnp.mean(jnp.var(flat, axis=0)):.3f}"
          f"  (target: mean=0, var=1)  info: {info}")
