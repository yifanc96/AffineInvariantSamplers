"""
sampler_peaches — ensemble preconditioned HMC, single file, JAX only.

Moves:
  h-walk : matrix momentum in the span of the complement ensemble
  h-side : scalar momentum along a random ensemble direction

Adaptation (warmup only, each independently toggleable):
  Dual averaging       → step size             (adapt_step_size)
  ChEES criterion      → integration length    (adapt_L)
  Heuristic line search→ initial step size     (find_init_step_size)

Setting all three flags to False reduces warmup to a plain burn-in at the
user-supplied (step_size, L).

Reference: https://arxiv.org/abs/2505.02987
"""

import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Leapfrog integrators
# ──────────────────────────────────────────────────────────────────────────────

def _leapfrog_walk(q, p, grad_U, eps, L, centered):
    """q:(W,D)  p:(W,W)  centered:(W,D) — returns (q', p', velocity)."""
    g = grad_U(q)
    p = p - 0.5 * eps * (g @ centered.T)
    def step(_, s):
        q, p = s
        q = q + eps * (p @ centered)
        p = p - eps * (grad_U(q) @ centered.T)
        return q, p
    q, p = jax.lax.fori_loop(0, L - 1, step, (q, p))
    q = q + eps * (p @ centered)
    p = p - 0.5 * eps * (grad_U(q) @ centered.T)
    return q, p, p @ centered          # velocity = p @ centered


def _leapfrog_side(q, p, grad_U, eps, L, diff):
    """q:(W,D)  p:(W,)  diff:(W,D) — returns (q', p', velocity)."""
    g = grad_U(q)
    p = p - 0.5 * eps * jnp.sum(g * diff, axis=1)
    def step(_, s):
        q, p = s
        q = q + eps * p[:, None] * diff
        p = p - eps * jnp.sum(grad_U(q) * diff, axis=1)
        return q, p
    q, p = jax.lax.fori_loop(0, L - 1, step, (q, p))
    q = q + eps * p[:, None] * diff
    p = p - 0.5 * eps * jnp.sum(grad_U(q) * diff, axis=1)
    return q, p, p[:, None] * diff    # velocity = p * diff


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble moves — return (proposed, log_alpha, velocity, proposed_lp, aux)
#   aux = centered  for h-walk  (used by affine ChEES if ever needed)
#   aux = diff      for h-side  (used by affine_1d ChEES)
# ──────────────────────────────────────────────────────────────────────────────

def h_walk(group, complement, eps, key, log_prob, grad_U, L, lp_group):
    """Hamiltonian walk move (Algorithm 3 of https://arxiv.org/abs/2505.02987)."""
    W = group.shape[0]
    centered = (complement - jnp.mean(complement, axis=0)) / jnp.sqrt(W)
    p0 = jax.random.normal(key, (W, W))
    proposed, p1, vel = _leapfrog_walk(group, p0, grad_U, eps, L, centered)
    lp1 = log_prob(proposed)
    dH  = (-lp1 + 0.5 * jnp.sum(p1**2, axis=1)) \
        - (-lp_group + 0.5 * jnp.sum(p0**2, axis=1))
    dH  = jnp.where(jnp.isnan(dH), jnp.inf, dH)
    return proposed, jnp.minimum(0., -dH), vel, lp1, centered


def h_side(group, complement, eps, key, log_prob, grad_U, L, lp_group):
    """Hamiltonian side move (Algorithm 4 of https://arxiv.org/abs/2505.02987)."""
    W, D = group.shape
    keys = jax.random.split(key, W + 1)
    idx  = jnp.arange(W)
    ch   = jax.vmap(lambda k: jax.random.choice(k, idx, (2,), replace=False))(keys[:W])
    diff = (complement[ch[:, 0]] - complement[ch[:, 1]]) / jnp.sqrt(2 * D)
    p0 = jax.random.normal(keys[-1], (W,))
    proposed, p1, vel = _leapfrog_side(group, p0, grad_U, eps, L, diff)
    lp1 = log_prob(proposed)
    dH  = (-lp1 + 0.5 * p1**2) - (-lp_group + 0.5 * p0**2)
    dH  = jnp.where(jnp.isnan(dH), jnp.inf, dH)
    return proposed, jnp.minimum(0., -dH), vel, lp1, diff


# ──────────────────────────────────────────────────────────────────────────────
# Metropolis accept / reject
# ──────────────────────────────────────────────────────────────────────────────

def _mh(current, proposed, log_alpha, key):
    accept = jnp.log(jax.random.uniform(key, log_alpha.shape, minval=1e-10)) < log_alpha
    return jnp.where(accept[:, None], proposed, current), accept


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

def _da_update(state, log_alpha, log_eps0, target, t0=10., gamma=0.05, kappa=0.75):
    it     = state.iteration + 1
    accept = log_alpha.size / jnp.sum(1. / jnp.clip(jnp.exp(log_alpha), 1e-10, 1.))
    eta    = 1. / (it + t0)
    H_bar  = (1. - eta) * state.H_bar + eta * (target - accept)
    log_e  = log_eps0 - jnp.sqrt(it) / ((it + t0) * gamma) * H_bar
    log_eb = it**(-kappa) * log_e + (1. - it**(-kappa)) * state.log_eps_bar
    return DAState(it, log_e, log_eb, H_bar)


# ──────────────────────────────────────────────────────────────────────────────
# ChEES — integration length via ADAM ascent on trajectory quality
# ──────────────────────────────────────────────────────────────────────────────

class ChEESState(NamedTuple):
    log_T: float;  log_T_bar: float
    m: float;      v: float
    iteration: int; halton: float

@jax.jit
def _halton(n, base=2):
    i, b = jnp.asarray(n, jnp.int32), jnp.asarray(base, jnp.int32)
    def body(s):
        i, f, r = s; f = f / jnp.float32(b)
        return i // b, f, r + f * jnp.mod(i, b)
    _, _, r = jax.lax.while_loop(lambda s: s[0] > 0, body, (i, 1., 0.))
    return r

def _chees_init(eps, L):
    T = eps * L
    return ChEESState(jnp.log(T), jnp.log(T), 0., 0., 1, _halton(1))

def _chees_update(state, log_alpha, pos_cur, pos_pro, vel, complement, aux, metric,
                  lr=0.025, beta1=0., beta2=0.95, reg=1e-7,
                  T_min=0.25, T_max=10., T_interp=0.9):
    """
    ChEES gradient signal:  g = t_n * diff_sq * inner,
    where diff_sq and inner are computed in the chosen metric.

    "affine"    (h-walk): full Sigma^{-1} metric.
                  diff_sq = c_pro^T S^{-1} c_pro - c_cur^T S^{-1} c_cur
                  inner   = (S^{-1} c_pro)^T vel        [reuses solve for c_pro]

    "affine_1d" (h-side): project onto each chain's diff direction.
                  aux = diff  (W, D)
                  proj = (c · diff) / ||diff||^2  →  recovers scalar displacement
                  Affine-invariant when c ∝ diff (exact for side move up to
                  the ensemble-mean correction, which is O(1/W)).
                  No matrix solve required.

    "euclidean": plain dot products for both terms.
    """
    alpha = jnp.clip(jnp.exp(log_alpha), 0., 1.)
    c_cur = pos_cur - jnp.mean(pos_cur, axis=0)
    c_pro = pos_pro - jnp.mean(pos_pro, axis=0)

    if metric == "affine-invariant":
        cov  = jnp.atleast_2d(jnp.cov(complement, rowvar=False))
        W    = c_cur.shape[0]
        sol  = jnp.linalg.solve(cov, jnp.concatenate([c_cur.T, c_pro.T], axis=1))
        Sc   = sol[:, :W].T;   Sp = sol[:, W:].T    # Sigma^{-1} c_cur, c_pro
        diff_sq = jnp.sum(c_pro * Sp, 1) - jnp.sum(c_cur * Sc, 1)
        inner   = jnp.sum(Sp * vel, 1)              # = c_pro^T Sigma^{-1} vel

    elif metric == "affine-invariant_1d":
        d    = aux                                   # diff directions  (W, D)
        nsq  = jnp.sum(d**2, axis=1)                # ||diff_i||^2     (W,)
        pp   = jnp.sum(c_pro * d, axis=1) / nsq     # scalar projection of c_pro
        pc   = jnp.sum(c_cur * d, axis=1) / nsq     # scalar projection of c_cur
        pv   = jnp.sum(vel   * d, axis=1) / nsq     # recovers scalar p_i
        diff_sq = pp**2 - pc**2
        inner   = pp * pv

    else:                                            # euclidean
        diff_sq = jnp.sum(c_pro**2, 1) - jnp.sum(c_cur**2, 1)
        inner   = jnp.sum(c_pro * vel, 1)

    g_m = state.halton * jnp.exp(state.log_T) * diff_sq * inner
    g_m = jnp.where((alpha > 1e-4) & jnp.isfinite(g_m), g_m, 0.)
    g   = jnp.sum(alpha * g_m) / (jnp.sum(alpha) + reg)

    it = state.iteration + 1
    m  = beta1 * state.m + (1 - beta1) * g
    v  = beta2 * state.v + (1 - beta2) * g**2
    delta = lr * (m / (1 - beta1**it)) / jnp.sqrt(v / (1 - beta2**it) + reg)
    delta = jnp.clip(delta, -0.35, 0.35)       # per-step clip (TFP/BlackJAX)
    log_T = jnp.clip(state.log_T + delta, jnp.log(T_min), jnp.log(T_max))
    log_Tb = jnp.logaddexp(jnp.log(T_interp) + state.log_T_bar,
                            jnp.log(1 - T_interp) + log_T)
    log_Tb = jnp.clip(log_Tb, jnp.log(T_min), jnp.log(T_max))
    return ChEESState(log_T, log_Tb, m, v, it, _halton(it))

def _chees_L(state, eps, jitter=0.6, bar=False, max_L=100):
    T = jnp.exp(state.log_T_bar if bar else state.log_T)
    T = (1 - jitter) * T + jitter * state.halton * T
    return jnp.clip(jnp.ceil(T / eps), 1, max_L).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search (~80% acceptance with L=1)
# ──────────────────────────────────────────────────────────────────────────────

def _find_init_eps(key, g1, g2, log_prob, grad_U, eps0, move_fn):
    lp1, lp2 = log_prob(g1), log_prob(g2)
    fi = jnp.finfo(jnp.result_type(eps0))

    def body(s):
        eps, _, d, k = s
        k, k1, k2 = jax.random.split(k, 3)
        eps = (2.**d) * eps
        _, la1, _, _, _ = move_fn(g1, g2, eps, k1, log_prob, grad_U, 1, lp1)
        _, la2, _, _, _ = move_fn(g2, g1, eps, k2, log_prob, grad_U, 1, lp2)
        la  = jnp.concatenate([la1, la2])
        avg = jnp.log(la.shape[0]) - jax.scipy.special.logsumexp(-la)
        return eps, d, jnp.where(jnp.log(.8) < avg, 1, -1), k

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps / 2.


# ──────────────────────────────────────────────────────────────────────────────
# sampler_peaches
# ──────────────────────────────────────────────────────────────────────────────

def sampler_peaches(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup           = 1000,
    move             = "h-walk",    # "h-walk" or "h-side"
    step_size        = 0.05,
    L                = 5,
    max_L            = 20,
    thin_by          = 1,
    target_accept    = 0.651,
    chees_metric     = "affine-invariant",    # "affine-invariant" (auto per move) or "euclidean"
    grad_log_prob_fn = None,        # (batch,D)->(batch,D), or None for JAX autodiff
    find_init_step_size   = True,
    adapt_step_size  = True,
    adapt_L          = True,
    seed             = 0,
    verbose          = True,
):
    """
    Ensemble preconditioned HMC with automatic step-size and integration-length tuning.

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).  n_chains must be even and >= 4.
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations.
        move             : "h-walk" or "h-side".
        step_size        : Initial step size (adapted during warmup).
        L                : Initial leapfrog steps (adapted during warmup).
        max_L            : Maximum leapfrog steps (default 100).
        thin_by          : Keep every thin_by-th sample.
        target_accept    : Target acceptance rate for dual averaging.
        chees_metric     : "affine-invariant"    — Sigma^{-1} metric for h-walk;
                                                   1-D diff projection for h-side.
                           "euclidean" — plain dot products for both.
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
                           If None, uses jax.vmap(jax.grad(log_prob_fn)).
        find_init_step_size : If True (default), run a short heuristic search at
                              the initial positions to scale `step_size` to
                              ~80% acceptance before warmup.
                              If False, use `step_size` as-is.
        adapt_step_size  : If True, tune step size by dual averaging during
                           warmup.  If False, use `step_size` as given.
        adapt_L          : If True, tune integration length by ChEES during
                           warmup.  If False, use `L` as given.
        seed             : Integer random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_step_size, final_L)
    """
    assert chees_metric in ("affine-invariant", "euclidean"), \
        "chees_metric must be 'affine-invariant' or 'euclidean'"
    state    = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert n_chains >= 4 and n_chains % 2 == 0, "Need >= 4 even chains"
    assert 1 <= L <= max_L, f"Need 1 <= L <= max_L (got L={L}, max_L={max_L})"

    move_fn = h_walk if move == "h-walk" else h_side
    # resolve internal metric
    _metric = ("affine-invariant" if move == "h-walk" else "affine-invariant_1d") \
              if chees_metric == "affine-invariant" else "euclidean"

    if grad_log_prob_fn is None:
        grad_U = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_U = grad_log_prob_fn
    # negate: leapfrog uses gradient of U = -log_prob
    _grad_U = lambda x: -grad_U(x)

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    key = jax.random.key(seed)

    if find_init_step_size:
        key, k = jax.random.split(key)
        step_size = _find_init_eps(k, g1, g2, log_prob_fn, _grad_U, step_size, move_fn)
    step_size = jnp.asarray(step_size)
    if verbose:
        print(f"move={move}  metric={_metric}  init_eps={float(step_size):.4f}"
              f"  find_init_step_size={find_init_step_size}"
              f"  adapt_step_size={adapt_step_size}  adapt_L={adapt_L}")

    log_eps0 = jnp.log(step_size)
    da  = _da_init(log_eps0)
    ch  = _chees_init(step_size, L)
    lp1 = log_prob_fn(g1);  lp2 = log_prob_fn(g2)

    # Closure-captured Python bools resolve at trace time, so unused
    # adaptation branches are removed from the compiled graph entirely.
    fixed_L = jnp.int32(L)

    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, ch, keys):
        k1, k2, ka1, ka2 = keys
        eps   = jnp.exp(da.log_eps) if adapt_step_size else step_size
        cur_L = _chees_L(ch, eps, max_L=max_L) if adapt_L else fixed_L
        p1, la1, v1, lp1n, aux1 = move_fn(g1, g2, eps, k1, log_prob_fn, _grad_U,
                                            cur_L, lp1)
        if adapt_step_size:
            da = _da_update(da, la1, log_eps0, target_accept)
        if adapt_L:
            ch = _chees_update(ch, la1, g1, p1, v1, g2, aux1, _metric)
        g1, a1 = _mh(g1, p1, la1, ka1);   lp1 = jnp.where(a1, lp1n, lp1)

        eps   = jnp.exp(da.log_eps) if adapt_step_size else step_size
        cur_L = _chees_L(ch, eps, max_L=max_L) if adapt_L else fixed_L
        p2, la2, v2, lp2n, aux2 = move_fn(g2, g1, eps, k2, log_prob_fn, _grad_U,
                                            cur_L, lp2)
        if adapt_step_size:
            da = _da_update(da, la2, log_eps0, target_accept)
        if adapt_L:
            ch = _chees_update(ch, la2, g2, p2, v2, g1, aux2, _metric)
        g2, a2 = _mh(g2, p2, la2, ka2);   lp2 = jnp.where(a2, lp2n, lp2)

        acc = (jnp.mean(a1.astype(float)) + jnp.mean(a2.astype(float))) / 2
        return g1, g2, lp1, lp2, da, ch, acc

    key, k = jax.random.split(key)
    flat  = jax.random.split(k, warmup * 4)
    wkeys = flat.reshape(warmup, 4, *flat.shape[1:])
    total_acc = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, ch, acc = _warmup_step(g1, g2, lp1, lp2, da, ch, wkeys[i])
        total_acc += acc

    final_eps   = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
    if adapt_L:
        final_log_T = ch.log_T_bar                # learned trajectory time (log)
        nominal_L   = _chees_L(ch, final_eps, bar=True, max_L=max_L)
    else:
        final_log_T = jnp.log(final_eps * L)
        nominal_L   = fixed_L
    if verbose:
        print(f"Warmup done.  eps={float(final_eps):.4f}  L={int(nominal_L)}"
              f"  accept={float(total_acc)/max(warmup,1):.3f}")

    # production: jitter L each step via Halton sequence (standard ChEES)
    jitter = 0.6
    halton_offset = ch.iteration          # continue Halton from where warmup left off

    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2, step_i = carry
        k1, k2, ka1, ka2 = keys
        if adapt_L:
            h = _halton(halton_offset + step_i)
            T = jnp.exp(final_log_T)
            T_jit = (1 - jitter) * T + jitter * h * T
            Lc = jnp.clip(jnp.ceil(T_jit / final_eps), 1, max_L).astype(int)
        else:
            Lc = fixed_L
        p1, la1, _, lp1n, _ = move_fn(g1, g2, final_eps, k1, log_prob_fn, _grad_U, Lc, lp1)
        g1, a1 = _mh(g1, p1, la1, ka1);   lp1 = jnp.where(a1, lp1n, lp1)
        p2, la2, _, lp2n, _ = move_fn(g2, g1, final_eps, k2, log_prob_fn, _grad_U, Lc, lp2)
        g2, a2 = _mh(g2, p2, la2, ka2);   lp2 = jnp.where(a2, lp2n, lp2)
        state  = jnp.concatenate([g1, g2])
        accept = jnp.concatenate([a1, a2]).astype(float)
        return (g1, g2, lp1, lp2, step_i + 1), (state, accept)

    key, k  = jax.random.split(key)
    flat    = jax.random.split(k, num_samples * thin_by * 4)
    skeys   = flat.reshape(num_samples * thin_by, 4, *flat.shape[1:])
    (g1, g2, lp1, lp2, _), (all_states, all_acc) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2, jnp.int32(0)), skeys)

    samples = all_states[::thin_by]
    # Production gradient evals: nominal_L per trajectory per walker, across
    # num_samples * thin_by iterations and n_chains walkers.  Gradient caching
    # between trajectories is ignored (off-by-one per iteration).
    n_grad_evals = int(num_samples * thin_by) * int(nominal_L) * int(n_chains)
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),
                final_step_size=float(final_eps),
                nominal_L=int(nominal_L),
                n_grad_evals=n_grad_evals)
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}")
    return samples, info


if __name__ == "__main__":

    # # --- Test 1: Ill-conditioned Gaussian (20D, kappa=1000) ---
    # print("=" * 60)
    # print("Test 1: Ill-conditioned Gaussian  (D=20, kappa=1000)")
    # print("=" * 60)
    # dim = 20
    # kappa = 1000.  # condition number
    # # Eigenvalues log-spaced from 1 to kappa
    # eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    # # Random orthogonal basis
    # Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    # cov_gauss = Q @ jnp.diag(eigvals) @ Q.T
    # prec_gauss = Q @ jnp.diag(1. / eigvals) @ Q.T
    # def log_prob_gauss(x):
    #     return -0.5 * jnp.sum((x @ prec_gauss) * x, axis=-1)

    # init = jax.random.normal(jax.random.key(42), (100, dim))
    # for ut in ["affine-invariant", "euclidean"]:
    #     samples, info = sampler_peaches(log_prob_gauss, init, num_samples=5000,
    #                                     warmup=1000, seed=123, step_size=0.01, chees_metric=ut)
    #     flat = samples.reshape(-1, dim)
    #     var_est = jnp.var(flat, axis=0)
    #     var_true = jnp.diag(cov_gauss)
    #     rel_err = jnp.mean(jnp.abs(var_est - var_true) / var_true)
    #     print(f"  {ut}:  mean_rel_err(var)={rel_err:.3f}"
    #           f"  var_range=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
    #           f"  (target: [{jnp.min(var_true):.2f}, {jnp.max(var_true):.2f}])")
    #     print(f"    info: {info}")

    # # --- Test 2: Rosenbrock (20D) ---
    # print()
    # print("=" * 60)
    # print("Test 2: Rosenbrock  (D=20, a=1, b=100)")
    # print("=" * 60)
    # # p(x) ~ exp(-(sum_i [ b*(x_{2i+1} - x_{2i}^2)^2 + (x_{2i} - a)^2 ]))
    # # x_e ~ N(a, 1/2): mean=a, var=0.5
    # # x_o | x_e ~ N(x_e^2, 1/(2b)):  E[x_o]=E[x_e^2]=1.5, Var(x_o)=Var(x_e^2)+1/(2b)≈2.505
    # a_ros, b_ros = 1.0, 100.0
    # dim_ros = 10
    # def log_prob_rosen(x):
    #     x_even = x[:, ::2]
    #     x_odd = x[:, 1::2]
    #     return -(b_ros * jnp.sum((x_odd - x_even**2)**2, axis=1)
    #              + jnp.sum((x_even - a_ros)**2, axis=1))

    # init_r = jax.random.normal(jax.random.key(42), (100, dim_ros))
    # samples, info = sampler_peaches(log_prob_rosen, init_r, num_samples=2000,
    #                                 warmup=500, seed=123, step_size=0.01)
    # flat = samples.reshape(-1, dim_ros)
    # mean_even = jnp.mean(flat[:, ::2])
    # mean_odd = jnp.mean(flat[:, 1::2])
    # var_even = jnp.mean(jnp.var(flat[:, ::2], axis=0))
    # var_odd = jnp.mean(jnp.var(flat[:, 1::2], axis=0))
    # print(f"  x_even: mean={mean_even:.3f}  var={var_even:.4f}"
    #       f"  (target: mean={a_ros}, var=0.5)")
    # print(f"  x_odd:  mean={mean_odd:.3f}  var={var_odd:.4f}"
    #       f"  (target: mean=1.5, var~2.505)")
    # print(f"  info: {info}")

    # --- Test 3: Neal's Funnel (3D) ---
    print()
    print("=" * 60)
    print("Test 3: Neal's Funnel  (D=3: v ~ N(0,9), x_i|v ~ N(0,exp(v)))")
    print("=" * 60)
    # Exact: E[v]=0, Var[v]=9, E[x_i]=0, Var[x_i]=E[exp(v)]=exp(9/2)~90.0
    funnel_dim = 3
    def log_prob_funnel(x):
        v = x[:, 0]
        xs = x[:, 1:]
        log_p_v = -0.5 * v**2 / 9.
        log_p_x = -0.5 * jnp.sum(xs**2 * jnp.exp(-v)[:, None], axis=1) \
                   - 0.5 * (funnel_dim - 1) * v
        return log_p_v + log_p_x

    init_f = jax.random.normal(jax.random.key(99), (100, funnel_dim))
    samples, info = sampler_peaches(log_prob_funnel, init_f, num_samples=50000,
                                    warmup=10000, seed=42, step_size=0.01, move="h-side",
                                    find_init_step_size=False)
    flat = samples.reshape(-1, funnel_dim)
    v_samples = flat[:, 0]
    x_samples = flat[:, 1:]
    print(f"  v:   mean={jnp.mean(v_samples):.3f}  var={jnp.var(v_samples):.2f}"
          f"  (target: mean=0, var=9)")
    print(f"  x_i: mean={jnp.mean(x_samples):.3f}  var={jnp.mean(jnp.var(x_samples, axis=0)):.1f}"
          f"  (target: mean=0, var~90.0)")
    print(f"  info: {info}")
    print("  (Funnel is a hard test — geometry varies drastically with v.)")