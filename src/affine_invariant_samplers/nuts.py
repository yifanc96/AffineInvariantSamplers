"""
sampler_nuts — standard NUTS (no ensemble preconditioning), single file, JAX only.

Standard HMC dynamics with identity mass matrix.
Integration length determined per-chain by the No-U-Turn criterion.

Adaptation:
  Dual averaging       → step size            (warmup, adapt_step_size)
  Heuristic line search→ initial step size    (pre-warmup, find_init_step_size)
  NUTS (No-U-Turn)     → integration length   (every step, intrinsic; controlled
                                               by max_tree_depth)

Setting adapt_step_size=False and find_init_step_size=False reduces step-size
handling to a plain burn-in at the user-supplied `step_size`.

Reference: https://arxiv.org/abs/1111.4246
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Single-step leapfrog for NUTS tree building (one chain, identity mass)
# ──────────────────────────────────────────────────────────────────────────────

def _leapfrog_single(q, p, grad_U_single, eps):
    """One leapfrog step.  q:(D,)  p:(D,) → (q', p').  M=I."""
    g = grad_U_single(q)
    p = p - 0.5 * eps * g
    q = q + eps * p
    g = grad_U_single(q)
    p = p - 0.5 * eps * g
    return q, p


# ──────────────────────────────────────────────────────────────────────────────
# U-turn condition:  (z - z0) · p < 0
# ──────────────────────────────────────────────────────────────────────────────

def _uturn(z_left, r_left, z_right, r_right):
    """Standard NUTS U-turn: (z_right - z_left)·r_left < 0 or ·r_right < 0."""
    d = z_right - z_left
    return (jnp.dot(d, r_left) < 0) | (jnp.dot(d, r_right) < 0)


# ──────────────────────────────────────────────────────────────────────────────
# NUTS tree building  (per-chain, designed for vmap)
#
# Uses iterative doubling with progressive sampling (slice-free NUTS).
# Two selection schemes:
#   "progressive" (default) — biased progressive: min(1, w_new / w_cur)
#   "multinomial"           — unbiased:           w_new / (w_new + w_cur)
# Checkpoints are stored at even leaf indices for efficient turning detection.
# ──────────────────────────────────────────────────────────────────────────────

class _TreeState(NamedTuple):
    z_left: jnp.ndarray      # leftmost position
    r_left: jnp.ndarray      # leftmost momentum
    z_right: jnp.ndarray     # rightmost position
    r_right: jnp.ndarray     # rightmost momentum
    z_prop: jnp.ndarray      # proposal position
    lp_prop: float            # log prob of proposal
    depth: int
    log_weight: float         # log sum-weight for progressive selection
    turning: bool
    diverging: bool
    sum_accept: float         # sum of min(1, exp(-ΔH))
    n_proposals: int


def _leaf_to_ckpt_idxs(n):
    """Leaf index → (ckpt_idx_min, ckpt_idx_max) for checkpoint-based turning."""
    idx_max = jnp.bitwise_count(n >> 1).astype(jnp.int32)
    num_sub = jnp.bitwise_count((~n & (n + 1)) - 1).astype(jnp.int32)
    return idx_max - num_sub + 1, idx_max


def _build_nuts_tree(z0, log_prob_single, grad_U_single, eps, key,
                     max_depth, max_dE, progressive=True):
    """
    Build a NUTS tree for one chain (standard dynamics, M=I).

    Returns:
        z_prop, log_alpha_da, lp_prop, n_proposals
    """
    D = z0.shape[0]
    lp0 = log_prob_single(z0)

    # sample momentum  p ~ N(0, I_D)
    key, mk = jax.random.split(key)
    r0 = jax.random.normal(mk, (D,))
    KE0 = 0.5 * jnp.sum(r0**2)
    E0 = -lp0 + KE0

    # checkpoint arrays for sub-subtree U-turn detection
    z_ckpts = jnp.zeros((max_depth, D))
    r_ckpts = jnp.zeros((max_depth, D))

    # initial tree (single point)
    tree0 = _TreeState(
        z_left=z0, r_left=r0, z_right=z0, r_right=r0,
        z_prop=z0, lp_prop=lp0, depth=0,
        log_weight=jnp.float32(0.), turning=False, diverging=False,
        sum_accept=jnp.float32(0.), n_proposals=0,
    )

    # --- helpers closed over shared state ---

    def _one_step(z, r, direction):
        signed_eps = jnp.where(direction, eps, -eps)
        return _leapfrog_single(z, r, grad_U_single, signed_eps)

    def _base_tree(z, r, direction):
        z1, r1 = _one_step(z, r, direction)
        lp1 = log_prob_single(z1)
        KE1 = 0.5 * jnp.sum(r1**2)
        E1 = -lp1 + KE1
        dE = E1 - E0
        dE = jnp.where(jnp.isnan(dE), jnp.inf, dE)
        accept = jnp.clip(jnp.exp(-dE), 0., 1.)
        return _TreeState(
            z_left=z1, r_left=r1, z_right=z1, r_right=r1,
            z_prop=z1, lp_prop=lp1, depth=0, log_weight=-dE,
            turning=False, diverging=dE > max_dE,
            sum_accept=accept, n_proposals=1,
        )

    def _combine(cur, new, direction, rng, biased_progressive=False):
        z_left  = jnp.where(direction, cur.z_left,  new.z_left)
        r_left  = jnp.where(direction, cur.r_left,  new.r_left)
        z_right = jnp.where(direction, new.z_right, cur.z_right)
        r_right = jnp.where(direction, new.r_right, cur.r_right)

        log_w = jnp.logaddexp(cur.log_weight, new.log_weight)
        # Inside build_tree: always multinomial (unbiased).
        # At the top-level doubling: progressive if requested.
        if biased_progressive:
            accept_log_p = jnp.minimum(0., new.log_weight - cur.log_weight)
        else:
            accept_log_p = new.log_weight - log_w
        accept_new = jax.random.bernoulli(rng, jnp.clip(jnp.exp(accept_log_p), 0., 1.))
        z_prop = jnp.where(accept_new, new.z_prop, cur.z_prop)
        lp_prop = jnp.where(accept_new, new.lp_prop, cur.lp_prop)

        # NOTE: no endpoint U-turn check here.  Inside _build_subtree the
        # checkpoint system handles all power-of-2 sub-subtree U-turn checks
        # (matching the recursive NUTS algorithm).  The full-tree endpoint
        # U-turn is checked explicitly in body_fn after combining with the
        # main tree.
        turning = cur.turning | new.turning

        return _TreeState(
            z_left=z_left, r_left=r_left, z_right=z_right, r_right=r_right,
            z_prop=z_prop, lp_prop=lp_prop,
            depth=cur.depth + 1,
            log_weight=log_w,
            turning=turning,
            diverging=cur.diverging | new.diverging,
            sum_accept=cur.sum_accept + new.sum_accept,
            n_proposals=cur.n_proposals + new.n_proposals,
        )

    # --- iterative subtree builder ---

    def _build_subtree(proto_depth, z_start, r_start, direction, rng):
        max_leaves = 2 ** proto_depth

        def cond(state):
            tree, turn, _, _, _, n = state
            return (n < max_leaves) & ~turn & ~tree.turning & ~tree.diverging

        def body(state):
            tree, _, z_ck, r_ck, rng, n = state
            rng, rk = jax.random.split(rng)

            z_leaf = jnp.where(direction, tree.z_right, tree.z_left)
            r_leaf = jnp.where(direction, tree.r_right, tree.r_left)

            new = _base_tree(z_leaf, r_leaf, direction)

            combined = _combine(tree, new, direction, rk)
            tree = jax.lax.cond(n == 0, lambda: new, lambda: combined)

            # Store checkpoint at even leaf indices
            ckpt_min, ckpt_max = _leaf_to_ckpt_idxs(n)
            z_ck = jax.lax.cond(
                n % 2 == 0,
                lambda: z_ck.at[ckpt_max].set(new.z_right),
                lambda: z_ck,
            )
            r_ck = jax.lax.cond(
                n % 2 == 0,
                lambda: r_ck.at[ckpt_max].set(new.r_right),
                lambda: r_ck,
            )

            # Checkpoint U-turn: standard criterion with direction-aware ordering.
            # The extending tip of the tree and each checkpoint form a
            # (left, right) pair whose ordering depends on direction.
            z_tip = jnp.where(direction, tree.z_right, tree.z_left)
            r_tip = jnp.where(direction, tree.r_right, tree.r_left)

            def check_ckpts(state):
                i, turning = state
                z_c, r_c = z_ck[i], r_ck[i]
                # For forward: checkpoint is LEFT, tip is RIGHT
                # For backward: tip is LEFT, checkpoint is RIGHT
                zl = jnp.where(direction, z_c, z_tip)
                rl = jnp.where(direction, r_c, r_tip)
                zr = jnp.where(direction, z_tip, z_c)
                rr = jnp.where(direction, r_tip, r_c)
                turning = turning | _uturn(zl, rl, zr, rr)
                return i - 1, turning

            _, turn = jax.lax.while_loop(
                lambda s: (s[0] >= ckpt_min) & ~s[1],
                check_ckpts,
                (ckpt_max, jnp.bool_(False)),
            )

            return tree, turn, z_ck, r_ck, rng, n + 1

        init_tree = _TreeState(
            z_left=z_start, r_left=r_start,
            z_right=z_start, r_right=r_start,
            z_prop=z_start, lp_prop=jnp.float32(-jnp.inf),
            depth=0, log_weight=jnp.float32(-jnp.inf),
            turning=False, diverging=False,
            sum_accept=jnp.float32(0.), n_proposals=0,
        )
        tree, turn, z_ck, r_ck, _, _ = jax.lax.while_loop(
            cond, body,
            (init_tree, jnp.bool_(False), z_ckpts, r_ckpts, rng, jnp.int32(0)),
        )
        return tree._replace(depth=proto_depth, turning=tree.turning | turn)

    # --- main doubling loop ---

    def cond_fn(state):
        tree, _ = state
        return (tree.depth < max_depth) & ~tree.turning & ~tree.diverging

    def body_fn(state):
        tree, k = state
        k, dk, tk, ck = jax.random.split(k, 4)
        direction = jax.random.bernoulli(dk)

        z_leaf = jnp.where(direction, tree.z_right, tree.z_left)
        r_leaf = jnp.where(direction, tree.r_right, tree.r_left)

        sub = _build_subtree(tree.depth, z_leaf, r_leaf, direction, tk)

        # Save current proposal — per Algorithm 6, only accept subtree's
        # proposal when the subtree has no U-turn (sprime == 1).
        prev_z_prop = tree.z_prop
        prev_lp_prop = tree.lp_prop

        tree = _combine(tree, sub, direction, ck,
                        biased_progressive=progressive)

        # Full-tree endpoint U-turn check (matches reference Algorithm 6:
        # s = sprime AND stop_criterion(θ−, θ+, r−, r+))
        full_turn = _uturn(tree.z_left, tree.r_left,
                           tree.z_right, tree.r_right)
        tree = tree._replace(turning=tree.turning | full_turn)

        # Revert proposal if the subtree had a U-turn or divergence
        keep_old = sub.turning | sub.diverging
        tree = tree._replace(
            z_prop=jnp.where(keep_old, prev_z_prop, tree.z_prop),
            lp_prop=jnp.where(keep_old, prev_lp_prop, tree.lp_prop),
        )
        return tree, k

    tree, _ = jax.lax.while_loop(
        cond_fn, body_fn,
        (tree0, key),
    )

    accept_rate = tree.sum_accept / jnp.maximum(tree.n_proposals, 1)
    log_alpha_da = jnp.log(jnp.clip(accept_rate, 1e-10, 1.))
    return tree.z_prop, log_alpha_da, tree.lp_prop, tree.n_proposals


# ──────────────────────────────────────────────────────────────────────────────
# NUTS move — vmapped over chains
# ──────────────────────────────────────────────────────────────────────────────

def _nuts_move(positions, eps, key, log_prob, grad_U_vec, lp_cur,
               max_depth, max_dE, progressive=True):
    """Standard NUTS move for all chains.  vmapped internally."""
    N = positions.shape[0]
    keys = jax.random.split(key, N)

    def _one_chain(z0, lp0, k):
        lp_fn = lambda z: log_prob(z[None])[0]
        gU_fn = lambda z: grad_U_vec(z[None])[0]
        return _build_nuts_tree(
            z0, lp_fn, gU_fn, eps, k, max_depth, max_dE, progressive)

    proposed, log_alpha, lp_new, n_steps = jax.vmap(_one_chain)(
        positions, lp_cur, keys)
    return proposed, log_alpha, lp_new, n_steps


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
    accept = jnp.mean(jnp.clip(jnp.exp(log_alpha), 1e-10, 1.))
    eta    = 1. / (it + t0)
    H_bar  = (1. - eta) * state.H_bar + eta * (target - accept)
    log_e  = log_eps0 - jnp.sqrt(it) / ((it + t0) * gamma) * H_bar
    log_eb = it**(-kappa) * log_e + (1. - it**(-kappa)) * state.log_eps_bar
    return DAState(it, log_e, log_eb, H_bar)


# ──────────────────────────────────────────────────────────────────────────────
# Initial step-size search  (L=1 HMC probe, identity mass)
# ──────────────────────────────────────────────────────────────────────────────

def _hmc_probe(positions, eps, key, log_prob, grad_U, lp_cur):
    """Single L=1 HMC step for step-size probing.  p ~ N(0, I)."""
    N, D = positions.shape
    p0 = jax.random.normal(key, (N, D))
    # one leapfrog step
    g = grad_U(positions)
    p = p0 - 0.5 * eps * g
    q = positions + eps * p
    p = p - 0.5 * eps * grad_U(q)
    lp1 = log_prob(q)
    dH = (-lp1 + 0.5 * jnp.sum(p**2, axis=1)) \
       - (-lp_cur + 0.5 * jnp.sum(p0**2, axis=1))
    dH = jnp.where(jnp.isnan(dH), jnp.inf, dH)
    return jnp.minimum(0., -dH)


def _find_init_eps(key, positions, log_prob, grad_U, eps0):
    lp = log_prob(positions)
    fi = jnp.finfo(jnp.result_type(eps0))

    def body(s):
        eps, _, d, k = s
        k, k1 = jax.random.split(k)
        eps = (2.**d) * eps
        la = _hmc_probe(positions, eps, k1, log_prob, grad_U, lp)
        avg = jnp.log(la.shape[0]) - jax.scipy.special.logsumexp(-la)
        return eps, d, jnp.where(jnp.log(.8) < avg, 1, -1), k

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps / 2.


# ──────────────────────────────────────────────────────────────────────────────
# sampler_nuts
# ──────────────────────────────────────────────────────────────────────────────

def sampler_nuts(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup           = 1000,
    sampling         = "progressive",  # "progressive" or "multinomial"
    step_size        = 0.1,
    max_tree_depth   = 10,
    thin_by          = 1,
    target_accept    = 0.80,
    max_delta_energy = 1000.,
    grad_log_prob_fn    = None,
    find_init_step_size = False,
    adapt_step_size     = True,
    seed                = 0,
    verbose             = True,
):
    """
    Standard NUTS with automatic step-size tuning.
    No ensemble preconditioning — identity mass matrix, p ~ N(0, I).

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations (dual averaging for step size).
        sampling         : "progressive" (biased progressive, default) or
                           "multinomial" (unbiased multinomial).
        step_size        : Initial step size (adapted during warmup).
        max_tree_depth   : Maximum NUTS tree depth (trajectory ≤ 2^depth steps).
        thin_by          : Keep every thin_by-th sample.
        target_accept    : Target acceptance rate for dual averaging.
        max_delta_energy : Energy threshold for divergence detection.
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
                           If None, uses jax.vmap(jax.grad(log_prob_fn)).
        find_init_step_size : If True, run a short heuristic search at
                              the initial positions to scale `step_size` to
                              ~80% acceptance before warmup.
                              If False, use `step_size` as-is.
        adapt_step_size  : If True, tune step size by dual averaging during
                           warmup.  If False, use `step_size` as given.
                           (Trajectory length is always NUTS-adaptive; to
                           cap it, lower `max_tree_depth`.)
        seed             : Integer random seed.
        verbose          : Print progress.

    Returns:
        samples : (num_samples, n_chains, D)
        info    : dict(acceptance_rate, final_step_size, mean_tree_depth)
    """
    state    = jnp.asarray(initial_state)
    n_chains, dim = state.shape
    assert sampling in ("progressive", "multinomial"), \
        f"sampling must be 'progressive' or 'multinomial', got '{sampling}'"

    _progressive = sampling == "progressive"

    if grad_log_prob_fn is None:
        grad_U = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_U = grad_log_prob_fn
    _grad_U = lambda x: -grad_U(x)

    key = jax.random.key(seed)

    # --- initial step size ---
    if find_init_step_size:
        key, k = jax.random.split(key)
        step_size = _find_init_eps(k, state, log_prob_fn, _grad_U, step_size)
    step_size = jnp.asarray(step_size)
    if verbose:
        print(f"sampling={sampling}  max_depth={max_tree_depth}"
              f"  init_eps={float(step_size):.4f}"
              f"  find_init_step_size={find_init_step_size}"
              f"  adapt_step_size={adapt_step_size}")

    log_eps0 = jnp.log(step_size)
    da  = _da_init(log_eps0)
    lp  = log_prob_fn(state)

    # --- warmup ---
    @jax.jit
    def _warmup_step(positions, lp, da, keys):
        k1, = keys
        eps = jnp.exp(da.log_eps) if adapt_step_size else step_size

        # NUTS progressive/multinomial: always accept (no MH).
        proposed, la, lp_new, ns = _nuts_move(
            positions, eps, k1, log_prob_fn, _grad_U, lp,
            max_tree_depth, max_delta_energy, _progressive)
        if adapt_step_size:
            da = _da_update(da, la, log_eps0, target_accept)

        acc = jnp.mean(jnp.exp(la))
        mean_ns = jnp.mean(ns)
        return proposed, lp_new, da, acc, mean_ns

    key, k = jax.random.split(key)
    flat  = jax.random.split(k, warmup)
    wkeys = flat.reshape(warmup, 1, *flat.shape[1:])
    total_acc = 0.
    total_ns  = 0.
    for i in range(warmup):
        state, lp, da, acc, mean_ns = _warmup_step(state, lp, da, wkeys[i])
        total_acc += acc
        total_ns  += mean_ns

    final_eps = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
    if verbose:
        print(f"Warmup done.  eps={float(final_eps):.4f}"
              f"  accept={float(total_acc)/max(warmup,1):.3f}"
              f"  mean_steps={float(total_ns)/max(warmup,1):.1f}")

    # --- main sampling ---
    @jax.jit
    def _step(carry, keys):
        positions, lp = carry
        k1, = keys

        proposed, la, lp_new, ns = _nuts_move(
            positions, final_eps, k1, log_prob_fn, _grad_U, lp,
            max_tree_depth, max_delta_energy, _progressive)

        accept = jnp.exp(la)  # tree acceptance diagnostic
        return (proposed, lp_new), (proposed, accept, ns)

    key, k  = jax.random.split(key)
    flat    = jax.random.split(k, num_samples * thin_by)
    skeys   = flat.reshape(num_samples * thin_by, 1, *flat.shape[1:])
    (state, lp), (all_states, all_acc, all_ns) = jax.lax.scan(
        _step, (state, lp), skeys)

    samples = all_states[::thin_by]
    info = dict(acceptance_rate=float(jnp.mean(all_acc)),  # mean tree energy acceptance
                final_step_size=float(final_eps),
                mean_tree_depth=float(jnp.mean(jnp.log2(all_ns + 1))))
    if verbose:
        print(f"Done.  accept={info['acceptance_rate']:.3f}"
              f"  mean_steps={float(jnp.mean(all_ns)):.1f}")
    return samples, info


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim  = 2
    cov  = jnp.array([[1., .95], [.95, 1.]])
    prec = jnp.linalg.inv(cov)
    def log_prob(x):
        return -0.5 * jnp.sum(x @ prec * x, axis=-1)

    init = jax.random.normal(jax.random.key(42), (20, dim))

    print("=" * 60)
    print("NUTS (standard)")
    print("=" * 60)
    samples, info = sampler_nuts(log_prob, init, num_samples=2000, warmup=400, seed=123)
    print(f"mean : {jnp.mean(samples, axis=(0,1))}")
    print(f"cov  :\n{jnp.cov(samples.reshape(-1, dim), rowvar=False)}")
    print(f"info : {info}")
