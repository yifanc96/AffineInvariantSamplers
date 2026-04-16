"""
sampler_peanuts — ensemble preconditioned PEANUTS, single file, JAX only.

Moves:
  h-walk : matrix momentum in the span of the complement ensemble
  h-side : scalar momentum along a random ensemble direction

Adaptation:
  Dual averaging       → step size            (warmup, adapt_step_size)
  Heuristic line search→ initial step size    (pre-warmup, find_init_step_size)
  PEANUTS (No-U-Turn)  → integration length   (every step, intrinsic; controlled
                                               by max_tree_depth)

Setting adapt_step_size=False and find_init_step_size=False reduces step-size
handling to a plain burn-in at the user-supplied `step_size`.

Reference: https://arxiv.org/abs/2505.02987
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Leapfrog integrators  (identical to ChEES version)
# ──────────────────────────────────────────────────────────────────────────────

def _leapfrog_walk(q, p, grad_U, eps, L, centered):
    """q:(W,D)  p:(W,W)  centered:(W,D) — returns (q', p')."""
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
    return q, p


def _leapfrog_side(q, p, grad_U, eps, L, diff):
    """q:(W,D)  p:(W,)  diff:(W,D) — returns (q', p')."""
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
    return q, p


# ──────────────────────────────────────────────────────────────────────────────
# Single-step leapfrog for PEANUTS tree building (operates on one chain at a time)
# ──────────────────────────────────────────────────────────────────────────────

def _leapfrog_walk_single(q, p, grad_U_single, eps, centered):
    """One leapfrog step.  q:(D,)  p:(W,)  centered:(W,D) → (q', p')."""
    g = grad_U_single(q)
    p = p - 0.5 * eps * (centered @ g)       # (W,D)@(D,) = (W,)
    q = q + eps * (centered.T @ p)            # (D,W)@(W,) = (D,)
    g = grad_U_single(q)
    p = p - 0.5 * eps * (centered @ g)
    return q, p


def _leapfrog_side_single(q, p, grad_U_single, eps, diff):
    """One leapfrog step.  q:(D,)  p:()  diff:(D,) → (q', p')."""
    g = grad_U_single(q)
    p = p - 0.5 * eps * jnp.dot(g, diff)
    q = q + eps * p * diff
    g = grad_U_single(q)
    p = p - 0.5 * eps * jnp.dot(g, diff)
    return q, p


# ──────────────────────────────────────────────────────────────────────────────
# U-turn conditions
#
# Two variants:
#   "euclidean" : standard NUTS criterion  (z_r - z_l) · v  < 0
#   "affine-invariant"    : affine-invariant criterion  (z - z0)^T Σ^{-1} v  < 0
#                 where v = C^T r (walk) or r·d (side)
# Both check BOTH endpoints to detect the U-turn.
# ──────────────────────────────────────────────────────────────────────────────

# --- euclidean ---

def _uturn_walk_euclidean(z_l, r_l, z_r, r_r, centering):
    """Walk U-turn (euclidean): (z_r - z_l) · v  < 0 for either endpoint."""
    dz = z_r - z_l
    v_l = centering.T @ r_l   # (D,W)@(W,) = (D,)
    v_r = centering.T @ r_r
    return (jnp.dot(dz, v_l) < 0) | (jnp.dot(dz, v_r) < 0)


def _uturn_side(z_l, r_l, z_r, r_r, diff):
    """Side U-turn: movement is 1D along diff, so check (z_r-z_l)·diff * r < 0."""
    proj = jnp.dot(z_r - z_l, diff)
    return (r_l * proj < 0) | (r_r * proj < 0)


# --- affine-invariant ---
# Uses Cholesky factor L of Σ for numerical stability and efficiency:
#   dz^T Σ^{-1} v = (L^{-1} dz)^T (L^{-1} v)
# Precomputed L_inv_CT = L^{-1} C^T avoids redundant work in the inner loop.
# Side mode is 1D (scalar momentum along diff) — no matrix needed.

def _uturn_walk_affine(z_l, r_l, z_r, r_r, L_inv, L_inv_CT):
    """Walk U-turn (affine-invariant): (z_r-z_l)^T Σ^{-1} C^T r < 0.
    L_inv   : (D,D) inverse of cholesky(Σ), precomputed.
    L_inv_CT: (D,W) = L^{-1} @ C^T, precomputed."""
    dz_w = L_inv @ (z_r - z_l)           # (D,)
    v_l = L_inv_CT @ r_l                  # (D,)
    v_r = L_inv_CT @ r_r
    return (jnp.dot(dz_w, v_l) < 0) | (jnp.dot(dz_w, v_r) < 0)




# ──────────────────────────────────────────────────────────────────────────────
# PEANUTS tree building  (per-chain, designed for vmap)
#
# Uses iterative doubling with progressive sampling (slice-free PEANUTS).
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


def _build_peanuts_tree(z0, log_prob_single, grad_U_single, eps, key,
                     max_depth, max_dE, leapfrog_one, is_walk,
                     centered=None, L_inv=None, L_inv_CT=None,
                     diff=None,
                     progressive=True, uturn="affine-invariant"):
    """
    Build a PEANUTS tree for one chain.

    Args:
        z0           : initial position (D,)
        log_prob_single : q → scalar log prob
        grad_U_single   : q → (D,) gradient of -log_prob
        eps          : step size (scalar)
        key          : PRNG key
        max_depth    : max tree depth
        max_dE       : max energy difference before divergence
        leapfrog_one : single-step leapfrog function
        is_walk      : bool (static), selects walk vs side mode
        centered     : (W, D) centering matrix (walk only)
        L_inv        : (D, D) inverse Cholesky of Σ (walk + affine-invariant only)
        L_inv_CT     : (D, W) = L^{-1} @ C^T, precomputed (walk + affine-invariant only)
        diff         : (D,) direction vector (side only)
        progressive  : bool (static), if True use biased progressive sampling,
                       otherwise use multinomial sampling
        uturn        : "euclidean" or "affine-invariant" U-turn criterion

    Returns:
        z_prop, log_alpha_da, lp_prop, n_proposals
    """
    D = z0.shape[0]
    lp0 = log_prob_single(z0)

    # sample momentum
    key, mk = jax.random.split(key)
    if is_walk:
        W = centered.shape[0]
        r0 = jax.random.normal(mk, (W,))
        KE0 = 0.5 * jnp.sum(r0**2)
        r_size = W
    else:
        r0 = jax.random.normal(mk, ())
        KE0 = 0.5 * r0**2
        r_size = 1

    E0 = -lp0 + KE0  # total energy

    # checkpoint arrays
    z_ckpts = jnp.zeros((max_depth, D))
    r_ckpts = jnp.zeros((max_depth, r_size))

    # initial tree (single point)
    tree0 = _TreeState(
        z_left=z0, r_left=r0, z_right=z0, r_right=r0,
        z_prop=z0, lp_prop=lp0, depth=0,
        log_weight=jnp.float32(0.), turning=False, diverging=False,
        sum_accept=jnp.float32(0.), n_proposals=0,
    )

    # --- helpers closed over shared state ---

    def _one_step(z, r, direction):
        """One leapfrog step in the given direction."""
        signed_eps = jnp.where(direction, eps, -eps)
        return leapfrog_one(z, r, grad_U_single, signed_eps)

    def _base_tree(z, r, direction):
        """Single leapfrog step → _TreeState."""
        z1, r1 = _one_step(z, r, direction)
        lp1 = log_prob_single(z1)
        if is_walk:
            KE1 = 0.5 * jnp.sum(r1**2)
        else:
            KE1 = 0.5 * r1**2
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

    def _check_uturn(z_l, r_l, z_r, r_r):
        """U-turn check between (left, right) endpoints."""
        if not is_walk:
            # Side mode: 1D motion along diff, no matrix needed
            return _uturn_side(z_l, r_l, z_r, r_r, diff)
        elif uturn == "euclidean":
            return _uturn_walk_euclidean(z_l, r_l, z_r, r_r, centered)
        else:  # affine-invariant
            return _uturn_walk_affine(z_l, r_l, z_r, r_r, L_inv, L_inv_CT)

    def _combine(cur, new, direction, rng, biased_progressive=False):
        """Merge two sub-trees, stochastically pick proposal."""
        # left/right leaves
        z_left  = jnp.where(direction, cur.z_left,  new.z_left)
        r_left  = jnp.where(direction, cur.r_left,  new.r_left)
        z_right = jnp.where(direction, new.z_right, cur.z_right)
        r_right = jnp.where(direction, new.r_right, cur.r_right)

        log_w = jnp.logaddexp(cur.log_weight, new.log_weight)
        if biased_progressive:
            accept_log_p = jnp.minimum(0., new.log_weight - cur.log_weight)
        else:
            accept_log_p = new.log_weight - log_w
        accept_new = jax.random.bernoulli(rng, jnp.clip(jnp.exp(accept_log_p), 0., 1.))
        z_prop = jnp.where(accept_new, new.z_prop, cur.z_prop)
        lp_prop = jnp.where(accept_new, new.lp_prop, cur.lp_prop)

        # No endpoint U-turn here; checkpoints handle internal sub-subtree
        # checks, and body_fn checks the full-tree endpoint.
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

    # --- iterative subtree builder (one direction, 2^depth leaves) ---

    def _build_subtree(proto_depth, z_start, r_start, direction, rng):
        """Build subtree of given depth iteratively."""
        max_leaves = 2 ** proto_depth

        def cond(state):
            tree, turn, _, _, _, n = state
            return (n < max_leaves) & ~turn & ~tree.turning & ~tree.diverging

        def body(state):
            tree, _, z_ck, r_ck, rng, n = state
            rng, rk = jax.random.split(rng)

            # get the extending leaf
            z_leaf = jnp.where(direction, tree.z_right, tree.z_left)
            r_leaf = jnp.where(direction, tree.r_right, tree.r_left)

            new = _base_tree(z_leaf, r_leaf, direction)

            # combine
            combined = _combine(tree, new, direction, rk)
            # first leaf: just use the base tree directly
            tree = jax.lax.cond(
                n == 0,
                lambda: new,
                lambda: combined,
            )

            # update checkpoints at even leaf indices
            ckpt_min, ckpt_max = _leaf_to_ckpt_idxs(n)
            r_ckpt_val = jnp.atleast_1d(new.r_right)[:r_size]
            z_ck = jax.lax.cond(
                n % 2 == 0,
                lambda: z_ck.at[ckpt_max].set(new.z_right),
                lambda: z_ck,
            )
            r_ck = jax.lax.cond(
                n % 2 == 0,
                lambda: r_ck.at[ckpt_max].set(r_ckpt_val),
                lambda: r_ck,
            )

            # Checkpoint U-turn: standard NUTS criterion with direction-aware
            # ordering.  Checkpoint is one endpoint, extending tip is the other.
            z_tip = jnp.where(direction, tree.z_right, tree.z_left)
            r_tip = jnp.where(direction, tree.r_right, tree.r_left)

            def check_ckpts(state):
                i, turning = state
                z_c = z_ck[i]
                r_c_raw = r_ck[i]
                # Restore momentum shape: walk→(W,), side→scalar
                r_c = r_c_raw if is_walk else r_c_raw[0]
                # Direction-aware left/right ordering
                z_l = jnp.where(direction, z_c, z_tip)
                r_l = jnp.where(direction, r_c, r_tip)
                z_r = jnp.where(direction, z_tip, z_c)
                r_r = jnp.where(direction, r_tip, r_c)
                turning = turning | _check_uturn(z_l, r_l, z_r, r_r)
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
        return tree._replace(depth=proto_depth, turning=tree.turning | turn), z_ck, r_ck

    # --- main doubling loop ---

    def cond_fn(state):
        tree, _, _, _ = state
        return (tree.depth < max_depth) & ~tree.turning & ~tree.diverging

    def body_fn(state):
        tree, k, z_ck, r_ck = state
        k, dk, tk, ck = jax.random.split(k, 4)
        direction = jax.random.bernoulli(dk)

        z_leaf = jnp.where(direction, tree.z_right, tree.z_left)
        r_leaf = jnp.where(direction, tree.r_right, tree.r_left)

        sub, z_ck, r_ck = _build_subtree(tree.depth, z_leaf, r_leaf, direction, tk)

        # Save current proposal — per Algorithm 6, only accept subtree's
        # proposal when the subtree has no U-turn (sprime == 1).
        prev_z_prop = tree.z_prop
        prev_lp_prop = tree.lp_prop

        tree = _combine(tree, sub, direction, ck,
                        biased_progressive=progressive)

        # Full-tree endpoint U-turn check
        full_turn = _check_uturn(tree.z_left, tree.r_left,
                                 tree.z_right, tree.r_right)
        tree = tree._replace(turning=tree.turning | full_turn)

        # Revert proposal if the subtree had a U-turn or divergence
        keep_old = sub.turning | sub.diverging
        tree = tree._replace(
            z_prop=jnp.where(keep_old, prev_z_prop, tree.z_prop),
            lp_prop=jnp.where(keep_old, prev_lp_prop, tree.lp_prop),
        )
        return tree, k, z_ck, r_ck

    tree, _, _, _ = jax.lax.while_loop(
        cond_fn, body_fn,
        (tree0, key, z_ckpts, r_ckpts),
    )

    accept_rate = tree.sum_accept / jnp.maximum(tree.n_proposals, 1)
    log_alpha_da = jnp.log(jnp.clip(accept_rate, 1e-10, 1.))
    return tree.z_prop, log_alpha_da, tree.lp_prop, tree.n_proposals


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble moves (PEANUTS) — return (proposed, log_alpha, proposed_lp, n_steps)
# ──────────────────────────────────────────────────────────────────────────────

def _peanuts_walk(group, complement, eps, key, log_prob, grad_U_vec, lp_group,
                  max_depth, max_dE, progressive=True, uturn="affine-invariant"):
    """PEANUTS walk move for one ensemble group.  vmapped over chains internally."""
    W, D = group.shape
    centered = (complement - jnp.mean(complement, axis=0)) / jnp.sqrt(W)

    # Precompute Cholesky-based quantities for affine-invariant U-turn
    if uturn == "affine-invariant":
        cov = jnp.atleast_2d(jnp.cov(complement, rowvar=False))
        reg = 1e-6 * jnp.trace(cov) / D * jnp.eye(D)
        L = jnp.linalg.cholesky(cov + reg)
        L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(D), lower=True)
        L_inv_CT = L_inv @ centered.T       # (D,W) precomputed
    else:
        L_inv = None
        L_inv_CT = None

    keys = jax.random.split(key, W)

    def _one_chain(z0, lp0, k):
        lp_fn = lambda z: log_prob(z[None])[0]
        gU_fn = lambda z: grad_U_vec(z[None])[0]
        leapfrog = lambda z, r, gU, e: _leapfrog_walk_single(z, r, gU, e, centered)
        return _build_peanuts_tree(
            z0, lp_fn, gU_fn, eps, k, max_depth, max_dE,
            leapfrog, True, centered=centered,
            L_inv=L_inv, L_inv_CT=L_inv_CT,
            progressive=progressive, uturn=uturn)

    proposed, log_alpha, lp_new, n_steps = jax.vmap(_one_chain)(group, lp_group, keys)
    return proposed, log_alpha, lp_new, n_steps


def _peanuts_side(group, complement, eps, key, log_prob, grad_U_vec, lp_group,
                  max_depth, max_dE, progressive=True, uturn="affine-invariant"):
    """PEANUTS side move for one ensemble group.  vmapped over chains internally.
    Note: side mode U-turn is always 1D (no matrix needed), uturn param unused."""
    W, D = group.shape
    keys = jax.random.split(key, 2 * W)
    idx = jnp.arange(W)
    ch = jax.vmap(lambda k: jax.random.choice(k, idx, (2,), replace=False))(keys[:W])
    diff = (complement[ch[:, 0]] - complement[ch[:, 1]]) / jnp.sqrt(2 * D)

    chain_keys = keys[W:]

    def _one_chain(z0, lp0, d, k):
        lp_fn = lambda z: log_prob(z[None])[0]
        gU_fn = lambda z: grad_U_vec(z[None])[0]
        leapfrog = lambda z, r, gU, e: _leapfrog_side_single(z, r, gU, e, d)
        return _build_peanuts_tree(
            z0, lp_fn, gU_fn, eps, k, max_depth, max_dE,
            leapfrog, False, diff=d,
            progressive=progressive, uturn=uturn)

    proposed, log_alpha, lp_new, n_steps = jax.vmap(_one_chain)(
        group, lp_group, diff, chain_keys)
    return proposed, log_alpha, lp_new, n_steps


# ──────────────────────────────────────────────────────────────────────────────
# Metropolis accept / reject  (not used by PEANUTS — both progressive and
# multinomial sampling maintain detailed balance without an MH correction)
# ──────────────────────────────────────────────────────────────────────────────

def _mh(current, proposed, log_alpha, key):
    accept = jnp.log(jax.random.uniform(key, log_alpha.shape, minval=1e-10)) < log_alpha
    return jnp.where(accept[:, None], proposed, current), accept


# ──────────────────────────────────────────────────────────────────────────────
# Dual averaging — step size  (identical to ChEES version)
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
# Initial step-size search  (adapted for PEANUTS — uses L=1 HMC as probe)
# ──────────────────────────────────────────────────────────────────────────────

def _hmc_probe_walk(group, complement, eps, key, log_prob, grad_U, lp_group):
    """Single L=1 HMC walk step for step-size probing."""
    W = group.shape[0]
    centered = (complement - jnp.mean(complement, axis=0)) / jnp.sqrt(W)
    p0 = jax.random.normal(key, (W, W))
    proposed, p1 = _leapfrog_walk(group, p0, grad_U, eps, 1, centered)
    lp1 = log_prob(proposed)
    dH = (-lp1 + 0.5 * jnp.sum(p1**2, axis=1)) \
       - (-lp_group + 0.5 * jnp.sum(p0**2, axis=1))
    dH = jnp.where(jnp.isnan(dH), jnp.inf, dH)
    return jnp.minimum(0., -dH)


def _hmc_probe_side(group, complement, eps, key, log_prob, grad_U, lp_group):
    """Single L=1 HMC side step for step-size probing."""
    W, D = group.shape
    keys = jax.random.split(key, W + 1)
    idx = jnp.arange(W)
    ch = jax.vmap(lambda k: jax.random.choice(k, idx, (2,), replace=False))(keys[:W])
    diff = (complement[ch[:, 0]] - complement[ch[:, 1]]) / jnp.sqrt(2 * D)
    p0 = jax.random.normal(keys[-1], (W,))
    proposed, p1 = _leapfrog_side(group, p0, grad_U, eps, 1, diff)
    lp1 = log_prob(proposed)
    dH = (-lp1 + 0.5 * p1**2) - (-lp_group + 0.5 * p0**2)
    dH = jnp.where(jnp.isnan(dH), jnp.inf, dH)
    return jnp.minimum(0., -dH)


def _find_init_eps(key, g1, g2, log_prob, grad_U, eps0, move):
    """Binary search for step size giving ~80% acceptance with L=1."""
    probe_fn = _hmc_probe_walk if move == "h-walk" else _hmc_probe_side
    lp1, lp2 = log_prob(g1), log_prob(g2)
    fi = jnp.finfo(jnp.result_type(eps0))

    def body(s):
        eps, _, d, k = s
        k, k1, k2 = jax.random.split(k, 3)
        eps = (2.**d) * eps
        la1 = probe_fn(g1, g2, eps, k1, log_prob, grad_U, lp1)
        la2 = probe_fn(g2, g1, eps, k2, log_prob, grad_U, lp2)
        la = jnp.concatenate([la1, la2])
        avg = jnp.log(la.shape[0]) - jax.scipy.special.logsumexp(-la)
        return eps, d, jnp.where(jnp.log(.8) < avg, 1, -1), k

    def cond(s):
        eps, ld, d, _ = s
        return (((eps > fi.tiny) | (d >= 0)) & ((eps < fi.max) | (d <= 0))
                & ((ld == 0) | (d == ld)))

    eps, *_ = jax.lax.while_loop(cond, body, (eps0, 0, 0, key))
    return eps / 2.


# ──────────────────────────────────────────────────────────────────────────────
# sampler_peanuts
# ──────────────────────────────────────────────────────────────────────────────

def sampler_peanuts(
    log_prob_fn,
    initial_state,
    num_samples,
    warmup           = 1000,
    move             = "h-walk",       # "h-walk" or "h-side"
    sampling         = "progressive",  # "progressive" or "multinomial"
    uturn            = "affine-invariant",  # "affine-invariant" or "euclidean"
    step_size        = 0.1,
    max_tree_depth   = 5,  # max trajectory length ≤ 2^depth
    thin_by          = 1,
    target_accept    = 0.80,
    max_delta_energy = 1000.,
    grad_log_prob_fn = None,
    find_init_step_size   = True,
    adapt_step_size  = True,
    seed             = 0,
    verbose          = True,
):
    """
    Ensemble preconditioned PEANUTS with automatic step-size tuning.

    Integration length is determined per-chain per-step by the No-U-Turn
    criterion (tree doubling), so there is no L or ChEES parameter.

    Args:
        log_prob_fn      : (batch, D) -> (batch,).  Vectorised log density.
        initial_state    : (n_chains, D).  n_chains must be even and >= 4.
        num_samples      : Post-warmup samples to return.
        warmup           : Warmup iterations (dual averaging for step size).
        move             : "h-walk" or "h-side".
        sampling         : "progressive" (biased progressive, default) or
                           "multinomial" (unbiased multinomial).
        step_size        : Initial step size (adapted during warmup).
        max_tree_depth   : Maximum PEANUTS tree depth (trajectory ≤ 2^depth steps).
        thin_by          : Keep every thin_by-th sample.
        target_accept    : Target acceptance rate for dual averaging.
        max_delta_energy : Energy threshold for divergence detection.
        grad_log_prob_fn : Vectorised gradient (batch,D)->(batch,D).
                           If None, uses jax.vmap(jax.grad(log_prob_fn)).
        find_init_step_size : If True (default), run a short heuristic search at
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
    assert n_chains >= 4 and n_chains % 2 == 0, "Need >= 4 even chains"
    assert sampling in ("progressive", "multinomial"), \
        f"sampling must be 'progressive' or 'multinomial', got '{sampling}'"

    _progressive = sampling == "progressive"
    assert uturn in ("euclidean", "affine-invariant"), \
        f"uturn must be 'euclidean' or 'affine-invariant', got '{uturn}'"
    _peanuts_base = _peanuts_walk if move == "h-walk" else _peanuts_side
    peanuts_fn = lambda *a, **kw: _peanuts_base(*a, **kw, progressive=_progressive, uturn=uturn)

    if grad_log_prob_fn is None:
        grad_U = jax.vmap(jax.grad(lambda x: log_prob_fn(x[None])[0]))
    else:
        grad_U = grad_log_prob_fn
    # negate: leapfrog uses gradient of U = -log_prob
    _grad_U = lambda x: -grad_U(x)

    W = n_chains // 2
    g1, g2 = state[:W], state[W:]
    key = jax.random.key(seed)

    # --- initial step size ---
    if find_init_step_size:
        key, k = jax.random.split(key)
        step_size = _find_init_eps(k, g1, g2, log_prob_fn, _grad_U, step_size, move)
    step_size = jnp.asarray(step_size)
    if verbose:
        print(f"move={move}  sampling={sampling}  max_depth={max_tree_depth}"
              f"  init_eps={float(step_size):.4f}"
              f"  find_init_step_size={find_init_step_size}"
              f"  adapt_step_size={adapt_step_size}")

    log_eps0 = jnp.log(step_size)
    da  = _da_init(log_eps0)
    lp1 = log_prob_fn(g1);  lp2 = log_prob_fn(g2)

    # --- warmup (dual averaging only, PEANUTS handles trajectory length) ---
    @jax.jit
    def _warmup_step(g1, g2, lp1, lp2, da, keys):
        k1, k2, ka1, ka2 = keys
        eps = jnp.exp(da.log_eps) if adapt_step_size else step_size

        # PEANUTS progressive/multinomial: proposal already satisfies detailed
        # balance, so always accept (no MH step). la1/la2 are tree
        # acceptance statistics used only for step-size adaptation (DA).
        p1, la1, lp1n, ns1 = peanuts_fn(g1, g2, eps, k1, log_prob_fn, _grad_U, lp1,
                                          max_tree_depth, max_delta_energy)
        if adapt_step_size:
            da = _da_update(da, la1, log_eps0, target_accept)
        g1 = p1
        lp1 = lp1n

        eps = jnp.exp(da.log_eps) if adapt_step_size else step_size
        p2, la2, lp2n, ns2 = peanuts_fn(g2, g1, eps, k2, log_prob_fn, _grad_U, lp2,
                                         max_tree_depth, max_delta_energy)
        if adapt_step_size:
            da = _da_update(da, la2, log_eps0, target_accept)
        g2 = p2
        lp2 = lp2n

        acc = (jnp.mean(jnp.exp(la1)) + jnp.mean(jnp.exp(la2))) / 2
        mean_ns = (jnp.mean(ns1) + jnp.mean(ns2)) / 2
        return g1, g2, lp1, lp2, da, acc, mean_ns

    key, k = jax.random.split(key)
    flat  = jax.random.split(k, warmup * 4)
    wkeys = flat.reshape(warmup, 4, *flat.shape[1:])
    total_acc = 0.
    total_ns  = 0.
    for i in range(warmup):
        g1, g2, lp1, lp2, da, acc, mean_ns = _warmup_step(
            g1, g2, lp1, lp2, da, wkeys[i])
        total_acc += acc
        total_ns  += mean_ns

    final_eps = jnp.exp(da.log_eps_bar) if adapt_step_size else step_size
    if verbose:
        print(f"Warmup done.  eps={float(final_eps):.4f}"
              f"  accept={float(total_acc)/max(warmup,1):.3f}"
              f"  mean_steps={float(total_ns)/max(warmup,1):.1f}")

    # --- main sampling (fixed step size, PEANUTS tree per step) ---
    @jax.jit
    def _step(carry, keys):
        g1, g2, lp1, lp2 = carry
        k1, k2, ka1, ka2 = keys

        p1, la1, lp1n, ns1 = peanuts_fn(g1, g2, final_eps, k1, log_prob_fn, _grad_U, lp1,
                                          max_tree_depth, max_delta_energy)
        g1 = p1
        lp1 = lp1n

        p2, la2, lp2n, ns2 = peanuts_fn(g2, g1, final_eps, k2, log_prob_fn, _grad_U, lp2,
                                         max_tree_depth, max_delta_energy)
        g2 = p2
        lp2 = lp2n

        state  = jnp.concatenate([g1, g2])
        # Report tree acceptance rate (integrator quality diagnostic)
        accept = jnp.concatenate([jnp.exp(la1), jnp.exp(la2)])
        n_steps = (jnp.mean(ns1) + jnp.mean(ns2)) / 2
        return (g1, g2, lp1, lp2), (state, accept, n_steps)

    key, k  = jax.random.split(key)
    flat    = jax.random.split(k, num_samples * thin_by * 4)
    skeys   = flat.reshape(num_samples * thin_by, 4, *flat.shape[1:])
    (g1, g2, lp1, lp2), (all_states, all_acc, all_ns) = jax.lax.scan(
        _step, (g1, g2, lp1, lp2), skeys)

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

    # --- Test 1: Ill-conditioned Gaussian (20D, kappa=1000) ---
    print("=" * 60)
    print("Test 1: Ill-conditioned Gaussian  (D=20, kappa=1000)")
    print("=" * 60)
    dim = 20
    kappa = 1000.  # condition number
    # Eigenvalues log-spaced from 1 to kappa
    eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
    # Random orthogonal basis
    Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(0), (dim, dim)))
    cov_gauss = Q @ jnp.diag(eigvals) @ Q.T
    prec_gauss = Q @ jnp.diag(1. / eigvals) @ Q.T
    def log_prob_gauss(x):
        return -0.5 * jnp.sum((x @ prec_gauss) * x, axis=-1)

    init = jax.random.normal(jax.random.key(42), (40, dim))
    for ut in ["affine-invariant", "euclidean"]:
        samples, info = sampler_peanuts(log_prob_gauss, init, num_samples=5000,
                                        warmup=1000, step_size=0.01, seed=123, uturn=ut)
        flat = samples.reshape(-1, dim)
        var_est = jnp.var(flat, axis=0)
        var_true = jnp.diag(cov_gauss)
        rel_err = jnp.mean(jnp.abs(var_est - var_true) / var_true)
        print(f"  {ut}:  mean_rel_err(var)={rel_err:.3f}"
              f"  var_range=[{jnp.min(var_est):.2f}, {jnp.max(var_est):.2f}]"
              f"  (target: [{jnp.min(var_true):.2f}, {jnp.max(var_true):.2f}])")
        print(f"    info: {info}")

    # --- Test 2: Rosenbrock (20D) ---
    print()
    print("=" * 60)
    print("Test 2: Rosenbrock  (D=20, a=1, b=100)")
    print("=" * 60)
    # p(x) ~ exp(-(sum_i [ b*(x_{2i+1} - x_{2i}^2)^2 + (x_{2i} - a)^2 ]))
    # Exact: x_even ~ N(a, 1/(2b) + 1) ≈ N(1, 1.005), x_odd ~ complicated
    a_ros, b_ros = 1.0, 100.0
    dim_ros = 10
    def log_prob_rosen(x):
        x_even = x[:, ::2]
        x_odd = x[:, 1::2]
        return -(b_ros * jnp.sum((x_odd - x_even**2)**2, axis=1)
                 + jnp.sum((x_even - a_ros)**2, axis=1))

    # x_e ~ N(a, 1/2): mean=a, var=0.5
    # x_o | x_e ~ N(x_e^2, 1/(2b)):  E[x_o]=E[x_e^2]=1.5, Var(x_o)=Var(x_e^2)+1/(2b)≈2.505
    init_r = jax.random.normal(jax.random.key(42), (100, dim_ros))
    samples, info = sampler_peanuts(log_prob_rosen, init_r, num_samples=2000,
                                    warmup=400, step_size=0.01, seed=123)
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

    init_f = jax.random.normal(jax.random.key(99), (20, funnel_dim)) * 0.5
    samples, info = sampler_peanuts(log_prob_funnel, init_f, num_samples=5000,
                                    warmup=1000, seed=42)
    flat = samples.reshape(-1, funnel_dim)
    v_samples = flat[:, 0]
    x_samples = flat[:, 1:]
    print(f"  v:   mean={jnp.mean(v_samples):.3f}  var={jnp.var(v_samples):.2f}"
          f"  (target: mean=0, var=9)")
    print(f"  x_i: mean={jnp.mean(x_samples):.3f}  var={jnp.mean(jnp.var(x_samples, axis=0)):.1f}"
          f"  (target: mean=0, var~90.0)")
    print(f"  info: {info}")
    print("  (Funnel is a hard test — geometry varies drastically with v.)")