"""Parallel tempering with a fully JIT-compiled inner block.

K replicas at temperatures τ_0 < τ_1 < … < τ_{K−1}.  The inner
sampler is a minimal batched BAOAB kinetic-Langevin step *without*
ensemble preconditioning — each replica has `n_chains` independent
particles in R^D, and each walker uses ordinary (mass-1, identity-
preconditioned) underdamped Langevin.  This is faster and simpler
than the full ensemble-preconditioned ``pickles_unadjusted`` and is
what we actually want inside PT: the cross-basin mixing comes from
temperature swaps, not from the preconditioner.

Structure::

    state : (K, n_chains, D) position, (K, n_chains, D) momentum
    one block:
        for _ in range(swap_every):  BAOAB step per-replica (each τ_k)
        swap proposal on alternating adjacent pairs

The whole block is wrapped in ``jax.jit`` and ``jax.lax.scan``, so the
sampler runs at full GPU throughput.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MD.energies import (
    dw4_energy, lj_energy, lj_pita_energy,
    init_dw4, init_lj,
)
from MD.metrics import energy_w2, distance_w2, pairwise_distances_flat
from MD.ground_truth import load, energies as gt_energies, pick_init, FILES

ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────
# MH-adjusted HMC trajectory (one replica), fully JIT-friendly.
# Full momentum refresh each call → pure HMC; L leapfrog steps; exact
# MH correction based on ΔH removes integrator bias.
#
# Target at temperature τ:  π(q) ∝ exp(-E(q)/τ).  HMC joint:
#     π(q, p) ∝ exp(-E(q)/τ - |p|²/2),   grad_q(-log π) = grad E / τ.
# ──────────────────────────────────────────────────────────────────────

def _hmc_trajectory(Q, h, tau, L, E_batched, grad_E_batched, key,
                     force_clip=1e4):
    """One HMC trajectory per walker at inverse temperature β = 1/τ.

    Returns (Q_new, n_accept_per_walker).
    """
    key_p, key_u = jax.random.split(key)

    # full momentum refresh
    P = jax.random.normal(key_p, Q.shape)

    def clip(g):
        return jnp.clip(g, -force_clip, force_clip)

    E_start = E_batched(Q)                                    # (n,)
    KE_start = 0.5 * jnp.sum(P * P, axis=-1)                  # (n,)
    H_start = E_start / tau + KE_start

    Q0, P0 = Q, P

    # leapfrog: p-half, q-full, p-full, ..., p-half
    g = clip(grad_E_batched(Q)) / tau
    P = P - 0.5 * h * g

    def body(i, carry):
        Q, P = carry
        Q = Q + h * P
        g = clip(grad_E_batched(Q)) / tau
        P = jnp.where(i < L - 1, P - h * g, P - 0.5 * h * g)
        return (Q, P)

    Q, P = jax.lax.fori_loop(0, L, body, (Q, P))

    # NaN guard: if any walker produced non-finite H_end, force rejection
    E_end = E_batched(Q)
    KE_end = 0.5 * jnp.sum(P * P, axis=-1)
    H_end = E_end / tau + KE_end

    log_alpha = H_start - H_end
    log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha,
                           -jnp.inf * jnp.ones_like(log_alpha))
    u = jax.random.uniform(key_u, log_alpha.shape)
    accept = jnp.log(u) < log_alpha                           # (n,)

    Q_new = jnp.where(accept[:, None], Q, Q0)
    return Q_new, accept.astype(jnp.float32)


# ──────────────────────────────────────────────────────────────────────
# One PT block: K replicas × swap_every BAOAB steps + even/odd swap
# ──────────────────────────────────────────────────────────────────────

def build_pt_step(batched_E_fn, K, n_chains, D, temps, step_sizes,
                  L, swap_every):
    """Return a JIT'd function that advances one PT block of MH-adjusted
    HMC trajectories + one adjacent-pair swap.

    Per replica: ``swap_every`` trajectories, each of ``L`` leapfrog
    steps at the replica's (h, τ).  MH accept removes integrator bias
    exactly.
    """
    temps      = jnp.asarray(temps,      dtype=jnp.float32)            # (K,)
    step_sizes = jnp.asarray(step_sizes, dtype=jnp.float32)            # (K,)
    betas      = 1.0 / temps                                            # (K,)

    grad_E_single = jax.grad(lambda x: batched_E_fn(x[None])[0])
    grad_E_batch  = jax.vmap(grad_E_single)

    def inner_traj_one_replica(Q, h, tau, key):
        return _hmc_trajectory(Q, h, tau, L,
                                 batched_E_fn, grad_E_batch, key)

    def one_swap_every_body(carry, key_block):
        Q_all, acc_accum = carry
        keys = jax.random.split(key_block, K)

        def step_rep(Q, h, tau, k):
            return inner_traj_one_replica(Q, h, tau, k)

        Q_all, acc = jax.vmap(step_rep)(Q_all, step_sizes, temps, keys)
        return (Q_all, acc_accum + acc), None

    def swap_move(Q_all, E_all, betas, key, parity):
        K_ = Q_all.shape[0]
        key_u, _ = jax.random.split(key)
        u = jax.random.uniform(key_u, (K_ - 1, n_chains))
        dE = E_all[1:] - E_all[:-1]
        dbeta = betas[1:] - betas[:-1]                                  # negative
        log_alpha = dbeta[:, None] * dE
        would_accept = jnp.log(u) < log_alpha

        i_idx = jnp.arange(K_ - 1)
        parity_mask = (i_idx % 2 == parity)
        accept = would_accept & parity_mask[:, None]

        def apply_pair(Q, idx):
            a = accept[idx]
            Qi, Qi1 = Q[idx], Q[idx + 1]
            Q = Q.at[idx]    .set(jnp.where(a[:, None], Qi1, Qi))
            Q = Q.at[idx + 1].set(jnp.where(a[:, None], Qi,  Qi1))
            return Q
        for idx in range(parity, K_ - 1, 2):
            Q_all = apply_pair(Q_all, idx)

        return Q_all, accept.astype(jnp.float32)

    def pt_block(Q_all, key, parity):
        acc_accum = jnp.zeros((K, n_chains), dtype=jnp.float32)
        keys = jax.random.split(key, swap_every + 2)
        (Q_all, acc_accum), _ = jax.lax.scan(
            one_swap_every_body, (Q_all, acc_accum), keys[:swap_every])
        E_all = jax.vmap(batched_E_fn)(Q_all)
        Q_all, swap_acc_mat = swap_move(Q_all, E_all, betas, keys[-1], parity)
        E_after = jax.vmap(batched_E_fn)(Q_all)
        return Q_all, E_all, E_after, swap_acc_mat, acc_accum / swap_every

    return jax.jit(pt_block, static_argnums=(2,))


# ──────────────────────────────────────────────────────────────────────
# Defaults per problem
# ──────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # (h, L) per replica: pick h/τ small enough that MH accept ≳ 0.5.
    # Each replica samples π_k(q) ∝ exp(-E/τ_k), so leapfrog force = grad E / τ.
    #
    # The *_pita ladders start at τ=1 — correct target under the pita formula
    # (2·E_LJ + E_osc).  Legacy iDEM ladders start at τ=0.5, the old empirical
    # calibration against DEM data under the no-factor-2 formula.
    "dw4":  dict(n_chains=64, K=6,
                 temps=[1.0, 1.3, 1.7, 2.2, 2.9, 3.8],
                 step_size_ladder=[0.05, 0.06, 0.07, 0.08, 0.10, 0.12],
                 L=10, warmup_blocks=50, n_blocks=1500, swap_every=10,
                 trace_stride=1),
    "lj13": dict(n_chains=64, K=8,
                 temps=[0.5, 0.65, 0.85, 1.1, 1.4, 1.8, 2.3, 3.0],
                 step_size_ladder=[0.006, 0.008, 0.011, 0.015, 0.019,
                                    0.024, 0.030, 0.038],
                 L=15, warmup_blocks=100, n_blocks=2500, swap_every=10,
                 trace_stride=1),
    # For the pita formula (2·LJ + osc) at τ=1 — same ladder shifted 2×.
    "lj13_pita": dict(n_chains=64, K=8,
                 temps=[1.0, 1.3, 1.7, 2.2, 2.9, 3.8, 5.0, 6.5],
                 step_size_ladder=[0.004, 0.005, 0.007, 0.009, 0.012,
                                    0.015, 0.019, 0.024],
                 L=15, warmup_blocks=100, n_blocks=2500, swap_every=10,
                 trace_stride=1),
    "lj55": dict(n_chains=32, K=10,
                 temps=[0.5, 0.62, 0.77, 0.96, 1.2, 1.5, 1.85, 2.3, 2.85, 3.6],
                 step_size_ladder=[0.003, 0.0038, 0.0047, 0.006, 0.0075,
                                    0.0094, 0.012, 0.015, 0.019, 0.024],
                 L=20, warmup_blocks=150, n_blocks=3000, swap_every=10,
                 trace_stride=1),
    "lj55_pita": dict(n_chains=32, K=10,
                 temps=[1.0, 1.25, 1.55, 1.9, 2.4, 3.0, 3.7, 4.6, 5.7, 7.2],
                 step_size_ladder=[0.0021, 0.0027, 0.0033, 0.0042, 0.0053,
                                    0.0066, 0.0084, 0.0106, 0.0134, 0.017],
                 L=20, warmup_blocks=150, n_blocks=3000, swap_every=10,
                 trace_stride=1),
}


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def run_pt(problem: str, n_chains: int, temps, step_sizes, L,
           warmup_blocks: int, n_blocks: int, swap_every: int,
           seed: int, init_from_gt: bool, trace_stride: int,
           formula: str = "idem"):
    cfg = FILES[problem]
    n = cfg["n_particles"]; sd = cfg["spatial_dim"]
    D = cfg["n_particles"] * cfg["spatial_dim"]

    if problem == "dw4":
        batched_E = dw4_energy
    elif formula == "pita":
        batched_E = lambda x: lj_pita_energy(x, n)
    else:
        batched_E = lambda x: lj_energy(x, n)

    K = len(temps)

    if init_from_gt:
        init_one = np.asarray(pick_init(problem, n_chains, seed=seed))
    else:
        k0 = jax.random.key(seed)
        if problem == "dw4":
            init_one = np.asarray(init_dw4(k0, n_chains))
        else:
            init_one = np.asarray(init_lj(k0, n_chains, n))
    Q = jnp.broadcast_to(jnp.asarray(init_one), (K, n_chains, D)).copy()

    pt_block = build_pt_step(batched_E, K, n_chains, D,
                             temps, step_sizes, L, swap_every)

    key = jax.random.key(seed)
    print(f"[PT {problem}]  K={K}  n_chains={n_chains}  L={L}  "
          f"warmup_blocks={warmup_blocks}  n_blocks={n_blocks}  "
          f"swap_every={swap_every}", flush=True)
    print(f"  temps       = {list(np.asarray(temps))}", flush=True)
    print(f"  step_sizes  = {list(np.asarray(step_sizes))}", flush=True)

    # ── warmup ────────────────────────────────────────────────────
    t0 = time.time()
    for b in range(warmup_blocks):
        key, k = jax.random.split(key)
        Q, _, _, _, _ = pt_block(Q, k, b % 2)
    Q.block_until_ready()
    warm_time = time.time() - t0
    print(f"  warmup {warmup_blocks} blocks done in {warm_time:.1f}s", flush=True)

    # ── production ────────────────────────────────────────────────
    trace_E_before = np.zeros((K, n_blocks))
    trace_E_after  = np.zeros((K, n_blocks))
    trace_hmc_acc  = np.zeros((K, n_blocks))
    swap_acc_sum   = np.zeros(K - 1)
    swap_acc_cnt   = np.zeros(K - 1)
    t0_samples     = []

    t0 = time.time()
    for b in range(n_blocks):
        key, k = jax.random.split(key)
        Q, E_before, E_after, swap_mat, hmc_acc_block = pt_block(Q, k, b % 2)
        trace_E_before[:, b] = np.asarray(E_before).mean(axis=1)
        trace_E_after [:, b] = np.asarray(E_after ).mean(axis=1)
        trace_hmc_acc [:, b] = np.asarray(hmc_acc_block).mean(axis=1)
        swap_acc_sum += np.asarray(swap_mat).sum(axis=1)
        row_parity = (np.arange(K - 1) % 2 == (b % 2)).astype(float)
        swap_acc_cnt += row_parity * n_chains

        if b % trace_stride == 0:
            t0_samples.append(np.asarray(Q[0]))

        if (b + 1) % max(1, n_blocks // 10) == 0:
            print(f"  block {b+1}/{n_blocks}   "
                  f"τ=1 E={trace_E_after[0, b]:+.3f}  "
                  f"HMCacc_τ0={trace_hmc_acc[0, b]:.2f}  "
                  f"top τ E={trace_E_after[-1, b]:+.3f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    prod_time = time.time() - t0
    t0_samples = np.stack(t0_samples, axis=0)

    info = dict(warmup_sec=float(warm_time), production_sec=float(prod_time),
                K=int(K), n_chains=int(n_chains), L=int(L),
                temps=list(np.asarray(temps, dtype=float)),
                step_sizes=list(np.asarray(step_sizes, dtype=float)),
                warmup_blocks=int(warmup_blocks), n_blocks=int(n_blocks),
                swap_every=int(swap_every),
                swap_accept_rate=list(swap_acc_sum / np.maximum(swap_acc_cnt, 1)),
                swap_propose=list(swap_acc_cnt),
                hmc_accept_rate=[float(trace_hmc_acc[k].mean()) for k in range(K)],
                final_temp_E=[float(trace_E_after[k, -1]) for k in range(K)])
    return t0_samples, info, trace_E_before, trace_E_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(DEFAULTS))
    ap.add_argument("--n-chains", type=int, default=None)
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--warmup-blocks", type=int, default=None)
    ap.add_argument("--init-from-gt", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--formula", default="idem", choices=["idem", "pita"],
                    help="'idem' = E_LJ + E_osc  (arXiv:2504.11713 / iDEM / our default); "
                         "'pita' = 2·E_LJ + E_osc  (github.com/taraak/pita — the formula "
                         "used to generate the published DEM GT data)")
    args = ap.parse_args()

    key = f"{args.problem}_{args.formula}" if f"{args.problem}_{args.formula}" in DEFAULTS \
          else args.problem
    d = DEFAULTS[key]
    n_chains = args.n_chains or d["n_chains"]
    n_blocks = args.n_blocks or d["n_blocks"]
    warmup_blocks = args.warmup_blocks or d["warmup_blocks"]

    t0_samples, info, trace_before, trace_after = run_pt(
        args.problem, n_chains, d["temps"], d["step_size_ladder"],
        d["L"], warmup_blocks, n_blocks, d["swap_every"],
        args.seed, args.init_from_gt, d["trace_stride"],
        formula=args.formula)

    cfg = FILES[args.problem]
    n = cfg["n_particles"]; sd = cfg["spatial_dim"]
    gt = load(args.problem, "test")
    rng = np.random.default_rng(args.seed)
    gt_sub = gt[rng.choice(gt.shape[0], size=min(10_000, gt.shape[0]),
                            replace=False)]
    gt_E = gt_energies(args.problem, gt_sub)

    # take last half of samples for analysis
    keep = t0_samples[t0_samples.shape[0] // 2:]
    flat = keep.reshape(-1, keep.shape[-1])
    sub = flat[rng.choice(flat.shape[0], size=min(20_000, flat.shape[0]),
                          replace=False)]
    if args.problem == "dw4":
        sub_E = np.asarray(dw4_energy(jnp.asarray(sub)))
    elif args.formula == "pita":
        sub_E = np.asarray(lj_pita_energy(jnp.asarray(sub), n))
        gt_E_pita = np.asarray(lj_pita_energy(jnp.asarray(gt_sub), n))
        gt_E = gt_E_pita                                                # compare apples-to-apples
    else:
        sub_E = np.asarray(lj_energy(jnp.asarray(sub), n))

    metrics = dict(
        problem=args.problem, init_from_gt=args.init_from_gt,
        n_chains=n_chains, n_blocks=n_blocks,
        warmup_blocks=warmup_blocks, seed=args.seed,
        info=info,
        gt=dict(E_mean=float(gt_E.mean()), E_std=float(gt_E.std())),
        sampler_tau1=dict(E_mean=float(sub_E.mean()),
                           E_std=float(sub_E.std()),
                           E_min=float(sub_E.min())),
        energy_W2_vs_gt=float(energy_w2(sub_E, gt_E)),
        distance_W2_vs_gt=float(distance_w2(sub, gt_sub, n, sd)),
    )

    tag = f"{args.problem}_pt" + ("_gt" if args.init_from_gt else "")
    if args.formula != "idem":
        tag += f"_{args.formula}"
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    fig = ROOT / "figs"; fig.mkdir(parents=True, exist_ok=True)
    np.savez(out / "data.npz", t0_samples=t0_samples, sub=sub, sub_E=sub_E,
             gt_sub=gt_sub, gt_E=gt_E,
             trace_E_before=trace_before, trace_E_after=trace_after)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # ─── plots ─────────────────────────────────────────────────
    # energy histogram vs GT
    f1, ax = plt.subplots(figsize=(6.5, 4.0))
    lo = min(sub_E.min(), gt_E.min()) - 1; hi = max(sub_E.max(), gt_E.max()) + 1
    bins = np.linspace(lo, hi, 80)
    ax.hist(gt_E, bins=bins, density=True, alpha=0.5, label="ground truth")
    ax.hist(sub_E, bins=bins, density=True, alpha=0.5,
            label=f"PT τ=1 (E W2 = {metrics['energy_W2_vs_gt']:.3f})")
    ax.set_xlabel("energy"); ax.set_ylabel("density")
    ax.set_title(f"{args.problem.upper()} — parallel tempering vs GT")
    ax.legend(); f1.tight_layout()
    f1.savefig(fig / f"{tag}_energy.png", dpi=140); plt.close(f1)

    # pair-distance
    f2, ax = plt.subplots(figsize=(6.5, 4.0))
    d_s  = pairwise_distances_flat(sub,    n, sd)
    d_gt = pairwise_distances_flat(gt_sub, n, sd)
    bins = np.linspace(0, np.percentile(np.concatenate([d_s, d_gt]), 99.5), 120)
    ax.hist(d_gt, bins=bins, density=True, alpha=0.5, label="ground truth")
    ax.hist(d_s,  bins=bins, density=True, alpha=0.5,
            label=f"PT τ=1 (d W2 = {metrics['distance_W2_vs_gt']:.3f})")
    ax.set_xlabel("d_ij"); ax.set_ylabel("density")
    ax.set_title(f"{args.problem.upper()} — pair-distance distribution")
    ax.legend(); f2.tight_layout()
    f2.savefig(fig / f"{tag}_distance.png", dpi=140); plt.close(f2)

    # energy trace per rung
    f3, ax = plt.subplots(figsize=(7.5, 4.0))
    for k, t in enumerate(info["temps"]):
        ax.plot(trace_after[k], label=f"τ={t:.2f}", alpha=0.8)
    ax.axhline(gt_E.mean(), color="k", linestyle="--",
               label=f"GT τ=1 mean = {gt_E.mean():.2f}")
    ax.set_xlabel("PT block"); ax.set_ylabel("mean E")
    ax.set_title(f"{args.problem.upper()} — energy trace per rung")
    ax.legend(fontsize=8, ncol=2)
    f3.tight_layout()
    f3.savefig(fig / f"{tag}_trace.png", dpi=140); plt.close(f3)

    # swap acceptance
    f4, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.bar(range(len(info["swap_accept_rate"])), info["swap_accept_rate"])
    ax.set_xlabel("rung i ↔ i+1"); ax.set_ylabel("swap accept")
    for y in [0.2, 0.5]:
        ax.axhline(y, color="red" if y == 0.2 else "green",
                    linestyle=":", alpha=0.5)
    ax.set_title(f"{args.problem.upper()} — PT swap acceptance")
    f4.tight_layout()
    f4.savefig(fig / f"{tag}_swapacc.png", dpi=140); plt.close(f4)

    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
