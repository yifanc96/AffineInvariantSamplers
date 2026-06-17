"""Parallel-tempering HMC for alanine dipeptide via OpenMM.

K replicas at temperatures τ·300 K for τ ∈ {1, 1.5, 2.0, 2.7, 3.6, 4.8, 6.4, 8.5}.
Each replica is one single-chain MH-adjusted HMC with L leapfrog steps.
After each block we propose adjacent-pair swaps (alternating parity).

Pure NumPy / OpenMM — no JAX intermediation (saves pure_callback overhead).
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openmm
import openmm.unit as unit

from MD.alanine.alanine import (
    build_system, phi_psi_atom_indices, dihedral_np, kT_kJmol_300K,
)
from MD.alanine.run_langevin import ramachandran_kl

ROOT = Path(__file__).resolve().parent


class MultiContext:
    """K OpenMM Contexts running the same System at different temperatures.
    Provides a single batched energy_forces call."""

    def __init__(self, system, K: int, platform_name: str = "OpenCL"):
        self.K = K
        self.n_atoms = system.getNumParticles()
        self.platform = openmm.Platform.getPlatformByName(platform_name)
        self.contexts = []
        for _ in range(K):
            integ = openmm.VerletIntegrator(1e-3)
            self.contexts.append(openmm.Context(system, integ, self.platform))

    def energy_forces(self, Q):                                         # (K, N, 3)
        E = np.empty(self.K, dtype=np.float64)
        F = np.empty((self.K, self.n_atoms, 3), dtype=np.float64)
        for k, ctx in enumerate(self.contexts):
            ctx.setPositions(Q[k] * unit.nanometer)
            st = ctx.getState(getEnergy=True, getForces=True)
            E[k] = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            F[k] = np.asarray(st.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer))
        return E, F


def hmc_trajectory(Q, mc, taus_kT, h_vec, L, rng):
    """One MH-adjusted HMC trajectory per replica.

    taus_kT  : (K,) effective kT per replica (= τ_k · kT_300K)
    h_vec    : (K,) leapfrog step size per replica
    Returns (Q_new, E_new, accept_mask).
    """
    K = Q.shape[0]
    P = rng.normal(size=Q.shape)                                       # mass = 1

    E0, F0 = mc.energy_forces(Q)
    KE0 = 0.5 * (P * P).sum(axis=(1, 2))
    H0  = E0 / taus_kT + KE0

    # Leapfrog: half-P, (full-Q, full-P)·(L−1), full-Q, half-P
    Q_new = Q.copy()
    P_new = P.copy()
    grad = -F0                                                        # ∇E = -F
    P_new -= 0.5 * h_vec[:, None, None] * grad / taus_kT[:, None, None]
    for step in range(L):
        Q_new = Q_new + h_vec[:, None, None] * P_new
        if step < L - 1:
            _, F_step = mc.energy_forces(Q_new)
            grad = -F_step
            P_new -= h_vec[:, None, None] * grad / taus_kT[:, None, None]
    # final half-step
    E1, F1 = mc.energy_forces(Q_new)
    grad = -F1
    P_new -= 0.5 * h_vec[:, None, None] * grad / taus_kT[:, None, None]

    KE1 = 0.5 * (P_new * P_new).sum(axis=(1, 2))
    H1  = E1 / taus_kT + KE1

    log_alpha = H0 - H1
    finite = np.isfinite(log_alpha)
    log_alpha = np.where(finite, log_alpha, -np.inf)
    u = rng.random(size=K)
    accept = np.log(u) < log_alpha
    Q_out = np.where(accept[:, None, None], Q_new, Q)
    E_out = np.where(accept, E1, E0)
    return Q_out, E_out, accept


def swap_adjacent(Q, E, taus_kT, rng, parity):
    """Propose swaps on replica pairs (parity, parity+2, parity+4, …)."""
    K = Q.shape[0]
    betas = 1.0 / taus_kT
    swap_log = []
    for i in range(parity, K - 1, 2):
        dbeta = betas[i + 1] - betas[i]                                # negative
        dE    = E[i + 1] - E[i]
        log_alpha = dbeta * dE
        u = rng.random()
        acc = (np.log(u) < log_alpha) and np.isfinite(log_alpha)
        swap_log.append((i, bool(acc)))
        if acc:
            Q[[i, i + 1]] = Q[[i + 1, i]]
            E[[i, i + 1]] = E[[i + 1, i]]
    return Q, E, swap_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[1.0, 1.5, 2.0, 2.7, 3.6, 4.8, 6.4, 8.5],
                    help="τ_k multipliers on T=300 K")
    ap.add_argument("--L", type=int, default=10,
                    help="leapfrog steps per HMC trajectory")
    ap.add_argument("--h-base", type=float, default=0.003,
                    help="base leapfrog step (nm · √amu/kJ·mol). "
                         "h_k scales as √τ_k so all replicas have similar accept.")
    ap.add_argument("--swap-every", type=int, default=10,
                    help="HMC trajectories between swap attempts")
    ap.add_argument("--warmup-blocks", type=int, default=200)
    ap.add_argument("--n-blocks",      type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--platform", default="OpenCL")
    ap.add_argument("--out", default="pt_hmc")
    ap.add_argument("--gt", default=str(ROOT / "reference" / "AD2_relaxed.npy"))
    ap.add_argument("--init", default="equilibrium",
                    choices=["equilibrium", "gt"],
                    help="equilibrium=bgmol minimum for all K replicas; "
                         "gt=draw K distinct GT configs, minimise each")
    args = ap.parse_args()

    out = ROOT / "results" / args.out
    out.mkdir(parents=True, exist_ok=True)

    topo, sys_, pos, atoms = build_system()
    N = sys_.getNumParticles()
    phi_ij, psi_ij = phi_psi_atom_indices(topo)

    taus = np.asarray(args.taus, dtype=np.float64)
    K = len(taus)
    taus_kT = taus * kT_kJmol_300K                                     # (K,)
    # h scales as √τ so (h / √τ_k) is constant — roughly matches accept rate.
    h_vec = args.h_base * np.sqrt(taus)                                # (K,)
    print(f"[alanine] PT-HMC  K={K}  τ={taus.tolist()}  h={h_vec.tolist()}  L={args.L}",
          flush=True)

    # Initial states: either bgmol equilibrium (all replicas in same basin)
    # or K distinct GT frames (built-in basin diversity).  Each is then
    # minimised in our FF so the chain starts at a local energy minimum.
    rng = np.random.default_rng(args.seed)
    if args.init == "gt":
        gt = np.load(args.gt)                                          # (N_gt, 22, 3) Å
        idx = rng.choice(gt.shape[0], size=K, replace=False)
        Q = gt[idx] * 0.1                                              # → nm
        Q = Q - Q.mean(axis=1, keepdims=True) + pos.mean(0)
    else:
        Q = np.broadcast_to(pos, (K, N, 3)).copy()
    Q += 0.005 * rng.normal(size=Q.shape)

    mc = MultiContext(sys_, K, platform_name=args.platform)

    print(f"Minimising {K} init configs...", flush=True)
    t_min = time.time()
    for k in range(K):
        mc.contexts[k].setPositions(Q[k] * unit.nanometer)
        openmm.LocalEnergyMinimizer.minimize(mc.contexts[k], tolerance=1.0,
                                               maxIterations=100)
        st = mc.contexts[k].getState(getPositions=True)
        Q[k] = np.asarray(st.getPositions(asNumpy=True)
                            .value_in_unit(unit.nanometer))
    E_init, _ = mc.energy_forces(Q)
    print(f"  minimised in {time.time()-t_min:.1f}s  "
          f"E={E_init}", flush=True)

    # Warmup
    print(f"Warmup: {args.warmup_blocks} blocks × "
          f"({args.swap_every} HMC × {args.L} leapfrog + swap)", flush=True)
    t0 = time.time()
    hmc_accepts = np.zeros(K)
    hmc_counts  = 0
    swap_counts = {i: [0, 0] for i in range(K - 1)}
    parity = 0
    for b in range(args.warmup_blocks):
        for _ in range(args.swap_every):
            Q, E, acc = hmc_trajectory(Q, mc, taus_kT, h_vec, args.L, rng)
            hmc_accepts += acc.astype(np.float64)
            hmc_counts  += 1
        Q, E, sw = swap_adjacent(Q, E, taus_kT, rng, parity)
        for i, a in sw:
            swap_counts[i][0] += int(a); swap_counts[i][1] += 1
        parity = 1 - parity
        if (b + 1) % 50 == 0:
            print(f"  warmup block {b+1}/{args.warmup_blocks}  "
                  f"HMC_acc={hmc_accepts/hmc_counts}  "
                  f"E_cold={E[0]:+.1f} kJ/mol", flush=True)
    warm_time = time.time() - t0
    print(f"warmup done in {warm_time:.1f}s "
          f"(HMC_acc={hmc_accepts/hmc_counts})", flush=True)

    # Production
    print(f"Production: {args.n_blocks} blocks ...", flush=True)
    traj_cold = np.empty((args.n_blocks, N, 3), dtype=np.float64)
    E_cold_trace = np.empty(args.n_blocks, dtype=np.float64)
    hmc_accepts = np.zeros(K); hmc_counts = 0
    swap_counts = {i: [0, 0] for i in range(K - 1)}
    t0 = time.time()
    for b in range(args.n_blocks):
        for _ in range(args.swap_every):
            Q, E, acc = hmc_trajectory(Q, mc, taus_kT, h_vec, args.L, rng)
            hmc_accepts += acc.astype(np.float64)
            hmc_counts  += 1
        Q, E, sw = swap_adjacent(Q, E, taus_kT, rng, parity)
        for i, a in sw:
            swap_counts[i][0] += int(a); swap_counts[i][1] += 1
        parity = 1 - parity
        traj_cold[b]   = Q[0]
        E_cold_trace[b] = E[0]
    prod_time = time.time() - t0
    print(f"production done in {prod_time:.1f}s "
          f"(HMC_acc={hmc_accepts/hmc_counts})", flush=True)
    swap_rates = {i: (v[0] / max(1, v[1])) for i, v in swap_counts.items()}
    print(f"swap_rates: {swap_rates}", flush=True)

    np.save(out / "trajectory_cold.npy", traj_cold)

    phi_our = dihedral_np(traj_cold, *phi_ij)
    psi_our = dihedral_np(traj_cold, *psi_ij)
    gt = np.load(args.gt)
    phi_gt = dihedral_np(gt, *phi_ij); psi_gt = dihedral_np(gt, *psi_ij)

    kl_ours_to_gt = ramachandran_kl(phi_our, psi_our, phi_gt, psi_gt)
    kl_gt_to_ours = ramachandran_kl(phi_gt,  psi_gt,  phi_our, psi_our)
    print(f"KL(ours‖gt) = {kl_ours_to_gt:.3f} bits", flush=True)
    print(f"KL(gt‖ours) = {kl_gt_to_ours:.3f} bits", flush=True)

    # Ramachandran
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, (p, s, name) in zip(axes,
                                  [(phi_gt, psi_gt, "GT (AD2_relaxed)"),
                                   (phi_our, psi_our,
                                    f"PT-HMC  K={K}  {args.n_blocks} blocks")]):
        h, xe, ye = np.histogram2d(np.rad2deg(p), np.rad2deg(s),
                                     bins=80, range=[[-180, 180], [-180, 180]])
        ax.pcolormesh(xe, ye, np.log(h.T + 1), cmap="viridis", shading="auto")
        ax.set_title(name)
        ax.set_xlabel("φ (deg)"); ax.set_ylabel("ψ (deg)")
        ax.set_aspect("equal")
    fig.suptitle(f"Ramachandran — KL(ours‖gt)={kl_ours_to_gt:.3f} bits")
    fig.tight_layout()
    fig.savefig(out / "ramachandran.png", dpi=120)

    m = dict(
        sampler="pt_hmc",
        taus=taus.tolist(), h_base=args.h_base, L=args.L,
        swap_every=args.swap_every,
        warmup_blocks=args.warmup_blocks, n_blocks=args.n_blocks,
        hmc_accept=(hmc_accepts / hmc_counts).tolist(),
        swap_accept={str(k): v for k, v in swap_rates.items()},
        warmup_sec=warm_time, production_sec=prod_time,
        elapsed_sec=warm_time + prod_time,
        n_frames=int(traj_cold.shape[0]),
        E_cold_mean=float(E_cold_trace.mean()),
        E_cold_std =float(E_cold_trace.std()),
        kl_ours_to_gt_bits=kl_ours_to_gt,
        kl_gt_to_ours_bits=kl_gt_to_ours,
    )
    with open(out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2, default=str)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
