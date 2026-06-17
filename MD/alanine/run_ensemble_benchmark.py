"""Ensemble-preconditioning benchmark on alanine dipeptide.

Compares:
    MALT                 — identity mass, kinetic Langevin, MH        (non-ensemble)
    pickles              — ensemble mass, kinetic Langevin, MH        (ensemble MH)
    pickles_unadjusted   — ensemble mass, BAOAB, no MH                (ensemble no-MH)
    plain BAOAB          — identity mass, kinetic Langevin, no MH     (non-ensemble no-MH)
    (optional) chess     — identity mass, HMC + ChEES, MH
    (optional) peaches   — ensemble mass, HMC + ChEES, MH

Target: OpenMM ff99SBildn + GB-OBC energy (same as our PT-HMC runs).  We
initialise N=256 chains from minimised GT configs (no jitter) to keep
chains well inside the basin where MH can work.

Numerical safeguards:
    - Gradient-norm clipping at `--grad-clip` (default 2000 /nm, ≈ 2× typical
      near-basin force), wraps the JAX gradient before passing to samplers.
    - All samplers receive the same `grad_lp_batched` so clipping is uniform.

Outputs (for later analysis without re-running):
    results/<sampler>/samples.npy        (T, N, D=66) — full trajectory
    results/<sampler>/phi_psi.npy        (T, N, 2)    — torsion time series
    results/<sampler>/metrics.json
    results/<sampler>/ramachandran.png
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import ShapeDtypeStruct

import openmm
import openmm.unit as unit

from affine_invariant_samplers import (
    sampler_pickles_unadjusted, sampler_pickles, sampler_malt,
    sampler_chess, sampler_peaches,
)

from MD.alanine.alanine import (
    build_system, phi_psi_atom_indices, dihedral_np, kT_kJmol_300K,
)


class SingleContextEnergy:
    """One OpenMM Context re-used across the batch.  Same semantics as
    OpenMMEnergy in ``alanine.py`` but no N-context limit — setPositions +
    getState in a Python for-loop.  Pure-callback dispatch overhead
    dominates anyway, so reusing one context costs us only the absent
    parallelism (which OpenMM's Context objects don't give us on one GPU)."""

    def __init__(self, system, platform_name: str = "OpenCL"):
        self.system = system
        self.n_atoms = system.getNumParticles()
        self.platform = openmm.Platform.getPlatformByName(platform_name)
        integ = openmm.VerletIntegrator(1e-3)
        self.ctx = openmm.Context(system, integ, self.platform)

    def energy_forces(self, coords_batch_nm: np.ndarray):
        B = coords_batch_nm.shape[0]
        e = np.empty(B, dtype=np.float64)
        f = np.empty((B, self.n_atoms, 3), dtype=np.float64)
        for i in range(B):
            self.ctx.setPositions(coords_batch_nm[i] * unit.nanometer)
            state = self.ctx.getState(getEnergy=True, getForces=True)
            e[i] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            f[i] = np.asarray(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer))
        return e, f
from MD.alanine.run_langevin import ramachandran_kl

ROOT = Path(__file__).resolve().parent


def make_callbacks(system, n_chains: int, kT: float,
                    grad_clip: float = 2000.0,
                    platform_name: str = "OpenCL"):
    """Build BOTH batched and single-point log_prob callbacks, plus a
    gradient-clipped batched gradient that all samplers can use.

    Gradient-norm clipping: if ‖∇log π‖ > grad_clip for a chain, rescale to
    that magnitude.  Defends against leapfrog blow-up at the repulsive wall
    without changing samples in the populated region (typical ‖∇‖ ≈ 100–1000)."""

    oe = SingleContextEnergy(system, platform_name=platform_name)
    n_atoms = oe.n_atoms
    D = 3 * n_atoms
    inv_kT = 1.0 / kT

    # ── Single-point (D,) -> scalar, the ONLY callback we declare.
    # Batched versions are jax.vmap over this; pure_callback's
    # vmap_method="sequential" handles the shape bookkeeping correctly.
    def _s_host(x_flat):
        x = np.asarray(x_flat, dtype=np.float64).reshape(1, n_atoms, 3)
        e, f = oe.energy_forces(x)
        return e[0].astype(np.float64), f[0].reshape(-1).astype(np.float64)

    e_spec_s = ShapeDtypeStruct((),    jnp.float64)
    f_spec_s = ShapeDtypeStruct((D,),  jnp.float64)

    @jax.custom_vjp
    def lp_single(x_flat):
        e, _ = jax.pure_callback(_s_host, (e_spec_s, f_spec_s), x_flat,
                                   vmap_method="sequential")
        return -e * inv_kT

    def lp_s_fwd(x_flat):
        e, f = jax.pure_callback(_s_host, (e_spec_s, f_spec_s), x_flat,
                                   vmap_method="sequential")
        return -e * inv_kT, f

    def lp_s_bwd(force, g):
        return (g * force * inv_kT,)

    lp_single.defvjp(lp_s_fwd, lp_s_bwd)

    # Batched lp + gradient built via vmap.  sampler_pickles_unadjusted and
    # sampler_pickles accept (batch, D) -> (batch,), so wrap with vmap.
    lp_batched = jax.vmap(lp_single)
    grad_single = jax.grad(lp_single)

    def grad_lp_clipped(x_flat):
        """(B, D) -> (B, D), with per-chain ‖grad‖ ≤ grad_clip."""
        g = jax.vmap(grad_single)(x_flat)
        norms = jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-12
        scale = jnp.minimum(1.0, grad_clip / norms)
        return g * scale

    return lp_batched, lp_single, grad_lp_clipped, oe


def _evaluate(arr_samples, phi_ij, psi_ij, gt_phi, gt_psi):
    """Compute (phi, psi, ramachandran KL) over a (T, N, D=66) trajectory."""
    T, N, D = arr_samples.shape
    coords = arr_samples.reshape(T * N, D // 3, 3)
    phi = dihedral_np(coords, *phi_ij).reshape(T, N)
    psi = dihedral_np(coords, *psi_ij).reshape(T, N)
    kl = ramachandran_kl(phi.ravel(), psi.ravel(), gt_phi, gt_psi)
    return phi, psi, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains",    type=int,   default=256,
                    help="MUST be ≥ 2d = 132 for full-rank ensemble covariance")
    ap.add_argument("--warmup",      type=int,   default=1000)
    ap.add_argument("--n-samples",   type=int,   default=5000)
    ap.add_argument("--step-size",   type=float, default=0.002,
                    help="h for MH-adjusted samplers (MALT, pickles, peaches)")
    ap.add_argument("--h-unadjusted",type=float, default=0.0015,
                    help="h for unadjusted BAOAB / pickles_unadj — smaller for "
                         "stability without MH")
    ap.add_argument("--L",           type=int,   default=10,
                    help="leapfrog steps per trajectory for MALT/pickles")
    ap.add_argument("--gamma",       type=float, default=2.0)
    ap.add_argument("--grad-clip",   type=float, default=2000.0,
                    help="clip ‖∇log π‖ per chain (defends against repulsive "
                         "wall blowup; typical near-basin ‖∇‖ ≈ 100–1000)")
    ap.add_argument("--samplers", nargs="+",
                    default=["malt", "pickles", "baoab", "pickles_unadjusted"])
    ap.add_argument("--save-thin-by", type=int, default=50,
                    help="thin full-coord save by this factor (default 50). "
                         "Energy + (phi,psi) time series always saved at full "
                         "rate — that's what autocorrelation needs.")
    ap.add_argument("--max-save-mb", type=float, default=150.0,
                    help="refuse to save coords if size exceeds this (MB). "
                         "Falls back to saving only the last N frames.")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--platform",    default="OpenCL")
    ap.add_argument("--gt", default=str(ROOT / "reference" / "AD2_relaxed.npy"))
    args = ap.parse_args()

    topo, sys_, pos, atoms = build_system()
    n_atoms = sys_.getNumParticles()
    D = 3 * n_atoms
    phi_ij, psi_ij = phi_psi_atom_indices(topo)

    print(f"[alanine ensemble benchmark]  n_chains={args.n_chains}  "
          f"warmup={args.warmup}  n_samples={args.n_samples}  "
          f"h_MH={args.step_size}  h_unadj={args.h_unadjusted}  "
          f"L={args.L}  grad_clip={args.grad_clip}", flush=True)

    # Callbacks
    print("Building OpenMM callbacks + grad clipper…", flush=True)
    lp_batched, lp_single, grad_clipped, oe = make_callbacks(
        sys_, n_chains=args.n_chains, kT=kT_kJmol_300K,
        grad_clip=args.grad_clip, platform_name=args.platform)

    # Init: K distinct GT frames, minimised in our FF
    gt = np.load(args.gt)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(gt.shape[0], size=args.n_chains, replace=False)
    chains = gt[idx] * 0.1                                              # Å → nm
    chains = chains - chains.mean(axis=1, keepdims=True) + pos.mean(0)

    print(f"Minimising {args.n_chains} init configs (CPU, ~1 min)…", flush=True)
    t0_min = time.time()
    plat_cpu = openmm.Platform.getPlatformByName("CPU")
    for i in range(args.n_chains):
        integ = openmm.VerletIntegrator(1e-3)
        ctx = openmm.Context(sys_, integ, plat_cpu)
        ctx.setPositions(chains[i] * unit.nanometer)
        openmm.LocalEnergyMinimizer.minimize(ctx, tolerance=1.0, maxIterations=100)
        chains[i] = np.asarray(ctx.getState(getPositions=True).getPositions(
            asNumpy=True).value_in_unit(unit.nanometer))
    print(f"  minimised in {time.time() - t0_min:.1f}s", flush=True)

    init_flat = jnp.asarray(chains.reshape(args.n_chains, D))
    e0 = np.asarray(lp_batched(init_flat)) * (-kT_kJmol_300K)
    print(f"init E: min={e0.min():.1f}  max={e0.max():.1f}  "
          f"mean={e0.mean():.1f} kJ/mol", flush=True)

    # GT for KL
    phi_gt = dihedral_np(gt, *phi_ij)
    psi_gt = dihedral_np(gt, *psi_ij)

    for sampler_name in args.samplers:
        out = ROOT / "results" / f"ensemble_{sampler_name}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {sampler_name} ===", flush=True)
        t0 = time.time()
        try:
            if sampler_name == "baoab":
                # Plain BAOAB per-chain independent.  We implement inline to
                # match the integrator of pickles_unadjusted but with identity
                # mass and no ensemble coupling.
                h = args.h_unadjusted
                alpha = jnp.exp(-args.gamma * h)
                sigma = jnp.sqrt(1.0 - alpha ** 2)

                @jax.jit
                def step(state, key):
                    x, p = state
                    grads = grad_clipped(x)
                    p = p + (h / 2.0) * grads
                    x = x + (h / 2.0) * p
                    noise = jax.random.normal(key, x.shape)
                    p = alpha * p + sigma * noise
                    x = x + (h / 2.0) * p
                    grads = grad_clipped(x)
                    p = p + (h / 2.0) * grads
                    return (x, p), x

                key = jax.random.key(args.seed)
                x = init_flat
                p = jax.random.normal(jax.random.key(args.seed + 1), x.shape)
                state = (x, p)
                # warmup
                for _ in range(args.warmup):
                    key, k = jax.random.split(key)
                    state, _ = step(state, k)
                samples = []
                for _ in range(args.n_samples):
                    key, k = jax.random.split(key)
                    state, xs = step(state, k)
                    samples.append(xs)
                samples = jnp.stack(samples, axis=0)
                samples.block_until_ready()
                info = {}
            elif sampler_name == "pickles_unadjusted":
                samples, info = sampler_pickles_unadjusted(
                    lp_batched, init_flat, args.n_samples,
                    warmup=args.warmup, step_size=args.h_unadjusted,
                    gamma=args.gamma, adapt=None,
                    grad_log_prob_fn=grad_clipped,
                    seed=args.seed, verbose=False)
            elif sampler_name == "malt":
                samples, info = sampler_malt(
                    lp_single, init_flat, args.n_samples,
                    warmup=args.warmup, step_size=args.step_size,
                    L=args.L, gamma=1.0,
                    seed=args.seed, verbose=False)
            elif sampler_name == "pickles":
                samples, info = sampler_pickles(
                    lp_batched, init_flat, args.n_samples,
                    warmup=args.warmup, step_size=args.step_size,
                    L=args.L, gamma=1.0, adapt_L=False,
                    grad_log_prob_fn=grad_clipped,
                    seed=args.seed, verbose=False)
            elif sampler_name == "chess":
                samples, info = sampler_chess(
                    lp_batched, init_flat, args.n_samples,
                    warmup=args.warmup, step_size=args.step_size,
                    grad_log_prob_fn=grad_clipped,
                    seed=args.seed, verbose=False)
            elif sampler_name == "peaches":
                samples, info = sampler_peaches(
                    lp_batched, init_flat, args.n_samples,
                    warmup=args.warmup, step_size=args.step_size,
                    grad_log_prob_fn=grad_clipped,
                    seed=args.seed, verbose=False)
            samples.block_until_ready()
            wall = time.time() - t0
            arr = np.asarray(samples)                                   # (T, N, D)
            T, N, _ = arr.shape

            # ── always save small, cheap observables ─
            coords = arr.reshape(T * N, D // 3, 3)
            phi_full = dihedral_np(coords, *phi_ij).reshape(T, N).astype(np.float32)
            psi_full = dihedral_np(coords, *psi_ij).reshape(T, N).astype(np.float32)
            np.save(out / "phi_psi.npy",
                     np.stack([phi_full, psi_full], axis=-1))          # (T, N, 2)

            E_all = -np.asarray(lp_batched(arr.reshape(-1, D))) * kT_kJmol_300K
            E_series = E_all.reshape(T, N).astype(np.float32)
            np.save(out / "energy.npy", E_series)                       # (T, N)

            # ── save coords, thinned or truncated per size budget ─
            full_mb = arr.nbytes / (1024 ** 2)
            thin = max(1, int(args.save_thin_by))
            thinned = arr[::thin].astype(np.float32)
            thin_mb = thinned.nbytes / (1024 ** 2)
            if thin_mb <= args.max_save_mb:
                np.save(out / f"samples_thin{thin}.npy", thinned)
                save_mode = f"thin{thin}"
            else:
                # fallback: save last K frames where K·N·D·4 ≈ max_save_mb
                K = max(1, int(args.max_save_mb * 1024 ** 2 / (N * D * 4)))
                np.save(out / f"samples_last{K}.npy",
                         arr[-K:].astype(np.float32))
                save_mode = f"last{K}"
            print(f"coords full={full_mb:.1f} MB → saved {save_mode}", flush=True)

            # KL
            phi_our = phi_full; psi_our = psi_full
            kl = ramachandran_kl(phi_our.ravel(), psi_our.ravel(),
                                  phi_gt, psi_gt)

            kl_rev = ramachandran_kl(phi_gt, psi_gt,
                                      phi_our.ravel(), psi_our.ravel())

            # Ramachandran plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                                      sharex=True, sharey=True)
            for ax, (p, s, name) in zip(axes,
                    [(phi_gt, psi_gt, "GT (AD2_relaxed)"),
                     (phi_our.ravel(), psi_our.ravel(), sampler_name)]):
                h, xe, ye = np.histogram2d(np.rad2deg(p), np.rad2deg(s),
                                             bins=80,
                                             range=[[-180, 180], [-180, 180]])
                ax.pcolormesh(xe, ye, np.log(h.T + 1), cmap="viridis",
                              shading="auto")
                ax.set_title(name); ax.set_aspect("equal")
                ax.set_xlabel("φ (deg)"); ax.set_ylabel("ψ (deg)")
            fig.suptitle(f"{sampler_name} — KL(ours‖gt) = {kl:.3f} bits")
            fig.tight_layout()
            fig.savefig(out / "ramachandran.png", dpi=120)
            plt.close(fig)

            # metrics
            m = dict(
                sampler=sampler_name,
                n_chains=args.n_chains, warmup=args.warmup,
                n_samples=args.n_samples,
                step_size=float(args.step_size),
                h_unadjusted=float(args.h_unadjusted),
                grad_clip=float(args.grad_clip),
                wall_sec=float(wall),
                acceptance_rate=float(info["acceptance_rate"])
                                  if "acceptance_rate" in info else None,
                final_step_size=float(info["final_step_size"])
                                  if "final_step_size" in info else None,
                kl_ours_to_gt_bits=float(kl),
                kl_gt_to_ours_bits=float(kl_rev),
                shape=list(arr.shape),
            )
            with open(out / "metrics.json", "w") as f:
                json.dump(m, f, indent=2)
            print(json.dumps(m, indent=2))
        except Exception as e:
            print(f"FAILED: {e!r}", flush=True)
            with open(out / "metrics.json", "w") as f:
                json.dump(dict(sampler=sampler_name,
                                error=repr(e),
                                wall_sec=time.time() - t0), f, indent=2)


if __name__ == "__main__":
    main()
