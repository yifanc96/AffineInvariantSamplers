"""Run sampler_gndr on LJ-{13,55} with an LJ-aware residual function.

Residual is the harmonic-approximation "square root" of the potential:

    r_pair_{ij}  = √(k_LJ·ε/τ) · (d_ij − r_m)     — LJ well curvature near r_m
                                                     k_LJ = 72 (idem) or 144 (pita)
    r_harm_{i,α} = (1/√τ) · (x_{i,α} − x_COM,α)   — COM-relative displacement

For ideal harmonic target  0.5·‖r‖² = E_tot/τ  — so the Gauss-Newton
Hessian J_r^T J_r matches the true Hessian in the harmonic regime.
Use ``--formula pita --tau 1.0`` to match the DEM reference data exactly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from affine_invariant_samplers import sampler_gndr

from MD.energies import _pairwise_distances, lj_energy, lj_pita_energy
from MD.metrics import energy_w2, distance_w2, pairwise_distances_flat
from MD.ground_truth import load, pick_init, FILES

ROOT = Path(__file__).resolve().parent


def make_lj_log_prob_single(n_particles: int, tau: float, rm: float = 1.0,
                              eps: float = 1.0, osc_scale: float = 1.0,
                              formula: str = "idem"):
    lj_prefactor = 2.0 if formula == "pita" else 1.0

    def lp(x):                                                       # (D,) -> scalar
        coords = x.reshape(n_particles, 3)
        iu, ju = jnp.triu_indices(n_particles, k=1)
        diffs = coords[iu] - coords[ju]
        d = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + 1e-12)
        inv6 = (rm / d) ** 6
        inv12 = inv6 * inv6
        E_lj = lj_prefactor * (eps / tau) * (inv12 - 2.0 * inv6).sum()
        com = coords.mean(axis=0, keepdims=True)
        E_osc = 0.5 * jnp.sum((coords - com) ** 2) / tau
        return -(E_lj + osc_scale * E_osc)
    return lp


def make_lj_residual(n_particles: int, tau: float, rm: float = 1.0,
                       eps: float = 1.0, osc_scale: float = 1.0,
                       formula: str = "idem"):
    """Residual whose 0.5·‖r‖² = E_tot/τ in the harmonic approximation.

    LJ pair term: V''(r_m) = 72·ε/r_m² (·2 under pita's factor-of-2 LJ)
                ⇒ pair residual √(k_LJ/τ)·(d − r_m).
    Harmonic term: spring constant osc_scale/τ per coordinate.
    """
    k_LJ_factor = 2.0 if formula == "pita" else 1.0
    sqrt_kLJ   = jnp.sqrt(k_LJ_factor * 72.0 * eps / tau)
    sqrt_kosc  = jnp.sqrt(osc_scale / tau)

    def r(x):
        coords = x.reshape(n_particles, 3)
        iu, ju = jnp.triu_indices(n_particles, k=1)
        diffs = coords[iu] - coords[ju]
        d = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + 1e-12)
        r_pair = sqrt_kLJ * (d - rm)                                 # (n_pairs,)

        com = coords.mean(axis=0, keepdims=True)
        r_harm = (sqrt_kosc * (coords - com)).reshape(-1)            # (3·n,)

        return jnp.concatenate([r_pair, r_harm])

    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=["lj13", "lj55"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n-chains", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--n-samples", type=int, default=20_000)
    ap.add_argument("--step-size", type=float, default=0.05)
    ap.add_argument("--n-try", type=int, default=3)
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--target-accept", type=float, default=0.574)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--formula", default="idem", choices=["idem", "pita"],
                    help="'pita' uses 2·E_LJ + E_osc (arXiv:2506.16471).")
    args = ap.parse_args()

    n = FILES[args.problem]["n_particles"]
    lp_single       = make_lj_log_prob_single(n, args.tau, formula=args.formula)
    residual_single = make_lj_residual(n, args.tau, formula=args.formula)

    init_split = "pita_test" if args.formula == "pita" else "train"
    try:
        init = pick_init(args.problem, args.n_chains, seed=args.seed,
                           split=init_split)
    except FileNotFoundError:
        init = pick_init(args.problem, args.n_chains, seed=args.seed)
    gt_split = "pita_test" if args.formula == "pita" else "test"
    gt = load(args.problem, gt_split)
    rng = np.random.default_rng(args.seed)
    gt_sub = gt[rng.choice(gt.shape[0], size=min(10_000, gt.shape[0]), replace=False)]
    E_fn = lj_pita_energy if args.formula == "pita" else lj_energy
    gt_E = np.asarray(E_fn(jnp.asarray(gt_sub), n))

    print(f"[{args.problem}] GNDR  τ={args.tau}  n_chains={args.n_chains}  "
          f"warmup={args.warmup}  n_samples={args.n_samples}  h={args.step_size}  "
          f"n_try={args.n_try}  shrink={args.shrink}", flush=True)
    print(f"GT (τ=1 evaluation convention): E = {gt_E.mean():+.3f} ± {gt_E.std():.3f}",
          flush=True)

    t0 = time.time()
    samples, info = sampler_gndr(
        lp_single, init, args.n_samples, warmup=args.warmup,
        step_size=args.step_size, n_try=args.n_try, shrink=args.shrink,
        target_accept=args.target_accept,
        residual_fn=residual_single,
        seed=args.seed, verbose=False)
    samples.block_until_ready()
    elapsed = time.time() - t0

    arr = np.asarray(samples)
    flat = arr.reshape(-1, arr.shape[-1])
    sub = flat[rng.choice(flat.shape[0],
                          size=min(20_000, flat.shape[0]), replace=False)]
    E_parts = []
    for i in range(0, sub.shape[0], 4096):
        E_parts.append(np.asarray(E_fn(jnp.asarray(sub[i:i + 4096]), n)))
    sub_E = np.concatenate(E_parts)

    m = dict(problem=args.problem, sampler="gndr",
             tau=args.tau, n_chains=args.n_chains, warmup=args.warmup,
             n_samples=args.n_samples, step_size=args.step_size,
             n_try=args.n_try, shrink=args.shrink,
             acceptance_rate=float(info["acceptance_rate"]),
             stage1_rate=float(info.get("stage1_rate", -1.0))
                           if "stage1_rate" in info else None,
             final_step_size=float(info["final_step_size"]),
             n_grad_evals=int(info.get("n_grad_evals", -1))
                           if info.get("n_grad_evals") else None,
             elapsed_sec=elapsed,
             gt=dict(E_mean=float(gt_E.mean()), E_std=float(gt_E.std())),
             sampler_stats=dict(E_mean=float(sub_E.mean()),
                                 E_std=float(sub_E.std()),
                                 E_min=float(sub_E.min())),
             energy_W2_vs_gt=float(energy_w2(sub_E, gt_E)),
             distance_W2_vs_gt=float(distance_w2(sub, gt_sub, n, 3)))
    tag = f"{args.problem}_gndr_tau{args.tau:g}"
    if args.formula != "idem":
        tag += f"_{args.formula}"
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
