"""Run GMM-VI (`sampler_gmbbvi` from dev/) on LJ-13 at τ=0.5.

The VI target is exp(−(E_LJ + E_osc)/τ) with our standard formula.
GMM-VI fits a K-component Gaussian mixture by minimising the reverse
KL divergence — unlike MCMC, it's an *optimisation* method whose
"samples" are draws from the fitted mixture.
"""
from __future__ import annotations

import argparse, json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "dev"))

import jax, jax.numpy as jnp, numpy as np
from gmbbvi import sampler_gmbbvi

from MD.energies import lj_energy
from MD.metrics import energy_w2, distance_w2
from MD.ground_truth import load, pick_init, FILES


def make_lj_log_prob_single(n_particles, tau, rm=1.0, eps=1.0, osc_scale=1.0):
    def lp(x):                                                       # (D,) -> scalar
        coords = x.reshape(n_particles, 3)
        iu, ju = jnp.triu_indices(n_particles, k=1)
        diffs = coords[iu] - coords[ju]
        d = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + 1e-12)
        inv6 = (rm / d) ** 6
        inv12 = inv6 * inv6
        E_lj = (eps / tau) * (inv12 - 2.0 * inv6).sum()
        com = coords.mean(axis=0, keepdims=True)
        E_osc = 0.5 * jnp.sum((coords - com) ** 2) / tau
        return -(E_lj + osc_scale * E_osc)
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=["lj13", "lj55", "dw4"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n-modes", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--n-ens", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--n-chains", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--init-jitter", type=float, default=0.2)
    ap.add_argument("--anneal", action="store_true", default=True)
    ap.add_argument("--no-anneal", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    if args.no_anneal:
        args.anneal = False

    n = FILES[args.problem]["n_particles"]
    lp_single = make_lj_log_prob_single(n, args.tau)

    init = pick_init(args.problem, args.n_chains, seed=args.seed)
    if args.init_jitter > 0:
        k = jax.random.key(args.seed + 1)
        init = init + args.init_jitter * jax.random.normal(k, init.shape)

    gt = load(args.problem, "test")
    rng = np.random.default_rng(args.seed)
    gt_sub = gt[rng.choice(gt.shape[0], size=min(10_000, gt.shape[0]), replace=False)]
    gt_E = np.asarray(lj_energy(jnp.asarray(gt_sub), n))

    print(f"[{args.problem}] GMBBVI  τ={args.tau}  n_modes={args.n_modes}  "
          f"warmup={args.warmup}  n_ens={args.n_ens}  dt={args.dt}  "
          f"n_chains={args.n_chains}  n_samples={args.n_samples}  "
          f"jitter={args.init_jitter}", flush=True)
    print(f"GT: E = {gt_E.mean():+.3f} ± {gt_E.std():.3f}", flush=True)

    t0 = time.time()
    samples, info = sampler_gmbbvi(
        lp_single, init, args.n_samples,
        warmup=args.warmup, n_modes=args.n_modes,
        n_ens=args.n_ens, dt=args.dt,
        anneal=args.anneal, seed=args.seed, verbose=True)
    samples.block_until_ready()
    elapsed = time.time() - t0

    arr = np.asarray(samples).reshape(-1, samples.shape[-1])
    sub = arr[rng.choice(arr.shape[0], size=min(20_000, arr.shape[0]),
                          replace=False)]
    E_parts = []
    for i in range(0, sub.shape[0], 4096):
        E_parts.append(np.asarray(lj_energy(jnp.asarray(sub[i:i+4096]), n)))
    sub_E = np.concatenate(E_parts)
    sub_E = sub_E[np.isfinite(sub_E)]

    m = dict(problem=args.problem, sampler="gmbbvi",
             tau=args.tau, n_modes=args.n_modes, warmup=args.warmup,
             n_ens=args.n_ens, dt=args.dt, n_samples=args.n_samples,
             init_jitter=args.init_jitter, anneal=args.anneal,
             elapsed_sec=elapsed,
             gt=dict(E_mean=float(gt_E.mean()), E_std=float(gt_E.std())),
             sampler_stats=dict(E_mean=float(sub_E.mean()),
                                 E_std=float(sub_E.std()),
                                 E_min=float(sub_E.min())),
             weights=[float(x) for x in info["weights"]],
             energy_W2_vs_gt=float(energy_w2(sub_E, gt_E)),
             distance_W2_vs_gt=float(distance_w2(sub, gt_sub, n, 3)))
    out = ROOT / "results" / f"{args.problem}_gmbbvi_K{args.n_modes}_tau{args.tau:g}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in m.items() if k != 'weights'}, indent=2))
    print(f"weights (top 10 of {len(m['weights'])}):",
          sorted(m['weights'], reverse=True)[:10])


if __name__ == "__main__":
    main()
