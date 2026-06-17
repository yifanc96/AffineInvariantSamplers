"""Run pickles / peaches / peams / pickles_unadjusted on LJ-{13,55}.

Use ``--formula pita`` with ``--tau 1.0`` to sample the same target as
pita's NUTS script (arXiv:2506.16471) — the target of the DEM reference
data.  Default ``--formula idem --tau 0.5`` kept for backwards comparison
with the pre-pita literature."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np
from affine_invariant_samplers import (
    sampler_pickles, sampler_pickles_unadjusted,
    sampler_peaches, sampler_peams,
)

from MD.energies import _pairwise_distances, lj_energy, lj_pita_energy
from MD.metrics import energy_w2, distance_w2
from MD.ground_truth import load, pick_init, FILES

ROOT = Path(__file__).resolve().parent


def make_lj_log_prob_batched(n_particles, tau, rm=1.0, eps=1.0, osc_scale=1.0,
                              formula="idem", smooth=False, r_cut=0.65):
    lj_prefactor = 2.0 if formula == "pita" else 1.0
    from MD.energies import _lj_pair_smooth

    def lp(x):
        coords = x.reshape(x.shape[0], n_particles, 3)
        d = _pairwise_distances(coords)
        if smooth:
            lj_pair = _lj_pair_smooth(d, rm=rm, eps=eps, r_cut=r_cut)
        else:
            inv6 = (rm / d) ** 6; inv12 = inv6 * inv6
            lj_pair = eps * (inv12 - 2.0 * inv6)
        E_lj = lj_prefactor * lj_pair.sum(-1) / tau
        com = coords.mean(-2, keepdims=True)
        E_osc = 0.5 * jnp.sum((coords - com) ** 2, axis=(-2, -1)) / tau
        return -(E_lj + osc_scale * E_osc)
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=["lj13", "lj55"])
    ap.add_argument("--sampler",
                    choices=["pickles", "pickles_unadjusted", "peaches", "peams"],
                    required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n-chains", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=20_000)
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--step-size", type=float, default=0.01)
    ap.add_argument("--L", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--init-jitter", type=float, default=0.0,
                    help="Gaussian jitter added to GT init to spread the ensemble "
                         "(break ensemble-preconditioner collapse).")
    ap.add_argument("--no-init-step-search", action="store_true",
                    help="Disable find_init_step_size in pickles / peaches / peams.")
    ap.add_argument("--formula", default="idem", choices=["idem", "pita"],
                    help="'pita' uses 2·E_LJ + E_osc (arXiv:2506.16471).")
    ap.add_argument("--filter-divergent", type=float, default=0.0,
                    help="If > 0, drop samples whose E exceeds (GT_mean + filter · GT_std). "
                         "Typical: 10–20.  Standard trick for MH-adjusted LJ.")
    ap.add_argument("--smooth", action="store_true",
                    help="Use soft-core LJ below r_cut (prevents chain getting "
                         "stuck in the 1/r^12 wall — needed for MH-adjusted "
                         "samplers on the pita formula).")
    ap.add_argument("--r-cut", type=float, default=0.65)
    args = ap.parse_args()

    n = FILES[args.problem]["n_particles"]
    lp = make_lj_log_prob_batched(n, args.tau, formula=args.formula,
                                    smooth=args.smooth, r_cut=args.r_cut)
    init_split = "pita_test" if args.formula == "pita" else "train"
    try:
        init = pick_init(args.problem, args.n_chains, seed=args.seed,
                           split=init_split)
    except (TypeError, FileNotFoundError):
        init = pick_init(args.problem, args.n_chains, seed=args.seed)
    if args.init_jitter > 0:
        k = jax.random.key(args.seed + 1)
        init = init + args.init_jitter * jax.random.normal(k, init.shape)

    gt_split = "pita_test" if args.formula == "pita" else "test"
    gt = load(args.problem, gt_split)
    rng = np.random.default_rng(args.seed)
    gt_sub = gt[rng.choice(gt.shape[0], size=min(10_000, gt.shape[0]), replace=False)]
    if args.formula == "pita":
        gt_E = np.asarray(lj_pita_energy(jnp.asarray(gt_sub), n))
    else:
        gt_E = np.asarray(lj_energy(jnp.asarray(gt_sub), n))

    print(f"[{args.problem}] {args.sampler}  τ={args.tau}  n_chains={args.n_chains}  "
          f"warmup={args.warmup}  n_samples={args.n_samples}  h={args.step_size}  "
          f"L={args.L}  gamma={args.gamma}", flush=True)
    print(f"GT (τ=1 eval conv): E = {gt_E.mean():+.3f} ± {gt_E.std():.3f}", flush=True)

    t0 = time.time()
    init_h_search = not args.no_init_step_search
    if args.sampler == "pickles":
        samples, info = sampler_pickles(
            lp, init, args.n_samples, warmup=args.warmup,
            step_size=args.step_size, L=args.L, gamma=args.gamma,
            find_init_step_size=init_h_search, adapt_step_size=True,
            adapt_L=False,
            seed=args.seed, verbose=False)
    elif args.sampler == "pickles_unadjusted":
        samples, info = sampler_pickles_unadjusted(
            lp, init, args.n_samples, warmup=args.warmup,
            step_size=args.step_size, gamma=args.gamma,
            seed=args.seed, verbose=False)
    elif args.sampler == "peaches":
        samples, info = sampler_peaches(
            lp, init, args.n_samples, warmup=args.warmup,
            step_size=args.step_size,
            find_init_step_size=init_h_search, adapt_step_size=True,
            seed=args.seed, verbose=False)
    elif args.sampler == "peams":
        samples, info = sampler_peams(
            lp, init, args.n_samples, warmup=args.warmup,
            step_size=args.step_size,
            find_init_step_size=init_h_search, adapt_step_size=True,
            seed=args.seed, verbose=False)
    samples.block_until_ready()
    elapsed = time.time() - t0

    arr = np.asarray(samples)
    flat = arr.reshape(-1, arr.shape[-1])
    sub = flat[rng.choice(flat.shape[0], size=min(20_000, flat.shape[0]), replace=False)]
    E_fn = lj_pita_energy if args.formula == "pita" else lj_energy
    E_parts = []
    for i in range(0, sub.shape[0], 4096):
        E_parts.append(np.asarray(E_fn(jnp.asarray(sub[i:i + 4096]), n)))
    sub_E = np.concatenate(E_parts)

    keep_frac = 1.0
    if args.filter_divergent > 0:
        E_cut = float(gt_E.mean() + args.filter_divergent * gt_E.std())
        mask = np.isfinite(sub_E) & (sub_E < E_cut)
        keep_frac = float(mask.mean())
        sub = sub[mask]
        sub_E = sub_E[mask]
        print(f"filter_divergent: kept {keep_frac:.3f} of samples "
              f"(E < {E_cut:.2f})", flush=True)

    m = dict(problem=args.problem, sampler=args.sampler, formula=args.formula,
             tau=args.tau, n_chains=args.n_chains, warmup=args.warmup,
             n_samples=args.n_samples, step_size=args.step_size,
             L=args.L, gamma=args.gamma,
             filter_divergent=args.filter_divergent,
             filter_keep_frac=keep_frac,
             acceptance_rate=(float(info["acceptance_rate"]) if "acceptance_rate" in info else None),
             final_step_size=(float(info["final_step_size"]) if "final_step_size" in info else None),
             n_grad_evals=(int(info["n_grad_evals"]) if info.get("n_grad_evals") else None),
             elapsed_sec=elapsed,
             gt=dict(E_mean=float(gt_E.mean()), E_std=float(gt_E.std())),
             sampler_stats=dict(E_mean=float(sub_E.mean()), E_std=float(sub_E.std()),
                                 E_min=float(sub_E.min())),
             energy_W2_vs_gt=float(energy_w2(sub_E, gt_E)),
             distance_W2_vs_gt=float(distance_w2(sub, gt_sub, n, 3)))
    tag = f"{args.problem}_{args.sampler}_tau{args.tau:g}"
    if args.formula != "idem":
        tag += f"_{args.formula}"
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
