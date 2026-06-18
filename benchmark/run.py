"""Unified benchmark runner: one (target, dim, sampler, warmup, seed) cell.

Writes ``metrics.json`` (accuracy + mixing) to
``benchmark/results/<target>_d<dim>/<sampler>__<tag>/seed<seed>/``.

Samplers:
  - package ensemble / gradient samplers: peaches, peams, pickles, chess,
    stretch, ... (anything in affine_invariant_samplers)
  - ``nuts`` : NumPyro NUTS with a dense mass matrix (reference HMC).

Examples
--------
    python -m benchmark.run --target rosenbrock --dim 8  --sampler peaches
    python -m benchmark.run --target gaussian   --dim 100 --log-cond 6 --sampler stretch
    python -m benchmark.run --target gaussian   --dim 50  --sampler nuts --warmup 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)   # ill-conditioned Gaussians (kappa up to
                                            # e^12 in variance) need float64; float32
                                            # makes the preconditioner/mass-matrix
                                            # inversion blow up.
import jax.numpy as jnp
import numpy as np

from affine_invariant_samplers import (
    sampler_peaches, sampler_peams, sampler_peanuts, sampler_pickles,
    sampler_chess, sampler_stretch, sampler_walk, sampler_malt, sampler_aldi,
)
from benchmark.distributions import make_target, init_ball
from benchmark import metrics as M

PKG_SAMPLERS = {
    "peaches": sampler_peaches, "peams": sampler_peams,
    "peanuts": sampler_peanuts, "pickles": sampler_pickles,
    "chess": sampler_chess, "stretch": sampler_stretch, "walk": sampler_walk,
    "malt": sampler_malt, "aldi": sampler_aldi,
}
GRADFREE = {"stretch", "walk"}


def run_pkg(name, log_prob, init, num_samples, warmup, seed, max_L=None,
            max_tree_depth=None, target_accept=None):
    sampler = PKG_SAMPLERS[name]
    import inspect
    sig = inspect.signature(sampler).parameters
    kw = dict(num_samples=num_samples, warmup=warmup, seed=seed, verbose=False)
    if max_L is not None and "max_L" in sig:
        kw["max_L"] = max_L
    if max_tree_depth is not None and "max_tree_depth" in sig:   # peanuts
        kw["max_tree_depth"] = max_tree_depth
    if target_accept is not None and "target_accept" in sig:
        kw["target_accept"] = target_accept
    t0 = time.perf_counter()
    samples, info = sampler(log_prob, init, **kw)
    samples.block_until_ready()
    wall = time.perf_counter() - t0
    return np.asarray(samples), dict(info), wall


def run_nuts(log_prob, init, num_samples, warmup, seed, dense_mass=True,
             target_accept=0.8, max_tree_depth=10):
    import numpyro
    from numpyro.infer import MCMC, NUTS

    init = np.asarray(init)
    n_chains, dim = init.shape
    numpyro.set_host_device_count(1)

    def potential(x):                               # x: (D,) -> scalar PE
        return -log_prob(x[None, :])[0]

    kernel = NUTS(potential_fn=potential, dense_mass=dense_mass,
                  target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=num_samples,
                num_chains=n_chains, chain_method="vectorized", progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.key(seed), init_params=jnp.asarray(init),
             extra_fields=("num_steps", "mean_accept_prob"))
    wall = time.perf_counter() - t0

    samp = np.asarray(mcmc.get_samples(group_by_chain=True))  # (C, N, D)
    samples = np.transpose(samp, (1, 0, 2))                   # (N, C, D)
    extra = mcmc.get_extra_fields()
    n_steps = extra.get("num_steps", None)
    n_grad_prod = int(np.sum(n_steps)) if n_steps is not None else \
        num_samples * n_chains * (2 ** max_tree_depth)
    try:
        accept = float(np.mean(extra.get("mean_accept_prob", np.nan)))
    except Exception:
        accept = float("nan")
    try:
        step = float(np.mean(np.asarray(mcmc.last_state.adapt_state.step_size)))
    except Exception:
        step = None
    info = dict(acceptance_rate=accept, n_grad_evals=n_grad_prod,
                final_step_size=step,
                mean_num_steps=float(np.mean(n_steps)) if n_steps is not None else None)
    return samples, info, wall


def production_grads(sampler, info, num_samples, n_chains):
    """Production-only leapfrog/gradient count, consistent & exact per sampler.

    Every sampler is counted as (production leapfrog steps) so ESS/grad is a
    fair cross-sampler efficiency. NUTS and peanuts use adaptive trajectory
    lengths, so we use their exact summed step counts (not a nominal L).
    """
    if sampler in GRADFREE:
        return num_samples * n_chains            # one log-prob per walker per iter
    if sampler == "nuts":
        return int(info.get("n_grad_evals") or 0)          # exact sum of num_steps
    if sampler == "peanuts":
        # exact production leapfrog count (added to info); fall back to mean_num_steps
        if info.get("n_grad_evals_prod"):
            return int(info["n_grad_evals_prod"])
        if info.get("mean_num_steps"):
            return int(num_samples * n_chains * info["mean_num_steps"])
    mean_L = info.get("mean_L")
    if mean_L:
        return int(num_samples * n_chains * mean_L)
    return int(info.get("n_grad_evals") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["gaussian", "rosenbrock"], required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--sampler", required=True,
                    choices=list(PKG_SAMPLERS) + ["nuts"])
    ap.add_argument("--log-cond", type=float, default=6.0)
    ap.add_argument("--no-rotate", action="store_true",
                    help="axis-aligned Gaussian covariance (default rotates)")
    ap.add_argument("--n-chains", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--init-scale", type=float, default=2.0)
    ap.add_argument("--max-L", type=int, default=None)
    ap.add_argument("--target-accept", type=float, default=None,
                    help="target acceptance for the step-size adapter "
                         "(peaches/peanuts/NUTS); pin across d to recover the "
                         "d^{-1/4} step-size law.")
    ap.add_argument("--max-tree-depth", type=int, default=10,
                    help="peanuts/NUTS trajectory cap (2^depth steps)")
    ap.add_argument("--nuts-mass", choices=["dense", "diag"], default="dense",
                    help="NUTS mass matrix; use 'diag' on isotropic/diagonal "
                         "targets to avoid noisy dense-mass estimation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="default")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--warm-start", type=int, default=0,
                    help="Stretch-move pre-warmup iters (gradient-free, "
                         "affine-invariant) to equilibrate the ensemble SHAPE "
                         "before the gradient sampler. The ensemble analogue of "
                         "NUTS mass-matrix warmup; adds no gradient evals. "
                         "Does not use the target covariance (not an oracle).")
    args = ap.parse_args()

    # Build target + analytic truth
    tkw = {}
    if args.target == "gaussian":
        tkw = dict(log_cond=args.log_cond, rotate=not args.no_rotate, seed=0)
    log_prob, truth, meta = make_target(args.target, args.dim, **tkw)

    # Init: overdispersed ball so burn-in bias is visible
    init = init_ball(jax.random.key(args.seed + 1), args.n_chains, args.dim,
                     scale=args.init_scale)

    # Optional stretch-move pre-warmup: gradient-free, affine-invariant shape
    # finding. Lets the ensemble cloud morph to the target's anisotropy before
    # the gradient sampler starts (so the empirical-covariance preconditioner
    # is well-conditioned). Uses no gradient and no knowledge of the target.
    if args.warm_start > 0 and args.sampler != "nuts":
        sw, _ = sampler_stretch(log_prob, init, num_samples=args.warm_start,
                                warmup=args.warm_start, seed=args.seed, verbose=False)
        sw.block_until_ready()
        init = sw[-1]

    print(f"[{args.target} D={args.dim}] sampler={args.sampler} "
          f"chains={args.n_chains} warmup={args.warmup} ws={args.warm_start} "
          f"N={args.num_samples} seed={args.seed}", flush=True)

    if args.sampler == "nuts":
        samples, info, wall = run_nuts(log_prob, init, args.num_samples,
                                       args.warmup, args.seed,
                                       dense_mass=(args.nuts_mass == "dense"),
                                       target_accept=(args.target_accept
                                                      if args.target_accept
                                                      else 0.8))
    else:
        samples, info, wall = run_pkg(args.sampler, log_prob, init,
                                      args.num_samples, args.warmup, args.seed,
                                      max_L=args.max_L,
                                      max_tree_depth=args.max_tree_depth,
                                      target_accept=args.target_accept)

    n_grad = production_grads(args.sampler, info, args.num_samples, args.n_chains)
    met = M.all_metrics(samples, truth, n_grad_evals=n_grad or None)

    out_dir = Path(args.out or
                   f"benchmark/results/{args.target}_d{args.dim}/"
                   f"{args.sampler}__{args.tag}/seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(
        target=args.target, dim=args.dim, sampler=args.sampler, tag=args.tag,
        log_cond=args.log_cond if args.target == "gaussian" else None,
        n_chains=args.n_chains, warmup=args.warmup, num_samples=args.num_samples,
        init_scale=args.init_scale, warm_start=args.warm_start,
        seed=args.seed, wall_s=wall,
        acceptance_rate=info.get("acceptance_rate"),
        mean_L=info.get("mean_L") or info.get("mean_num_steps"),
        final_step_size=info.get("final_step_size"),
        n_grad_evals_prod=int(n_grad),
        **met,
    )
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("sampler", "dim", "acceptance_rate", "mean_err_sigma_max",
                       "bias_z_max", "iat_max", "ess_bulk_min", "ess_per_grad",
                       "wall_s") if k in summary}, indent=2))


if __name__ == "__main__":
    main()
