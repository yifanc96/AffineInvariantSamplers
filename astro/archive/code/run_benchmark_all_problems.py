"""Benchmark ensemble / HMC samplers against emcee on the two astro posteriors.

Usage
-----
    python -m astro.run_benchmark --problem hogg     --sampler peaches
    python -m astro.run_benchmark --problem hogg     --sampler stretch
    python -m astro.run_benchmark --problem hogg     --sampler emcee
    python -m astro.run_benchmark --problem pantheon --sampler pickles
    python -m astro.run_benchmark --problem pantheon --sampler nuts

Writes samples + info + ESS/wall to ``astro/results/<problem>/<sampler>/``.

`--sampler emcee` uses the reference emcee stretch-move implementation
(`pip install emcee`) on the *same* log-probability function so the
per-gradient / per-likelihood comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from affine_invariant_samplers import (
    sampler_stretch, sampler_walk,
    sampler_peaches, sampler_peams, sampler_peanuts,
    sampler_pickles, sampler_pickles_unadjusted,
    sampler_chess, sampler_nuts, sampler_malt,
    sampler_langevin_walk, sampler_aldi, sampler_gndr,
    effective_sample_size,
)

from astro.likelihoods import make_hogg, init_hogg, make_pantheon, init_pantheon
from astro.likelihoods_transit import make_transit, init_transit
from astro.likelihoods_gw import make_gw_population, init_gw_population
from astro.likelihoods_extra import (make_funnel, init_funnel,
                                      make_illcond_gauss, init_illcond_gauss)
from astro.likelihoods_multi_transit import make_multi_transit, init_multi_transit
from astro.likelihoods_multi_transit_reparam import (
    make_multi_transit_reparam, init_multi_transit_reparam)
from astro.likelihoods_des_y1 import make_des_y1, init_des_y1
from astro.likelihoods_des_y1_phys import make_des_y1_phys, init_des_y1_phys
from astro.mode_finder import ball_around_mode


def _warm_start_stretch(log_prob, init, n_warm: int, seed: int):
    """Run `n_warm` iters of the stretch move; return the final walker positions
    as a new initial ensemble.  Standard emcee practice — gives gradient-based
    samplers a typical-set start so dual-averaging sees reasonable local curvature."""
    samples, _ = sampler_stretch(log_prob, init, num_samples=n_warm,
                                  warmup=n_warm, seed=seed, verbose=False)
    return samples[-1]


SAMPLERS_GRADFREE = {"stretch": sampler_stretch,
                     "walk":    sampler_walk}

SAMPLERS_GRAD    = {"peaches": sampler_peaches,
                    "peams":   sampler_peams,
                    "peanuts": sampler_peanuts,
                    "pickles": sampler_pickles,
                    "pickles_unadjusted": sampler_pickles_unadjusted,
                    "nuts":    sampler_nuts,
                    "malt":    sampler_malt,
                    "langevin_walk":       sampler_langevin_walk,
                    "aldi":                sampler_aldi,
                    "gndr":                sampler_gndr,
                    "chess":   sampler_chess}          # was mis-placed in GRADFREE


def _run_jax(sampler_name, log_prob, init, num_samples, warmup, step_size, seed,
             find_init_step_size=True, max_L=None, target_accept=None,
             residual_fn=None):
    if sampler_name in SAMPLERS_GRADFREE:
        sampler = SAMPLERS_GRADFREE[sampler_name]
        kw = dict(num_samples=num_samples, warmup=warmup, seed=seed)
        t0 = time.perf_counter()
        samples, info = sampler(log_prob, init, **kw)
    else:
        sampler = SAMPLERS_GRAD[sampler_name]
        import inspect
        sig = inspect.signature(sampler).parameters
        kw = dict(num_samples=num_samples, warmup=warmup,
                  step_size=step_size, seed=seed)
        # Only pass find_init_step_size if the sampler accepts it (aldi /
        # pickles_unadjusted do not).
        if "find_init_step_size" in sig:
            kw["find_init_step_size"] = find_init_step_size
        if max_L is not None and "max_L" in sig:             kw["max_L"] = max_L
        if max_L is not None and "max_tree_depth" in sig:
            # NUTS: max_tree_depth = log2(max_L).  Round up.
            import math
            kw["max_tree_depth"] = max(1, int(math.ceil(math.log2(max_L))))
        if target_accept is not None and "target_accept" in sig:
            kw["target_accept"] = target_accept
        # gndr: pass Gauss-Newton residual function (PSD-guaranteed Hessian).
        if residual_fn is not None and "residual_fn" in sig:
            kw["residual_fn"] = residual_fn
        t0 = time.perf_counter()
        samples, info = sampler(log_prob, init, **kw)
    samples.block_until_ready()
    wall = time.perf_counter() - t0
    return np.asarray(samples), dict(info), wall


def _run_emcee(log_prob, init, num_samples, warmup, seed):
    try:
        import emcee
    except ImportError as e:
        raise SystemExit("pip install emcee  (needed for --sampler emcee)") from e

    init_np = np.asarray(init)
    n_walkers, dim = init_np.shape

    # emcee wants a scalar log-prob; reuse the batched JAX one via vmap-of-1.
    jlp = jax.jit(log_prob)
    def scalar_lp(x):
        v = float(jlp(jnp.asarray(x)[None, :])[0])
        return v if np.isfinite(v) else -np.inf

    rng = np.random.default_rng(seed)
    sampler = emcee.EnsembleSampler(n_walkers, dim, scalar_lp)
    t0 = time.perf_counter()
    state = sampler.run_mcmc(init_np, warmup, progress=False,
                             rstate0=rng.bit_generator.state)
    sampler.reset()
    sampler.run_mcmc(state, num_samples, progress=False)
    wall = time.perf_counter() - t0
    samples = sampler.get_chain()                      # (N, n_walkers, D)
    info = dict(acceptance_rate=float(np.mean(sampler.acceptance_fraction)),
                 n_lp_evals=int(num_samples * n_walkers + warmup * n_walkers))
    return samples, info, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem",
                    choices=["hogg", "pantheon", "transit", "gw_pop",
                              "funnel", "illcond", "multi_transit",
                              "multi_transit_reparam", "des_y1", "des_y1_phys"],
                    required=True)
    ap.add_argument("--dim", type=int, default=20,
                    help="Dimension for funnel / illcond problems")
    ap.add_argument("--log-cond", type=float, default=6.0,
                    help="Log condition number for illcond Gaussian")
    ap.add_argument("--max-L", type=int, default=None)
    ap.add_argument("--target-accept", type=float, default=None)
    ap.add_argument("--sampler",  required=True,
                    choices=list(SAMPLERS_GRADFREE) + list(SAMPLERS_GRAD) + ["emcee"])
    ap.add_argument("--n-chains",     type=int,   default=64)
    ap.add_argument("--num-samples",  type=int,   default=5_000)
    ap.add_argument("--warmup",       type=int,   default=1_000)
    ap.add_argument("--step-size",    type=float, default=0.3)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--out",          type=str,   default=None)
    ap.add_argument("--warm-start",   type=int,   default=0,
                    help="Run this many stretch iters before the target sampler, "
                         "then use the last walker positions as the init. Equalises "
                         "the starting point across samplers.")
    ap.add_argument("--no-init-step-search", action="store_true",
                    help="Pass find_init_step_size=False to gradient-based samplers "
                         "— use --step-size as the literal initial step.")
    ap.add_argument("--find-mode", action="store_true",
                    help="Before sampling, run Adam on -log_prob to locate the MAP "
                         "and re-spawn walkers in a small ball around it. Standard "
                         "pymc/numpyro/bilby practice for HMC on new posteriors.")
    ap.add_argument("--init-scale", type=float, default=None,
                    help="Override per-problem default init ball width (in z-space "
                         "for des_y1). Use init-scale=1.0 for a wide-ball start that "
                         "matches the post-stretch-pre-warmup spread without needing "
                         "stretch pre-warmup itself.  des_y1 only.")
    ap.add_argument("--mode-steps", type=int, default=300)
    ap.add_argument("--mode-ball-scale", type=float, default=1e-2)
    args = ap.parse_args()

    if args.problem == "hogg":
        log_prob, meta = make_hogg()
        init = init_hogg(jax.random.key(args.seed + 1), args.n_chains)
    elif args.problem == "pantheon":
        log_prob, meta = make_pantheon()
        init = init_pantheon(jax.random.key(args.seed + 1), args.n_chains)
    elif args.problem == "transit":
        log_prob, meta = make_transit()
        init = init_transit(jax.random.key(args.seed + 1), args.n_chains)
    elif args.problem == "gw_pop":
        log_prob, meta = make_gw_population()
        init = init_gw_population(jax.random.key(args.seed + 1), args.n_chains)
    elif args.problem == "funnel":
        log_prob, meta = make_funnel(dim=args.dim)
        init = init_funnel(jax.random.key(args.seed + 1), args.n_chains, dim=args.dim)
    elif args.problem == "illcond":
        log_prob, meta = make_illcond_gauss(dim=args.dim, log_cond=args.log_cond)
        init = init_illcond_gauss(jax.random.key(args.seed + 1), args.n_chains,
                                    dim=args.dim)
    elif args.problem == "multi_transit":
        log_prob, meta = make_multi_transit()
        init = init_multi_transit(jax.random.key(args.seed + 1), args.n_chains)
    elif args.problem == "multi_transit_reparam":
        log_prob, meta = make_multi_transit_reparam()
        init = init_multi_transit_reparam(jax.random.key(args.seed + 1), args.n_chains)
    elif args.problem == "des_y1_phys":
        log_prob, meta = make_des_y1_phys()
        init_kw = {} if args.init_scale is None else {"init_scale": args.init_scale}
        init = init_des_y1_phys(jax.random.key(args.seed + 1), args.n_chains, **init_kw)
    else:  # des_y1
        log_prob, meta = make_des_y1()
        init_kw = {} if args.init_scale is None else {"init_scale": args.init_scale}
        init = init_des_y1(jax.random.key(args.seed + 1), args.n_chains, **init_kw)

    print(f"[{args.problem}] dim={meta['dim']} n_data={meta['n_data']} "
          f"sampler={args.sampler} walkers={args.n_chains} "
          f"warmup={args.warmup} samples={args.num_samples} "
          f"warm_start={args.warm_start}")

    if args.find_mode:
        bounds = (meta["prior_lo"], meta["prior_hi"])
        # Preserve the user-specified initial ensemble spread when re-centring
        # on MAP — too-tight a ball after find_mode collapses dual averaging.
        ens_sigma = np.asarray(jnp.std(init, axis=0))
        init, mode_info = ball_around_mode(log_prob, init[0], args.n_chains,
                                           scale=args.mode_ball_scale,
                                           seed=args.seed + 2,
                                           n_steps=args.mode_steps,
                                           bounds=bounds,
                                           ensemble_scale=ens_sigma)
        print(f"  MAP found: lp={mode_info['lp_map']:.3f}  "
              f"x_map={np.round(mode_info['x_map'], 3).tolist()}  "
              f"ball_sigma={np.round(mode_info['sigma'], 4).tolist()}")

    if args.warm_start > 0:
        init = _warm_start_stretch(log_prob, init, args.warm_start, args.seed)

    if args.sampler == "emcee":
        samples, info, wall = _run_emcee(log_prob, init,
                                          args.num_samples, args.warmup, args.seed)
    else:
        samples, info, wall = _run_jax(args.sampler, log_prob, init,
                                        args.num_samples, args.warmup,
                                        args.step_size, args.seed,
                                        find_init_step_size=not args.no_init_step_search,
                                        max_L=args.max_L,
                                        target_accept=args.target_accept,
                                        residual_fn=meta.get("residual_fn"))

    # Per-dimension ESS (min across dims is the bottleneck; mean is headline).
    # effective_sample_size handles (N, n_chains, D) directly and returns (D,).
    ess = np.asarray(effective_sample_size(samples))
    out_dir = Path(args.out or f"astro/results/{args.problem}/{args.sampler}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "samples.npz", samples=samples.astype(np.float32))

    # Unified "target-density-call" cost metric.  Gradient-based samplers
    # report n_grad_evals; gradient-free report n_lp_evals (or we back it
    # out from iterations × walkers × moves-per-iter).  For a fair plot,
    # gradient evaluations are ~1–3× the cost of a log-prob call in practice,
    # but we leave that conversion to the reader.
    n_grad  = int(info.get("n_grad_evals")  or 0)
    n_lp    = int(info.get("n_lp_evals")    or 0)
    if not n_grad and not n_lp:
        # Gradient-free ensemble samplers in this repo call log_prob once per
        # walker per iteration (stretch / walk), plus warmup.
        n_lp = (args.num_samples + args.warmup) * args.n_chains
    n_evals = n_grad if n_grad else n_lp
    eval_kind = "grad" if n_grad else "logprob"

    # Sanity: if acceptance ≈ 0, walkers froze at init and ESS is fake
    # (IID init draws look uncorrelated under Sokal's estimator).
    accept = float(info.get("acceptance_rate", 1.0))
    fake_ess_warning = None
    if accept < 0.05:
        fake_ess_warning = (
            f"acceptance_rate={accept:.3g} — walkers likely frozen at init; "
            "reported ESS is of the IID init distribution, NOT the posterior."
        )

    summary = dict(problem=args.problem, sampler=args.sampler,
                   dim=meta["dim"], n_data=meta["n_data"],
                   n_chains=args.n_chains, warmup=args.warmup,
                   num_samples=args.num_samples,
                   warm_start=args.warm_start, find_mode=bool(args.find_mode),
                   wall_s=wall,
                   ess=ess.tolist(), ess_min=float(ess.min()),
                   ess_mean=float(ess.mean()),
                   ess_per_sec=float(ess.min() / wall),
                   n_target_evals=n_evals, eval_kind=eval_kind,
                   ess_per_eval=float(ess.min() / n_evals) if n_evals else None,
                   fake_ess_warning=fake_ess_warning,
                   info={k: (float(v) if np.ndim(v) == 0 else None)
                         for k, v in info.items()
                         if isinstance(v, (int, float, np.floating, np.integer))})
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
