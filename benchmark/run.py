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
import hashlib
import inspect
import importlib.metadata
import subprocess
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
            max_tree_depth=None, target_accept=None, gamma=None,
            chees_metric=None):
    sampler = PKG_SAMPLERS[name]
    sig = inspect.signature(sampler).parameters
    kw = dict(num_samples=num_samples, warmup=warmup, seed=seed, verbose=False)
    if max_L is not None and "max_L" in sig:
        kw["max_L"] = max_L
    if max_tree_depth is not None and "max_tree_depth" in sig:   # peanuts
        kw["max_tree_depth"] = max_tree_depth
    if target_accept is not None and "target_accept" in sig:
        kw["target_accept"] = target_accept
    if gamma is not None and "gamma" in sig:                     # pickles damping
        kw["gamma"] = gamma
    if chees_metric is not None and "chees_metric" in sig:       # ChEES norm
        kw["chees_metric"] = chees_metric
    t0 = time.perf_counter()
    # MALT's public API evaluates one point, unlike the batched ensemble APIs.
    density = (lambda x: log_prob(x[None, :])[0]) if name == "malt" else log_prob
    samples, info = sampler(density, init, **kw)
    samples.block_until_ready()
    wall = time.perf_counter() - t0
    info = dict(info)
    info["benchmark_effective_settings"] = {
        key: kw.get(key, param.default) for key, param in list(sig.items())[2:]
        if key in kw or param.default is not inspect.Parameter.empty
    }
    return np.asarray(samples), info, wall


def run_nuts(log_prob, init, num_samples, warmup, seed, dense_mass=True,
             target_accept=0.8, max_tree_depth=10, adapt_mass=True):
    import numpyro
    from numpyro.infer import MCMC, NUTS

    init = np.asarray(init)
    n_chains, dim = init.shape
    numpyro.set_host_device_count(1)

    def potential(x):                               # x: (D,) -> scalar PE
        return -log_prob(x[None, :])[0]

    # adapt_mass=False -> identity mass matrix throughout (only the scalar step
    # size is adapted): the "naive HMC / NUTS without mass adaptation" baseline.
    kernel = NUTS(potential_fn=potential, dense_mass=dense_mass,
                  adapt_mass_matrix=adapt_mass,
                  target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=num_samples,
                num_chains=n_chains, chain_method="vectorized", progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.key(seed), init_params=jnp.asarray(init),
             extra_fields=("num_steps", "mean_accept_prob", "diverging"))
    raw_samples = mcmc.get_samples(group_by_chain=True)
    jax.block_until_ready(raw_samples)
    wall = time.perf_counter() - t0

    samp = np.asarray(raw_samples)  # (C, N, D)
    samples = np.transpose(samp, (1, 0, 2))                   # (N, C, D)
    extra = mcmc.get_extra_fields()
    n_steps = extra.get("num_steps", None)
    n_grad_prod = int(np.sum(n_steps)) if n_steps is not None else \
        None
    try:
        accept = float(np.mean(extra.get("mean_accept_prob", np.nan)))
    except Exception:
        accept = float("nan")
    try:
        step = float(np.mean(np.asarray(mcmc.last_state.adapt_state.step_size)))
    except Exception:
        step = None
    try:
        n_div = int(np.sum(np.asarray(extra.get("diverging"))))   # production divergences
    except Exception:
        n_div = None
    info = dict(acceptance_rate=accept, n_grad_evals=n_grad_prod,
                final_step_size=step, n_divergences=n_div,
                mean_num_steps=float(np.mean(n_steps)) if n_steps is not None else None,
                benchmark_effective_settings=dict(num_samples=num_samples, warmup=warmup,
                    seed=seed, dense_mass=dense_mass, adapt_mass=adapt_mass,
                    target_accept=target_accept, max_tree_depth=max_tree_depth,
                    chain_method="vectorized", num_chains=n_chains))
    return samples, info, wall


def production_grads(sampler, info, num_samples, n_chains):
    """Production gradient evaluations; excludes warmup and initial probes.

    Trust production counters before trajectory lengths: PEACHES/PEAMS/CHESS
    evaluate the initial gradient as well as L integration gradients. Cached-
    gradient samplers have a different count. Gradient-free methods return None.
    """
    if sampler in GRADFREE:
        return None
    prod = info.get("n_grad_evals_prod")
    if prod is not None:
        return int(prod)
    # PEANUTS's generic counter includes warmup; other supported samplers' do not.
    exact = info.get("n_grad_evals")
    if sampler != "peanuts" and exact is not None:
        return int(exact)
    # MALT caches the starting gradient and computes one new gradient per step.
    if sampler == "malt" and info.get("L") is not None:
        thin = info.get("benchmark_effective_settings", {}).get("thin_by", 1)
        return int(num_samples * thin * n_chains * info["L"])
    return None


def production_cost(sampler, info, num_samples, n_chains):
    n_grad = production_grads(sampler, info, num_samples, n_chains)
    thin = info.get("benchmark_effective_settings", {}).get("thin_by", 1)
    n_logp = num_samples * thin * n_chains if sampler in GRADFREE else None
    count = n_logp if sampler in GRADFREE else n_grad
    return dict(n_grad_evals_prod=n_grad, n_log_prob_evals_prod=n_logp,
                n_evals_prod=count,
                evaluation_unit=("log_density" if sampler in GRADFREE else "gradient"),
                production_cost_available=count is not None,
                cost_scope="production only; excludes warmup, initial scale probes and preparation")


def source_metadata():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    files = list((root / "src").rglob("*.py")) + [
        root / "benchmark" / name for name in ("run.py", "metrics.py", "distributions.py")]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    versions = {}
    for name in ("jax", "jaxlib", "numpy", "arviz", "numpyro"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(git_revision=git("rev-parse", "HEAD"),
                git_branch=git("branch", "--show-current"),
                git_tracked_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                source_sha256=digest.hexdigest(), library_versions=versions,
                devices=[str(device) for device in jax.devices()],
                jax_enable_x64=bool(jax.config.jax_enable_x64))


def json_safe(value):
    """Strict JSON: unavailable/nonfinite numbers become null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["gaussian", "rosenbrock"], required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--sampler", required=True,
                    choices=list(PKG_SAMPLERS) + ["nuts"])
    ap.add_argument("--log-cond", type=float, default=6.0,
                    help="log eigen-standard-deviation condition; covariance condition is exp(2*log_cond)")
    ap.add_argument("--no-rotate", action="store_true",
                    help="axis-aligned Gaussian covariance (default rotates)")
    ap.add_argument("--n-chains", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--init-scale", type=float, default=2.0)
    ap.add_argument("--max-L", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=None,
                    help="PICKLES kinetic-damping coefficient (recommended 0.5).")
    ap.add_argument("--target-accept", type=float, default=None,
                    help="target acceptance for the step-size adapter "
                         "(peaches/peanuts/NUTS); pin across d to recover the "
                         "d^{-1/4} step-size law.")
    ap.add_argument("--max-tree-depth", type=int, default=10,
                    help="peanuts/NUTS trajectory cap (2^depth steps)")
    ap.add_argument("--nuts-mass", choices=["dense", "diag", "identity"],
                    default="dense",
                    help="NUTS mass matrix; 'diag' on isotropic targets, "
                         "'identity' disables mass adaptation entirely (naive "
                         "HMC baseline: only the step size is adapted).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-seed", type=int, default=0,
                    help="random seed for the target realization (Gaussian "
                         "rotation). Vary across runs to get problem-instance "
                         "error bands rather than sampling-noise-only bands.")
    ap.add_argument("--tag", type=str, default="default")
    ap.add_argument("--chees-metric", choices=["affine-invariant","euclidean"],
                    default=None, help="norm used inside the ChEES ESJD criterion")
    ap.add_argument("--fast-ess", action="store_true",
                    help="score with the ensemble-mean ESS (O(D N log N), no "
                         "per-chain FFT) instead of ArviZ rank ESS; use for "
                         "large walker counts (high-d scaling).")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--warm-start", type=int, default=0,
                    help="Stretch-move pre-warmup iters (gradient-free, "
                         "affine-invariant) to equilibrate the ensemble SHAPE "
                         "before the gradient sampler. The ensemble analogue of "
                         "NUTS mass-matrix warmup; adds no gradient evals. "
                         "Does not use the target covariance (not an oracle).")
    args = ap.parse_args()

    provenance = source_metadata()

    # Build target + analytic truth
    tkw = {}
    if args.target == "gaussian":
        tkw = dict(log_cond=args.log_cond, rotate=not args.no_rotate,
                   seed=args.target_seed)
    log_prob, truth, meta = make_target(args.target, args.dim, **tkw)

    # Init: overdispersed ball so burn-in bias is visible
    init = init_ball(jax.random.key(args.seed + 1), args.n_chains, args.dim,
                     scale=args.init_scale)

    # Optional stretch-move pre-warmup: gradient-free, affine-invariant shape
    # finding. Lets the ensemble cloud morph to the target's anisotropy before
    # the gradient sampler starts (so the empirical-covariance preconditioner
    # is well-conditioned). Uses no gradient and no knowledge of the target.
    preparation_wall = 0.0
    preparation_iterations = 0
    if args.warm_start > 0 and args.sampler != "nuts":
        preparation_t0 = time.perf_counter()
        sw, _ = sampler_stretch(log_prob, init, num_samples=args.warm_start,
                                warmup=args.warm_start, seed=args.seed, verbose=False)
        sw.block_until_ready()
        init = sw[-1]
        preparation_wall = time.perf_counter() - preparation_t0
        preparation_iterations = 2 * args.warm_start

    print(f"[{args.target} D={args.dim}] sampler={args.sampler} "
          f"chains={args.n_chains} warmup={args.warmup} ws={args.warm_start} "
          f"N={args.num_samples} seed={args.seed}", flush=True)

    if args.sampler == "nuts":
        samples, info, wall = run_nuts(log_prob, init, args.num_samples,
                                       args.warmup, args.seed,
                                       dense_mass=(args.nuts_mass == "dense"),
                                       adapt_mass=(args.nuts_mass != "identity"),
                                       max_tree_depth=args.max_tree_depth,
                                       target_accept=(args.target_accept
                                                      if args.target_accept is not None
                                                      else 0.8))
    else:
        samples, info, wall = run_pkg(args.sampler, log_prob, init,
                                      args.num_samples, args.warmup, args.seed,
                                      max_L=args.max_L,
                                      max_tree_depth=args.max_tree_depth,
                                      target_accept=args.target_accept,
                                      gamma=args.gamma,
                                      chees_metric=args.chees_metric)

    cost = production_cost(args.sampler, info, args.num_samples, args.n_chains)
    n_grad = cost["n_grad_evals_prod"]
    # Thin before the (expensive) ESS/R-hat computation ONLY for gradient-free
    # samplers (emcee/stretch), whose IAT is large (~hundreds), so thinning by
    # k << IAT preserves ESS while shrinking the array at large N. Gradient HMC
    # samplers (peaches/peanuts/nuts) have IAT ~ 1, so ANY thinning destroys
    # their ESS -- they are never thinned (n_grad keeps the full production
    # count regardless; thinning and ESS estimation are recorded separately).
    thin = max(1, samples.shape[0] // 5000) if args.sampler in GRADFREE else 1
    samples_m = samples[::thin] if thin > 1 else samples
    met = M.all_metrics(samples_m, truth, n_grad_evals=n_grad or None,
                        fast=args.fast_ess)

    if args.sampler in GRADFREE and cost["n_evals_prod"]:
        met["ess_per_log_prob"] = met["ess_bulk_min"] / cost["n_evals_prod"]
    if cost["n_evals_prod"]:
        met["ess_per_eval"] = met["ess_bulk_min"] / cost["n_evals_prod"]

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
        n_divergences=info.get("n_divergences"),
        **cost,
        requested_settings=vars(args),
        effective_sampler_settings=info.get("benchmark_effective_settings"),
        target_metadata=meta,
        source_metadata=provenance,
        initialization=dict(method="isotropic Gaussian ball", seed=args.seed + 1,
                            scale=args.init_scale, center=0.0),
        preparation=dict(sampler="stretch" if preparation_iterations else None,
                         iterations=preparation_iterations,
                         log_prob_evals=preparation_iterations * args.n_chains,
                         wall_s=preparation_wall,
                         excludes_initial_density=True),
        wall_s_with_preparation=wall + preparation_wall,
        metrics_thin_by=thin, metrics_num_draws=len(samples_m),
        sampler_info=info,
        **met,
    )
    summary["unavailable_metrics"] = [
        key for key, value in {**met, **cost}.items()
        if value is None or isinstance(value, (float, np.floating)) and not np.isfinite(value)]
    summary = json_safe(summary)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({k: summary[k] for k in
                      ("sampler", "dim", "acceptance_rate", "mean_err_sigma_max",
                       "bias_z_max", "iat_max", "ess_bulk_min", "ess_per_grad",
                       "wall_s") if k in summary}, indent=2))


if __name__ == "__main__":
    main()
