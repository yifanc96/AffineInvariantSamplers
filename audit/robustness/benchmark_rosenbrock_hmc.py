"""Controlled Rosenbrock checks for the ensemble Hamiltonian samplers.

Run from the checkout: JAX_PLATFORMS=cpu PYTHONPATH=src python
audit/robustness/benchmark_rosenbrock_hmc.py --output-dir /tmp/rosenbrock-hmc
"""
import argparse
import hashlib
import inspect
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import affine_invariant_samplers as samplers
from benchmark_stretch import _mean_trace_ess
from _support import save_metadata, write_results, metadata


def log_prob(x):
    a, b = x[:, ::2], x[:, 1::2]
    return -jnp.sum(100. * (b - a*a)**2 + (a - 1.)**2, axis=-1)


def measure(samples, info):
    x = np.asarray(samples, dtype=np.float64)
    dim = x.shape[-1]
    mean_true = np.tile([1., 1.5], dim // 2)
    variance_true = np.tile([.5, 2.505], dim // 2)
    mean = x.mean(axis=(0, 1))
    variance = x.var(axis=(0, 1))
    residual = x[:, :, 1::2] - x[:, :, ::2]**2
    observables = np.concatenate([
        x.mean(axis=1), (x*x).mean(axis=1),
        residual.mean(axis=1), (residual*residual).mean(axis=1)], axis=1)
    stats = np.asarray([_mean_trace_ess(trace) for trace in observables.T])
    cost = info.get("n_grad_evals_prod", info.get("n_grad_evals"))
    esjd = np.mean(np.sum(np.diff(x, axis=0)**2 / variance_true, axis=-1))
    return dict(
        mean=mean.tolist(), variance=variance.tolist(),
        mean_error_sigma_max=float(np.max(np.abs(mean - mean_true) / np.sqrt(variance_true))),
        variance_relative_error_max=float(np.max(np.abs(variance / variance_true - 1.))),
        banana_residual_mean=float(residual.mean()),
        banana_residual_variance=float(residual.var()), banana_residual_variance_target=.005,
        coordinate_ensemble_mean_ess_min=float(stats[:dim, 0].min()),
        squared_coordinate_ensemble_mean_ess_min=float(stats[dim:2*dim, 0].min()),
        observable_ensemble_mean_ess_min=float(stats[:, 0].min()),
        reliable_observable_fraction=float(stats[:, 2].mean()),
        actual_move_rate=float(np.mean(np.any(np.diff(x, axis=0) != 0., axis=-1))),
        standardized_esjd=float(esjd), finite_samples=bool(np.isfinite(x).all()),
        production_gradient_evaluations=cost,
        effective_draw_equivalent_per_million_gradients=(
            float(x.shape[1] * stats[:, 0].min() / cost * 1e6) if cost else None),
        info=info,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--samplers", nargs="+", default=["peaches", "pickles", "peanuts", "peams"])
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=750)
    parser.add_argument("--production", type=int, default=3000)
    parser.add_argument("--max-L", type=int, default=20)
    parser.add_argument("--max-tree-depth", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data/rosenbrock_hmc")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(args.output_dir / "run.json", dict(
        study="Rosenbrock ensemble HMC", settings=vars(args) | {"output_dir": str(args.output_dir)},
        initial_seed="42 + seed", initialization="Gaussian ball, scale one",
        preparation="stretch warmup100 plus100 draws; final ensemble passed identically to each method",
        precision="default JAX float32", trajectory_profile="public defaults except explicit limits",
        ess="ensemble-mean coordinate, square, banana residual and residual-square; IAT capped at one",
        cost="production counter; preparation/warmup separate; equivalent draws assume product target ensemble",
    ))
    rows = []
    for dim in args.dimensions:
        if dim % 2:
            raise ValueError("paired Rosenbrock requires even dimension")
        for seed in args.seeds:
            initial = jax.random.normal(jax.random.key(42 + seed), (args.walkers, dim))
            preparation_start = time.perf_counter()
            prepared, preparation_info = samplers.sampler_stretch(
                log_prob, initial, 100, warmup=100, seed=seed, verbose=False)
            prepared.block_until_ready()
            preparation_wall = time.perf_counter() - preparation_start
            initial = prepared[-1]
            for name in args.samplers:
                sampler = getattr(samplers, "sampler_" + name)
                parameters = inspect.signature(sampler).parameters
                settings = dict(warmup=args.warmup, seed=seed, verbose=False)
                if "max_L" in parameters:
                    settings["max_L"] = args.max_L
                if "max_tree_depth" in parameters:
                    settings["max_tree_depth"] = args.max_tree_depth
                start = time.perf_counter()
                samples, info = sampler(log_prob, initial, args.production, **settings)
                samples.block_until_ready()
                row = dict(sampler=name, dim=dim, seed=seed, walkers=args.walkers,
                           warmup=args.warmup, production=args.production,
                           max_L=settings.get("max_L"), max_tree_depth=settings.get("max_tree_depth"),
                           preparation_iterations=200, preparation_log_density_evaluations=201*args.walkers,
                           preparation_wall_seconds=preparation_wall,
                           wall_seconds_including_jit=time.perf_counter() - start,
                           source=metadata({})["source_sha256"],
                           sampler_source_sha256=hashlib.sha256(
                               Path(inspect.getfile(sampler)).read_bytes()).hexdigest())
                row.update(measure(samples, info))
                rows.append(row)
                write_results(args.output_dir / "results", rows)
                print(f"{name} D={dim} seed={seed}: accept={info['acceptance_rate']:.3f} "
                      f"h={info['final_step_size']:.4g} "
                      f"meanerr={row['mean_error_sigma_max']:.3f} "
                      f"varerr={row['variance_relative_error_max']:.3f} "
                      f"ESS={row['observable_ensemble_mean_ess_min']:.1f} "
                      f"cost={row['production_gradient_evaluations']}", flush=True)


if __name__ == "__main__":
    main()
