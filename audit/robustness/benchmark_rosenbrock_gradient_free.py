"""Paired gradient-free Rosenbrock study with separately counted preparation.

Run from the checkout root with JAX_PLATFORMS=cpu PYTHONPATH=src. The target
is a product of Rosenbrock pairs, not the chained Rosenbrock function.
"""

import argparse
import gc
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from affine_invariant_samplers.stretch import sampler_stretch
from affine_invariant_samplers.ensemble_dr import sampler_ensemble_dr_stretch
from affine_invariant_samplers.side import sampler_side
from affine_invariant_samplers.walk import sampler_walk
from benchmark_stretch import _mean_trace_ess
from _support import save_metadata, write_results


OUTPUT_ROOT = Path(__file__).resolve().parent / "data"


def log_prob(x):
    even, odd = x[..., ::2], x[..., 1::2]
    return -jnp.sum((even - 1.)**2 + 100. * (odd - even**2)**2, axis=-1)


def _moments(dim):
    # X~N(1,1/2), Y=X^2+E, E~N(0,1/200) independent of X.
    return (np.tile([1., 1.5], dim // 2),
            np.tile([.5, 2.505], dim // 2))


def _measure(samples, info, dim, stages, production, walkers):
    raw = np.asarray(samples)
    true_mean, true_variance = _moments(dim)
    standardized = (raw - true_mean) / np.sqrt(true_variance)
    means = np.mean(raw, axis=(0, 1), dtype=np.float64)
    variances = np.var(raw, axis=(0, 1), dtype=np.float64)
    coordinate_traces = np.mean(standardized, axis=1, dtype=np.float64)
    second_traces = np.mean(standardized * standardized, axis=1, dtype=np.float64)
    residual_even = np.sqrt(2.) * (raw[..., ::2] - 1.)
    residual_odd = np.sqrt(200.) * (raw[..., 1::2] - raw[..., ::2]**2)
    radius = np.mean(np.sum(residual_even**2 + residual_odd**2, axis=-1),
                     axis=1, dtype=np.float64)
    stats = [_mean_trace_ess(coordinate_traces[:, i]) for i in range(dim)]
    stats += [_mean_trace_ess(second_traces[:, i]) for i in range(dim)]
    stats += [_mean_trace_ess(radius)]
    ess = np.asarray([item[0] for item in stats])
    tau = np.asarray([item[1] for item in stats])
    reliable = np.asarray([item[2] for item in stats])
    delta = standardized[1:] - standardized[:-1]
    esjd = np.mean(np.sum(delta * delta, axis=-1, dtype=np.float64))
    evals = production * walkers * stages
    output = dict(
        finite_samples=bool(np.isfinite(raw).all()),
        acceptance_rate=info['acceptance_rate'],
        stage1_rate=info.get('stage1_rate', info['acceptance_rate']),
        stage2_rate=info.get('stage2_rate', 0.),
        actual_move_rate=float(np.mean(np.any(raw[1:] != raw[:-1], axis=-1))),
        normalized_mean_rms_error=float(np.sqrt(np.mean(
            (means - true_mean)**2 / true_variance))),
        relative_variance_rms_error=float(np.sqrt(np.mean(
            (variances / true_variance - 1.)**2))),
        means=means.tolist(), variances=variances.tolist(),
        even_mean=float(np.mean(means[::2])), odd_mean=float(np.mean(means[1::2])),
        even_variance=float(np.mean(variances[::2])),
        odd_variance=float(np.mean(variances[1::2])),
        residual_squared_radius_mean=float(np.mean(radius)),
        esjd=float(esjd), esjd_per_density_eval=float(esjd / stages),
        coordinate_ess_min=float(np.min(ess[:dim])),
        centered_second_moment_ess_min=float(np.min(ess[dim:2 * dim])),
        residual_squared_radius_ess=float(ess[-1]),
        observable_ess_min=float(np.min(ess)), observable_max_tau=float(np.max(tau)),
        reliable_ess_fraction=float(np.mean(reliable)),
        ensemble_mean_ess_per_million_density_evals=float(np.min(ess) / evals * 1e6),
        production_density_evals=evals,
    )
    output.update({key: value for key, value in info.items() if key.startswith('final_')})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='D=2, seed=0, all four samplers')
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--production', type=int, default=5000)
    parser.add_argument('--prepare-warmup', type=int, default=100)
    parser.add_argument('--prepare-production', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [('stretch', sampler_stretch, 1),
                ('dr_stretch', sampler_ensemble_dr_stretch, 2),
                ('side', sampler_side, 1), ('walk', sampler_walk, 1)]
    cases = [(dim, seed) for dim in (2, 8) for seed in range(3)]
    if args.quick:
        cases = [(2, 0)]
    save_metadata(args.output_dir / 'rosenbrock_gradient_free_run.json', {
        'study': 'gradient-free product Rosenbrock', 'a': 1., 'b': 100.,
        'cases': cases, 'walkers': 64, 'samplers': [entry[0] for entry in variants],
        'warmup': args.warmup, 'production': args.production,
        'initial': 'unit Gaussian centered at zero, key 42 + seed',
        'preparation': {'sampler': 'stretch', 'warmup': args.prepare_warmup,
                        'production': args.prepare_production, 'seed': '10000 + seed'},
        'main_sampler_seed': 'seed', 'density_cost_per_walker_iteration': '1 ordinary; 2 DR',
        'initial_search_cost': 'not instrumented; excluded from production cost',
        'analytical_moments': {'even_mean': 1., 'odd_mean': 1.5,
                               'even_variance': .5, 'odd_variance': 2.505},
    })
    results, preparation = [], []
    total = len(cases) * len(variants)
    for dim, seed in cases:
        initial = jax.random.normal(jax.random.key(42 + seed), (64, dim))
        start = time.perf_counter()
        prepared, prep_info = sampler_stretch(
            log_prob, initial, args.prepare_production, warmup=args.prepare_warmup,
            seed=10000 + seed, verbose=False)
        prepared.block_until_ready()
        prep_elapsed = time.perf_counter() - start
        prep_mean = np.mean(np.asarray(prepared[-1]), axis=0, dtype=np.float64)
        prep_var = np.var(np.asarray(prepared[-1]), axis=0, dtype=np.float64)
        prep_row = dict(dim=dim, seed=seed, warmup=args.prepare_warmup,
                        production=args.prepare_production, n_walkers=64,
                        density_evals=64 * (1 + args.prepare_warmup + args.prepare_production),
                        wall_seconds_including_jit=prep_elapsed,
                        acceptance_rate=prep_info['acceptance_rate'],
                        final_a=prep_info['final_a'], last_ensemble_mean=prep_mean.tolist(),
                        last_ensemble_variance=prep_var.tolist())
        preparation.append(prep_row)
        write_results(args.output_dir / 'rosenbrock_preparation', preparation)
        same_initial = prepared[-1]
        for name, sampler, stages in variants:
            start = time.perf_counter()
            samples, info = sampler(log_prob, same_initial, args.production,
                                    warmup=args.warmup, seed=seed, verbose=False)
            samples.block_until_ready()
            elapsed = time.perf_counter() - start
            result = dict(dim=dim, seed=seed, variant=name, n_walkers=64,
                          warmup=args.warmup, production=args.production,
                          preparation_density_evals=prep_row['density_evals'],
                          warmup_density_evals_excluding_initial_search=args.warmup * 64 * stages,
                          wall_seconds_including_jit=elapsed)
            result.update(_measure(samples, info, dim, stages, args.production, 64))
            results.append(result)
            write_results(args.output_dir / 'rosenbrock_gradient_free', results)
            print(f"{len(results)}/{total} D={dim} seed={seed} {name}: "
                  f"move={result['actual_move_rate']:.3f}, "
                  f"ESJD/eval={result['esjd_per_density_eval']:.4g}, "
                  f"ESS={result['observable_ess_min']:.1f}, "
                  f"meanRMS={result['normalized_mean_rms_error']:.3f}, "
                  f"varRMS={result['relative_variance_rms_error']:.3f}", flush=True)
            del samples
            gc.collect()


if __name__ == '__main__':
    main()
