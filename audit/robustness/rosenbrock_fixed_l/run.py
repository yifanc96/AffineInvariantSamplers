"""Eight paired Rosenbrock PEACHES/PICKLES checks with fixed L=16.

Use JAX_PLATFORMS=cpu PYTHONPATH=src python
audit/robustness/rosenbrock_fixed_l/run.py from the checkout root.
"""

import argparse
import hashlib
import inspect
from pathlib import Path
import sys
import time

import jax


RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

import affine_invariant_samplers as samplers
from benchmark_rosenbrock_hmc import log_prob, measure
from _support import metadata, save_metadata, write_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='D2, seed0, both samplers')
    parser.add_argument('--warmup', type=int, default=750)
    parser.add_argument('--production', type=int, default=3000)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [(dim, seed) for dim in (2, 8) for seed in (0, 1)]
    if args.quick:
        cases = [(2, 0)]
    measure_path = RESEARCH_DIR / 'benchmark_rosenbrock_hmc.py'
    save_metadata(args.output_dir / 'run.json', {
        'study': 'fixed-L Rosenbrock PEACHES/PICKLES', 'cases': cases, 'walkers': 64,
        'warmup': args.warmup, 'production': args.production,
        'L': 16, 'adapt_L': False, 'max_L': 128, 'other_parameters': 'public defaults',
        'initial_seed': '42 + seed',
        'preparation': 'stretch, warmup100 plus100 production, seed=seed; share final ensemble',
        'preparation_density_evaluations_per_case': 201 * 64,
        'measurement_helper_sha256': hashlib.sha256(measure_path.read_bytes()).hexdigest(),
        'measurement': 'shared benchmark_rosenbrock_hmc.measure, including coordinate squares/residuals',
        'reference': 'data/rosenbrock_hmc adaptive-L cases, 750+3000, max_L20',
        'cost': 'production gradient counter; excludes preparation, warmup and initial search',
        'wall': 'includes JIT; excludes host diagnostics; no steady-state speed claim',
    })
    results = []
    for dim, seed in cases:
        initial = jax.random.normal(jax.random.key(42 + seed), (64, dim))
        start = time.perf_counter()
        prepared, prep_info = samplers.sampler_stretch(
            log_prob, initial, 100, warmup=100, seed=seed, verbose=False)
        prepared.block_until_ready()
        prep_wall = time.perf_counter() - start
        initial = prepared[-1]
        for name in ('peaches', 'pickles'):
            sampler = getattr(samplers, 'sampler_' + name)
            start = time.perf_counter()
            samples, info = sampler(log_prob, initial, args.production,
                                    warmup=args.warmup, seed=seed, verbose=False,
                                    L=16, adapt_L=False, max_L=128)
            samples.block_until_ready()
            elapsed = time.perf_counter() - start
            row = dict(sampler=name, dim=dim, seed=seed, walkers=64,
                       warmup=args.warmup, production=args.production,
                       L=16, adapt_L=False, max_L=128,
                       preparation_iterations=200,
                       preparation_log_density_evaluations=201 * 64,
                       preparation_wall_seconds=prep_wall,
                       preparation_info=prep_info,
                       wall_seconds_including_jit=elapsed,
                       source=metadata({})['source_sha256'],
                       sampler_source_sha256=hashlib.sha256(
                           Path(inspect.getfile(sampler)).read_bytes()).hexdigest())
            row.update(measure(samples, info))
            results.append(row)
            write_results(args.output_dir / 'results', results)
            print(f"{len(results)}/{2 * len(cases)} {name} D={dim} seed={seed}: "
                  f"move={row['actual_move_rate']:.3f}, "
                  f"h={info['final_step_size']:.4g}, "
                  f"meanerr={row['mean_error_sigma_max']:.3f}, "
                  f"varerr={row['variance_relative_error_max']:.3f}, "
                  f"ESS={row['observable_ensemble_mean_ess_min']:.1f}, "
                  f"cost={row['production_gradient_evaluations']}", flush=True)


if __name__ == '__main__':
    main()
