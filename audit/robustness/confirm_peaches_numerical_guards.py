"""Three Gaussian confirmations after the numerical ChEES guards.

These repeat selected stored cases without overwriting either size grid.
"""

import argparse
import csv
from pathlib import Path

import jax

import benchmark_peaches_ensemble_size as study
from _support import peaches_from_checkout, save_metadata, write_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'data' / 'gaussian_guard_confirmations')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Include the prior tiny-ensemble outlier, the proposed low-D floor, and a
    # high-dimensional 4D ensemble. Preserve each stored production length.
    cases = [(2, 6, 2, .1, 1000), (2, 24, 0, 1., 1000), (64, 256, 0, 1., 500)]
    save_metadata(args.output_dir / 'peaches_numerical_guard_check_run.json', {
        'study': 'Gaussian confirmation after ChEES numerical guards',
        'cases_dim_walkers_seed_initial_scale_production': cases,
        'warmup': 200, 'initial_seed': '42 + sampler seed',
        'maximum_initial_walkers': {'2': 48, '64': 512},
        'reference': 'stored post actual-time-correction, pre numerical-guard size rows',
    })
    study.SAMPLER = peaches_from_checkout()
    study.WARMUP = 200
    stored = []
    data = Path(__file__).resolve().parent / 'data'
    for filename in ('peaches_size_post_time_fix.csv', 'peaches_size_high_dimension.csv'):
        with (data / filename).open() as handle:
            stored.extend(csv.DictReader(handle))
    results = []
    for dim, walkers, seed, scale, production in cases:
        study.PRODUCTION = production
        maximum_walkers = 48 if dim == 2 else 8 * dim
        initial = jax.random.normal(jax.random.key(42 + seed), (maximum_walkers, dim))
        result = study._run(dim, walkers, seed, scale, initial)
        previous = next(row for row in stored if (
            int(row['dim']), int(row['n_walkers']), int(row['seed']),
            float(row['initial_scale'])) == (dim, walkers, seed, scale))
        for field in ('final_step_size', 'mean_L', 'esjd_per_gradient_eval',
                      'observable_ess_min', 'variance_rms_error'):
            result['pre_guard_' + field] = float(previous[field])
        results.append(result)
        write_results(args.output_dir / 'peaches_numerical_guard_check', results)
        print(f"D={dim} N={walkers} seed={seed} scale={scale}: "
              f"h={result['final_step_size']:.6g} (previous {float(previous['final_step_size']):.6g}), "
              f"ESS={result['observable_ess_min']:.1f}, "
              f"varRMS={result['variance_rms_error']:.4g}", flush=True)


if __name__ == '__main__':
    main()
