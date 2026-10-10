"""Review-only PEACHES ensemble-size studies on standard Gaussians."""

import argparse
import json
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmark_stretch import _mean_trace_ess
from _support import peaches_from_checkout, save_metadata, write_results


OUTPUT_ROOT = Path(__file__).resolve().parent / "data"
WARMUP = 200
PRODUCTION = 1000
RESULT_STEM = "peaches_size_post_time_fix"
SAMPLER = None


def _condition_p95(values):
    values = np.sort(values)
    upper = int(np.ceil(.95 * (len(values) - 1)))
    if not np.isfinite(values[upper]):
        return float(values[upper])
    return float(np.percentile(values, 95))


def _metric_summary(states, final_step_size):
    spectra = []
    ranks = []
    global_spectra = []
    global_ranks = []
    for state in states:
        whole = np.asarray(state, dtype=np.float64)
        whole = whole - np.mean(whole, axis=0)
        global_ranks.append(np.linalg.matrix_rank(whole))
        global_spectra.append(np.linalg.eigvalsh(whole.T @ whole / len(whole)))
        half = state.shape[0] // 2
        for group in (state[:half], state[half:]):
            centered = np.asarray(group, dtype=np.float64) - np.mean(group, axis=0)
            ranks.append(np.linalg.matrix_rank(centered))
            spectra.append(np.linalg.eigvalsh(centered.T @ centered / len(group)))
    spectra = np.asarray(spectra)
    condition = spectra[:, -1] / np.maximum(spectra[:, 0], 1e-300)
    condition = np.where(np.asarray(ranks) < states.shape[-1], np.inf, condition)
    global_spectra = np.asarray(global_spectra)
    global_condition = global_spectra[:, -1] / np.maximum(global_spectra[:, 0], 1e-300)
    global_condition = np.where(np.asarray(global_ranks) < states.shape[-1], np.inf, global_condition)
    return dict(metric_min_rank=int(np.min(ranks)),
                metric_median_condition=float(np.median(condition)),
                metric_p95_condition=_condition_p95(condition),
                metric_min_eigenvalue=float(np.min(spectra[:, 0])),
                global_metric_min_rank=int(np.min(global_ranks)),
                global_metric_median_condition=float(np.median(global_condition)),
                global_metric_p95_condition=_condition_p95(global_condition),
                leapfrog_stable_fraction=float(np.mean(
                    final_step_size * np.sqrt(spectra[:, -1]) < 2.)))


def _run(dim, walkers, seed, initial_scale, maximum_initial, chees_metric=None):
    initial = initial_scale * maximum_initial[:walkers]
    log_prob = lambda x: -0.5 * jnp.sum(x * x, axis=-1)
    start = time.perf_counter()
    settings = dict(warmup=WARMUP, seed=seed, verbose=False)
    if chees_metric is not None:
        settings['chees_metric'] = chees_metric
    samples, info = SAMPLER(log_prob, initial, PRODUCTION, **settings)
    samples.block_until_ready()
    elapsed = time.perf_counter() - start
    samples = np.asarray(samples)
    ensemble_means = np.mean(samples, axis=1, dtype=np.float64)
    radius = np.mean(np.sum(samples * samples, axis=-1), axis=1, dtype=np.float64)
    traces = [ensemble_means[:, d] for d in range(dim)] + [radius]
    stats = [_mean_trace_ess(trace) for trace in traces]
    ess = np.asarray([entry[0] for entry in stats])
    tau = np.asarray([entry[1] for entry in stats])
    reliable = np.asarray([entry[2] for entry in stats])
    mean = np.mean(samples, axis=(0, 1), dtype=np.float64)
    variance = np.var(samples, axis=(0, 1), dtype=np.float64)
    delta = samples[1:] - samples[:-1]
    esjd = float(np.mean(np.sum(delta * delta, axis=-1, dtype=np.float64)))
    moves = float(np.mean(np.any(delta != 0., axis=-1)))
    gradient_cost_per_walker_iteration = info['n_grad_evals'] / (PRODUCTION * walkers)
    result = dict(dim=dim, n_walkers=walkers, seed=seed, initial_scale=initial_scale,
                  chees_metric=chees_metric or 'affine-invariant',
                  warmup=WARMUP, production=PRODUCTION,
                  final_step_size=info['final_step_size'],
                  nominal_L=info['nominal_L'], mean_L=info['mean_L'],
                  acceptance_rate=info['acceptance_rate'], actual_move_rate=moves,
                  n_grad_evals=info['n_grad_evals'], esjd=esjd,
                  esjd_per_gradient_eval=esjd / gradient_cost_per_walker_iteration,
                  mean_rms_error=float(np.sqrt(np.mean(mean * mean))),
                  variance_rms_error=float(np.sqrt(np.mean((variance - 1.)**2))),
                  variance_mean=float(np.mean(variance)),
                  squared_radius_mean=float(np.mean(radius)),
                  mean_coordinate_ess_min=float(np.min(ess[:-1])),
                  squared_radius_ess=float(ess[-1]),
                  observable_ess_min=float(np.min(ess)),
                  observable_max_tau=float(np.max(tau)),
                  reliable_ess_fraction=float(np.mean(reliable)),
                  effective_draws_equivalent_min=float(walkers * np.min(ess)),
                  effective_draws_per_million_gradients=float(
                      walkers * np.min(ess) / info['n_grad_evals'] * 1e6),
                  wall_seconds_including_jit=elapsed,
                  output_bytes=int(samples.nbytes),
                  momentum_matrix_bytes_per_half_update=int((walkers // 2)**2 * samples.dtype.itemsize),
                  estimated_walk_matmul_flops_per_iteration=float(
                      8. * info['mean_L'] * (walkers // 2)**2 * dim),
                  process_peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  finite_samples=bool(np.isfinite(samples).all()))
    result.update(_metric_summary(samples[::20], info['final_step_size']))
    initial_metric = _metric_summary(np.asarray(initial)[None, ...], info['final_step_size'])
    result.update({"initial_" + key: value for key, value in initial_metric.items()})
    return result


def _save(results):
    write_results(OUTPUT_ROOT / RESULT_STEM, results)


def main():
    global OUTPUT_ROOT, WARMUP, PRODUCTION, RESULT_STEM, SAMPLER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run D=2, N=6, seed=0, stationary start")
    parser.add_argument("--representative", action="store_true", help="Run D=2 N=6/24/48 and D=8 N=18/72")
    parser.add_argument("--high-dimensional", action="store_true",
                        help="Current PEACHES: D=32/64, N=2(D+1)/4D/8D, two stationary-start seeds")
    parser.add_argument("--n2d-comparison", action="store_true",
                        help="D=32/64, N=2D, two stationary-start seeds, default and Euclidean ChEES")
    parser.add_argument("--near-minimum-extra", action="store_true",
                        help="D=32/64, N=2D+4, two stationary-start seeds, default ChEES")
    parser.add_argument("--underdispersed-high-dimensional", action="store_true",
                        help="D=32/64, N=2D/4D, seed0, initial scale .1, default ChEES")
    parser.add_argument("--pre-time-fix", action="store_true", help="Reconstruct the historical ChEES time prefactor")
    parser.add_argument("--complement-tuner-reference", action="store_true",
                        help="Restore the recorded complement-only tuner before the small-ensemble fallback")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--production", type=int, default=PRODUCTION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    OUTPUT_ROOT, WARMUP, PRODUCTION = args.output_dir, args.warmup, args.production
    RESULT_STEM = "peaches_size_pre_time_fix" if args.pre_time_fix else "peaches_size_post_time_fix"
    if args.high_dimensional:
        if args.pre_time_fix or args.representative:
            parser.error("--high-dimensional requires the current time signal and its own size grid")
        RESULT_STEM = "peaches_size_high_dimension"
    if args.n2d_comparison:
        if args.pre_time_fix or args.representative or args.high_dimensional:
            parser.error("--n2d-comparison has its own current-source size and metric grid")
        RESULT_STEM = "peaches_size_n2d"
    if args.near_minimum_extra:
        if args.pre_time_fix or args.representative or args.high_dimensional or args.n2d_comparison:
            parser.error("--near-minimum-extra has its own current-source size grid")
        RESULT_STEM = "peaches_size_n2d_plus4"
    if args.underdispersed_high_dimensional:
        if (args.pre_time_fix or args.representative or args.high_dimensional
                or args.n2d_comparison or args.near_minimum_extra):
            parser.error("--underdispersed-high-dimensional has its own current-source size grid")
        RESULT_STEM = "peaches_size_high_dimension_underdispersed"
    SAMPLER = peaches_from_checkout(pre_time_fix=args.pre_time_fix,
                                   complement_tuner_reference=args.complement_tuner_reference)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = [(2, walkers) for walkers in (6, 12, 24, 48)]
    cases += [(8, walkers) for walkers in (18, 36, 72)]
    if args.representative:
        cases = [(2, walkers) for walkers in (6, 24, 48)] + [(8, walkers) for walkers in (18, 72)]
    seeds, scales = list(range(3)), [1., .1]
    metrics = [None]
    if args.high_dimensional:
        cases = [(dim, walkers) for dim in (32, 64)
                 for walkers in (2 * (dim + 1), 4 * dim, 8 * dim)]
        seeds, scales = [0, 1], [1.]
    if args.n2d_comparison:
        cases = [(32, 64), (64, 128)]
        seeds, scales, metrics = [0, 1], [1.], [None, 'euclidean']
    if args.near_minimum_extra:
        cases = [(32, 68), (64, 132)]
        seeds, scales = [0, 1], [1.]
    if args.underdispersed_high_dimensional:
        cases = [(dim, walkers) for dim in (32, 64) for walkers in (2 * dim, 4 * dim)]
        seeds, scales = [0], [.1]
    if args.quick:
        cases, seeds, scales = [(2, 6)], [0], [1.]
    save_metadata(OUTPUT_ROOT / (RESULT_STEM + "_run.json"), {
        "study": "PEACHES ensemble size", "quick": args.quick, "warmup": WARMUP,
        "production": PRODUCTION, "initial_seed": "42 + seed", "sampler_seeds": seeds,
        "dimensions_and_walkers": {str(dim): [w for d, w in cases if d == dim]
                                   for dim in sorted(set(d for d, _ in cases))},
        "initial_scales": scales, "initial_arrays_nested": True,
        "initial_array_walkers_by_dimension": {
            str(dim): {2: 48, 8: 72}.get(dim, 8 * dim) for dim in set(d for d, _ in cases)},
        "representative": args.representative, "pre_time_fix_reference": args.pre_time_fix,
        "high_dimension": args.high_dimensional, "actual_cases": cases,
        "n2d_comparison": args.n2d_comparison,
        "near_minimum_extra": args.near_minimum_extra,
        "underdispersed_high_dimension": args.underdispersed_high_dimensional,
        "chees_metrics": ['affine-invariant' if m is None else m for m in metrics],
        "complement_tuner_reference": args.complement_tuner_reference,
        "actual_sampler_seeds": seeds, "actual_initial_scales": scales,
        "time_signal": "halton*T" if args.pre_time_fix else "actual eps*L",
        "memory_note": "RSS is cumulative process peak; per-case matrix/output bytes are structural estimates",
        "wall_note": "Includes sampler JIT; excludes host diagnostics; not a steady-state speed benchmark",
    })
    results = []
    total = len(cases) * len(seeds) * len(scales) * len(metrics)
    for dim in sorted(set(case[0] for case in cases)):
        # Keep the same nested starting arrays as the full protocol in quick mode.
        maximum_walkers = {2: 48, 8: 72}.get(dim, 8 * dim)
        for seed in seeds:
            maximum_initial = jax.random.normal(jax.random.key(42 + seed),
                                                 (maximum_walkers, dim))
            for _, walkers in [case for case in cases if case[0] == dim]:
                for initial_scale in scales:
                    for metric in metrics:
                        result = _run(dim, walkers, seed, initial_scale, maximum_initial,
                                      chees_metric=metric)
                        result['covariance_profile'] = ('complement reference'
                            if args.complement_tuner_reference else 'checkout default')
                        results.append(result)
                        _save(results)
                        print(f"{len(results)}/{total} D={dim} N={walkers} seed={seed} "
                              f"metric={result['chees_metric']} init_scale={initial_scale}: "
                              f"h={result['final_step_size']:.3g}, L={result['mean_L']:.1f}, "
                              f"accept={result['acceptance_rate']:.3f}, "
                              f"ESJD/grad={result['esjd_per_gradient_eval']:.3f}, "
                              f"ESS={result['observable_ess_min']:.1f}, "
                              f"varRMS={result['variance_rms_error']:.3f}", flush=True)


if __name__ == '__main__':
    main()
