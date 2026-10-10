"""Controlled review-only comparison of stretch warmup and retry widths.

Run from the repository root with JAX_PLATFORMS=cpu PYTHONPATH=src.
No repository files are changed. Runtime includes JIT compilation; efficiency
comparisons use movement and ESS per target evaluation instead of that timing.
"""

import csv
import gc
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np


from _support import sampler_from_checkout, save_metadata, write_results

OUTPUT_ROOT = Path(__file__).resolve().parent / "data"
WARMUP = 200
PRODUCTION = 1000


def _variants():
    return [
        ("stretch_original", sampler_from_checkout("stretch.py", "sampler_stretch", "original"), 1),
        ("stretch_positive_width", sampler_from_checkout("stretch.py", "sampler_stretch", "positive"), 1),
        ("dr_original", sampler_from_checkout("ensemble_dr.py", "sampler_ensemble_dr_stretch", "original"), 2),
        ("dr_positive_width", sampler_from_checkout("ensemble_dr.py", "sampler_ensemble_dr_stretch", "positive"), 2),
        ("dr_positive_width_narrow_retry", sampler_from_checkout(
            "ensemble_dr.py", "sampler_ensemble_dr_stretch", "positive", narrow_retry=True), 2),
    ]


def _mean_trace_ess(trace):
    """Sokal-window ESS for an ensemble-mean trace; stuck traces have zero ESS.

    This measures effective ensemble averages, not independent walker draws.
    It is a finite-run comparison and does not certify convergence.
    """
    trace = np.asarray(trace, dtype=np.float64)
    if np.ptp(trace) == 0:
        return 0., np.inf, False
    centered = trace - np.mean(trace)
    variance = np.dot(centered, centered)
    if variance == 0 or not np.isfinite(variance):
        return 0., np.inf, False
    size = len(centered)
    fft = np.fft.rfft(centered, n=2 * size)
    acf = np.fft.irfft(fft * np.conjugate(fft), n=2 * size)[:size]
    acf /= acf[0]
    taus = 2. * np.cumsum(acf) - 1.
    indices = np.arange(size)
    windows = np.flatnonzero(indices >= 5. * taus)
    window = int(windows[0]) if len(windows) else size // 2
    tau = max(1., float(taus[window]))
    return size / tau, tau, size >= 50. * tau


def _problem(target, dim, seed):
    walkers = max(20, 2 * dim)
    initial_key = jax.random.key(42 + seed)
    gaussian = jax.random.normal(initial_key, (walkers, dim))
    if target == 'gaussian':
        log_prob = lambda x: -0.5 * jnp.sum(x * x, axis=-1)
        return gaussian, log_prob, lambda samples: samples
    if target == 'correlated_gaussian':
        covariance = 0.9 ** jnp.abs(jnp.arange(dim)[:, None] - jnp.arange(dim)[None, :])
        chol = jnp.linalg.cholesky(covariance)
        inverse = jnp.linalg.inv(chol)
        initial = gaussian @ chol.T
        log_prob = lambda x: -0.5 * jnp.sum((x @ inverse.T)**2, axis=-1)
        inverse = np.asarray(inverse)
        return initial, log_prob, lambda samples: samples @ inverse.T
    if target == 'funnel':
        v = 3. * gaussian[:, 0]
        initial = gaussian.at[:, 0].set(v)
        initial = initial.at[:, 1:].set(jnp.exp(v[:, None] / 2.) * gaussian[:, 1:])

        def log_prob(x):
            v, latent = x[:, 0], x[:, 1:]
            return (-0.5 * v * v / 9. - 0.5 * (dim - 1) * v
                    - 0.5 * jnp.exp(-v) * jnp.sum(latent * latent, axis=-1))

        def standardize(samples):
            output = samples.copy()
            output[:, :, 0] /= 3.
            output[:, :, 1:] *= np.exp(-samples[:, :, :1] / 2.)
            return output

        return initial, log_prob, standardize
    raise ValueError(target)


def _measure(target, dim, seed, name, sampler, stages, initial, log_prob, standardize):
    start = time.perf_counter()
    samples, info = sampler(log_prob, initial, PRODUCTION, warmup=WARMUP,
                            seed=seed, verbose=False)
    samples.block_until_ready()
    elapsed = time.perf_counter() - start
    raw = np.asarray(samples)
    standard = standardize(raw)
    mean = np.mean(standard, axis=(0, 1), dtype=np.float64)
    variance = np.var(standard, axis=(0, 1), dtype=np.float64)
    delta = standard[1:] - standard[:-1]
    jump = np.mean(np.sum(delta * delta, axis=-1, dtype=np.float64))
    moves = np.mean(np.any(raw[1:] != raw[:-1], axis=-1))
    ensemble_mean = np.mean(standard, axis=1, dtype=np.float64)
    ess, tau, reliable = zip(*[_mean_trace_ess(ensemble_mean[:, d]) for d in range(dim)])
    walkers = initial.shape[0]
    evals = PRODUCTION * walkers * stages
    result = dict(
        target=target, dim=dim, seed=seed, variant=name, n_walkers=walkers,
        warmup=WARMUP, production=PRODUCTION,
        final_a=info['final_a'], acceptance_rate=info['acceptance_rate'],
        stage1_rate=info.get('stage1_rate', info['acceptance_rate']),
        stage2_rate=info.get('stage2_rate', 0.), actual_move_rate=float(moves),
        mean_rms_error=float(np.sqrt(np.mean(mean * mean))),
        variance_rms_error=float(np.sqrt(np.mean((variance - 1.)**2))),
        variance_mean=float(np.mean(variance)), esjd=float(jump),
        esjd_per_density_eval=float(jump / stages),
        ensemble_mean_min_ess=float(np.min(ess)),
        ensemble_mean_median_ess=float(np.median(ess)),
        ensemble_mean_max_tau=float(np.max(tau)),
        ess_per_million_density_evals=float(np.min(ess) / evals * 1e6),
        reliable_ess_fraction=float(np.mean(reliable)),
        production_density_evals=evals, wall_seconds_including_jit=elapsed,
        finite_samples=bool(np.isfinite(raw).all()),
    )
    if target == 'funnel':
        result.update(v_mean=float(np.mean(raw[:, :, 0])),
                      v_variance=float(np.var(raw[:, :, 0])),
                      x_variance_mean=float(np.mean(np.var(raw[:, :, 1:], axis=(0, 1)))))
    del samples, raw, standard, delta, ensemble_mean
    gc.collect()
    return result


def _save(results):
    write_results(OUTPUT_ROOT / 'stretch', results)


def main():
    global OUTPUT_ROOT, WARMUP, PRODUCTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run only D=2, seed=0, all five variants")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--production", type=int, default=PRODUCTION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    OUTPUT_ROOT, WARMUP, PRODUCTION = args.output_dir, args.warmup, args.production
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_metadata(OUTPUT_ROOT / "stretch_run.json", {
        "study": "stretch warmup", "quick": args.quick, "warmup": WARMUP,
        "production": PRODUCTION, "initial_seed": "42 + seed", "sampler_seeds": [0, 1, 2],
        "gaussian_dimensions": [2, 10, 50, 100], "correlated_case": "D=10, seed=0, rho=.9",
        "funnel_case": "D=5, seeds=0/1, variance(v)=9", "density_cost_per_walker": "1 ordinary, 2 DR",
    })
    variants = _variants()
    cases = [('gaussian', dim, seed) for dim in (2, 10, 50, 100) for seed in range(3)]
    cases += [('correlated_gaussian', 10, 0)]
    cases += [('funnel', 5, seed) for seed in range(2)]
    if args.quick:
        cases = [('gaussian', 2, 0)]
    results = []
    total = len(cases) * len(variants)
    for target, dim, seed in cases:
        initial, log_prob, standardize = _problem(target, dim, seed)
        for name, sampler, stages in variants:
            result = _measure(target, dim, seed, name, sampler, stages,
                              initial, log_prob, standardize)
            results.append(result)
            _save(results)
            print(f"{len(results)}/{total} {target} D={dim} seed={seed} {name}: "
                  f"a={result['final_a']:.4g}, move={result['actual_move_rate']:.3f}, "
                  f"ESJD/eval={result['esjd_per_density_eval']:.3f}, "
                  f"mean-ESS={result['ensemble_mean_min_ess']:.1f}", flush=True)


if __name__ == '__main__':
    main()
