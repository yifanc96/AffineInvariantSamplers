"""Reproduce the README's three-seed Gaussian comparison (requires NumPyro).

From the repository root:
    python -m benchmark.gaussian_nuts_comparison
    python -m benchmark.gaussian_nuts_comparison --summarize

Each run writes a separate JSON file. Set CUDA_VISIBLE_DEVICES externally to
choose a GPU; --samplers permits splitting independent cells across devices.
All methods start from N(0, I), then receive 1,000 warmup iterations.
The table measures production ESS/gradient; moment errors are also recorded.
It does not measure warmup efficiency or wall-clock performance.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

DIM, WARMUP, DRAWS = 32, 1000, 4096
SEEDS = (5, 6, 7)
SAMPLERS = ("peaches", "peams", "pickles", "peanuts", "nuts_dense", "nuts_diag")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "gaussian_nuts_comparison"


def log_prob(q):
    # Keep the identity affine map explicit to reproduce the exploratory runs.
    x = (q - jnp.zeros(DIM)) @ jnp.eye(DIM)
    return -.05 * jnp.sum((x - 1.)**2, axis=-1)


def precision(samples, gradients):
    """Batch means of complete-ensemble observables include walker correlations."""
    result = {}
    for name, observable, marginal_variance in (
        ("mean", samples, 10.), ("variance", (samples - 1.)**2, 200.)
    ):
        series = observable.mean(axis=1)
        for batch in (64, 128):
            means = series.reshape(DRAWS // batch, batch, DIM).mean(axis=1)
            mcse_sq = batch * means.var(axis=0, ddof=1) / DRAWS
            if not np.all(np.isfinite(mcse_sq) & (mcse_sq > 0)):
                raise ValueError("Nonfinite or constant ensemble observable")
            efficiency = marginal_variance / mcse_sq / gradients
            suffix = "" if batch == 64 else "_double_batch"
            result[name + "_ess_per_gradient" + suffix] = float(np.median(efficiency))
    result["mean_rmse_sigma"] = float(np.sqrt(np.mean(
        ((samples.mean(axis=(0, 1)) - 1.) / np.sqrt(10.))**2)))
    result["variance_relative_rmse"] = float(np.sqrt(np.mean(
        (samples.var(axis=(0, 1)) / 10. - 1.)**2)))
    return result


def run_nuts(initial, seed, dense):
    from numpyro.infer import MCMC, NUTS
    kernel = NUTS(potential_fn=lambda x: -log_prob(x[None])[0],
                  dense_mass=dense, step_size=.1, target_accept_prob=.8,
                  max_tree_depth=10, find_heuristic_step_size=False)
    mcmc = MCMC(kernel, num_warmup=WARMUP, num_samples=DRAWS,
                num_chains=len(initial), chain_method="vectorized", progress_bar=False)
    fields = ("num_steps", "accept_prob", "diverging")
    mcmc.warmup(jax.random.key(seed), init_params=jnp.asarray(initial),
                extra_fields=fields, collect_warmup=True)
    warm = mcmc.get_extra_fields()
    warm_gradients = int(np.asarray(warm["num_steps"]).sum())
    warm_divergences = int(np.asarray(warm["diverging"]).sum())
    # Reuse the adapted state, with a separate production random key.
    mcmc.run(jax.random.key(seed + 100000), extra_fields=fields)
    samples = mcmc.get_samples(group_by_chain=True)
    samples.block_until_ready()
    extra = mcmc.get_extra_fields(group_by_chain=True)
    steps = np.asarray(extra["num_steps"])
    return np.asarray(samples).transpose(1, 0, 2), dict(
        n_grad_evals=int(steps.sum()), n_grad_warmup=warm_gradients,
        n_divergences=int(np.asarray(extra["diverging"]).sum()),
        n_warmup_divergences=warm_divergences,
        acceptance_rate=float(np.asarray(extra["accept_prob"]).mean()),
        mean_L=float(steps.mean()),
        final_step_size=float(np.asarray(mcmc.last_state.adapt_state.step_size).mean()),
        max_tree_depth_hit_rate=float(np.mean(steps >= 2**10 - 1)))


def run(sampler, seed, output):
    import numpyro
    nuts = sampler.startswith("nuts_")
    walkers = 16 if nuts else 128
    initial = np.random.default_rng(2026 + seed).normal(0., 1., (walkers, DIM))
    settings = dict(target="N(1, 10 I)", dimension=DIM, walkers_or_chains=walkers,
                    warmup=WARMUP, draws_per_walker=DRAWS, initialization="N(0, I)",
                    initialization_seed=2026 + seed, sampler_seed=seed, dtype="float64",
                    initial_step_size=.1, target_accept=.8 if nuts or sampler == "peanuts" else .651)
    if nuts:
        settings.update(implementation="NumPyro", mass="dense" if sampler == "nuts_dense" else "diagonal",
                        max_tree_depth=10, chain_method="vectorized",
                        find_heuristic_step_size=False, adapt_step_size=True, adapt_mass_matrix=True,
                        production_seed=seed + 100000)
        samples, info = run_nuts(initial, seed, sampler == "nuts_dense")
        sampler_source = Path(importlib.import_module("numpyro.infer.hmc").__file__)
    else:
        module = importlib.import_module("affine_invariant_samplers." + sampler)
        sampler_source = Path(module.__file__)
        kwargs = dict(warmup=WARMUP, L=10, max_L=128, step_size=.1,
                      target_accept=.651, seed=seed, verbose=False,
                      find_init_step_size=True, adapt_step_size=True)
        if sampler not in ("pickles", "peanuts"):
            kwargs["chees_metric"] = "affine-invariant"
        if sampler not in ("peams", "peanuts"):
            kwargs["adapt_L"] = True
        if sampler == "peaches":
            kwargs.update(move="h-walk", trajectory_jitter=.6)
        if sampler == "pickles":
            kwargs["gamma"] = 2.
        if sampler == "peanuts":
            kwargs = dict(warmup=WARMUP, step_size=.1, target_accept=.8,
                          max_tree_depth=10, move="h-walk", uturn="affine-invariant",
                          sampling="progressive", seed=seed, verbose=False,
                          find_init_step_size=True, adapt_step_size=True)
            settings.update(length_adaptation="No-U-Turn", thin_by=1)
        else:
            settings.update(length_adaptation="ChEES, affine metric", thin_by=1,
                            trajectory_jitter=.6)
        settings.update(kwargs)
        samples, info = getattr(module, "sampler_" + sampler)(
            log_prob, jnp.asarray(initial), DRAWS, **kwargs)
        samples.block_until_ready()
        samples = np.asarray(samples)
    if samples.shape != (DRAWS, walkers, DIM) or not np.isfinite(samples).all():
        raise ValueError("Invalid samples")
    gradients = int(info.get("n_grad_evals_prod", info["n_grad_evals"]))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    record = dict(sampler=sampler, seed=seed, settings=settings, info=info,
                  metrics=precision(samples, gradients),
                  sampler_source_sha256=hashlib.sha256(sampler_source.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sample_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                  source_revision=revision,
                  environment=dict(python=platform.python_version(), jax=jax.__version__,
                                   numpy=np.__version__, numpyro=numpyro.__version__,
                                   devices=[str(d) for d in jax.devices()],
                                   device_kinds=[d.device_kind for d in jax.devices()]))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{sampler}_s{seed}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(sampler, seed, json.dumps(record["metrics"]), flush=True)


def summarize(output):
    print("| Sampler | Mean ESS/gradient | Variance ESS/gradient |")
    print("|---|---:|---:|")
    for sampler in SAMPLERS:
        rows = [json.loads((output / f"{sampler}_s{seed}.json").read_text()) for seed in SEEDS]
        cells = []
        for observable in ("mean", "variance"):
            values = np.array([r["metrics"][observable + "_ess_per_gradient"] for r in rows])
            cells.append(f"{values.mean():.3f} ± {values.std(ddof=1):.3f}")
        print(f"| {sampler} | {' | '.join(cells)} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samplers", nargs="+", choices=SAMPLERS, default=SAMPLERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output_dir)
    else:
        for name in args.samplers:
            for s in args.seeds:
                run(name, s, args.output_dir)
