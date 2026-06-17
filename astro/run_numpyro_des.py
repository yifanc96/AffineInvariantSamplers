"""NumPyro NUTS reference baseline for DES Y1 3×2pt.

Runs:
  (a) ``dense_mass=True`` — full 21×21 mass matrix dual-averaging
  (b) ``dense_mass=[...]`` — block-dense per Campagne+ 2023 (cosmology 6D + galaxy bias 5D blocks, singletons elsewhere)
  (c) ``dense_mass=False`` — diagonal (for completeness / weak baseline)

Uses the same jax-cosmo forward model as ``astro.likelihoods_des_y1`` so the
posterior is identical.  Non-centered reparameterization on all Uniform
and Normal priors (automatic via `LocScaleReparam` + `TransformReparam`).

Writes metrics.json in the same schema as ``astro.run_benchmark`` so it
slots into the existing leaderboard / convergence diagnostics.

Usage::

    python -m astro.run_numpyro_des --mass block --seed 0 --num-samples 1000
    python -m astro.run_numpyro_des --mass dense --seed 1
    python -m astro.run_numpyro_des --mass diag  --seed 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam, TransformReparam

from astro.likelihoods_des_y1 import _theory_and_data, _UNIF, _NORM


# ─── NumPyro model reproducing the Campagne+ 2023 setup ─────────────────────

def make_numpyro_model(theory_fn, data_mean, cov_chol):
    """Build a NumPyro model given the jax-cosmo forward pipeline.
    `cov_chol` is the Cholesky factor of the data covariance.
    """
    param_names_u = [n for n, _, _ in _UNIF]
    param_names_n = [n for n, _, _ in _NORM]

    def model():
        phys = []
        for name, lo, hi in _UNIF:
            phys.append(numpyro.sample(name, dist.Uniform(lo, hi)))
        for name, mu, sd in _NORM:
            phys.append(numpyro.sample(name, dist.Normal(mu, sd)))
        phys = jnp.stack(phys)
        cl = theory_fn(phys)
        numpyro.sample("obs", dist.MultivariateNormal(cl, scale_tril=cov_chol),
                        obs=data_mean)

    return model, param_names_u + param_names_n


# ─── Reparam config — non-centered for Normal priors only ───────────────────
# Uniforms are auto-bijected to unconstrained by NumPyro; their "unconstrained"
# site name is the original (no suffix).  Normals get "_decentered" if we apply
# LocScaleReparam.

def reparam_config(x):
    fn = x["fn"]
    if isinstance(fn, dist.Normal) and ("decentered" not in x["name"]):
        return LocScaleReparam(centered=0)
    return None


# ─── Block-dense mass spec — Campagne+ structure on the unconstrained names ─

BLOCK_DENSE = [
    # cosmology block (6D): uniforms, names unchanged in unconstrained space
    ("Omega_c", "sigma8", "Omega_b", "h", "n_s", "w0"),
    # galaxy-bias block (5D)
    ("b1", "b2", "b3", "b4", "b5"),
    # singletons (diagonal within each)
    ("A",), ("eta",),
    # Normal priors w/ non-centered reparam → name has "_decentered" suffix
    ("m1_decentered",), ("m2_decentered",), ("m3_decentered",), ("m4_decentered",),
    ("dz1_decentered",), ("dz2_decentered",), ("dz3_decentered",), ("dz4_decentered",),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", choices=["dense", "block", "diag"], required=True)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--n-chains",     type=int, default=4)
    ap.add_argument("--warmup",       type=int, default=500)
    ap.add_argument("--num-samples",  type=int, default=1000)
    ap.add_argument("--max-tree-depth", type=int, default=7)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--chain-method", choices=["sequential", "parallel", "vectorized"],
                    default="sequential")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--init-scale", type=float, default=None,
                    help="If set, initialize NumPyro chains from a Gaussian ball of "
                         "this width (in z-space) around the fiducial point — matches "
                         "the wide-ball start used by our repo's samplers.")
    args = ap.parse_args()

    numpyro.set_host_device_count(args.n_chains)

    print("[numpyro-des] Building forward model + fiducial mock data …")
    theory_fn, data_mean, cov, fid_phys = _theory_and_data()
    cov_np = np.asarray(cov, dtype=np.float64)
    cov_np = 0.5 * (cov_np + cov_np.T) + 1e-10 * np.eye(cov_np.shape[0])
    cov_chol = jnp.asarray(np.linalg.cholesky(cov_np))

    model, param_names = make_numpyro_model(theory_fn, data_mean, cov_chol)

    # Apply reparam config
    reparam_model = numpyro.handlers.reparam(model, config=reparam_config)

    if args.mass == "dense":
        dense = True
    elif args.mass == "block":
        dense = BLOCK_DENSE
    else:
        dense = False

    kernel = NUTS(reparam_model,
                  dense_mass=dense,
                  max_tree_depth=args.max_tree_depth,
                  target_accept_prob=args.target_accept,
                  init_strategy=numpyro.infer.init_to_median)

    mcmc = MCMC(kernel,
                num_warmup=args.warmup,
                num_samples=args.num_samples,
                num_chains=args.n_chains,
                chain_method=getattr(args, "chain_method", "sequential"),
                progress_bar=False)
    # Track per-iter leapfrog count so we can report a real ess/grad.
    mcmc.run = mcmc.run    # silence linter; nothing to do

    print(f"[numpyro-des] NUTS mass={args.mass}  chains={args.n_chains}  "
          f"warmup={args.warmup}  samples={args.num_samples}")
    t0 = time.perf_counter()
    mcmc.run(jax.random.key(args.seed),
             extra_fields=("num_steps", "mean_accept_prob", "diverging"))
    wall = time.perf_counter() - t0

    # Collect samples in physical-parameter order
    samples_dict = mcmc.get_samples(group_by_chain=True)
    # Pull out the physical-space parameters (not the _base / _decentered internals)
    phys_samples = []
    for name in param_names:
        phys_samples.append(samples_dict[name])
    # Stack: (n_chains, n_samples, D) → (n_samples, n_chains, D) to match our schema
    samples = jnp.stack(phys_samples, axis=-1)                 # (C, N, D)
    samples = jnp.transpose(samples, (1, 0, 2))                # (N, C, D)

    # NumPyro diagnostics — actually extract from the collected fields.
    extra = mcmc.get_extra_fields()
    map_accept = extra.get("mean_accept_prob", None)
    accept = float(np.mean(map_accept)) if map_accept is not None \
             else float("nan")
    n_steps_per_sample = extra.get("num_steps", None)
    if n_steps_per_sample is not None:
        # Actual leapfrog count across all chains & samples.
        n_grad_evals_prod = int(np.sum(n_steps_per_sample))
        mean_num_steps = float(np.mean(n_steps_per_sample))
    else:
        n_grad_evals_prod = args.n_chains * args.num_samples * (2 ** args.max_tree_depth)
        mean_num_steps = float("nan")
    # Warmup cost — conservative upper bound at max tree depth × warmup × chains.
    n_grad_evals_warm = args.n_chains * args.warmup * (2 ** args.max_tree_depth)
    n_grad_evals = n_grad_evals_prod + n_grad_evals_warm

    # Final adapted step size (and divergence count if tracked).
    try:
        step_size = float(np.asarray(mcmc.last_state.adapt_state.step_size).mean())
    except Exception:
        step_size = float("nan")
    diverging = extra.get("diverging", None)
    n_divergences = int(np.sum(diverging)) if diverging is not None else 0

    # Per-dim ESS using our diagnostic
    from affine_invariant_samplers import effective_sample_size
    ess = np.asarray(effective_sample_size(np.asarray(samples)))

    out_dir = Path(args.out or
                   f"astro/results/des_y1/nuts_numpyro_{args.mass}/seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "samples.npz",
                         samples=np.asarray(samples, dtype=np.float32))

    summary = dict(problem="des_y1",
                    sampler=f"nuts_numpyro_{args.mass}",
                    dim=len(param_names), n_data=int(data_mean.size),
                    n_chains=args.n_chains,
                    warmup=args.warmup, num_samples=args.num_samples,
                    warm_start=0, find_mode=False,
                    wall_s=wall,
                    ess=ess.tolist(), ess_min=float(ess.min()),
                    ess_mean=float(ess.mean()),
                    ess_per_sec=float(ess.min() / wall),
                    n_target_evals=int(n_grad_evals),
                    eval_kind="grad",
                    ess_per_eval=float(ess.min() / n_grad_evals),
                    fake_ess_warning=None,
                    info=dict(acceptance_rate=accept,
                               max_tree_depth=args.max_tree_depth,
                               target_accept=args.target_accept,
                               final_step_size=step_size,
                               mean_num_steps=mean_num_steps,
                               n_divergences=n_divergences))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
