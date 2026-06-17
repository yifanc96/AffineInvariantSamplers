"""NumPyro NUTS reference for des_y1_phys (NO logit/decentered reparam).

Same forward model as ``run_numpyro_des.py`` but parameters are sampled in
physical space with wide-Gaussian priors (matching ``likelihoods_des_y1_phys``).
This is the test bed for whether dense-mass NUTS can recover what the manual
reparam gives — i.e., whether mass-matrix adaptation can do affine-invariance's
job from data alone.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import jax, jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from astro.likelihoods_des_y1 import _UNIF, _NORM, _theory_and_data


# Wide-Gaussian replacement for Uniforms (matches likelihoods_des_y1_phys)
_UNIF_LO = jnp.asarray([lo for _, lo, _ in _UNIF])
_UNIF_HI = jnp.asarray([hi for _, _, hi in _UNIF])
_UNIF_MU = 0.5 * (_UNIF_LO + _UNIF_HI)
_UNIF_SD = (_UNIF_HI - _UNIF_LO) / 4.0


def make_phys_model(theory_fn, data_mean, cov_chol):
    param_names = [n for n, _, _ in _UNIF] + [n for n, _, _ in _NORM]

    def model():
        phys_u = []
        for i, (name, lo, hi) in enumerate(_UNIF):
            phys_u.append(numpyro.sample(name,
                dist.Normal(_UNIF_MU[i], _UNIF_SD[i])))
        phys_n = []
        for name, mu, sd in _NORM:
            phys_n.append(numpyro.sample(name, dist.Normal(mu, sd)))
        phys = jnp.stack(phys_u + phys_n)
        cl = theory_fn(phys)
        numpyro.sample("obs", dist.MultivariateNormal(cl, scale_tril=cov_chol),
                       obs=data_mean)

    return model, param_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", choices=["dense", "diag"], required=True)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--n-chains",     type=int, default=4)
    ap.add_argument("--warmup",       type=int, default=500)
    ap.add_argument("--num-samples",  type=int, default=1000)
    ap.add_argument("--max-tree-depth", type=int, default=7)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    numpyro.set_host_device_count(args.n_chains)

    print("[numpyro-des-phys] Building forward model + fiducial mock data …")
    theory_fn, data_mean, cov, fid_phys = _theory_and_data()
    cov_np = np.asarray(cov, dtype=np.float64)
    cov_np = 0.5 * (cov_np + cov_np.T) + 1e-10 * np.eye(cov_np.shape[0])
    cov_chol = jnp.asarray(np.linalg.cholesky(cov_np))

    model, param_names = make_phys_model(theory_fn, data_mean, cov_chol)

    dense = True if args.mass == "dense" else False

    kernel = NUTS(model,
                  dense_mass=dense,
                  max_tree_depth=args.max_tree_depth,
                  target_accept_prob=args.target_accept,
                  init_strategy=numpyro.infer.init_to_median)

    mcmc = MCMC(kernel,
                num_warmup=args.warmup,
                num_samples=args.num_samples,
                num_chains=args.n_chains,
                chain_method="sequential",
                progress_bar=False)

    print(f"[numpyro-des-phys] NUTS mass={args.mass}  chains={args.n_chains}  "
          f"warmup={args.warmup}  samples={args.num_samples}")
    t0 = time.perf_counter()
    mcmc.run(jax.random.key(args.seed),
             extra_fields=("num_steps", "mean_accept_prob", "diverging"))
    wall = time.perf_counter() - t0

    samples_dict = mcmc.get_samples(group_by_chain=True)
    phys_samples = [samples_dict[name] for name in param_names]
    samples = jnp.stack(phys_samples, axis=-1)
    samples = jnp.transpose(samples, (1, 0, 2))

    extra = mcmc.get_extra_fields()
    accept = float(np.mean(extra.get("mean_accept_prob", float("nan"))))
    n_steps = extra.get("num_steps", None)
    if n_steps is not None:
        n_grad_prod = int(np.sum(n_steps))
        mean_num_steps = float(np.mean(n_steps))
    else:
        n_grad_prod = args.n_chains * args.num_samples * (2 ** args.max_tree_depth)
        mean_num_steps = float("nan")
    n_grad_warm = args.n_chains * args.warmup * (2 ** args.max_tree_depth)
    n_grad = n_grad_prod + n_grad_warm

    try:
        step_size = float(np.asarray(mcmc.last_state.adapt_state.step_size).mean())
    except Exception:
        step_size = float("nan")
    n_div = int(np.sum(extra.get("diverging", 0)))

    from affine_invariant_samplers import effective_sample_size
    ess = np.asarray(effective_sample_size(np.asarray(samples)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "samples.npz",
                        samples=np.asarray(samples, dtype=np.float32))
    summary = dict(problem="des_y1_phys",
                   sampler=f"nuts_numpyro_{args.mass}",
                   dim=len(param_names), n_data=int(data_mean.size),
                   n_chains=args.n_chains,
                   warmup=args.warmup, num_samples=args.num_samples,
                   warm_start=0, find_mode=False,
                   wall_s=wall,
                   ess=ess.tolist(), ess_min=float(ess.min()),
                   ess_mean=float(ess.mean()),
                   ess_per_sec=float(ess.min() / wall),
                   n_target_evals=int(n_grad),
                   eval_kind="grad",
                   ess_per_eval=float(ess.min() / n_grad),
                   fake_ess_warning=None,
                   info=dict(acceptance_rate=accept,
                             max_tree_depth=args.max_tree_depth,
                             target_accept=args.target_accept,
                             final_step_size=step_size,
                             mean_num_steps=mean_num_steps,
                             n_divergences=n_div))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "ess"}, indent=2))


if __name__ == "__main__":
    main()
