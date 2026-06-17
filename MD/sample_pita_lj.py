"""JAX / numpyro port of pita's ``sampling/sample_lj13.py``.

Exact same model:
    positions ~ Uniform(-100, 100)^{n_particles * 3}
    log π(x) ∝ − [ 2 · E_LJ(x) + E_osc(x) ] / temperature
    E_LJ(x)  = Σ_{i<j} [ (r_m/r_ij)^{12} − 2·(r_m/r_ij)^6 ]     (ε = r_m = 1)
    E_osc(x) = ½·Σ_i ‖x_i − x̄‖²

Sampler: NUTS (same as pita).  Runs on GPU via JAX; multiple chains in
parallel (pmap over devices or sequential on a single GPU).
"""
from __future__ import annotations

import argparse, time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

ROOT = Path(__file__).resolve().parent


def lj_pair(r, eps=1.0, rm=1.0):
    return eps * ((rm / r) ** 12 - 2.0 * (rm / r) ** 6)


def remove_mean(x):
    return x - jnp.mean(x, axis=0, keepdims=True)


def energy_pita(x_flat, n_particles):
    """Matches pita `energy2`:  2·E_LJ + E_osc  (temperature = 1)."""
    x = x_flat.reshape(n_particles, 3)
    iu, ju = jnp.triu_indices(n_particles, k=1)
    diffs = x[iu] - x[ju]
    d = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + 1e-12)
    lj = lj_pair(d).sum()
    osc = 0.5 * jnp.sum(remove_mean(x) ** 2)
    return 2.0 * lj + osc                                              # pita's `energy2` returns −(2·LJ+osc); here return +energy


def make_model(n_particles: int, temperature: float):
    def model():
        positions = numpyro.sample(
            "positions",
            dist.Uniform(-1e2, 1e2).expand([n_particles * 3]).to_event(1),
        )
        U = energy_pita(positions, n_particles) / temperature
        numpyro.factor("potential_energy", -U)                          # log-weight = −E/τ
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-particles", type=int, default=13)
    ap.add_argument("--num-samples", type=int, default=20_000)
    ap.add_argument("--warmup-steps", type=int, default=2_000)
    ap.add_argument("--num-chains", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", choices=["uniform", "gt"], default="gt",
                    help="pita's script defaults to Uniform(-100,100) init; "
                         "here we support initialising chains from DEM GT to "
                         "skip the long basin-search warmup.  Equilibrium "
                         "under the pita target is reached quickly either way.")
    ap.add_argument("--gt-path", default=str(ROOT / "reference" / "lj13_test.npy"))
    ap.add_argument("--out", default=str(ROOT / "reference" / "lj13_pita_test.npy"))
    args = ap.parse_args()

    numpyro.set_host_device_count(args.num_chains)
    model = make_model(args.n_particles, args.temperature)

    if args.init_from == "gt":
        gt = np.load(args.gt_path)                                      # (N_gt, 39)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(gt.shape[0], size=args.num_chains, replace=False)
        init_positions = gt[idx].astype(np.float64)                     # (K, 39)
        init_params = {"positions": jnp.asarray(init_positions)}
    else:
        init_params = None

    print(f"[pita NUTS LJ-{args.n_particles}]  "
          f"num_chains={args.num_chains}  warmup={args.warmup_steps}  "
          f"samples={args.num_samples}  T={args.temperature}  "
          f"init={args.init_from}", flush=True)

    kernel = NUTS(model, target_accept_prob=0.8)
    mcmc = MCMC(kernel,
                num_warmup=args.warmup_steps,
                num_samples=args.num_samples,
                num_chains=args.num_chains,
                chain_method="vectorized",
                progress_bar=True)
    key = jax.random.key(args.seed)
    t0 = time.time()
    if init_params is not None:
        mcmc.run(key, init_params=init_params)
    else:
        mcmc.run(key)
    elapsed = time.time() - t0
    print(f"sampling done in {elapsed:.1f}s", flush=True)

    samples = np.asarray(mcmc.get_samples(group_by_chain=True)["positions"])
    print(f"samples shape (chain, sample, dim) = {samples.shape}", flush=True)

    flat = samples.reshape(-1, samples.shape[-1])                       # (K·N, 39)
    np.save(args.out, flat.astype(np.float32))
    print(f"saved → {args.out}  shape={flat.shape}", flush=True)

    # Quick diagnostic energies
    E = jax.vmap(lambda x: energy_pita(x, args.n_particles))(jnp.asarray(flat))
    E = np.asarray(E)
    print(f"⟨E_pita⟩ on pita-NUTS samples:  {E.mean():+.3f} ± {E.std():.3f}",
          flush=True)


if __name__ == "__main__":
    main()
