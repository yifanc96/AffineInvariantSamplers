"""Pickles runner — supports both des_y1 (z-space) and des_y1_phys (no reparam)."""
import argparse, json, sys, time, os
sys.path.insert(0, "/home/yifanchen/research/AffineInvariantSamplers")
import jax, jax.numpy as jnp, numpy as np
# Persistent compilation cache so repeated runs don't pay the 25-min jax-cosmo
# JIT compile each time.
_JAX_CACHE = os.environ.get("JAX_PERSIST_CACHE",
                             "/home/yifanchen/.cache/jax_pickles_des")
os.makedirs(_JAX_CACHE, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _JAX_CACHE)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
from pathlib import Path
from affine_invariant_samplers import sampler_pickles, sampler_stretch, effective_sample_size

ap = argparse.ArgumentParser()
ap.add_argument("--problem", choices=["des_y1", "des_y1_phys"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--L", type=int, default=20)
ap.add_argument("--max-L", type=int, default=20)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--step-size", type=float, default=0.01)
ap.add_argument("--init-scale", type=float, required=True)
ap.add_argument("--init-mode", choices=["fid_ball", "prior_median"], default="fid_ball",
                help="fid_ball: Gaussian ball around fiducial truth (oracle warm-start). "
                     "prior_median: Gaussian ball at z=0 — matches numpyro NUTS init_to_median "
                     "center for fair comparison.")
ap.add_argument("--n-chains", type=int, default=64)
ap.add_argument("--num-samples", type=int, default=1000)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--warm-start", type=int, default=0,
                help="Stretch-move pre-warmup iterations (gradient-free). "
                     "Final stretch positions become the gradient sampler's "
                     "initial state.")
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.problem == "des_y1_phys":
    from astro.likelihoods_des_y1_phys import make_des_y1_phys as _make
    from astro.likelihoods_des_y1_phys import init_des_y1_phys as _init
    lp, _ = _make()
    init = _init(jax.random.key(args.seed + 1), args.n_chains, init_scale=args.init_scale)
else:
    from astro.likelihoods_des_y1 import make_des_y1 as _make
    from astro.likelihoods_des_y1 import init_des_y1 as _init
    lp, _ = _make()
    init = _init(jax.random.key(args.seed + 1), args.n_chains,
                 init_scale=args.init_scale, init_mode=args.init_mode)

# Optional stretch pre-warmup: disperses walkers from Gaussian ball to typical set.
if args.warm_start > 0:
    print(f"[{args.problem}] stretch pre-warmup ({args.warm_start} iters)...", flush=True)
    samples_stretch, _ = sampler_stretch(lp, init, num_samples=args.warm_start,
                                          warmup=0, seed=args.seed, verbose=False)
    samples_stretch.block_until_ready()
    init = samples_stretch[-1]    # final stretch state, shape (n_chains, dim)

print(f"[{args.problem}] pickles L_init={args.L} max_L={args.max_L} γ={args.gamma} "
      f"init={args.init_scale}({args.init_mode}) N={args.num_samples} warm={args.warmup} "
      f"ws={args.warm_start}", flush=True)

t0 = time.perf_counter()
samples, info = sampler_pickles(lp, init,
                                num_samples=args.num_samples, warmup=args.warmup,
                                step_size=args.step_size, L=args.L,
                                max_L=args.max_L, gamma=args.gamma,
                                seed=args.seed, verbose=False)
samples.block_until_ready()
wall = time.perf_counter() - t0
ess = np.asarray(effective_sample_size(samples))
mean_L = float(info.get("mean_L", args.L))
# Production-only n_grad (sampler-native count is total_L * n_chains)
n_grad_prod = int(args.num_samples * args.n_chains * mean_L)
n_grad_tot  = int((args.num_samples + args.warmup) * args.n_chains * mean_L)
print(f"wall={wall:.0f}s mean_L={mean_L:.1f} ess_min={ess.min():.0f} "
      f"ess/grad_prod={ess.min()/n_grad_prod:.3e} ess/grad_tot={ess.min()/n_grad_tot:.3e} "
      f"acc={info['acceptance_rate']:.3f}")
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "samples.npz", samples=np.asarray(samples, dtype=np.float32))
(out / "metrics.json").write_text(json.dumps(dict(
    problem=args.problem, sampler="pickles",
    dim=int(samples.shape[-1]), n_data=2250,
    n_chains=args.n_chains, warmup=args.warmup, num_samples=args.num_samples,
    warm_start=args.warm_start, wall_s=wall,
    ess=ess.tolist(), ess_min=float(ess.min()), ess_mean=float(ess.mean()),
    ess_per_sec=float(ess.min()/wall),
    n_target_evals=n_grad_prod, eval_kind="grad",
    ess_per_eval=float(ess.min()/n_grad_prod),
    fake_ess_warning=None,
    info={**{k: float(v) for k,v in info.items() if isinstance(v,(int,float,np.floating,np.integer))},
          "L_setting": args.L, "max_L_setting": args.max_L,
          "gamma_setting": args.gamma, "init_scale": args.init_scale,
          "init_mode": args.init_mode,
          "n_grad_evals": n_grad_prod,
          "n_grad_evals_total": n_grad_tot}), indent=2))
