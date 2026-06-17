"""Ensemble-preconditioning ablation on LJ-{13,55} with the pita target.

Three A/B pairs — integrator + adaptation identical, only the mass matrix
(identity vs ensemble-covariance) differs:

    sampler_chess    HMC + ChEES,          identity M    ← no preconditioner
    sampler_peaches  HMC + ChEES,          ensemble M    ← ensemble preconditioner

    sampler_malt     kinetic Langevin,     identity M    ← no preconditioner
    sampler_pickles  kinetic Langevin,     ensemble M    ← ensemble preconditioner

    baoab            BAOAB kinetic Langevin, identity M  ← no preconditioner
    pickles_unadjusted  BAOAB,              ensemble M   ← ensemble preconditioner

(``plain ULA`` — overdamped, identity M — is also available as an extra
no-ensemble reference via ``--samplers ula``.)

All runs use the same soft-core LJ (pita + smooth below r=0.65) so the
comparison isolates preconditioning from numerical stability."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np
from affine_invariant_samplers import (
    sampler_pickles_unadjusted, sampler_peaches, sampler_chess,
    sampler_pickles, sampler_malt, sampler_peams,
)

from MD.energies import _pairwise_distances, lj_pita_energy, _lj_pair_smooth
from MD.metrics import energy_w2, distance_w2
from MD.ground_truth import load, pick_init, FILES

ROOT = Path(__file__).resolve().parent


def make_lp(n_particles, tau=1.0, smooth=True, r_cut=0.65):
    """Return (lp_batched, lp_single).  Samplers that take (D,)→scalar use
    lp_single; those that take (B,D)→(B,) use lp_batched."""
    def lp_batched(x):
        coords = x.reshape(x.shape[0], n_particles, 3)
        d = _pairwise_distances(coords)
        lj = _lj_pair_smooth(d, r_cut=r_cut) if smooth else \
             ((1 / d) ** 12 - 2.0 * (1 / d) ** 6)
        E_lj = 2.0 * lj.sum(-1) / tau
        com = coords.mean(-2, keepdims=True)
        E_osc = 0.5 * jnp.sum((coords - com) ** 2, axis=(-2, -1)) / tau
        return -(E_lj + E_osc)

    def lp_single(x):                                        # (D,) → scalar
        return lp_batched(x[None])[0]
    return lp_batched, lp_single


def plain_ula(lp, init, num_samples, warmup, step_size, seed=0):
    """Plain unadjusted Langevin (first-order, no MH, no ensemble coupling)."""
    g = jax.grad(lambda q: lp(q[None])[0])
    grad_batched = jax.jit(jax.vmap(g))

    @jax.jit
    def step(x, key):
        noise = jax.random.normal(key, x.shape)
        return x + step_size * grad_batched(x) + jnp.sqrt(2 * step_size) * noise

    key = jax.random.key(seed)
    x = jnp.asarray(init)
    for _ in range(warmup):
        key, k = jax.random.split(key); x = step(x, k)
    samples = []
    for _ in range(num_samples):
        key, k = jax.random.split(key); x = step(x, k); samples.append(x)
    arr = jnp.stack(samples, axis=0); arr.block_until_ready()
    return arr


def plain_baoab(lp, init, num_samples, warmup, step_size, gamma=2.0, seed=0):
    """Plain BAOAB kinetic Langevin (same integrator family as pickles_unadj
    but with identity mass, per-chain independent).  The direct no-ensemble
    counterpart to ``sampler_pickles_unadjusted``."""
    g = jax.grad(lambda q: lp(q[None])[0])
    grad_batched = jax.jit(jax.vmap(g))
    alpha = jnp.exp(-gamma * step_size)
    sigma = jnp.sqrt(1.0 - alpha ** 2)
    half_h = step_size / 2.0

    @jax.jit
    def step(state, key):
        x, p = state
        # B: half-kick using grad at x
        grads = grad_batched(x)
        p = p + half_h * grads
        # A: half-drift
        x = x + half_h * p
        # O: full OU
        noise = jax.random.normal(key, x.shape)
        p = alpha * p + sigma * noise
        # A: half-drift
        x = x + half_h * p
        # B: half-kick using grad at new x
        grads = grad_batched(x)
        p = p + half_h * grads
        return x, p

    key = jax.random.key(seed)
    x = jnp.asarray(init)
    p = jax.random.normal(jax.random.key(seed + 1), x.shape)
    state = (x, p)
    for _ in range(warmup):
        key, k = jax.random.split(key)
        state = step(state, k)
    samples = []
    for _ in range(num_samples):
        key, k = jax.random.split(key)
        state = step(state, k)
        samples.append(state[0])
    arr = jnp.stack(samples, axis=0); arr.block_until_ready()
    return arr




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="lj13", choices=["lj13", "lj55"])
    ap.add_argument("--n-chains", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--n-samples", type=int, default=50_000)
    ap.add_argument("--step-size", type=float, default=0.002)
    ap.add_argument("--ula-step-size", type=float, default=0.001,
                    help="separate for plain ULA since no preconditioner; "
                         "smaller for stability")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--init-jitter", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-thin-by", type=int, default=10,
                    help="thin full-coord save by this factor; energy "
                         "time series always saved at full rate")
    ap.add_argument("--samplers", nargs="+",
                    default=["baoab", "pickles_unadjusted",
                             "malt", "pickles",
                             "chess", "peaches", "peams"])
    args = ap.parse_args()

    n = FILES[args.problem]["n_particles"]
    lp, lp_single = make_lp(n, tau=args.tau, smooth=True)

    init = pick_init(args.problem, args.n_chains, seed=args.seed,
                      split="pita_test")
    if args.init_jitter > 0:
        k = jax.random.key(args.seed + 1)
        init = init + args.init_jitter * jax.random.normal(k, init.shape)
    init = jnp.asarray(init)

    gt = load(args.problem, "pita_test")
    rng = np.random.default_rng(args.seed)
    gt_sub = gt[rng.choice(gt.shape[0], size=min(10_000, gt.shape[0]),
                            replace=False)]
    gt_E = np.asarray(lj_pita_energy(jnp.asarray(gt_sub), n))

    results = {}
    for sampler_name in args.samplers:
        print(f"\n=== {sampler_name} ===", flush=True)
        t0 = time.time()
        try:
            if sampler_name == "ula":
                samples = plain_ula(lp, init, args.n_samples, args.warmup,
                                     step_size=args.ula_step_size, seed=args.seed)
                info = {}
            elif sampler_name == "baoab":
                samples = plain_baoab(lp, init, args.n_samples, args.warmup,
                                        step_size=args.ula_step_size, gamma=2.0,
                                        seed=args.seed)
                info = {}
            elif sampler_name == "pickles_unadjusted":
                samples, info = sampler_pickles_unadjusted(
                    lp, init, args.n_samples, warmup=args.warmup,
                    step_size=args.ula_step_size, gamma=2.0,
                    adapt=None, seed=args.seed, verbose=False)
            elif sampler_name == "malt":
                samples, info = sampler_malt(
                    lp_single, init, args.n_samples, warmup=args.warmup,
                    step_size=args.step_size, L=15, gamma=1.0,
                    seed=args.seed, verbose=False)
            elif sampler_name == "pickles":
                samples, info = sampler_pickles(
                    lp, init, args.n_samples, warmup=args.warmup,
                    step_size=args.step_size, L=15, gamma=1.0,
                    adapt_L=False,
                    seed=args.seed, verbose=False)
            elif sampler_name == "chess":
                samples, info = sampler_chess(
                    lp, init, args.n_samples, warmup=args.warmup,
                    step_size=args.step_size,
                    seed=args.seed, verbose=False)
            elif sampler_name == "peaches":
                samples, info = sampler_peaches(
                    lp, init, args.n_samples, warmup=args.warmup,
                    step_size=args.step_size,
                    seed=args.seed, verbose=False)
            elif sampler_name == "peams":
                samples, info = sampler_peams(
                    lp, init, args.n_samples, warmup=args.warmup,
                    step_size=args.step_size,
                    seed=args.seed, verbose=False)
            samples.block_until_ready()
            elapsed = time.time() - t0

            arr = np.asarray(samples)                                   # (T, N, D)
            T_save, N_save, D_save = arr.shape

            # Full energy time series (T, N) — small, always save
            E_all = np.asarray(
                lj_pita_energy(jnp.asarray(arr.reshape(-1, D_save)), n)
            ).reshape(T_save, N_save).astype(np.float32)

            # Thinned coords for later reanalysis
            thin = max(1, args.save_thin_by)
            seed_out = (ROOT / "results" /
                        f"{args.problem}_ablation_seed{args.seed}" /
                        sampler_name)
            seed_out.mkdir(parents=True, exist_ok=True)
            np.save(seed_out / "energy.npy", E_all)
            thinned = arr[::thin].astype(np.float32)
            np.save(seed_out / f"samples_thin{thin}.npy", thinned)

            flat = arr.reshape(-1, D_save)
            sub = flat[rng.choice(flat.shape[0],
                                  size=min(20_000, flat.shape[0]), replace=False)]
            sub_E = np.asarray(lj_pita_energy(jnp.asarray(sub), n))
            finite = np.isfinite(sub_E)
            sub_E_f = sub_E[finite]; sub_f = sub[finite]

            if sub_f.shape[0] == 0:
                ew2 = float("nan"); dw2 = float("nan")
                E_mean = float("nan"); E_std = float("nan")
            else:
                ew2 = float(energy_w2(sub_E_f, gt_E))
                dw2 = float(distance_w2(sub_f, gt_sub, n, 3))
                E_mean = float(sub_E_f.mean()); E_std = float(sub_E_f.std())
            entry = dict(
                sampler=sampler_name,
                E_mean=E_mean, E_std=E_std,
                energy_W2=ew2, distance_W2=dw2,
                finite_frac=float(finite.mean()),
                acceptance_rate=float(info["acceptance_rate"])
                                 if "acceptance_rate" in info else None,
                final_step_size=float(info["final_step_size"])
                                 if "final_step_size" in info else None,
                elapsed_sec=elapsed,
                saved_dir=str(seed_out),
            )
        except Exception as e:
            entry = dict(sampler=sampler_name, error=repr(e),
                          elapsed_sec=time.time() - t0)
            print(f"FAILED: {e}", flush=True)
        print(json.dumps(entry, indent=2), flush=True)
        results[sampler_name] = entry

    out = ROOT / "results" / f"{args.problem}_ablation_ensemble"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(dict(problem=args.problem,
                        gt=dict(E_mean=float(gt_E.mean()), E_std=float(gt_E.std())),
                        results=results), f, indent=2)
    print(f"\nsaved → {out / 'metrics.json'}")

    # Summary
    print(f"\n{'sampler':<25} {'E_mean':>10} {'E W2':>10} {'d W2':>8} {'accept':>8} {'wall':>8}")
    for s, r in results.items():
        if "error" in r:
            print(f"{s:<25} FAILED ({r['error'][:60]})  wall={r['elapsed_sec']:.1f}s")
            continue
        a = f"{r['acceptance_rate']:.3f}" if r['acceptance_rate'] is not None else "—"
        print(f"{s:<25} {r['E_mean']:>10.3f} {r['energy_W2']:>10.3f} "
              f"{r['distance_W2']:>8.4f} {a:>8} {r['elapsed_sec']:>7.1f}s")


if __name__ == "__main__":
    main()
