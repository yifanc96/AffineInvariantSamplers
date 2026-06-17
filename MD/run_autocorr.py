"""Autocorrelation function comparison on LJ-13 (pita target, smoothed).

For each working sampler, runs a long trajectory, saves E(x_t) per chain,
and computes the integrated autocorrelation time τ_int and the per-step
decorrelation curve ρ(lag).  Produces a comparison plot.

ESS_per_step ≈ 1 / (2 τ_int + 1)
ESS_per_wall ≈ ESS_per_step / (wall / n_samples)
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from affine_invariant_samplers import (
    sampler_pickles_unadjusted, sampler_peaches, sampler_chess,
    sampler_pickles, sampler_malt,
)
from MD.energies import _pairwise_distances, lj_pita_energy, _lj_pair_smooth
from MD.ground_truth import load, pick_init, FILES
from MD.run_ablation import make_lp, plain_baoab, plain_ula

ROOT = Path(__file__).resolve().parent


def autocorr_1d(x, max_lag=None):
    """Normalised autocorrelation ρ(τ) of a 1-D time series."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    # FFT-based: O(n log n)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]
    return acf[:max_lag + 1]


def tau_int(rho, cutoff=5):
    """Sokal integrated autocorrelation time with the 'early-stopping'
    heuristic: stop summing when lag > cutoff · τ_int."""
    tau = 0.5
    for k in range(1, len(rho)):
        tau += rho[k]
        if k > cutoff * tau:
            break
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="lj13", choices=["lj13", "lj55"])
    ap.add_argument("--n-chains", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--step-size", type=float, default=0.01)
    ap.add_argument("--ula-step-size", type=float, default=0.0005)
    ap.add_argument("--init-jitter", type=float, default=0.0)
    ap.add_argument("--samplers", nargs="+",
                    default=["malt", "pickles", "peaches", "baoab",
                              "pickles_unadjusted"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n = FILES[args.problem]["n_particles"]
    lp, lp_single = make_lp(n, smooth=True)

    init = pick_init(args.problem, args.n_chains, seed=args.seed,
                      split="pita_test")
    if args.init_jitter > 0:
        k = jax.random.key(args.seed + 1)
        init = init + args.init_jitter * jax.random.normal(k, init.shape)
    init = jnp.asarray(init)

    out_dir = ROOT / "results" / f"{args.problem}_autocorr"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for sampler_name in args.samplers:
        print(f"\n=== {sampler_name} ===", flush=True)
        t0 = time.time()
        try:
            if sampler_name == "baoab":
                samples = plain_baoab(lp, init, args.n_samples, args.warmup,
                                        step_size=args.ula_step_size, seed=args.seed)
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
            samples.block_until_ready()
            wall = time.time() - t0

            # Energy time series per chain, shape (T, N)
            arr = np.asarray(samples)                            # (T, N, D)
            T, N, D = arr.shape
            E = np.asarray(lj_pita_energy(
                jnp.asarray(arr.reshape(-1, D)), n)).reshape(T, N)

            # drop non-finite and clip outliers
            finite_mask = np.isfinite(E).all(axis=0)              # chains with all finite E
            E = E[:, finite_mask]
            if E.shape[1] == 0:
                print(f"  ALL CHAINS NAN — skipping")
                continue

            # Autocorrelation per chain, then average
            max_lag = min(T // 4, 2000)
            rhos = np.stack([autocorr_1d(E[:, c], max_lag=max_lag)
                             for c in range(E.shape[1])], axis=0)
            rho_mean = rhos.mean(axis=0)
            taus = np.array([tau_int(rhos[c]) for c in range(E.shape[1])])
            tau_med = float(np.median(taus))

            results[sampler_name] = dict(
                E_mean=float(E.mean()), E_std=float(E.std()),
                tau_int_median=tau_med,
                tau_int_mean=float(taus.mean()),
                n_finite_chains=int(E.shape[1]),
                wall_sec=float(wall),
                steps_per_sec=float(args.n_samples / wall),
                effective_samples_per_sec=float(args.n_samples / wall / (2 * tau_med + 1)),
                acceptance_rate=float(info["acceptance_rate"])
                                  if "acceptance_rate" in info else None,
                rho=rho_mean.tolist()[:max_lag + 1],
            )
            print(f"  τ_int (median): {tau_med:.1f} steps")
            print(f"  wall: {wall:.1f}s  | steps/s: {args.n_samples/wall:.1f}  |  "
                  f"eff samples/s: {results[sampler_name]['effective_samples_per_sec']:.2f}")

        except Exception as e:
            print(f"  FAILED: {e!r}")
            results[sampler_name] = dict(error=repr(e))

    with open(out_dir / "autocorr.json", "w") as f:
        json.dump(dict(problem=args.problem,
                        n_samples=args.n_samples,
                        n_chains=args.n_chains,
                        results=results), f, indent=2)

    # Plot ρ(lag) for each sampler
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, r in results.items():
        if "rho" not in r:
            continue
        rho = np.asarray(r["rho"])
        ax.plot(rho, label=f"{s}  (τ={r['tau_int_median']:.1f}, "
                           f"{r['effective_samples_per_sec']:.1f} eff/s)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(1/np.e, color="gray", linestyle="--", linewidth=0.5,
               label="1/e")
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel(r"$\rho(k)$ — energy autocorrelation")
    ax.set_xscale("symlog", linthresh=1)
    ax.legend(fontsize=8)
    ax.set_title(f"{args.problem.upper()} — energy autocorrelation by sampler "
                  f"(n_chains={args.n_chains}, n_samples={args.n_samples})")
    fig.tight_layout()
    fig.savefig(out_dir / "autocorr.png", dpi=140)

    # Table
    print(f"\n{'sampler':<22} {'τ_int':>8} {'wall(s)':>10} {'steps/s':>10} {'eff/s':>10}")
    for s, r in results.items():
        if "error" in r:
            print(f"{s:<22}  FAILED")
            continue
        print(f"{s:<22} {r['tau_int_median']:>8.1f} "
              f"{r['wall_sec']:>10.1f} {r['steps_per_sec']:>10.1f} "
              f"{r['effective_samples_per_sec']:>10.2f}")
    print(f"\nsaved → {out_dir / 'autocorr.json'}")
    print(f"plot  → {out_dir / 'autocorr.png'}")


if __name__ == "__main__":
    main()
