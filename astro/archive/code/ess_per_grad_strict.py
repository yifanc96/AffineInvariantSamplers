"""Strict ess/grad leaderboard — no burnin discard, exact n_grad_evals from
metrics.json, same samples used everywhere.

Four numbers per sampler:
    A/prod  = ess_A(all_samples) / (n_grad_evals × N_prod / N_total_iter)
    A/tot   = ess_A(all_samples) / n_grad_evals
    B/prod  = ess_B(all_samples) × n_chains / (n_grad_evals × N_prod / N_total_iter)
    B/tot   = ess_B(all_samples) × n_chains / n_grad_evals

Convention A: per-chain Sokal τ averaged, ess_A = N·C / τ̄
Convention B: chain-mean-first, ess_B = N / τ(ensemble-mean trajectory)

prod denominator accounts only for production-iteration gradient evaluations;
tot includes warmup iterations as well.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from affine_invariant_samplers.diagnostics import integrated_autocorr_time_1d


def ess_A(x):
    """(N, C, D) → (D,).  Per-chain τ averaged, ESS = N·C / max(τ̄, 1)."""
    N, C, D = x.shape
    taus = np.zeros((C, D))
    for c in range(C):
        for d in range(D):
            taus[c, d], _, _ = integrated_autocorr_time_1d(x[:, c, d])
    tau_avg = taus.mean(axis=0)
    return N * C / np.maximum(tau_avg, 1.0)


def ess_B(x):
    """Chain-mean-first: x̄(t) = mean_c x_c(t), then ESS = N / max(τ(x̄), 1)."""
    N, C, D = x.shape
    xbar = x.mean(axis=1)
    tau = np.zeros(D)
    for d in range(D):
        tau[d], _, _ = integrated_autocorr_time_1d(xbar[:, d])
    return N / np.maximum(tau, 1.0)


def main():
    root = Path("astro/results/des_y1")
    rows = []
    for sampler_dir in sorted(root.iterdir()):
        if not sampler_dir.is_dir(): continue
        sampler = sampler_dir.name
        per_seed_A = []; per_seed_B = []; per_seed_Atot = []; per_seed_Aprod = []
        per_seed_Btot = []; per_seed_Bprod = []
        for p in sorted(sampler_dir.glob("seed*/samples.npz")):
            seed_dir = p.parent
            mj = json.loads((seed_dir / "metrics.json").read_text())
            arr = np.load(p)["samples"].astype(np.float64)      # (N, C, D)
            N, C, D = arr.shape
            n_grad_tot = mj.get("n_target_evals") or mj["info"].get("n_grad_evals") or 0
            n_warm = mj["warmup"]
            n_prod = mj["num_samples"]
            # prod-only grads = tot × (n_prod / (n_prod + n_warm))
            n_grad_prod = n_grad_tot * n_prod / (n_prod + n_warm) if n_grad_tot else 0
            # Fix for chess/peanuts where n_grad_tot in metrics is wrong
            info = mj.get("info", {})
            if sampler in ("chess", "chess_tuned") and "nominal_L" in info and n_grad_tot < 1e6:
                L = info["nominal_L"]
                n_grad_tot = int((n_prod + n_warm) * C * L)
                n_grad_prod = int(n_prod * C * L)
            if sampler == "peanuts" and "mean_tree_depth" in info and n_grad_tot < 1e6:
                mtd = info["mean_tree_depth"]
                n_grad_tot = int((n_prod + n_warm) * C * (2 ** mtd))
                n_grad_prod = int(n_prod * C * (2 ** mtd))

            A = float(np.min(ess_A(arr)))
            B = float(np.min(ess_B(arr)))
            per_seed_A.append(A); per_seed_B.append(B)
            if n_grad_tot:
                per_seed_Atot.append(A / n_grad_tot)
                per_seed_Aprod.append(A / n_grad_prod)
                per_seed_Btot.append(B * C / n_grad_tot)
                per_seed_Bprod.append(B * C / n_grad_prod)
        if not per_seed_A: continue
        med = np.median
        rows.append((
            sampler, len(per_seed_A),
            med(per_seed_A), med(per_seed_B),
            med(per_seed_Aprod) if per_seed_Aprod else float("nan"),
            med(per_seed_Atot)  if per_seed_Atot  else float("nan"),
            med(per_seed_Bprod) if per_seed_Bprod else float("nan"),
            med(per_seed_Btot)  if per_seed_Btot  else float("nan"),
        ))

    print(f"{'sampler':<22s} {'seeds':>5s} {'ess_A':>9s} {'ess_B':>7s}  "
          f"{'A/prod':>10s} {'A/tot':>10s} {'B/prod':>10s} {'B/tot':>10s}")
    print('-' * 94)
    def f(v, w=10, precision=2):
        if not np.isfinite(v): return f"{'—':>{w}s}"
        return f"{v:>{w}.{precision}e}"
    def fe(v, w=9):
        if not np.isfinite(v): return f"{'—':>{w}s}"
        if abs(v) >= 1e4: return f"{v:>{w}.0f}"
        if abs(v) < 1: return f"{v:>{w}.2f}"
        return f"{v:>{w}.1f}"
    for r in rows:
        print(f"{r[0]:<22s} {r[1]:>5d} {fe(r[2])} {fe(r[3], 7)}  "
              f"{f(r[4])} {f(r[5])} {f(r[6])} {f(r[7])}")


if __name__ == "__main__":
    main()
