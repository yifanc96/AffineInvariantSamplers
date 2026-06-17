"""Fair cross-sampler ESS comparison using multiple ESS conventions.

Computes four ESS conventions on the same saved samples so ratios between
samplers can be cross-validated.  All four use Sokal's self-consistent-window
ACF estimator under the hood; they differ in when/how chains are combined.

  A — emcee-style: per-chain τ, averaged across chains, ESS = N·C / τ̄
      (our repo's default; upper-bounded because walkers are treated as
       independent chains even if they share the same ensemble move)
  B — chain-mean-first: x̄(t) = mean_c x_c(t), ESS = N / τ_Sokal(x̄)
      (honest for ensemble samplers; exposes walker-shared correlations)
  C — pooled: ACF on the concatenated N·C-long sequence, ESS = N·C / τ
      (penalises between-chain disagreement; arviz-adjacent)
  D — arviz rank-normalised split-chain ESS (if arviz is installed)
      (rank-robust to heavy tails; Vehtari+ 2021 standard)

Usage::

    python -m astro.ess_comparison                # all DES Y1 samplers
    python -m astro.ess_comparison --problem hogg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from affine_invariant_samplers.diagnostics import (
    integrated_autocorr_time_1d, autocorrelation_fft)


# ─── The four ESS conventions ─────────────────────────────────────────────────

def ess_A_per_chain_avg(x):
    """(N, C, D) → (D,).  Per-chain τ averaged, ESS = N·C / max(τ̄, 1)."""
    x = np.asarray(x, dtype=np.float64)
    N, C, D = x.shape
    taus = np.zeros((C, D))
    for c in range(C):
        for d in range(D):
            taus[c, d], _, _ = integrated_autocorr_time_1d(x[:, c, d])
    tau_avg = taus.mean(axis=0)
    return N * C / np.maximum(tau_avg, 1.0)


def ess_B_chain_mean_first(x):
    """Average across chains first, THEN ACF.

    This is the fair convention for ensemble samplers: x̄(t) = mean over
    walkers of x(t).  If walkers are truly independent, τ(x̄) ≈ τ_single/√C;
    if walkers move in lockstep (correlated ensemble), τ(x̄) ≈ τ_single and
    no chain-count bonus — revealing the true ensemble mixing rate.

    ESS here is comparable ACROSS different chain counts (our 64 vs NumPyro's 4).
    """
    x = np.asarray(x, dtype=np.float64)
    N, C, D = x.shape
    xbar = x.mean(axis=1)                 # (N, D)
    tau = np.zeros(D)
    for d in range(D):
        tau[d], _, _ = integrated_autocorr_time_1d(xbar[:, d])
    return N / np.maximum(tau, 1.0)


def ess_C_pooled(x):
    """Concatenate chains into one N·C flat sequence, compute ACF on that."""
    x = np.asarray(x, dtype=np.float64)
    N, C, D = x.shape
    ess = np.zeros(D)
    for d in range(D):
        flat = x[:, :, d].reshape(-1, order="F")       # chain-major
        tau, _, _ = integrated_autocorr_time_1d(flat)
        ess[d] = (N * C) / max(tau, 1.0)
    return ess


def ess_D_arviz(x):
    """arviz rank-normalised split-chain ESS (if arviz is installed)."""
    try:
        import arviz as az
    except ImportError:
        return None
    x = np.asarray(x, dtype=np.float64).transpose(1, 0, 2)   # (C, N, D)
    dset = az.convert_to_dataset({"x": x})
    ess = az.ess(dset, method="bulk").to_array().values      # (1, D)
    return ess.flatten()


# ─── Drive ────────────────────────────────────────────────────────────────────

def _analyse_one(npz_path: Path):
    arr = np.load(npz_path)["samples"]
    if arr.ndim != 3:
        return None
    # burn 10%
    arr = arr[arr.shape[0] // 10:]
    return dict(
        shape=arr.shape,
        A_per_chain_avg   = ess_A_per_chain_avg(arr),
        B_chain_mean_first = ess_B_chain_mean_first(arr),
        C_pooled           = ess_C_pooled(arr),
        D_arviz            = ess_D_arviz(arr),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="des_y1")
    ap.add_argument("--results-dir", default="astro/results")
    ap.add_argument("--per-grad", action="store_true",
                    help="Divide each ESS by total gradient evaluations "
                         "(read from metrics.json).")
    args = ap.parse_args()

    root = Path(args.results_dir) / args.problem
    cells: dict[str, list] = {}
    for p in sorted(root.glob("*/seed*/samples.npz")):
        sampler = p.parts[-3]
        r = _analyse_one(p)
        if r is None:
            continue
        # Attach metrics.json info (total grads, wall, etc.)
        mj = p.parent / "metrics.json"
        if mj.exists():
            m = json.loads(mj.read_text())
            r["n_grad_evals"] = m.get("n_target_evals") or m.get("info", {}).get("n_grad_evals")
            # For chess / peanuts, post-hoc correction
            info = m.get("info", {}) or {}
            if sampler == "chess" and "nominal_L" in info:
                L = info["nominal_L"]
                iters = (m["num_samples"] + m["warmup"]) * m["n_chains"]
                r["n_grad_evals"] = iters * int(L)
            elif sampler == "peanuts" and "mean_tree_depth" in info:
                mtd = info["mean_tree_depth"]
                iters = (m["num_samples"] + m["warmup"]) * m["n_chains"]
                r["n_grad_evals"] = int(iters * (2 ** mtd))
            r["wall_s"]       = m.get("wall_s")
            r["n_chains"]     = m["n_chains"]
        cells.setdefault(sampler, []).append(r)

    if not cells:
        print(f"no samples under {root}")
        return

    def min_med(results, key):
        vals = [np.min(r[key]) for r in results if r.get(key) is not None]
        return float(np.median(vals)) if vals else float("nan")

    # Four-way per-grad comparison: A_prod, A_total, B_prod, B_total
    # plus the raw ESS under A and B for reference.
    print(f"\nDES Y1 — ess/grad by convention and denominator (median across seeds)\n")
    print(f"{'sampler':<22s} {'(N,C,D)':<14s} "
          f"{'ess_A':>10s} {'ess_B':>10s} "
          f"{'A/grad_prod':>12s} {'A/grad_tot':>12s} "
          f"{'B/grad_prod':>12s} {'B/grad_tot':>12s}")
    print('-' * 120)
    N_prod_default = 1000
    N_warm_default = 500
    for s in sorted(cells):
        rs = cells[s]
        shape = "×".join(str(i) for i in rs[0]["shape"])
        # Per-seed: compute A, B raw; compute the 4 per-grad numbers
        per_seed = {k: [] for k in ("A", "B",
                                      "A_prod", "A_tot", "B_prod", "B_tot")}
        for r in rs:
            if r.get("A_per_chain_avg") is None: continue
            essA = float(np.min(r["A_per_chain_avg"]))
            essB = float(np.min(r["B_chain_mean_first"])) \
                   if r.get("B_chain_mean_first") is not None else float("nan")
            n_tot = r.get("n_grad_evals")
            if n_tot is None: continue
            # Production-only: subtract warmup grads proportionally.
            # For our runs (N_prod=1000, N_warm=500), prod/(prod+warm) = 2/3.
            # This assumes per-iter cost is roughly constant warmup vs prod,
            # true for all HMC-family samplers here.
            n_prod = n_tot * N_prod_default / (N_prod_default + N_warm_default)
            n_ch = r.get("n_chains", 1)
            per_seed["A"].append(essA)
            per_seed["B"].append(essB)
            per_seed["A_prod"].append(essA / n_prod)
            per_seed["A_tot"].append(essA / n_tot)
            # B: multiply by n_chains (per-walker normalisation) before dividing
            per_seed["B_prod"].append(essB * n_ch / n_prod if essB == essB else float("nan"))
            per_seed["B_tot"].append(essB * n_ch / n_tot if essB == essB else float("nan"))

        med = {k: float(np.median(v)) if v else float("nan")
               for k, v in per_seed.items()}
        def fg(v, w=12):
            if not np.isfinite(v): return f"{'—':>{w}s}"
            return f"{v:>{w}.2e}"
        def fe(v, w=10):
            if not np.isfinite(v): return f"{'—':>{w}s}"
            if abs(v) >= 1e4: return f"{v:>{w}.0f}"
            return f"{v:>{w}.1f}"
        print(f"{s:<22s} {shape:<14s} "
              f"{fe(med['A'])} {fe(med['B'])} "
              f"{fg(med['A_prod'])} {fg(med['A_tot'])} "
              f"{fg(med['B_prod'])} {fg(med['B_tot'])}")


if __name__ == "__main__":
    main()
