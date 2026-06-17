"""Print a consolidated leaderboard from `astro/results/**/metrics.json`.

For each (problem, sampler) cell, reports the median over seeds of:
  - ess_min, ess_mean       (raw effective sample size)
  - ess/sec                 (efficiency per wall-clock)
  - ess/grad or ess/density (efficiency per target-density call;
                             grad for HMC-family, density for stretch/emcee)
  - acceptance              (sanity)
  - #seeds with fake_ess_warning

Usage:
    python -m astro.leaderboard
    python -m astro.leaderboard --csv astro/results/sweep_summary.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

from astro.convergence import analyze as rhat_analyze


def _gather_from_tree(root: Path):
    rows = []
    for p in root.glob("*/*/seed*/metrics.json"):
        rows.append(json.loads(p.read_text()))
    # Fall back to the single-shot layout for problems without seed subdirs.
    for p in root.glob("*/*/metrics.json"):
        if "seed" not in p.parent.name:
            rows.append(json.loads(p.read_text()))
    return rows


def _gather_from_csv(path: Path):
    rows = []
    for r in csv.DictReader(open(path)):
        # Coerce numeric fields
        out = dict(r)
        for k in ("ess_min", "ess_mean", "ess_per_sec", "ess_per_eval",
                  "acceptance_rate", "final_step_size", "n_target_evals"):
            if r.get(k):
                try:
                    out[k] = float(r[k])
                except ValueError:
                    pass
        out["info"] = {"acceptance_rate": out.get("acceptance_rate", 0)}
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="astro/results")
    ap.add_argument("--csv", default=None,
                    help="If set, read the sweep CSV instead of walking the tree.")
    args = ap.parse_args()

    if args.csv:
        rows = _gather_from_csv(Path(args.csv))
    else:
        rows = _gather_from_tree(Path(args.results_dir))
    if not rows:
        print(f"no results under {args.csv or args.results_dir}")
        return

    data = collections.defaultdict(list)
    for r in rows:
        data[(r["problem"], r["sampler"])].append(r)

    def med(xs):
        xs = [x for x in xs if x is not None and isinstance(x, (int, float))]
        return statistics.median(xs) if xs else float("nan")

    print(f"{'problem':<10s} {'sampler':<12s}  {'accept':>6s}  "
          f"{'ess_min':>10s}  {'[lo – hi]':>20s}  "
          f"{'ess/sec':>10s}  {'ess/grad':>10s}  "
          f"{'ess/dens':>10s}  {'R̂':>5s}  seeds  status")
    print("-" * 120)
    root = Path(args.results_dir) if not args.csv else None
    for k in sorted(data.keys()):
        seeds = data[k]
        n = len(seeds)
        accept = med(r.get("info", {}).get("acceptance_rate", r.get("acceptance_rate"))
                     for r in seeds)
        ess_min = med(r.get("ess_min") for r in seeds)
        ess_s   = med(r.get("ess_per_sec") for r in seeds)
        # Split ess_per_eval by eval_kind (grad for HMC, logprob for gradient-free).
        # Post-hoc correction for samplers whose info dict pre-dated the
        # `n_grad_evals` patch: chess (nominal_L), peanuts (mean_tree_depth).
        grads, dens = [], []
        for r in seeds:
            info = r.get("info", {}) or {}
            name = r.get("sampler", k[1])
            ne   = r.get("n_target_evals") or 0
            kind = r.get("eval_kind", "?")
            emin = r.get("ess_min", 0) or 0
            if name == "chess" and "nominal_L" in info and kind != "grad":
                L = info["nominal_L"]
                iters = (r.get("num_samples", 0) + r.get("warmup", 0)) \
                         * r.get("n_chains", 1)
                ne, kind = iters * int(L), "grad"
            elif name == "peanuts" and "mean_tree_depth" in info and kind != "grad":
                mtd = info["mean_tree_depth"]
                iters = (r.get("num_samples", 0) + r.get("warmup", 0)) \
                         * r.get("n_chains", 1)
                ne, kind = int(iters * (2 ** mtd)), "grad"
            val = (emin / ne) if ne else None
            if kind == "grad":
                grads.append(val)
            else:
                dens.append(val)
        ess_g = med(grads) if grads else None
        ess_d = med(dens)  if dens  else None
        n_fake = sum(1 for r in seeds
                     if (r.get("fake_ess_warning") or "").strip() not in ("", "None"))
        # Cross-seed R̂ from the saved sample files (skipped if reading CSV only).
        rhat_max = float("nan")
        if root is not None:
            rh = rhat_analyze(k[0], k[1], root)
            if rh:
                rhat_max = rh["rhat_max"]
        # Fake detection: ess < n_chains AND (accept is NaN or < 0.05 — i.e.,
        # walkers aren't actually moving).  Low-accept-but-real mixing
        # (e.g., stretch/langevin_walk) is NOT fake — it's just slow.
        ncw = seeds[0].get("n_chains", 64) if seeds else 64
        n_frozen = 0
        n_low    = 0
        for r in seeds:
            info = r.get("info", {}) or {}
            a = info.get("acceptance_rate", float("nan"))
            em = r.get("ess_min") or 0
            if em < ncw:
                # Is the chain actually moving?
                bad_accept = (a != a) or (isinstance(a, (int, float)) and a < 0.05)
                if bad_accept:
                    n_frozen += 1
                else:
                    n_low += 1
        if n_fake or n_frozen == n:
            status = f"FAKE:{max(n_fake, n_frozen)}/{n}"
        elif n_frozen:
            status = f"frozen:{n_frozen}/{n}"
        elif rhat_max > 1.2:
            status = "BAD (R̂)"
        elif rhat_max > 1.05:
            status = "warn (R̂)"
        elif n_low == n:
            status = "slow-mix"           # real but low ESS (e.g., stretch 21D)
        elif n_low:
            status = f"slow:{n_low}/{n}"
        else:
            status = "ok"

        def f(v, w=10, d=2):
            if v is None:
                return f"{'—':>{w}s}"
            if v == 0:
                return f"{0:>{w}d}"
            if abs(v) >= 1e4:
                return f"{v:>{w}.0f}"
            if abs(v) < 1e-2:
                return f"{v:>{w}.2e}"
            return f"{v:>{w}.{d}f}"

        rh_str = f"{rhat_max:5.2f}" if rhat_max == rhat_max else f"{'—':>5s}"
        # Show [min - max] spread so outliers aren't hidden by median.
        ess_range = [r.get("ess_min") for r in seeds if r.get("ess_min") is not None]
        if ess_range:
            spread = f"[{min(ess_range):7.1f}–{max(ess_range):7.1f}]"
        else:
            spread = f"{'—':>20s}"
        print(f"{k[0]:<10s} {k[1]:<12s}  {accept:6.3f}  "
              f"{f(ess_min, 10, 1)}  {spread:>20s}  "
              f"{f(ess_s, 10, 2)}  {f(ess_g, 10, 3)}  "
              f"{f(ess_d, 10, 3)}  {rh_str}  {n:>5d}  {status}")


if __name__ == "__main__":
    main()
