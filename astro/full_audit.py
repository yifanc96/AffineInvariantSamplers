"""Comprehensive DES Y1 audit: every sampler × {with stretch warmup, without}
× {ess_A, ess_B} × {per-grad-prod, per-grad-tot}.

Gradients are counted RIGOROUSLY from each metrics.json's stored n_grad_evals
(reflects actual mean_L for adaptive-L samplers). For samplers that don't
populate this field, we compute from mean_L × iters × n_chains.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from affine_invariant_samplers.diagnostics import integrated_autocorr_time_1d


def ess_A(x):
    N, C, D = x.shape
    taus = np.empty((C, D))
    for c in range(C):
        for d in range(D):
            taus[c, d], _, _ = integrated_autocorr_time_1d(x[:, c, d])
    return N * C / np.maximum(taus.mean(axis=0), 1.0)


def ess_B(x):
    N, C, D = x.shape
    xbar = x.mean(axis=1)
    tau = np.empty(D)
    for d in range(D):
        tau[d], _, _ = integrated_autocorr_time_1d(xbar[:, d])
    return N / np.maximum(tau, 1.0)


def true_n_grad(metrics):
    """Production-only and total gradient counts, using stored values
    (fixed in the May 2026 strict 2nd-pass audit) supplemented by recomputation
    when the sampler exposes the relevant trajectory length."""
    info = metrics.get("info", {}) or {}
    sampler = metrics.get("sampler", "")
    iters_total = metrics["num_samples"] + metrics["warmup"]
    iters_prod  = metrics["num_samples"]
    n_ch = metrics["n_chains"]

    # Get effective trajectory length per sample
    if "mean_L" in info and info["mean_L"] > 0:
        eff = float(info["mean_L"])
    elif "mean_num_steps" in info and info["mean_num_steps"] > 0:
        eff = float(info["mean_num_steps"])
    elif "mean_tree_depth" in info:
        eff = float(2 ** info["mean_tree_depth"])
    elif sampler in ("aldi", "pickles_unadjusted", "langevin_walk",
                      "stretch", "emcee"):
        eff = 1.0
    elif sampler == "malt" and "L" in info:
        eff = float(info["L"])
    elif sampler == "malt" and "L_setting" in info:
        eff = float(info["L_setting"])
    else:
        # Use stored as production-only (post-fix convention)
        n_prod = info.get("n_grad_evals") or metrics.get("n_target_evals") or 0
        n_tot = info.get("n_grad_evals_total") or n_prod  # fallback
        return int(n_tot), int(n_prod)
    n_prod = int(iters_prod * n_ch * eff)
    n_tot  = int(iters_total * n_ch * eff)
    return n_tot, n_prod


def analyse_dir(d: Path):
    """Return median over seeds of (ess_A, ess_B, A_prod, A_tot, B_prod, B_tot)."""
    rows = []
    for sd in sorted(d.glob("seed*")):
        npz = sd / "samples.npz"
        mj  = sd / "metrics.json"
        if not (npz.exists() and mj.exists()): continue
        m = json.loads(mj.read_text())
        x = np.load(npz)["samples"].astype(np.float64)
        if x.ndim != 3: continue
        N, C, _ = x.shape
        eA = float(np.min(ess_A(x)))
        eB = float(np.min(ess_B(x)))
        n_tot, n_prod = true_n_grad(m)
        if not n_tot:
            continue
        rows.append(dict(eA=eA, eB=eB,
                          A_tot=eA / n_tot, A_prod=eA / n_prod,
                          B_tot=eB * C / n_tot, B_prod=eB * C / n_prod,
                          n_chains=C, n_grad_tot=n_tot, n_grad_prod=n_prod,
                          mean_L=m.get("info", {}).get("mean_L",
                                m.get("info", {}).get("nominal_L",
                                m.get("info", {}).get("mean_tree_depth", "—")))))
    if not rows: return None
    med = lambda k: float(np.median([r[k] for r in rows]))
    return dict(seeds=len(rows), nch=rows[0]["n_chains"],
                eA=med("eA"), eB=med("eB"),
                A_tot=med("A_tot"), A_prod=med("A_prod"),
                B_tot=med("B_tot"), B_prod=med("B_prod"),
                mean_L=rows[0]["mean_L"])


root = Path("astro/results/des_y1")

# Map: (label, dir)
RUNS = [
    # Original (with stretch pre-warmup)
    ("peaches",                 "peaches",                    "STRETCH-WS"),
    ("peaches max_L=50",        "peaches_maxL50",             "STRETCH-WS"),
    ("peaches max_L=100",       "peaches_maxL100",            "STRETCH-WS"),
    ("peams",                   "peams",                      "STRETCH-WS"),
    ("peanuts mtd=5",           "peanuts",                    "STRETCH-WS"),
    ("peanuts mtd=7",           "peanuts_mtd7",               "STRETCH-WS"),
    ("pickles γ=2 default",     "pickles",                    "STRETCH-WS"),
    ("pickles L=20 γ=0.5",      "pickles_tune_L20_g05",       "STRETCH-WS"),
    ("pickles L=50 γ=0.5",      "pickles_tune_L50_g05",       "STRETCH-WS"),
    ("pickles L=100 γ=0.5",     "pickles_tune_L100_g05",      "STRETCH-WS"),
    ("chess (default buggy)",   "chess",                      "STRETCH-WS"),
    ("chess (tuned)",           "chess_tuned",                "STRETCH-WS"),
    ("malt n_ch=16",            "malt_corrected_16ch",        "STRETCH-WS"),
    ("malt L=50 γ=0.5",         "malt_tune_L50_g05",          "STRETCH-WS"),
    ("malt L=100 γ=0.5",        "malt_tune_L100_g05",         "STRETCH-WS"),
    ("aldi",                    "aldi_corrected",             "STRETCH-WS"),
    ("langevin_walk",           "langevin_walk",              "STRETCH-WS"),
    ("stretch",                 "stretch",                    "STRETCH-WS"),
    # Fair-config (init=0.5, no warmup)
    ("peaches",                 "peaches_init05_no_ws",       "FAIR-NO-WS"),
    ("peams",                   "peams_init05_no_ws",         "FAIR-NO-WS"),
    ("peanuts mtd=5",           "peanuts_init05_no_ws",       "FAIR-NO-WS"),
    ("pickles γ=2 default",     "pickles_init05_no_ws",       "FAIR-NO-WS"),
    ("pickles γ=0.5 max_L=128", "pickles_g05_init05_no_ws",   "FAIR-NO-WS"),
    ("pickles γ=0.5 max_L=50",  "pickles_g05_maxL50_init05",  "FAIR-NO-WS"),
    ("pickles γ=0.5 max_L=20",  "pickles_g05_maxL20_init05",  "FAIR-NO-WS"),
    ("chess (tuned)",           "chess_init05_no_ws",         "FAIR-NO-WS"),
    ("malt L=50 γ=0.5",         "malt_L50g05_init05_no_ws",   "FAIR-NO-WS"),
    ("aldi",                    "aldi_init05_no_ws",          "FAIR-NO-WS"),
    ("pickles_unadjusted",      "pickles_unadj_init05_no_ws", "FAIR-NO-WS"),
    # NumPyro NUTS reference (no stretch pre-warmup by design)
    ("nuts-numpyro diag",       "nuts_numpyro_diag",          "NUMPYRO-REF"),
    ("nuts-numpyro block",      "nuts_numpyro_block",         "NUMPYRO-REF"),
    ("nuts-numpyro dense",      "nuts_numpyro_dense",         "NUMPYRO-REF"),
]

results = []
for label, sub, group in RUNS:
    r = analyse_dir(root / sub)
    if r is None: continue
    r.update(label=label, group=group, dir=sub)
    results.append(r)

print(f"DES Y1 — comprehensive ess_per_grad audit  "
      f"(median over seeds; gradients counted with mean_L / mean_tree_depth)\n")
for grp_label, grp in [
    ("WITH STRETCH PRE-WARMUP (original setup)", "STRETCH-WS"),
    ("FAIR CONFIG (init_scale=0.5, no stretch warmup)", "FAIR-NO-WS"),
    ("NUMPYRO NUTS REFERENCE (own internal warmup)", "NUMPYRO-REF"),
]:
    print(f"### {grp_label} ###")
    print(f"{'sampler':<28s} {'sd':>3s} {'nch':>3s} {'L̄':>5s} "
          f"{'ess_A':>8s} {'ess_B':>7s} "
          f"{'A/grad_prod':>11s} {'A/grad_tot':>11s} "
          f"{'B/grad_prod':>11s} {'B/grad_tot':>11s}")
    print('-' * 110)
    for r in results:
        if r["group"] != grp: continue
        ml = r["mean_L"]
        ml_str = f"{ml:.0f}" if isinstance(ml, (int, float)) else f"{ml}"
        def f(v):
            if not np.isfinite(v): return f"{'—':>11s}"
            return f"{v:>11.2e}"
        def fe(v):
            if not np.isfinite(v): return f"{'—':>8s}"
            if v >= 1e4: return f"{v:>8.0f}"
            return f"{v:>8.1f}"
        print(f"{r['label']:<28s} {r['seeds']:>3d} {r['nch']:>3d} {ml_str:>5s} "
              f"{fe(r['eA'])} {fe(r['eB'])[-7:]:>7s} "
              f"{f(r['A_prod'])} {f(r['A_tot'])} "
              f"{f(r['B_prod'])} {f(r['B_tot'])}")
    print()
