# Synthetic benchmarks — affine-invariant ensemble HMC

Clean, fully-analytic demonstrations of sampler behavior on two controllable
targets, complementing the real-data DES Y1 experiment in `../astro/`.

Targets (both with **closed-form ground truth**, so bias needs no reference run):

- **Anisotropic Gaussian** `N(0, Σ)` — eigenvalues log-spaced over condition
  number `κ = exp(log_cond)`, optionally rotated. Isolates *global*
  anisotropy / conditioning and *dimension*.
- **Paired Rosenbrock** (even `D`) — `π(q) ∝ exp(−Σ_i [100(q_{2i-1}²−q_{2i})² + (q_{2i-1}−1)²])`.
  Banana *curvature* that no global affine map removes. Per pair
  `a~N(1,½)`, `b|a~N(a²,1/200)` ⇒ `E[a]=1, E[b]=3/2, Var[a]=½, Var[b]=2.505`.

## Three things the experiments show

1. **peaches scales with dimension; the stretch move does not.**
   On the Gaussian, fixed 64 walkers, stretch acceptance/mixing collapses as
   `D` grows (it needs walkers ≥ 2D); on Rosenbrock its gradient-free linear
   proposal cannot cross the curvature. peaches stays efficient. Seen in both
   **burn-in bias** (mean error after a fixed budget) and **stationary IAT**.
2. **Affine invariance matters: peaches beats non-affine chess.**
   On the rotated Gaussian at fixed `D=50`, peaches (and the affine stretch)
   are essentially `κ`-invariant; the non-affine `chess` degrades as `κ` grows.
3. **peaches is comparable to / beats well-tuned NUTS, but far more robust to
   warmup.** Against NumPyro NUTS (dense mass), peaches matches or beats its
   accuracy and IAT, and—because it keeps adapting the ensemble covariance
   *during production*—barely degrades when the warmup budget is cut, whereas
   NUTS (which must fit its mass matrix in warmup) degrades sharply.

## Files

| File | What |
|---|---|
| `distributions.py` | targets + analytic truth + init |
| `metrics.py` | moment bias vs truth, integrated autocorr time (IAT), ESS |
| `run.py` | one (target, dim, sampler, warmup, seed) cell → `metrics.json` |
| `experiments.py` | the 3 experiment matrices + multi-GPU launcher |
| `plots.py` | aggregate → `figs/*.png` + `results/summary.csv` |
| `benchmark_note.tex` | written-up results |

## Reproduce

From the repo root (needs `pip install -e .` and `numpyro`):

```bash
# one cell
python -m benchmark.run --target rosenbrock --dim 8  --sampler peaches --max-L 256
python -m benchmark.run --target gaussian   --dim 100 --sampler stretch --log-cond 6
python -m benchmark.run --target gaussian   --dim 50  --sampler nuts --warmup 200

# full experiments across GPUs, then figures
python -m benchmark.experiments --exp all --gpus 0 1 2 3 4 5 6
python -m benchmark.plots
```

## Settings (corrected protocol)

- **float64** throughout — the ill-conditioned Gaussian (κ up to e¹² in
  variance) overflows float32 (preconditioner / mass-matrix inversion blows up).
- **n_walkers = 4·D** for ensemble samplers — the empirical-covariance
  preconditioner is rank-deficient (and biased) if n_walkers < D.
- **Stretch pre-warmup `--warm-start 200`** — gradient-free, affine-invariant
  shape-finding so the ensemble matches the target anisotropy before the
  gradient sampler runs. Without it an isotropic init collapses the adapted
  step size. The ensemble analogue of NUTS' mass-matrix warmup; uses no
  gradients and no knowledge of the target. NUTS gets its own warmup instead.
- 5000 production samples, 3 seeds, median reported.
- peaches `max_L=20` (Gaussian) / `256` (Rosenbrock); peanuts/NUTS
  `max_tree_depth=10`, `target_accept=0.8`; NUTS is NumPyro dense-mass.

## Metrics (rigorous)

ESS and MCSE via ArviZ rank-normalized split-chain estimators;
IAT = N·C / ESS; bias judged by **z-score** |mean−truth|/MCSE (O(1) when
converged, ≫1 flags real bias); per-gradient efficiency uses the *exact*
production leapfrog count (NUTS/peanuts adapt trajectory length).

Raw samples (`*.npz`) are not stored; rerun to regenerate. The small
`metrics.json` per cell + `results/summary.csv` are the committed artifacts.

## Headline result (vs NUTS, Gaussian D=100)

With the corrected protocol, peaches/peanuts **match or beat tuned NUTS at
full warmup** (ESS/grad 1.8e-1 / 8e-2 vs 1.1e-1) and are **~1000–5000× more
robust at short warmup** (NUTS collapses to 3.7e-5 at warmup=50 — unadapted
mass matrix → max-depth trees — while the ensemble methods are unchanged,
adapting their preconditioner during production). All equally accurate
(bias z ≈ 2–3).
