# Synthetic benchmarks — affine-invariant ensemble HMC

Clean, fully-analytic demonstrations of sampler behavior on two controllable
targets, complementing the real-data DES Y1 experiment in `../astro/`.

Targets (both with **closed-form ground truth**, so bias needs no reference run):

- **Anisotropic Gaussian** `N(0, Σ)` — eigen-standard-deviations have condition
  `κσ = exp(log_cond)`; the covariance condition is `κ = exp(2*log_cond)`
  (one for `D=1`), optionally rotated. Isolates *global*
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

From the repo root (needs `pip install -e . numpyro arviz`):

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

## Metrics and recorded costs

Rank-normalized ArviZ ESS/R-hat and ensemble-mean Sokal ESS are reported as
separate methods. Walkers in an interacting ensemble are dependent, so the
usual independent-chain interpretation of rank ESS/R-hat does not establish
convergence here. Check independent ensemble seeds, errors against analytic
first/second moments, and ensemble-mean diagnostics. The fast estimator clamps
IAT to at least one and differs from rank ESS on antithetic trajectories.
Frozen or constant observables receive zero ESS rather than a successful mixing
score; unavailable diagnostics are stored as JSON `null` and identified.

Gradient efficiency uses reported production gradient counts, including the
initial gradient on every PEACHES/PEAMS/CHESS trajectory. Gradient-free samplers
report log-density evaluations and ESS per log-density evaluation separately.
Warmup and initial step-size searches are excluded from production efficiency.
`--warm-start W` currently runs W stretch warmup iterations plus W retained
iterations: 2W preparation iterations, recorded separately with their cost and
elapsed time. Package/NUTS wall time includes device synchronization and
compilation; it excludes this preparation, with a second field including it.

New result cells record requested and effective sampler settings, target
metadata, source revision and hash, dependency versions, devices, preparation,
and diagnostic thinning. Historical cells lack this metadata and raw samples;
comparisons involving them should identify assumptions instead of inferring
unrecorded tuning settings or recomputing unavailable diagnostics.

Raw samples (`*.npz`) are not stored; rerun to regenerate. The small
`metrics.json` per cell + `results/summary.csv` are the committed artifacts.

## Headline result (vs NUTS, Gaussian D=100)

With the corrected protocol, peaches/peanuts **match or beat tuned NUTS at
full warmup** (ESS/grad 1.8e-1 / 8e-2 vs 1.1e-1) and are **~1000–5000× more
robust at short warmup** (NUTS collapses to 3.7e-5 at warmup=50 — unadapted
mass matrix → max-depth trees — while the ensemble methods are unchanged,
adapting their preconditioner during production). All equally accurate
(bias z ≈ 2–3).
