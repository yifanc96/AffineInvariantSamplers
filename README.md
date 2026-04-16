# AffineInvariantSamplers

JAX implementations of affine-invariant ensemble MCMC samplers and related
Hamiltonian Monte Carlo variants.

Paper: [**New affine invariant ensemble samplers and their dimensional
scaling**](https://arxiv.org/abs/2505.02987)

The original numpy implementation that accompanied the paper lives on the
[`initial-samplers`](https://github.com/yifanc96/AffineInvariantSamplers/tree/initial-samplers)
branch.  This `main` branch is a redesigned JAX-based package with a
much larger family of samplers that target high-dimensional distributions,
curved geometry, and heterogeneous / multiscale geometry — with or without
access to gradients of the target.  More papers on analysis and
methodological development are forthcoming.

## Install

```bash
git clone https://github.com/yifanc96/AffineInvariantSamplers.git
cd AffineInvariantSamplers
pip install -e .
```

Requires Python ≥ 3.10, `jax`, `jaxlib`, `numpy`.  For plotting utilities,
add `[plot]`; for tests, add `[test]`.

## Which sampler should I use?

- **Curved geometry** (e.g. Rosenbrock) — start with `sampler_peaches`.
  Other strong choices in the same family: `sampler_peanuts` (NUTS),
  `sampler_pickles` (kinetic Langevin), `sampler_peams` (microcanonical
  HMC).  All are ensemble affine-invariant versions of well-tuned HMC.
- **Multiscale geometry** (e.g. Neal's funnel) — use a delayed-rejection
  sampler: `sampler_ensemble_dr_stretch` (gradient-free) or
  `sampler_gndr` (gradient + Gauss–Newton Hessian).
- **High dimension** — prefer gradient-based ensemble HMC (peaches,
  peanuts, pickles, peams).  If you have no gradient, `sampler_kalman_move`
  achieves Langevin-like scaling on approximately Gaussian targets.

## Quick start

```python
import jax, jax.numpy as jnp
from affine_invariant_samplers import sampler_peaches

# 10-D Rosenbrock  (a = 1, b = 100).  Batched log density (n_chains, D) -> (n_chains,).
a, b = 1.0, 100.0
def log_prob(x):
    xe, xo = x[:, ::2], x[:, 1::2]
    return -(b * jnp.sum((xo - xe ** 2) ** 2, axis=1)
             + jnp.sum((xe - a) ** 2, axis=1))

init = jax.random.normal(jax.random.key(0), (100, 10))   # 100 walkers, D=10
samples, info = sampler_peaches(log_prob, init, num_samples=5000, warmup=1000,
                                 step_size=0.01)
```

**Output** (seed 0, with `affine_invariant_samplers.effective_sample_size`
for the last line):

```
samples.shape  = (5000, 100, 10)                            # 500 000 total samples
info           = {'acceptance_rate': 0.993, 'final_step_size': 0.0118,
                  'nominal_L': 20, 'n_grad_evals': 10_000_000}
x_even moments : mean = 0.99  var = 0.50   (target: 1.00, 0.500)
x_odd  moments : mean = 1.48  var = 2.44   (target: 1.50, 2.505)
min ESS        : 1031                                       # worst-mixing of the 10 coordinates
```

`min_ESS` is the smallest entry of `effective_sample_size(samples)`, i.e.
the **worst-mixing dimension**.  ESS = N<sub>total</sub> / τ where τ is the
integrated autocorrelation time; it tells you how many *independent* draws
would give the same Monte-Carlo variance.  Here ≈ 1 000 of the 500 000
samples' worth of information is realised in the hardest direction (the
long Rosenbrock axis) — that dimension sets the bottleneck for joint
statistics.

<p align="center">
  <img src="assets/quickstart_peaches_rosenbrock.png" width="620">
</p>

Blue histograms = posterior samples, red curves/contours = exact Rosenbrock
marginals.  

Every sampler in the package has this same shape:

```python
samples, info = sampler_xxx(
    log_prob_fn,                  # see table below — batched or single-point
    initial_state,                # (n_chains, D)
    num_samples,
    warmup          = 1000,
    step_size       = <default>,
    seed            = 0,
    verbose         = True,
    # sampler-specific kwargs (target_accept, L, gamma, a, chees_metric, ...)
    find_init_step_size = True,   # heuristic initial step-size search
    adapt_step_size     = True,   # dual averaging during warmup
    # adapt_L / adapt_gamma / adapt_a where applicable
)
```

If the default `find_init_step_size` heuristic picks a bad starting step
(rare, but possible when the initial ensemble is under-dispersed relative
to the target), set `find_init_step_size=False` and supply a `step_size`
of your own; dual averaging will refine it during warmup.

### Import styles

Every sampler is re-exported at the package top level, so:

```python
from affine_invariant_samplers import sampler_peaches           # flat
from affine_invariant_samplers.peaches import sampler_peaches   # namespaced
from affine_invariant_samplers import peaches                   # module
peaches.sampler_peaches(...)
```

### `log_prob_fn` convention

Not all samplers accept the same form:

| Form                                     | Samplers |
|------------------------------------------|----------|
| batched  `(n_chains, D) → (n_chains,)`   | `sampler_walk`, `sampler_stretch`, `sampler_side`, `sampler_ensemble_dr_{stretch,side}`, `sampler_langevin_walk`, `sampler_kalman_move`, `sampler_kalman_dr`, `sampler_nuts`, `sampler_peaches`, `sampler_peams`, `sampler_peanuts`, `sampler_pickles`, `sampler_chess`, `sampler_aldi`, `sampler_pickles_unadjusted` |
| single-point  `(D,) → scalar`            | `sampler_malt`, `sampler_mams`, `sampler_gndr`  |

See each sampler's docstring for its full signature and specific toggles.

## Samplers

All samplers expose toggles for (a) a heuristic initial step-size search
and (b) dual-averaging adaptation during warmup.  Where applicable they
also expose length- or scale-adaptation toggles (ChEES, NUTS tree depth,
etc.).

### Ensemble affine-invariant (gradient-free)

| Function                               | Idea                                         |
|----------------------------------------|----------------------------------------------|
| `sampler_walk`                         | Goodman–Weare walk move (k-subset variant). |
| `sampler_stretch`                      | Goodman–Weare stretch move.                 |
| `sampler_side`                         | Side move (differential-evolution style).   |
| `sampler_ensemble_dr_stretch`          | 2-stage delayed-rejection stretch.          |
| `sampler_ensemble_dr_side`             | 2-stage delayed-rejection side.             |

### Ensemble gradient-based

| Function                 | Idea                                                         |
|--------------------------|--------------------------------------------------------------|
| `sampler_langevin_walk`  | Affine-invariant Langevin walk (MALA in the complement span).|
| `sampler_kalman_move`    | Ensemble Kalman move (derivative-free drift from forward G). |
| `sampler_kalman_dr`      | Multi-stage delayed-rejection Kalman move.                   |
| `sampler_gndr`           | Gauss–Newton proposal Langevin with multi-stage DR.          |

### HMC family (single chain, batched)

| Function        | Idea                                                              |
|-----------------|-------------------------------------------------------------------|
| `sampler_malt`  | Metropolis Adjusted Langevin Trajectories (BABO+O, HMC/MALA).     |
| `sampler_mams`  | Metropolis Adjusted Microcanonical Sampler (ChEES-L or τ-tuned).  |
| `sampler_nuts`  | Classical NUTS with dual averaging.                               |

### Ensemble HMC / microcanonical / NUTS

| Function           | Idea                                                                    |
|--------------------|-------------------------------------------------------------------------|
| `sampler_peaches`  | **PEACHES**: ensemble-preconditioned ChEES-tuned HMC (walk + HMC).      |
| `sampler_peams`    | **PEAMS**: ensemble-preconditioned microcanonical HMC (walk + MAMS).    |
| `sampler_peanuts`  | **PEANUTS**: ensemble-preconditioned NUTS.                              |
| `sampler_pickles`  | **PICKLES**: parallel interacting covariance-preconditioned kinetic Langevin. |
| `sampler_chess`    | Standard HMC with joint dual-averaging + ChEES length tuning.           |

### Unadjusted Langevin (ensemble / interacting)

No Metropolis correction — these target the continuous-time invariant
distribution.  Discretisation introduces an O(h²) bias, but often allows
larger step sizes than the Metropolised counterparts.

| Function                       | Idea                                                              |
|--------------------------------|-------------------------------------------------------------------|
| `sampler_aldi`                 | Affine-invariant Langevin dynamics (overdamped).                  |
| `sampler_pickles_unadjusted`   | Unadjusted PICKLES: kinetic Langevin (BAOAB) + ensemble precond.  |

### `dev/` — related methods, not in the main package

Samplers retained for comparison but outside the affine-invariant MCMC
family live under [`dev/`](./dev/):

- **PDMPs**: `bps.py`, `bps_walk.py`, `zigzag.py`, `zigzag_walk.py`
- **Variational inference / normalizing flows**: `gvi.py`, `gmbbvi.py`,
  `dfgmvi.py`, `ig.py`

## Diagnostics

`autocorrelation`, `integrated_autocorr_time`, and `effective_sample_size`
are re-exported at the package level.  All three accept samples in any of
`(N,)`, `(N, D)`, or `(N, n_chains, D)` shape — chains are averaged per
dimension.

```python
from affine_invariant_samplers import (
    effective_sample_size, integrated_autocorr_time,
)

tau  = integrated_autocorr_time(samples)   # array, shape (D,)
ess  = effective_sample_size(samples)      # array, shape (D,) — one ESS per dim
```

For each coordinate,  **ESS = N<sub>total</sub> / τ**, where τ is the
integrated autocorrelation time.  It measures how many *independent*
draws would give the same Monte-Carlo variance as the correlated chain:
IID ⇒ ESS ≈ N<sub>total</sub>; a chain with τ = 50 gives ESS ≈
N<sub>total</sub>/50.  When examples report `min_ESS`, they mean the
smallest ESS across the D dimensions — the worst-mixing direction, which
is the bottleneck for joint statistics.

The estimator is Sokal's self-consistent-window rule (same as emcee /
arviz) with pairwise-averaging fallback for short chains.
`effective_sample_size` clamps τ at 1 so that ESS never exceeds the
sample count (HMC with long trajectories is often slightly antithetic,
so raw τ can be < 1 — you can still see that via
`integrated_autocorr_time`).

## Plotting

Install the `plot` extra: `pip install "affine-invariant-samplers[plot]"`.

```python
from affine_invariant_samplers.plotting import (
    corner_plot, trace_plot, autocorrelation_plot,
)

fig = corner_plot(samples, labels=["x", "y"], truths=[0.0, 0.0],
                   truth_1d={...}, truth_2d={...})
```

`corner_plot` produces a lower-triangular grid with 1D histograms on the
diagonal and 2D histograms below.  Optional `truth_1d` and `truth_2d`
dicts overlay analytical marginals (red curves) and joint contours.
Pure matplotlib — no `corner` package dependency.

## Examples

Four comparison scripts in [`examples/`](./examples/).  Each reports
mean/variance accuracy, acceptance rate, minimum ESS, number of gradient
evaluations (where applicable), and wall-clock time, and displays a
contour-comparison figure plus per-method corner plots with analytical
truth overlays.

| Script                                 | Target                              | Samplers compared                                   |
|----------------------------------------|-------------------------------------|-----------------------------------------------------|
| `example_gaussian.py`                  | 50-D anisotropic Gaussian, κ=1000   | `stretch`, `langevin_walk`, `kalman_move`, `peaches`|
| `example_rosenbrock.py`                | 10-D Rosenbrock, (a, b)=(1, 100)    | `peaches`, `pickles`, `peams`                       |
| `example_rosenbrock_unadjusted.py`     | 10-D Rosenbrock, (a, b)=(1, 100)    | `aldi`, `pickles_unadjusted`                        |
| `example_funnel.py`                    | 5-D Neal's funnel                   | `stretch`, `stretch-DR`, `gndr`                     |

```bash
python examples/example_gaussian.py
python examples/example_rosenbrock.py
python examples/example_rosenbrock_unadjusted.py
python examples/example_funnel.py
```

## Tests

```bash
pip install -e ".[test]"
pytest tests/
```

The smoke test runs every main-package sampler briefly on a 2-D correlated
Gaussian and a 10-D Rosenbrock and checks finite-sample mean / variance;
the diagnostics test covers ACF / IAT / ESS and the corner/trace/acf plots.

## Citation

```bibtex
@article{chen2025new,
  title={New affine invariant ensemble samplers and their dimensional scaling},
  author={Chen, Yifan},
  journal={arXiv preprint arXiv:2505.02987},
  year={2025}
}
```
