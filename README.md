# AffineInvariantSamplers

JAX implementations of affine-invariant ensemble MCMC samplers and related
Hamiltonian Monte Carlo variants.

Paper: [New affine invariant ensemble samplers and their dimensional scaling](https://arxiv.org/abs/2505.02987)

The original numpy implementation that accompanied the paper lives on the
[`initial-samplers`](https://github.com/yifanc96/AffineInvariantSamplers/tree/initial-samplers)
branch.  This `main` branch contains a redesigned, JAX-based package with much more samplers that can handle high dimensional distributions, curved geometry, and heterogeneous/multiscale geometry, with or without gradient evaluations of the target distributions. More papers are coming associated with the analysis and methodological development of these samplers.

## Install

```bash
git clone https://github.com/yifanc96/AffineInvariantSamplers.git
cd AffineInvariantSamplers
pip install -e .
```

Requires Python ≥ 3.10, `jax`, `jaxlib`, `numpy`.

## Recommendation of samplers
- For sampling curved geometry such as Rosenbrock, we recommend peaches (affine-invariant ChEES), peanuts (affine-invariant NUTs), pickles (affine-invariant kinetic Langevin), peams (affine-invariant microcanonical HMC) which are ensemble affine-invariant version of well tuned HMC sampler. Try peaches first.
- For sampling multiscale distributions such as Funnel, we recommend ensemble sampler with delayed rejection such as sampler_stretch_dr (gradient-free) and sampler_gndr (gauss-Newton-delayed-rejection; gradient based). 
- For sampling high dimensional distribuions, we recommend gradient based approaches such as affine invariant Hamiltonian sampler: peaches, peanuts, pickles, peams, or use gradient-free ensemble kalman move which achieves gradient Langevin-like scaling if the target is Gaussian-like.

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

init = jax.random.normal(jax.random.key(0), (100, 10))      # 100 walkers, D=10
samples, info = sampler_peaches(log_prob, init,
                                 num_samples=5000, warmup=1000, step_size=0.01)
print(samples.shape)   # (5000, 100, 10)
print(info)            # {'acceptance_rate': ..., 'final_step_size': ...,
                       #  'nominal_L': ..., 'n_grad_evals': ...}
```

Every sampler is re-exported at top level, so you can import it either way:

```python
# flat
from affine_invariant_samplers import sampler_peaches
# namespaced
from affine_invariant_samplers.peaches import sampler_peaches
# module
from affine_invariant_samplers import peaches; peaches.sampler_peaches(...)
```

### Calling convention

```
samples, info = sampler_xxx(
    log_prob_fn,               # see table below — batched or single-point
    initial_state,             # (n_chains, D)
    num_samples,
    warmup         = 1000,
    step_size      = <default>,
    seed           = 0,
    verbose        = True,
    # sampler-specific kwargs (target_accept, L, gamma, a, chees_metric, ...)
    find_init_step_size = True,   # heuristic initial step-size search
    adapt_step_size     = True,   # dual averaging during warmup
    # adapt_L / adapt_gamma / adapt_a where applicable
)
```
The samplers may select a bad initial step size. For such cases, please set find_init_step_size = False, and set your own initial stepsize. Then the algorithm will automatically adapt the stepsize using dual averaging to reach target acceptance probability.  

**`log_prob_fn` convention** — not all samplers accept the same form:

| Form                            | Samplers                                                                                                             |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| batched  `(n_chains, D) -> (n_chains,)` | `sampler_walk`, `sampler_stretch`, `sampler_side`, `sampler_ensemble_dr_{stretch,side}`, `sampler_langevin_walk`, `sampler_kalman_move`, `sampler_kalman_dr`, `sampler_nuts`, `sampler_peaches`, `sampler_peams`, `sampler_peanuts`, `sampler_pickles`, `sampler_chess`, `sampler_aldi`, `sampler_pickles_unadjusted` |
| single-point  `(D,) -> scalar`  | `sampler_malt`, `sampler_mams`, `sampler_gndr`                                                                       |

See each sampler's docstring for the full signature and its specific toggles.

## Samplers

All samplers support toggles for (a) heuristic initial step-size search and
(b) dual averaging during warmup.  Where applicable they also expose a
length-adaptation toggle (ChEES-based, NUTS tree-depth, etc.).

### Ensemble affine-invariant (gradient-free)

| Function                               | Idea                                          |
|----------------------------------------|-----------------------------------------------|
| `sampler_walk`                         | Goodman–Weare walk move (k-subset variant).  |
| `sampler_stretch`                      | Goodman–Weare stretch move.                  |
| `sampler_side`                         | Side move (differential-evolution style).    |
| `sampler_ensemble_dr_stretch`          | Delayed-rejection stretch, 2 stages.         |
| `sampler_ensemble_dr_side`             | Delayed-rejection side, 2 stages.            |

### Ensemble gradient-based

| Function                 | Idea                                                        |
|--------------------------|-------------------------------------------------------------|
| `sampler_langevin_walk`  | Affine-invariant Langevin (MALA in the complement subspace).|
| `sampler_kalman_move`    | Ensemble Kalman move (derivative-free drift from forward G).|
| `sampler_kalman_dr`      | Multi-stage delayed-rejection Kalman move.                  |
| `sampler_gndr`           | Gauss–Newton proposal Langevin with multi-stage DR.         |

### HMC family (single chain, batched)

| Function        | Idea                                                             |
|-----------------|------------------------------------------------------------------|
| `sampler_malt`  | Metropolis Adjusted Langevin Trajectories (BABO+O, HMC/MALA).    |
| `sampler_mams`  | Metropolis Adjusted Microcanonical Sampler (ChEES-L or τ tuned). |
| `sampler_nuts`  | Classical NUTS with dual averaging.                              |

### Ensemble HMC / microcanonical / NUTS

| Function           | Idea                                                             |
|--------------------|------------------------------------------------------------------|
| `sampler_peaches`  | Ensemble-preconditioned chess (Parallel ensemble affine-invariant ChEES).                   |
| `sampler_peams`    | Ensemble-preconditioned microcanonical HMC (walk move + MAMS).   |
| `sampler_peanuts`  | Ensemble-preconditioned NUTS (walk move + NUTS).                 |
| `sampler_pickles`  | Parallel interacting covariance-preconditioned kinetic Langevin. |
| `sampler_chess`    | Standard HMC with joint DA + ChEES integration-length tuning.    |

### Unadjusted Langevin dynamics (ensemble / interacting)

No Metropolis correction — these target the continuous-time invariant
distribution; discretisation introduces bias but often allows larger stepsize than the Metropolized counterpart.

| Function                       | Idea                                                              |
|--------------------------------|-------------------------------------------------------------------|
| `sampler_aldi`                 | Affine-invariant Langevin dynamics (interacting particles).       |
| `sampler_pickles_unadjusted`   | Unadjusted (no-MH) variant of PICKLES: kinetic Langevin with BAOAB.|

## `dev/` — related methods

Samplers that don't belong in the main affine-invariant MCMC family but are
retained for comparison live under [`dev/`](./dev/):

- **PDMPs**: `bps.py`, `bps_walk.py`, `zigzag.py`, `zigzag_walk.py`
- **Variational inference / normalizing flows**: `gvi.py`, `gmbbvi.py`, `dfgmvi.py`, `ig.py`

See [`dev/README.md`](./dev/README.md).

## Diagnostics

`autocorrelation`, `integrated_autocorr_time`, and `effective_sample_size` are
re-exported at the package level.  All three accept samples in any of
`(N,)`, `(N, D)`, or `(N, n_chains, D)` shape — chains are averaged per
dimension.

```python
from affine_invariant_samplers import (
    sampler_walk, effective_sample_size, integrated_autocorr_time,
)

samples, _ = sampler_walk(log_prob, init, num_samples=5000, warmup=1000)
tau  = integrated_autocorr_time(samples)   # array, shape (D,)
ess  = effective_sample_size(samples)      # array, shape (D,)
print(tau, ess)
```

The integrated autocorrelation time uses Sokal's self-consistent-window rule
(the same estimator as emcee / arviz), with pairwise-averaging fallback when
the chain is too short for a robust window.

## Plotting

Requires `matplotlib` — install as `pip install "affine-invariant-samplers[plot]"`.

```python
from affine_invariant_samplers.plotting import (
    corner_plot, trace_plot, autocorrelation_plot,
)

fig = corner_plot(samples, labels=["x", "y"], truths=[0.0, 0.0])
fig.savefig("posterior.png", dpi=150)

trace_plot(samples, labels=["x", "y"])
autocorrelation_plot(samples, labels=["x", "y"], max_lag=200)
```

`corner_plot` produces a lower-triangular grid with 1D histograms on the
diagonal and 2D histograms below.  It pure-matplotlib — no `corner` package
dependency.

## Examples

Three comparison scripts in [`examples/`](./examples/) — each reports
mean/variance accuracy, acceptance rate, minimum ESS, and wall-clock time.

| Script                                 | Target                           | Samplers compared                                     |
|----------------------------------------|----------------------------------|-------------------------------------------------------|
| `example_gaussian.py`                  | 20-D anisotropic Gaussian, κ=1000 | `stretch`, `langevin_walk`, `kalman_move`, `peaches`  |
| `example_rosenbrock.py`                | 10-D Rosenbrock, (a, b)=(1, 100)  | `peaches`, `pickles`, `peams`                         |
| `example_rosenbrock_unadjusted.py`     | 10-D Rosenbrock, (a, b)=(1, 100)  | `aldi`, `pickles_unadjusted`                          |
| `example_funnel.py`                    | 5-D Neal's funnel                 | `stretch`, `stretch-DR`, `gndr`                       |

Each script reports mean/variance accuracy, acceptance rate, min ESS,
number of gradient evaluations (where applicable), and wall-clock time; and
displays a contour-comparison figure plus per-method corner plots with
analytical truth marginals overlaid in red.

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

Or run the smoke test directly:

```bash
python tests/test_smoke.py
```

The smoke test runs every main-package sampler briefly on a 2D correlated
Gaussian and a 10D Rosenbrock and checks finite-sample mean/variance.

## Citation

```bibtex
@article{chen2025new,
  title={New affine invariant ensemble samplers and their dimensional scaling},
  author={Chen, Yifan},
  journal={arXiv preprint arXiv:2505.02987},
  year={2025}
}
```
