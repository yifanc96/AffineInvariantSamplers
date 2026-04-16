# AffineInvariantSamplers

JAX implementations of affine-invariant ensemble MCMC samplers and related
Hamiltonian Monte Carlo variants.

Paper: [New affine invariant ensemble samplers and their dimensional scaling](https://arxiv.org/abs/2505.02987)

The original numpy implementation that accompanied the paper lives on the
[`initial-samplers`](https://github.com/yifanc96/AffineInvariantSamplers/tree/initial-samplers)
branch.  This `main` branch contains a redesigned, JAX-based package.

## Install

```bash
git clone https://github.com/yifanc96/AffineInvariantSamplers.git
cd AffineInvariantSamplers
pip install -e .
```

Requires Python ≥ 3.10, `jax`, `jaxlib`, `numpy`.

## Quick start

```python
import jax, jax.numpy as jnp
from affine_invariant_samplers import sampler_walk

# Batched log density: (n_chains, D) -> (n_chains,)
def log_prob(x):
    return -0.5 * jnp.sum(x * x, axis=-1)

init = jax.random.normal(jax.random.key(0), (20, 2))        # 20 walkers, D=2
samples, info = sampler_walk(log_prob, init, num_samples=2000, warmup=500)
print(samples.shape)   # (2000, 20, 2)
print(info)            # {'acceptance_rate': ..., 'final_step_size': ...}
```

Every sampler is re-exported at top level, so you can import it either way:

```python
# flat
from affine_invariant_samplers import sampler_walk
# namespaced
from affine_invariant_samplers.walk import sampler_walk
# module
from affine_invariant_samplers import walk; walk.sampler_walk(...)
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
| `sampler_peaches`  | Ensemble-preconditioned HMC (walk move + HMC).                   |
| `sampler_peams`    | Ensemble-preconditioned microcanonical HMC (walk move + MAMS).   |
| `sampler_peanuts`  | Ensemble-preconditioned NUTS (walk move + NUTS).                 |
| `sampler_pickles`  | Parallel interacting covariance-preconditioned kinetic Langevin. |
| `sampler_chess`    | Standard HMC with joint DA + ChEES integration-length tuning.    |

### Unadjusted Langevin dynamics (ensemble / interacting)

No Metropolis correction — these target the continuous-time invariant
distribution; discretisation introduces an O(h²) bias.

| Function                       | Idea                                                              |
|--------------------------------|-------------------------------------------------------------------|
| `sampler_aldi`                 | Affine-invariant Langevin dynamics (interacting particles).       |
| `sampler_pickles_unadjusted`   | Unadjusted (no-MH) variant of PICKLES: kinetic Langevin with BAOAB.|

## `develop/` — related methods

Samplers that don't belong in the main affine-invariant MCMC family but are
retained for comparison live under [`develop/`](./develop/):

- **PDMPs**: `bps.py`, `bps_walk.py`, `zigzag.py`, `zigzag_walk.py`
- **Variational inference / normalizing flows**: `gvi.py`, `gmbbvi.py`, `dfgmvi.py`, `ig.py`

See [`develop/README.md`](./develop/README.md).

## Examples

Runnable comparison scripts are in [`examples/`](./examples/):

- `run_samplers.py`        — full cross-sampler comparison on Gaussian + Rosenbrock + Funnel
- `run_ensemble.py`        — ensemble-sampler showcase
- `run_funnel_*.py`        — Neal's-funnel comparisons for each family

```bash
python examples/run_samplers.py
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
