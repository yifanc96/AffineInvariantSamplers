# develop/ — related samplers

Experimental / related sampling methods not included in the main
`affine_invariant_samplers` package.

Each file is a standalone JAX module with the same calling convention as the
main samplers (`sampler_xxx(log_prob_fn, initial_state, num_samples, ...)`).
To use one, import the file directly, e.g.

```python
import importlib, sys
sys.path.insert(0, "develop")
bps = importlib.import_module("bps")
samples, info = bps.sampler_bps(log_prob, init, num_samples=1000)
```

## Piecewise-deterministic Markov processes (PDMP)

| File               | Description                                             |
|--------------------|---------------------------------------------------------|
| `bps.py`           | Bouncy particle sampler.                                |
| `bps_walk.py`      | Ensemble-walk variant of BPS.                           |
| `zigzag.py`        | Zig-zag sampler.                                        |
| `zigzag_walk.py`   | Ensemble-walk variant of zig-zag.                       |

## Unadjusted Langevin

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `pickles_unadjusted.py`   | Unadjusted (no-MH) variant of PICKLES.           |

## Variational inference / normalizing flows

| File             | Description                                               |
|------------------|-----------------------------------------------------------|
| `gvi.py`         | Gaussian variational inference.                           |
| `gmbbvi.py`      | Gaussian mean-based black-box VI.                         |
| `dfgmvi.py`      | Derivative-free Gaussian mixture VI.                      |
| `ig.py`          | Iterative Gaussianization (rational-quadratic spline flow).|
