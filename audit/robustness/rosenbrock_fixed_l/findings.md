# Fixed trajectory length on product Rosenbrock

Fixed `L=16` improved PEACHES' estimated slow-observable ESS per gradient in
these paired runs, but did not improve every moment estimate. PICKLES had mixed
results, including one substantially worse cost result. This supports a fixed
trajectory option to evaluate when automatic length tuning varies; it does
not support a new default.

## Protocol

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/rosenbrock_fixed_l/run.py
```

The eight cells are PEACHES/PICKLES, dimensions 2/8, 64 walkers, and seeds
0/1. Each initial ensemble is `normal(key(42+s),(64,D))`, prepared by stretch
with 100 warmup and 100 production iterations at seed `s`. Both methods receive
the same final ensemble. Preparation costs 12,864 density evaluations including
the initial density and is separate from production cost. Each method then
uses 750 warmup and 3000 production iterations, `L=16`, `adapt_L=False`,
`max_L=128`, and other public defaults, including step-size adaptation.

The comparison uses the earlier eight matching cells from the 16-cell
adaptive study: the same preparation, seeds, target, warmup, and production,
with default adaptive length and explicit `max_L=20`. Those rows are preserved
in [adaptive_reference.json](data/adaptive_reference.json), so later experiments
cannot replace the reference. The larger fixed-run cap does not constrain its
16-step trajectory. Every fixed/adaptive pair has identical sampler source
hashes: PEACHES `9b89705330863f069817a40cf3959a237aa134df824eb31a85660b03048f8050`
and PICKLES `cc7a167367b014e0d365ee4f248564f4952682bf1e5e2677f7069b7a8c3cad6d`.
The run metadata records the dirty base revision and shared measurement-helper
hash. These runs use the guarded complement tuner; their 32-walker complements
have full rank at dimensions 2/8. A fallback restricted to halves with at most
`D` walkers would not change these cases.

The target is `log pi=-sum((X-1)^2+100*(Y-X^2)^2)` over independent pairs.
Its coordinate means are 1/1.5 and variances .5/2.505. Measurement reuses
`benchmark_rosenbrock_hmc.measure()` for both profiles: ensemble-mean
coordinates, raw coordinate squares, banana residuals, and residual squares;
the minimum observable ESS uses the same conservative ACF window. Cost is
the production gradient counter, excluding preparation, warmup, and initial
scale search. PEACHES costs `L+1` gradients per walker/trajectory; PICKLES
caches the starting gradient and costs `L`. Thus each fixed case costs
3,264,000 and 3,072,000 production gradients respectively. Runtime includes
JIT compilation and is not a steady-state speed measurement.

## Paired results

Each table entry is **adaptive / fixed**. Mean error is the maximum coordinate
error in target standard-deviation units; variance error is the maximum
relative coordinate error. Equivalent ESS per million gradients multiplies
ensemble-mean ESS by 64 under the stationary product-ensemble law.

| Method | Dimension | Seed | Adaptive mean L | Mean error | Variance error | Minimum observable ESS | Equivalent ESS/million gradients |
|:---|---:|---:|---:|:---|:---|:---|:---|
| PEACHES | 2 | 0 | 15.34 | .001 / .030 | .013 / .089 | 36.5 / 57.9 | 745 / 1135 |
| PEACHES | 2 | 1 | 6.07 | .065 / .006 | .254 / .052 | 20.4 / 57.5 | 963 / 1127 |
| PICKLES | 2 | 0 | 9.53 | .011 / .016 | .082 / .014 | 27.7 / 12.3 | 968 / 257 |
| PICKLES | 2 | 1 | 5.14 | .038 / .015 | .148 / .020 | 15.5 / 48.3 | 1003 / 1005 |
| PEACHES | 8 | 0 | 9.73 | .029 / .049 | .080 / .120 | 16.7 / 28.6 | 517 / 560 |
| PEACHES | 8 | 1 | 12.14 | .020 / .026 | .062 / .119 | 14.9 / 22.1 | 378 / 434 |
| PICKLES | 8 | 0 | 19.95 | .037 / .043 | .104 / .116 | 22.9 / 20.2 | 382 / 421 |
| PICKLES | 8 | 1 | 15.72 | .045 / .044 | .151 / .075 | 19.8 / 15.6 | 420 / 326 |

All fixed runs returned finite samples and actual movement rates .868–.937.
PEACHES' mean equivalent ESS per gradient rose about 32% in dimension 2 and
11% in dimension 8. Its two fixed 8-D variance errors were slightly larger
than the adaptive errors. PICKLES improved the two 2-D variance estimates,
yet longer trajectories made the first seed's slowest-observable cost much
worse; the other seed was almost unchanged per gradient. Its 8-D comparisons
also split across seeds and observables.

Two seeds and short traces give noisy IAT/moment estimates. Most minimum ESS
values fail the 50-effective-average reliability heuristic; fixed 2-D PEACHES
passes it, while the other profiles should not be treated as converged.
The full cells and comparison fields are retained in
[results.json](data/results.json) and [comparison.csv](data/comparison.csv).
No cells were discarded and no settings were tuned further after inspecting
them. These HMC comparisons are internally paired; they should remain separate
from the gradient-free Rosenbrock study's different preparation seeds and
second-moment observables. No production defaults were changed.
