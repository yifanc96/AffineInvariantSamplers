# Findings from the paired adaptation check

The actual-time correction fixes the adaptation derivative but does not make
joint step-size/trajectory-length tuning robust on the standard paired
Rosenbrock. At this short budget, several seeds still choose short trajectories
and have material first/second-moment errors. This is finite-run evidence of
poor tuning/mixing; it does not establish that the Metropolis transition has
an incorrect invariant law. Longer runs and independent ensembles are needed
before calling a configuration reliable.

The table gives medians across the same three seeds, without discarding
unsuccessful runs. Standard deviation error is the largest relative coordinate
error. ESS/gradient uses the actual production counter.

| Target | Walkers | Time signal | Mean L | ESS/gradient | Maximum std error | Maximum second-moment error |
|---|---:|---|---:|---:|---:|---:|
| Gaussian | 8 | historical | 6.595 | 0.1121 | 1.70% | 3.37% |
| Gaussian | 8 | corrected | 6.529 | 0.1001 | 1.73% | 3.43% |
| Gaussian | 64 | historical | 1.366 | 0.2031 | 0.47% | 0.95% |
| Gaussian | 64 | corrected | 1.537 | 0.3696 | 0.11% | 0.21% |
| Rosenbrock | 32 | historical | 9.047 | 0.00226 | 6.24% | 6.51% |
| Rosenbrock | 32 | corrected | 6.246 | 0.00146 | 7.47% | 7.17% |
| Rosenbrock | 64 | historical | 6.658 | 0.00265 | 10.11% | 4.00% |
| Rosenbrock | 64 | corrected | 7.480 | 0.00232 | 4.26% | 8.07% |

The larger Gaussian ensemble improves efficiency in these runs, while larger
Rosenbrock ensembles do not guarantee a good length. These are four specific
configurations, not a universal ensemble-size rule. The time-gradient bug fix
and a more robust warmup policy are separate changes; no staged policy or
production adaptation default is applied by this study.

Historical benchmark accounting also needs care. PEACHES/PEAMS/CHESS evaluate
an initial gradient plus L integration gradients per trajectory, so the old
`N*walkers*mean_L` driver omitted one evaluation. PEANUTS recomputes two
gradients per tree leaf, while old counters counted one leaf as one gradient.
Corrected PEANUTS counters describe logical work along each walker's tree:
JAX vectorization may execute padded/masked work until the longest tree stops,
so the count is not a measurement of device FLOPs or elapsed cost. Preparation,
warmup and initial scale-search costs are separate from production efficiency.

The historical Gaussian target parameter `log_cond` controlled log eigen-
standard-deviation condition. For dimension above one its covariance condition
is `exp(2*log_cond)`: the commonly used value 6 gives approximately 162,755,
rather than 403.43. New target metadata and plot generators label this
correctly while preserving target distributions and saved numeric results.
Previously generated figures retain their historical labels; they were not
silently replaced. Historic cells generally lack source revisions, full
settings and raw trajectories, so correcting their costs is conditional and
cannot repair or verify their ESS estimates.

A frozen ensemble with walkers `[-1,+1,-1,+1]` has the exact Gaussian target
mean and standard deviation yet previously received ESS=400 over 100 draws.
Benchmark diagnostics now return zero ESS and mark constant/nonmoving
observables unestimable. Independent ensemble seeds and squared observables
help avoid reporting a stuck target-initialized cloud as successful sampling.
