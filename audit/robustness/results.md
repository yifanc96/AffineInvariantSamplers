# Sampler robustness studies

These short controlled studies test stretch warmup and walker-count tradeoffs.
They do not establish universal sampler rankings. Acceptance alone does not
measure mixing.

## Reproduce

From this checkout, using the installed package dependencies:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_stretch.py
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_peaches_ensemble_size.py --pre-time-fix
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_peaches_ensemble_size.py --representative
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_peaches_ensemble_size.py --high-dimensional --production 500
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_peaches_ensemble_size.py --n2d-comparison --complement-tuner-reference --production 500
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_peaches_ensemble_size.py --near-minimum-extra --production 500
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_peaches_ensemble_size.py --underdispersed-high-dimensional --complement-tuner-reference --production 500
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/benchmark_rosenbrock_gradient_free.py
JAX_PLATFORMS=cpu PYTHONPATH=src python audit/robustness/confirm_peaches_numerical_guards.py
```

The three benchmark scripts accept `--quick`, `--warmup`, `--production`, and
`--output-dir`. The confirmation script repeats three fixed settings and accepts
`--output-dir`.
Use a separate output directory to preserve the stored data. The scripts load
this checkout's source; historical comparisons change only the reviewed
expressions in memory. The `reference_*.patch` files show those changes and
are for inspection, not application to the checkout. The optional narrower DR
retry is an experiment in the script; production retains `a2 = a * shrink`.
The stored `2*D` rows used the complement-only ChEES tuner before the later
small-ensemble fallback. `--complement-tuner-reference` reproduces that profile
in memory; omit it to benchmark the checkout's current default. No old sampler
copy or temporary worktree is needed.

Stored runs used CPU, Python 3.12.13, JAX 0.9.2, NumPy 2.4.4, and float32 JAX
arithmetic. Their source base was `main` revision
`b3db5bae94cabc18a48137336710cb15ab5feb13`, with uncommitted audit fixes.
[Provenance](data/stored_provenance.json) distinguishes historical snapshots from
the post-correction run's source hashes and dirty status. Results are copied to
both maintained branches; the study was collected on `main`.
The Rosenbrock and high-dimensional follow-ups record their own complete
protocol, source hashes, revision, and dirty paths in the corresponding
`*_run.json` files. Their base revision is the same, with the later audit fixes.
The size grids labeled post time correction were collected before the final
ChEES numerical range/aggregation/trajectory guards. Those guards preserve
ordinary finite Gaussian formulas and address overflow or nonfinite proposals.
Stored grids retain their earlier hashes; loading a newer checkout may change
adaptive trajectories. Historical modes isolate their reference expressions
within the checkout being run, rather than restoring an entire old sampler.
After the numerical guards, three confirmations repeated `D=2,N=6,seed=2,
scale=.1`, `D=2,N=24,seed=0,scale=1`, and `D=64,N=256,seed=0,scale=1` with their
original lengths. Recorded step size, trajectory length, ESJD, observable ESS,
and variance error matched the corresponding earlier rows exactly. The tiny
ensemble's `.24` variance error therefore persists as a finite-run limitation
after the overflow fixes. This check covers those cases, not every possible
warmup trajectory; see
[confirmation data](data/gaussian_guard_confirmations/peaches_numerical_guard_check.csv).
These studies and confirmations change no sampler defaults.

## Stretch warmup: 75 runs

Each run used 200 warmup iterations, 1000 production iterations, sampler seed
`s`, and initial key `42 + s`. Standard Gaussians used dimensions 2/10/50/100,
three seeds `s=0,1,2`, and `max(20, 2*D)` walkers drawn from the target. Paired
variants shared the same ensemble and random seed. A correlated 10-D Gaussian
with covariance `0.9**abs(i-j)` used seed 0; a 5-D Neal's funnel used seeds 0/1
and exact-target starts. Initial `a=2`, `shrink=.5`, and other defaults were held
fixed.

The original reference adapts `log(a)`; the corrected version adapts
`log(a-1)`, keeping `a>1`. DR acceptance is corrected in every variant. A third
DR variant separately tries `a2 = 1 + shrink*(a-1)`.

Mean squared jump distance (ESJD) per production target-density evaluation,
averaged across three Gaussian seeds:

| Dimension | Stretch original | Stretch constrained | DR original | DR constrained |
|---:|---:|---:|---:|---:|
| 2 | .419 | .424 | .272 | .269 |
| 10 | .553 | .545 | .285 | .280 |
| 50 | .569 | .576 | .349 | .342 |
| 100 | 0 | .582 | .213 | .319 |

At dimension 100, original stretch froze in all three seeds and original DR in
one of three; no constrained run froze. The Gaussian moment errors remained
comparable at lower dimensions and improved in the frozen case. This is a
robustness improvement, not a general speedup: the two short funnel DR runs had
about 11% lower average ESJD with constrained warmup, while ordinary stretch was
almost unchanged. The optional retry width increased Gaussian DR ESJD by about
27%/7%/15% at dimensions 10/50/100, but does not justify a universal default.
All run-level values are in [stretch.csv](data/stretch.csv).

## PEACHES ensemble size, before and after the ChEES time correction

The historical 42-run study used the previous `halton*T` tuning prefactor.
The 30-run post-time-correction study uses each trajectory's actual `eps*L`.
It predates the later numerical guards. Both used
standard Gaussians, 200 warmup and 1000 production iterations, default `h-walk`,
three sampler seeds 0/1/2, and initial key `42 + s`. Initial arrays were nested
prefixes of 48 walkers in dimension 2 or 72 in dimension 8. Each case compared
target-scale starts with the same points multiplied by `.1`.

Current results below average the three seeds. `L` is the realized mean length;
the condition column averages each run's sampled covariance p95 condition.

| Dimension | Walkers | Start scale | Mean L | ESJD/gradient | Variance RMS error | Covariance condition p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 1 | 5.15 | .471 | .033 | 3273 |
| 2 | 24 | 1 | 1.79 | 1.664 | .015 | 4.4 |
| 2 | 48 | 1 | 1.47 | 1.592 | .006 | 2.9 |
| 2 | 6 | .1 | 5.03 | .426 | .109 | 25 million |
| 2 | 24 | .1 | 5.80 | .504 | .011 | 4.8 |
| 2 | 48 | .1 | 5.97 | .461 | .010 | 3.0 |
| 8 | 18 | 1 | 7.35 | 1.794 | .013 | 88354 |
| 8 | 72 | 1 | 2.38 | 5.445 | .008 | 8.3 |
| 8 | 18 | .1 | 9.20 | 1.294 | .018 | 74055 |
| 8 | 72 | .1 | 7.81 | 1.409 | .005 | 7.8 |

Every sampled complement had full rank, yet one current `D=2,N=6,scale=.1`
run had variance RMS error `.24` and minimum observable ESS `14.9`, despite
acceptance `.676`. Its historical error was `.027`: the time correction changes
adaptive trajectories and does not improve every finite run. The earlier
[42-run data](data/peaches_size_pre_time_fix.csv) and current
[30-run data](data/peaches_size_post_time_fix.csv) remain separate.

Centering gives rank at most `N/2-1`, so `N >= 2*(D+1)` is necessary for full
complement covariance rank. That minimum is not a practical stability guarantee.
For these examples, 24–48 walkers at dimension 2 and 36–72 at dimension 8 gave
better conditioned metrics; the 36-walker case is from the historical study.
Initialize near the target scale and inspect `h`, `mean_L`, covariance
conditioning, moments, and cost together. Increasing walkers alone did not
remove the longer learned trajectories from underdispersed starts.

## PEACHES high-dimensional size margin

Standard Gaussians at dimensions 32 and 64 used sampler seeds 0/1, 200 warmup
and 500 production iterations, and nested exact-target starts from the first
`N` rows of `normal(key(42+s), (8*D,D))`. The original 12 runs tested
`N=2*D+2,4*D,8*D` after the time correction and before numerical guards. The
12 added runs tested `N=2*D` with default and Euclidean ChEES tuning, plus
`N=2*D+4` with default tuning, after the numerical guards. Their metadata and
source hashes remain separate. Other defaults were unchanged.

These target-scale starts test production mixing and preservation of target
moments, rather than equilibration from a difficult initial state. All 24
stationary-start runs were finite, with coordinate variance RMS errors about
.004–.011. The full ensemble covariance always had rank `D`. Complement
covariance had rank `D-1` at `N=2*D` and full rank at the larger sizes.

The table averages two seeds. Condition is the sampled covariance p95;
runtime includes JIT compilation and excludes host diagnostics.

| Dimension | Walkers | Mean L | ESJD/gradient | Minimum observable ESS | Equivalent ESS/million gradients | Complement condition p95 | Sampler seconds including JIT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 12.05 | 3.87 | 148 | 22,668 | Singular | 2.88 |
| 32 | 66 | 12.08 | 3.89 | 149 | 22,821 | 782,500 | 2.58 |
| 32 | 68 | 12.00 | 3.96 | 93 | 14,369 | 38,928 | 3.19 |
| 32 | 128 | 5.84 | 9.44 | 123 | 36,142 | 36.8 | 2.87 |
| 32 | 256 | 3.00 | 16.96 | 125 | 62,341 | 9.25 | 3.64 |
| 64 | 128 | 14.09 | 6.60 | 137 | 18,169 | Singular | 3.58 |
| 64 | 130 | 10.44 | 9.79 | 96 | 19,688 | 2,402,300 | 3.21 |
| 64 | 132 | 14.00 | 6.77 | 155 | 20,605 | 235,040 | 3.69 |
| 64 | 256 | 6.41 | 17.54 | 109 | 29,384 | 36.9 | 4.56 |
| 64 | 512 | 3.70 | 30.12 | 174 | 73,901 | 9.18 | 9.04 |

The recorded complement-only ChEES tuner solves the complement covariance. At `N=2*D`
that inverse is mathematically undefined, although floating-point solves and
the numerical guards yielded finite tuning paths in these runs. The `h-walk`
proposal uses the complement's centered velocity covariance without this
inverse. A full-rank global ensemble and accurate finite production samples
therefore do not validate the singular tuning solve. The additional Euclidean
tuning profile avoids that solve; its warmup signal depends on the coordinate
system, while production freezes the resulting parameters.
The later production fix selects the current full ensemble for ChEES only
when the complement is structurally too small to have full rank and the full
ensemble has enough points. That covariance transforms covariantly under
affine maps. It leaves the larger-complement profiles and the production
proposal unchanged; the stored singular-tuner rows remain explicitly historical.

| Dimension | Walkers | ChEES profile | Mean L | ESJD/gradient | Minimum observable ESS | Equivalent ESS/million gradients |
|---:|---:|:---|---:|---:|---:|---:|
| 32 | 64 | Complement inverse | 12.05 | 3.87 | 148 | 22,668 |
| 32 | 64 | Euclidean | 6.39 | 7.96 | 83 | 26,956 |
| 64 | 128 | Complement inverse | 14.09 | 6.60 | 137 | 18,169 |
| 64 | 128 | Euclidean | 2.69 | 18.76 | 76 | 41,422 |

The two Euclidean 32-D seeds learned quite different mean lengths, 9.6 and
3.2. At 64-D, the shorter Euclidean trajectories gave fewer effective averages
per iteration but more per gradient evaluation. Adding two or four walkers
did not give a consistent cost improvement. `N=2*D` already gave accurate
moments for these target-scale starts; `4*D` was a useful conditioning/cost
comparison, not an accuracy requirement. Data for the extra rows are in
[peaches_size_n2d.csv](data/peaches_size_n2d.csv) and
[peaches_size_n2d_plus4.csv](data/peaches_size_n2d_plus4.csv).

Four additional initialization pilot cells used the same high-dimensional
initial arrays multiplied by `.1`, seed 0, and 200 warmup plus 500 production
iterations. These used the guarded complement-tuner source recorded in their
metadata. All returned finite samples and full global rank. Mean RMS errors
were .0034–.0061; both `2*D` and `4*D` reached reasonable Gaussian moments.

| Dimension | Walkers | Final step | Mean L | Movement | Variance RMS error | Production gradients | Equivalent ESS/million gradients |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | .597 | 12.22 | .791 | .0106 | 423,104 | 11,553 |
| 32 | 128 | .653 | 6.06 | .776 | .0068 | 451,712 | 24,007 |
| 64 | 128 | .521 | 13.92 | .772 | .0063 | 955,008 | 23,234 |
| 64 | 256 | .560 | 6.72 | .772 | .0043 | 987,904 | 48,450 |

The shorter learned trajectories at `4*D` made its production gradient budget
close to `2*D` and improved these particular observable cost metrics. Matrix
work still grows with walker count. One seed on a Gaussian target does not
establish general initialization robustness, and the singular complement
tuning issue at `2*D` remains distinct from the observed sampling results.
The pilot was preserved without extending it until moments matched; see
[peaches_size_high_dimension_underdispersed.csv](data/peaches_size_high_dimension_underdispersed.csv).

Larger ensembles improved conditioning and displacement per gradient here,
but do not improve every observable's ESS per iteration. The near-minimum
64-D seeds learned mean lengths 6.9 and 13.9, illustrating adaptation
variability. With `h-walk`, each half update draws an `(N/2,N/2)` momentum
matrix and its matrix products cost order `N^2*D` per leapfrog step. Doubling
walkers quadruples that matrix's storage. At dimension 64, `N=130/256/512`
returned approximately 16.6/32.8/65.5 MB of float32 samples, while one momentum
matrix occupies 16.9/65.5/262.1 kB. These are structural sizes, not whole-process
peak memory; the CSV's RSS is a cumulative process maximum. Raw wall times are
context for that tradeoff, not steady-state throughput measurements.

For independent stationary standard-Gaussian complement walkers, let
`W=N/2` and `S=(1/W) sum (x-mean(x))(x-mean(x))^T`, the `h-walk` velocity
covariance measured in the condition diagnostics. ChEES uses the unbiased
sample covariance, with denominator `W-1`, which has the same rank and condition.
The Wishart inverse-mean formula gives
`E[S^-1] = W/(W-D-2) I` only when `W>D+2`; at the full-rank minimum `W=D+1`
that mean does not exist. This follows by applying the inverse-Wishart mean
to a centered Gaussian sample covariance; see the
[primary derivation and Appendix A](https://academic.oup.com/mnras/article/510/3/3207/6449384).
It explains why algebraic rank alone is a weak conditioning criterion. It
does not imply a mixing guarantee. The production proposal uses the centered
complement directly; its ChEES tuner solves the corresponding sample covariance.

For these Gaussian-like `h-walk` examples, a modest low-dimensional floor of
roughly 24–48 walkers, with around 32 as a first try, avoids relying on tiny
ensembles. At larger dimension, evaluate `2*D` with a well-defined tuner and
compare `2*D+2`, `2*D+4`, or `4*D` if mixing or conditioning is poor. The
tested `4*D` cases worked reasonably, but the data do not make that size a
requirement. These are empirical starting points, not new defaults. The
`8*D` runs are an exploratory cost comparison and do not establish a universal
improvement. A larger ensemble is worth comparing only when its observable
mixing or conditioning justifies the extra matrix work and storage for the
particular problem. Non-Gaussian targets and initialization can require a
different choice. Original high-dimensional run-level data are in
[peaches_size_high_dimension.csv](data/peaches_size_high_dimension.csv).

## Gradient-free product Rosenbrock: 24 paired runs

The target is the repository's product of pairs,
`log pi(x)=-sum((x_even-1)^2+100*(x_odd-x_even^2)^2)`. It has the exact
representation `X~N(1,1/2)` and `Y=X^2+E`, with independent
`E~N(0,1/200)`. Thus even/odd coordinate means are `1/1.5` and variances
`.5/2.505`; it differs from a chained Rosenbrock target.

At dimensions 2 and 8, each sampler used 64 walkers and seeds 0/1/2. Every
case started from a unit Gaussian ball using key `42+s`, prepared with 100
warmup plus 100 production stretch iterations at seed `10000+s`. The final
prepared ensemble was shared across stretch, DR stretch, side, and walk.
Each then used 500 warmup and 5000 production iterations with defaults. The
preparation costs 12,864 density evaluations per case including its initial
density, recorded separately in [preparation data](data/rosenbrock_preparation.csv).
Side/walk's initial scale-search cost is not instrumented and is excluded from
the production comparison. Production cost is exactly one density evaluation
per walker/iteration for ordinary methods and two for DR.

Errors below average the three seeds. Mean RMS is normalized by the target
coordinate standard deviations; variance RMS is relative to target variance.
The ESS column is the minimum over ensemble-mean coordinates, their centered
second moments, and a residual-space squared radius.

| Dimension | Sampler | Actual movement | ESJD/density | Mean RMS error | Variance RMS error | Minimum observable ESS |
|---:|:---|---:|---:|---:|---:|---:|
| 2 | Stretch | .408 | .0218 | .022 | .043 | 13.6 |
| 2 | DR stretch | .831 | .0115 | .045 | .087 | 14.3 |
| 2 | Side | .458 | .0316 | .067 | .194 | 10.1 |
| 2 | Walk | .442 | .0137 | .079 | .157 | 7.7 |
| 8 | Stretch | .286 | .0101 | .564 | .572 | 6.9 |
| 8 | DR stretch | .356 | .00578 | .474 | .473 | 6.8 |
| 8 | Side | .378 | .0120 | .544 | .639 | 6.9 |
| 8 | Walk | .293 | .0119 | .529 | .597 | 6.8 |

Every run returned finite samples. Ordinary stretch gave reasonable 2-D
moments in this budget, although slow second moments still had few effective
ensemble averages and the ESS length heuristic mostly failed. DR's high
acceptance did not give a comparable cost gain. In 8-D, all four methods showed
large moment errors and low observable ESS: the shared preparation and short
run did not equilibrate this curved target. These failures remain in the
report; the experiment was not extended until moments matched. They show a
finite-run mixing limitation, not evidence of a new detailed-balance error.
Affine invariance addresses linear transformations, while the narrow
Rosenbrock valley is curved. Treat longer warmup/production, target-appropriate
initialization, or a model-specific nonlinear reparameterization as choices to
evaluate with observable diagnostics. Run-level values are in
[rosenbrock_gradient_free.csv](data/rosenbrock_gradient_free.csv).

An additional internally paired HMC check compares automatic length with
fixed `L=16` for PEACHES/PICKLES, using the HMC study's exact preparation and
measurement definitions. Fixed length helps PEACHES' estimated cost mixing
in those eight cells, while moment accuracy and PICKLES results are mixed;
see [protocol and paired findings](rosenbrock_fixed_l/findings.md). Its
preparation seeds and second-moment observables differ from the gradient-free
study, so the reports do not form one eight-method cost ranking.

## Related Hamiltonian studies

The [Rosenbrock HMC report](rosenbrock_hmc.md) covers the internally paired
Hamiltonian sampler families, including the fixed-length comparison. Its
protocols and cost counters are documented separately.

## Definitions and limits

ESJD is the mean squared displacement between consecutive production states.
Gaussian and funnel displacements/moments use standardized coordinates: the
correlated Gaussian is whitened; the funnel uses `(v/3, exp(-v/2)*x)`. Values
should be compared within a target, not across targets. Density cost is one
evaluation per walker/iteration for stretch and two for DR, because both DR
stages are evaluated. PEACHES uses its production `n_grad_evals` counter.
Warmup and initial evaluations are excluded from these cost metrics. Wall times
include JIT compilation and are not used for performance recommendations.

ESS uses a self-consistent ACF window on ensemble-mean coordinate traces; the
PEACHES study also includes the ensemble-mean squared radius. Rosenbrock adds
centered coordinate second moments and the squared radius of the independent
Gaussian residual coordinates `(sqrt(2)*(X-1), sqrt(200)*(Y-X^2))`. Constant traces
score ESS zero. ESS counts effective ensemble averages; the optional
effective-draw equivalents multiply by the walker count under the target
product-ensemble law. These are not rank-normalized convergence diagnostics.
Many 1000-step high-dimensional stretch traces and 5000-step Rosenbrock traces
fail the conservative `N >= 50*tau` reliability check. Two or three seeds do
not provide precise uncertainty intervals.
Funnel physical tail moments are especially noisy. The displacement, observable
variation, conditioning, and paired failure triggers carry the conclusions;
these runs do not prove convergence or identify universally optimal settings.
