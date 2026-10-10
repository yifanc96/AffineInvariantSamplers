# Paired ChEES time-signal check

These 24 cells compare the historical `halton*T` adaptation signal with the
corrected actual integration time `eps*L`. They use immutable source snapshots,
float64, identical saved initial arrays, and seeds 0–2. Each target is two
dimensional: Gaussian uses 8/64 walkers and `max_L=20`; paired Rosenbrock uses
32/64 walkers and `max_L=256`. Warmup is 1,000 iterations, production is 1,500,
and `target_accept=0.8`. Initial arrays came from an isotropic Gaussian ball
(scale 2, seed `seed+1`), followed by 100 stretch warmup and 100 stretch
production iterations. This 200-iteration preparation is excluded from the
production gradient denominator and sampler timing.

`cells/` preserves the original measurements. `summary.json` contains their
protocol, snapshot hashes, aggregate statistics and complete cells.
`initials.json` stores the shared arrays and their hashes. Snapshot locations
inside old measurements are provenance; rerunning requires no `/tmp` files.

Install this checkout and the optional diagnostic dependencies, then run:

```bash
pip install -e . arviz
PYTHONPATH=src JAX_PLATFORMS=cpu python -m audit.robustness.paired_hmc.run \
  --target rosenbrock --walkers 32 --seed 0 --version original
PYTHONPATH=src JAX_PLATFORMS=cpu python -m audit.robustness.paired_hmc.run \
  --target rosenbrock --walkers 32 --seed 0 --version corrected
```

The portable original reference loads the current checkout and restores only
the historical time-signal expression through `_support.peaches_from_checkout`.
It retains current input guards and diagnostic fields. New measurements go to
`rerun/`, preserving saved cells. An explicit `--out` path is also supported.
Exact floating-point reproduction depends on JAX versions and backend; source
and environment metadata are recorded on every rerun.

The output reports actual production gradient counts, including each initial
trajectory gradient, analytic first/second-moment errors, ensemble-mean ESS
and squared-observable ESS. The Sokal estimator clamps IAT to at least one;
its ESS is distinct from rank ESS. Walkers within an ensemble are dependent,
so their R-hat is a descriptive warning rather than a convergence certificate.
Wall time includes compilation and excludes preparation; these small CPU
measurements do not establish a hardware throughput ranking.
