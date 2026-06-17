# DES Y1 3×2pt — PICKLES cosmology benchmark

This directory contains the headline astrophysics experiment for the
`affine_invariant_samplers` package: benchmarking **PICKLES** (and the other
ensemble / gradient samplers) on the **DES Y1 3×2pt** cosmology posterior
(D = 21, 2250-element data vector) built on the `jax-cosmo` forward model.

The focus is **PICKLES vs NUTS vs emcee**; results for the other samplers
(peaches, peams, peanuts, chess, MALT, ALDI, langevin_walk, stretch) are
reported too, for completeness.

## Contents

| Path | What |
|---|---|
| `des_section_v2_focus_on_pickles.{tex,pdf}` | **Main note** — PICKLES vs NUTS vs emcee, per-dim ESS, warmup robustness, corner figures, per-seed + config appendices |
| `des_section_v1_many_samplers.{tex,pdf}` | **Broader note** — all samplers benchmarked on the same posterior |
| `TUNING_REPORT.md` | Hyperparameter-sweep provenance for every sampler |
| `likelihoods_des_y1.py` | DES Y1 forward model + priors + init (canonical z-space) |
| `likelihoods_des_y1_phys.py` | physical-space side experiment (appendix) |
| `run_benchmark.py` | unified runner for all 14 samplers (`--sampler ...`) |
| `run_pickles.py` | PICKLES runner with `--init-mode {fid_ball,prior_median}` and stretch pre-warmup |
| `run_numpyro_des.py` | NUTS reference (`--mass {dense,block,diag}`) |
| `full_audit.py` | recompute every table number from `results/` |
| `plot_corner.py` | regenerate the corner figures in `figs/` |
| `sweep.py` | multi-GPU driver (one `run_benchmark` per GPU) |
| `launch/` | exact reproducer scripts for each table/sweep |
| `results/des_y1/` | per-run outputs; see `results/des_y1/MANIFEST.md` for the dir→table map |
| `figs/` | corner figures (regenerable by `plot_corner.py`) |
| `archive/` | other posteriors (transit, gw_pop, illcond, …), their code + a summary note `other_benchmarks.tex`, and the exploratory DES Y1 runs (`results/des_y1_exploratory.tar.gz`) |

## Install

From the repo root:

```bash
pip install -e .          # JAX, jax-cosmo, numpyro, the sampler package
pip install emcee         # for the --sampler emcee baseline
```

The DES Y1 n(z) histograms (`data/2pt_NG_mcal_1110.fits`) are included. If
missing, fetch from the DES public data release (URL in `likelihoods_des_y1.py`).

## Reproduce

All commands run from the **repo root** (module form `astro.<x>`).

**One sampler, one seed:**
```bash
python -m astro.run_benchmark --problem des_y1 --sampler peaches --seed 0
python -m astro.run_pickles   --problem des_y1 --init-mode fid_ball --seed 0 \
       --gamma 0.5 --max-L 20 --num-samples 1000
python -m astro.run_numpyro_des --mass dense --seed 0 --num-samples 1000
```

**Full table / sweep reproducers** (multi-GPU, multi-seed) are in `launch/`:
```bash
bash launch/launch_pickles_priormed.sh    # PICKLES prior-median init grid
bash launch/launch_warm150_parallel.sh    # warmup-sweep cells
bash launch/launch_nuts_multiseed.sh      # NUTS dense/block multi-seed
bash launch/launch_g2_Lmax.sh             # PICKLES gamma=2 L_max sweep
```

**Regenerate the paper artifacts:**
```bash
python -m astro.full_audit       # prints the ESS/grad table numbers
python astro/plot_corner.py      # writes figs/des_y1_corner_optB.{pdf,png}, optC1, optC2
pdflatex des_section_v2_focus_on_pickles.tex
pdflatex des_section_v1_many_samplers.tex
```

## Conventions

- **ESS/grad** is production-only: `ESS_min / (n_chains × N_prod × L̄)`. The
  `_total` variant divides by `(N_prod + warmup)` instead.
- Multi-seed cells report the **3-seed median**; single-seed cells are flagged.
- `init`: `fid_ball` (Gaussian ball at the fiducial truth — oracle warm-start)
  vs `prior_median` (ball at z = 0, matching NUTS `init_to_median`).
- `ws` = stretch pre-warmup iterations; `warm` = gradient-sampler warmup.

See `results/des_y1/MANIFEST.md` for which directory feeds which table, and
`archive/other_benchmarks.tex` for the auxiliary posteriors.
