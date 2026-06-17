# DES Y1 result directories — provenance manifest

Each directory holds `seed{0,1,2}/{samples.npz, metrics.json}` for one
(sampler, config) cell. This manifest maps every directory to the table or
figure it feeds in the two LaTeX notes:

- **v1** = `des_section_v1_many_samplers.tex` (all samplers)
- **v2** = `des_section_v2_focus_on_pickles.tex` (PICKLES vs NUTS vs emcee)

Exploratory / superseded runs (hyperparameter sweeps, step-size scans, bare
first-draft dirs) were moved to `../../archive/results/des_y1_exploratory.tar.gz`
(61 dirs). The `illcond_d200` set was deleted (regenerable).

Config legend: `init` = fid_ball (Gaussian ball at fiducial truth) or
prior_median (ball at z=0, NUTS-fair); `ws` = stretch pre-warmup iters;
`warm` = gradient-sampler warmup iters; `N` = production samples.

## Main comparison, N=1000, fid_ball, ws=0, warm=500, 3-seed (v1 tab:desMain, v2 tab:desMainN1k)

| directory | sampler / config |
|---|---|
| `peaches_init05_no_ws` | peaches |
| `peams_init05_no_ws` | peams |
| `peanuts_init05_no_ws` | peanuts (mtd=5) |
| `peanuts_mtd7` | peanuts (mtd=7, alt) |
| `pickles_init05_no_ws` | PICKLES γ=2 L_max=20 (default) |
| `pickles_g05_maxL20_init05` | PICKLES γ=0.5 L_max=20 (recommended) |
| `pickles_g05_maxL50_init05` | PICKLES γ=0.5 L_max=50 |
| `pickles_g05_init05_no_ws` | PICKLES γ=0.5 L_max=128 |
| `pickles_g2_maxL50_init05` | PICKLES γ=2 L_max=50 |
| `pickles_g2_maxL128_init05` | PICKLES γ=2 L_max=128 |
| `chess_tuned` | chess (tuned) |
| `malt_L50g05_init05_no_ws` | MALT L=50 γ=0.5 (ws=0) |
| `malt_corrected_16ch` | MALT 16-chain (ws=500) |
| `stretch` | emcee / stretch (gradient-free) |
| `aldi_corrected` | ALDI |
| `langevin_walk` | unadjusted PICKLES (MALA-style) |
| `nuts_numpyro_diag` | NUTS-diag |
| `nuts_numpyro_block` | NUTS-block (Campagne+ structure) |
| `nuts_numpyro_dense` | NUTS-dense |

## Long production, N=5000, single seed (v1 tab:desLongN, v2 tab:desMainN5k)

| directory | sampler / config |
|---|---|
| `peaches_init0.5_N5k_no_ws` | peaches |
| `peams_init0.5_N5k_no_ws` | peams |
| `pickles_g05_init0.5_N5k_no_ws` | PICKLES γ=0.5 L_max=20 |
| `pickles_init0.5_N5k_no_ws` | PICKLES γ=2 default |
| `nuts_numpyro_dense_N5k` | NUTS-dense |
| `stretch_long_N50000_init05` | emcee / stretch (N=50,000) |

## Per-dimension ESS, N=1000 (v2 tab:desPerDimEss)

Uses `pickles_g05_maxL20_init05` + `nuts_numpyro_dense` (same as main table,
uncapped τ).

## Warmup-length sweep, N=1000, prior_median, 3-seed (v2 tab:desWarmupSweep)

| directory | γ | ws | warm |
|---|:-:|:-:|:-:|
| `pickles_g05_maxL20_warm50_ws200_init05`  | 0.5 | 200 | 50  |
| `pickles_g05_maxL20_warm100_ws200_init05` | 0.5 | 200 | 100 |
| `pickles_priormed_g05_ws200_w150`         | 0.5 | 200 | 150 |
| `pickles_g05_maxL20_warm200_ws200_init05` | 0.5 | 200 | 200 |
| `pickles_g05_maxL20_warm50_ws50_init05`   | 0.5 | 50  | 50  |
| `pickles_g05_maxL20_warm100_ws50_init05`  | 0.5 | 50  | 100 |
| `pickles_g05_maxL20_warm200_ws50_init05`  | 0.5 | 50  | 200 |
| `pickles_g05_maxL20_warm50_init05`        | 0.5 | 0   | 50  |
| `pickles_g05_maxL20_warm100_init05`       | 0.5 | 0   | 100 |
| `pickles_priormed_g05_ws0_w150`           | 0.5 | 0   | 150 |
| `pickles_g05_maxL20_warm200_init05`       | 0.5 | 0   | 200 |
| `pickles_priormed_g2_ws0_w50`             | 2   | 0   | 50  |
| `pickles_priormed_g2_ws0_w100`            | 2   | 0   | 100 |
| `pickles_priormed_g2_ws0_w150`            | 2   | 0   | 150 |
| `pickles_priormed_g2_ws0`                 | 2   | 0   | 200 |
| `nuts_numpyro_dense_warm50/100/200`, `nuts_numpyro_dense_w150` | NUTS-dense |
| `nuts_numpyro_block_warm50/100/200`, `nuts_numpyro_block_w150` | NUTS-block |
| `pickles_g2_warm50_ws50_init05`, `..._warm100_ws50_init05`, `..._warm200_ws50_init05` | γ=2 ws=50 (footnote context) |
| `pickles_g05_maxL20_warm200_ws100_init05`, `..._ws500_init05` | extra ws variants |

## v1 warmup sweep, fid_ball (v1 tab:desWarmupSweep)

| directory | sampler |
|---|---|
| `peaches_init0.5_warm50_no_ws`, `..._warm100_no_ws`, `..._warm200_no_ws` | peaches |
| `peams_init0.5_warm50_no_ws`, `..._warm100_no_ws`, `..._warm200_no_ws` | peams |
| `pickles_init0.5_warm50_no_ws`, `..._warm100_no_ws`, `..._warm200_no_ws` | PICKLES γ=2 |

## Prior-median canonical + unadjusted appendix

| directory | role |
|---|---|
| `pickles_priormed_g05_ws0` | PICKLES γ=0.5 prior_median ws=0 (per-seed appendix) |
| `pickles_priormed_g05_ws200` | PICKLES γ=0.5 prior_median ws=200 (corner figure source) |
| `pickles_priormed_g2_ws200` | PICKLES γ=2 prior_median ws=200 |
| `pickles_unadj_init05_no_ws` | unadjusted PICKLES (tab:desUnadjLongN) |
| `pickles_unadjusted_corrected_44ch` | unadjusted PICKLES 44-chain (tab:desUnadjLongN) |
