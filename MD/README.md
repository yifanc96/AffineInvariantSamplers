# MD benchmarks — DW-4, LJ-13, LJ-55

Reproductions of the DW-4 / LJ-13 / LJ-55 benchmarks using the correct
target formula (see "Energies" below).  The target and its numpyro-NUTS
reference dataset come from [pita](https://github.com/taraak/pita) /
arXiv:2506.16471; this is the *generator* for the iDEM / Adjoint-
Sampling ground-truth data.

## Headline result — PT-HMC at τ=1, no calibration hack

| Problem | ⟨E⟩ sampler | ⟨E⟩ GT | E W2 vs GT | d W2 vs GT | Wall (H100) |
| ------- | :---: | :---: | :---: | :---: | :---: |
| DW-4  | matches | matches | **0.10** | 0.21 |  9 s |
| LJ-13 | −42.81 | −43.13 | **0.35** | **0.008** | ~15 s |
| LJ-55 | −325.36 | −325.15 | **0.65** | **0.002** | ~45 s |

*(vs the previous iDEM-formula, τ=0.5 calibration: E W2 was 3.08 / 11.78,
d W2 was 0.10 / 0.06 on LJ-13 / LJ-55.  Correcting the formula improves
every metric by an order of magnitude without any τ tuning.)*

On LJ-13 with the Metropolis-adjusted **peaches** sampler + post-hoc
divergent-chain filter (iDEM-formula τ=0.5 regime) the geometry was
already very tight (`E W2 = 0.52, d W2 = 0.015`); we haven't re-run
peaches under the pita formula yet.

## Energies — what the code actually samples

**pita's MCMC script** (`sampling/sample_lj13.py`):

```
E(x)     = 2·E_LJ(x) + E_osc(x)
E_LJ(x)  = Σ_{i<j} [ (r_m/d_ij)^12 − 2·(r_m/d_ij)^6 ]       (ε = r_m = 1)
E_osc(x) = (1/2) · Σ_i ‖x_i − x̄‖²
log π(x) = −E(x) / τ,  τ = 1,  uniform prior on [−100, 100]^{3N}
```

**iDEM paper (arXiv:2406.07042) / Adjoint-Samplers (arXiv:2504.11713)
write the same formula without the factor of 2 on `E_LJ`**.  pita's own
`LennardJonesPotential` class also uses the no-factor-2 version — but
the dataset those papers cite as ground truth was generated from the
2× version, because that's what `sample_lj13.py` / `sample_lj55.py`
actually runs.  Evaluating the 2×-formula energy on the published DEM
`.npy` gives `⟨E⟩ = −43.13` (LJ-13) / `−325.15` (LJ-55), which matches
our NUTS re-generation of the pita target to within sampling noise.
This resolves the "τ calibration mystery" that the first draft of this
README described; see the section below.

- **DW-4** — 4 particles, 2D (d = 8):
  ``E(x) = Σ_{i<j} [a(d_ij − d₀) + b(d_ij − d₀)² + c(d_ij − d₀)⁴]``
  with ``a=0, b=−4, c=0.9, d₀=4, τ=1``.  (No controversy here.)

## Reproduce

```bash
# LJ-13 / LJ-55 at the pita target (factor-of-2 formula, τ=1)
python -m MD.parallel_tempering --problem lj13 --init-from-gt --formula pita
python -m MD.parallel_tempering --problem lj55 --init-from-gt --formula pita

# DW-4 (unaffected by the formula issue)
python -m MD.parallel_tempering --problem dw4 --init-from-gt

# Regenerate the pita-NUTS reference (200 k samples, ~50 min on H200):
python -m MD.sample_pita_lj --n-particles 13
python -m MD.sample_pita_lj --n-particles 55 \
    --gt-path  reference/lj55_test_part1.npy \
    --out      reference/lj55_pita_test.npy
```

## Data

The `.npy` datasets and sample arrays under `reference/` and `results/`
(~1 GB total) are **not committed** (`MD/.gitignore`).  Only the small
`metrics.json` summaries are versioned.  To obtain the data:

| File(s) | How to obtain |
| --- | --- |
| `reference/{dw4,lj13}_{train,test}.npy`, `reference/lj55_test_part1.npy` | DEM ground-truth arrays — download from [github.com/jarridrb/DEM](https://github.com/jarridrb/DEM/tree/main/data) into `reference/`. |
| `reference/lj{13,55}_pita_test.npy` | pita-NUTS reference — regenerate with `python -m MD.sample_pita_lj` (see **Reproduce** above; needs the DEM `*_test.npy` as init). |
| `results/**/*.npy` | Sampler outputs — regenerate by re-running the corresponding `run_*.py` / `parallel_tempering.py` (see **Reproduce**). |
| `alanine/reference/AD2_relaxed{,_holdout}.npy` | Alanine-dipeptide GT from the ASBS / Midgley 2023 paper (OSF) — see `alanine/README.md`. |

## Sampler comparison on the pita target

All numbers: `log π = −(2·E_LJ + E_osc)/τ, τ = 1`, init from GT, E W2 and
pair-distance W2 against the pita-NUTS reference.

**LJ-13** (d = 39, GT ⟨E⟩ = −42.92 ± 7.0):

| Sampler | ⟨E_pita⟩ | E W2 | d W2 | accept | Wall |
|---|---:|---:|---:|---:|---:|
| **peaches (MH + soft-core LJ, `--smooth`)** | −42.91 | **0.13** | **0.006** | 0.035 | 7.5 min |
| **PT-HMC (pita τ-ladder)** | −42.81 | 0.35 | 0.008 | 0.57 | **~15 s** |
| GNDR (bonded+harmonic residual) | −42.87 | 1.02 | 0.011 | 0.16 | 82 s |
| peaches (MH, raw LJ, no filter) | 2.7 × 10⁸ | 4.2 × 10⁹ | 0.11 | 0.99 | 23 min |
| peaches (MH, raw LJ, filter=15) | +13.6 | 59.7 | 0.146 | 0.99 | 9 min |
| pita-NUTS (numpyro reference) | −42.92 | 0 | 0 | — | 47 min |

**Quality winner**: peaches with soft-core LJ below `r=0.65` — E W2 almost
3× tighter than PT-HMC and barely distinguishable from the NUTS reference
itself.  **Speed winner**: PT-HMC (30× faster than peaches, W2 still below
the noise floor of the reference).

**LJ-55** (d = 165, GT ⟨E⟩ = −325.6 ± 13.8):

| Sampler | ⟨E_pita⟩ | E W2 | d W2 | accept | Wall |
|---|---:|---:|---:|---:|---:|
| **PT-HMC (pita τ-ladder)** | −325.4 | **0.65** | **0.002** | 0.57 | **~45 s** |
| pickles_unadjusted (h=0.001, 256 chains) | −324.8 | 0.96 | 0.003 | — | 63 s |
| pita-NUTS (numpyro reference) | −325.6 | 0 | 0 | — | 51 min |

## Does ensemble preconditioning help? — LJ ablation

*(see `MD/run_ablation.py`. Matched pairs — integrator + adaptation identical,
only the mass matrix (identity vs ensemble-covariance) differs — same soft-core
LJ target, same init, 50 k samples × 256 chains, warmup 10 k.  E W2 / d W2 vs the
pita-NUTS reference; both W2 estimates carry ≈ ±0.05 subsample noise.)*

> **Reproducibility note.** The LJ-13 numbers are recomputed from the (large,
> gitignored) per-sampler sample arrays and committed to
> `results/lj13_ablation_seed0/recomputed_metrics.json`; the LJ-55 numbers come
> from `results/lj55_ablation_ensemble/metrics.json`.  The original LJ-13
> `peaches`, `pickles_unadjusted`, and `baoab` runs were lost to a 256-chain GPU
> OOM and are not yet re-run, so the LJ-13 unadjusted pair is incomplete; the
> fuller matched-pair picture comes from LJ-55 below.  Wall times were not
> preserved for the surviving LJ-13 runs.

**LJ-13** (d = 39, near-isotropic basin) — surviving runs only:

| Pair | sampler | mass | E W2 | d W2 |
|---|---|---|---:|---:|
| Kinetic + MH | malt    | identity (no ens.) | **0.13** | 0.005 |
| Kinetic + MH | pickles | ensemble           | 0.38 | 0.006 |
| HMC + ChEES + MH | chess | identity (no ens.) | 0.18 | 0.004 |

On the isotropic LJ-13 basin the **no-ensemble** kinetic+MH sampler (MALT)
is the best of the surviving runs; the ensemble version (pickles) is ~3× worse
on energy W2.  (The ensemble HMC+ChEES counterpart, `peaches`, was the OOM
casualty, so that pair is incomplete here.)

**LJ-55** (d = 165) — `results/lj55_ablation_ensemble/metrics.json`:

| Pair | sampler | mass | E W2 | d W2 |
|---|---|---|---:|---:|
| Unadjusted Langevin | baoab              | identity (no ens.) | 290 (diverges) | 1.21 |
| Unadjusted Langevin | pickles_unadjusted | ensemble           | 21.1 | 0.042 |
| Kinetic + MH        | malt               | identity (no ens.) | diverges (≈2.6e9) | 0.005 |
| Kinetic + MH        | pickles            | ensemble           | **0.27** | 0.0006 |
| HMC + ChEES + MH    | peaches            | ensemble           | 0.69 | 0.0019 |

*(`chess` — the no-ensemble HMC+ChEES run — errored on LJ-55, so that pair is
incomplete.)*

Key takeaways:

- **Ensemble preconditioning rescues unadjusted samplers.** The empirical
  covariance rescales each direction's step size and stops the integrator
  blowing up: on LJ-55 plain BAOAB diverges (E W2 = 290) while
  `pickles_unadjusted` at the same step size converges (E W2 = 21.1).
- **Whether ensemble preconditioning helps MH-adjusted samplers is
  dimension-dependent.** On the near-isotropic LJ-13 basin (d = 39) it does not
  — no-ensemble MALT (0.13) beats ensemble pickles (0.38).  On LJ-55 (d = 165)
  it flips: no-ensemble MALT *diverges* while ensemble pickles is the best
  sampler in the table (E W2 = 0.27).  Higher dimension / anisotropy is where
  the per-step covariance update earns its cost.
- **Raw `1/r^12` traps MH samplers**, soft-core LJ (smooth=True below r=0.65)
  fixes this for all MH-family samplers — same target near the basin, no
  singular wall.

Related findings (numbers from the sampler-comparison tables above, not the
ablation):

- **peaches + soft-core LJ** reaches E W2 = 0.13 at 7.5 min wall on LJ-13 — the
  tightest geometry of any sampler here.
- **PT-HMC** gets E W2 = 0.35 at ~15 s on LJ-13 — fastest if you don't need
  noise-floor precision.
- **GNDR** runs cleanly (accept 16 %, correct energy) but the bond-stiffness-free
  LJ geometry doesn't reward Hessian-aware proposals.  Useful on targets
  with genuinely stiff modes (alanine bond vibrations — see `MD/alanine/`).
- **Why raw LJ breaks MH samplers but not Langevin**: at `r < r_m / 2` the
  gradient `~1/r^13` ejects a Langevin walker (pickles_unadjusted recovers
  naturally).  MH-adjusted samplers instead get *trapped*: from a wall state,
  proposals to similarly-bad states are accepted (`Δ log π ≈ 0` between two
  wall states), but proposals that would reach the populated basin require a
  finite transit through high-energy intermediates and are rejected with
  probability `1 − exp(−10⁸)`.  The problem is MH's locality, not the
  preconditioner.  NUTS avoids this via its `|ΔH| > 1000` divergence test;
  soft-core LJ avoids it by removing the trap entirely.

## Files

| File | Purpose |
| ---- | ------- |
| `energies.py`            | DW-4, LJ-n batched log-prob — both iDEM-formula and pita-formula variants |
| `ground_truth.py`        | Loader for DEM `.npy` arrays and the pita-NUTS reference |
| `metrics.py`             | 1-D W2 (capped), energy W2, pairwise-distance W2 |
| `parallel_tempering.py`  | JIT'd K-replica MH-HMC with temperature swaps  —  `--formula {idem,pita}` |
| `sample_pita_lj.py`      | numpyro / JAX port of pita's NUTS script — generates `lj{13,55}_pita_test.npy` |
| `run_pickles_cmp.py`     | Comparison dispatcher for pickles / peaches / peams / pickles_unadj  —  `--formula pita` |
| `run_gndr.py`            | Gauss-Newton preconditioned Langevin with DR  —  `--formula pita` |
| `run_gmbbvi.py`          | GMM-VI baseline (iDEM formula) |
| `run_ablation.py`        | Ensemble-preconditioning A/B ablation (identity vs ensemble mass) |
| `run_autocorr.py`        | Energy-autocorrelation / ESS-per-wall comparison across samplers |
| `reference/`             | `.npy` datasets: DEM originals + pita-NUTS-generated |

## Alanine dipeptide (separate benchmark)

See `MD/alanine/README.md`.  Unrelated to the LJ formula issue above.
