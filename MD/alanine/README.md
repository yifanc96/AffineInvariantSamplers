# Alanine dipeptide — ASBS benchmark (arXiv:2506.22565)

Sampling alanine dipeptide in AMBER ff99SBildn + GB-OBC implicit solvent at
T = 300 K.  Target is the Boltzmann distribution at `kT = 8.314 × 300 ≈ 2.494 kJ/mol`;
ground-truth samples are `AD2_relaxed.npy` (100 k × 22 × 3, in Å) and its
hold-out from the ASBS / Midgley 2023 paper, stored under `reference/`.

## Setup

- System built from `bgmol.systems.ala2.AlanineDipeptideImplicit`
  (22 atoms, AMBER force field + GBSA-OBC, `CustomGBForce`).
- Energy & forces via `openmm.Context` on the OpenCL platform
  (~15 k MD steps / s at 2 fs).
- JAX samplers talk to OpenMM through `jax.pure_callback` with a
  `custom_vjp` so that backward uses OpenMM's forces (`F = −∇E`).

## Results (comparable GT φ/ψ target)

| Sampler | Frames | KL(ours‖gt) [bits] | KL(gt‖ours) | Wall | Basin coverage |
| --- | ---: | :---: | :---: | ---: | --- |
| Langevin (1 chain, 20 ns MD) | 20,000 | **1.29** | 2.53 | 662 s | 2 major, missed minor |
| PT-HMC  (K=8, GT-init) | 2,000 | 2.60 | 8.80 | 335 s | 2 major |
| GNDR  (8 chains, 2k iter) | 16,000 | 2.89 | 12.88 | 172 s | 1 (step size collapsed) |
| PT-HMC  (K=8, eq-init) — baseline † | 2,000 | 3.97 | 10.07 | 511 s | 1 (no basin diversity) |

† The eq-init baseline run was not preserved under `results/` (only the
GT-init PT-HMC run is saved); its numbers are from the original experiment
log and are not currently reproducible from the checked-in data.  The first
three rows match the saved `results/*/metrics.json`.

Langevin wins on KL but needed 10× more frames.  PT-HMC with GT-diverse
init gets both major basins at 2,000 cold-replica frames already.

### Ramachandran plots

- `results/langevin/ramachandran.png` — 20 ns single chain
- `results/pt_hmc/ramachandran.png`   — PT-HMC, GT-diverse init
- `results/gndr/ramachandran.png`     — GNDR (illustrates step collapse)

## Key lessons

1. **Minimise init configs in our force field before sampling.**  GT data
   was generated with a slightly different AMBER variant — raw GT configs
   carry ~35 kT of hidden hydrogen-position strain, and any MH-based chain
   rejects every proposal from such a start.  A few LBFGS iterations drop
   the energy from +700 kT to −68 kT with only 0.6 Å of relaxation.
2. **Bond CFL caps `h ≈ 0.002 nm` for plain HMC.**  Sweep across
   `(h, L) ∈ [0.0015..0.003] × [3..15]` — `h = 0.003` always rejects,
   `h = 0.002` gives 50–60 % acceptance robustly.  The theoretical CFL
   bound `2·√(kT/k) ≈ 0.006` is 3× optimistic because of integrator
   non-linearities and replica-specific stability at τ > 1.
3. **PT-HMC with all replicas starting in the same basin doesn't mix.**
   200 warmup blocks × 10 HMC × 5 leapfrog at τ_max = 8.5 was not enough
   for the hottest replica to cross the φ ≈ 0 barrier.  GT-diverse init
   (each replica from a distinct minimised GT frame) fixes this.
4. **GNDR with a bonded-only residual has wrong preconditioning in soft
   modes.**  The isotropic-prior patch (`σ_iso` hyperparameter) is a
   band-aid.  The principled fix is to add LJ pair-well residuals for
   heavy-atom pairs, analogous to the LJ-13 setup in `../energies.py`.
   Left for a future iteration.
5. **Coulomb + GBSA can't be folded into the Gauss-Newton residual.**
   Coulomb is not a quadratic form; GBSA has non-local Born radii.  They
   stay in the full log-prob (OpenMM callback) and the MH step corrects
   for the preconditioner mismatch — same pattern as Köhler / LJ-13.

## Files

| File | Purpose |
| --- | --- |
| `alanine.py`                | OpenMM system build, batched energy evaluator, native Langevin, dihedral helpers |
| `jax_energy.py`             | Batched JAX log-prob wrapping OpenMM via `pure_callback` + `custom_vjp` |
| `residual.py`               | Gauss-Newton residual (bonds + angles + isotropic prior) |
| `run_langevin.py`           | OpenMM LangevinIntegrator reference baseline |
| `run_pickles_unadjusted.py` | Ensemble PICKLES (not competitive at d=66: requires N ≥ 2d chains, each step dispatches N OpenMM calls) |
| `run_pt_hmc.py`             | Parallel-tempering HMC, K = 8 replicas, pure NumPy loop + OpenMM |
| `run_gndr.py`               | Gauss-Newton DR with bonded + isotropic residual |
| `reference/AD2_relaxed.npy` | GT samples (100 k × 22 × 3, Å) from ASBS paper / OSF |
| `reference/AD2_relaxed_holdout.npy` | GT hold-out |

## Reproduce

```bash
# Reference MD
python -m MD.alanine.run_langevin --n-steps 10000000

# PT-HMC (recommended)
python -m MD.alanine.run_pt_hmc --init gt --warmup-blocks 200 --n-blocks 2000 \
    --L 5 --h-base 0.002 --swap-every 10

# GNDR
python -m MD.alanine.run_gndr --n-chains 8 --warmup 300 --n-samples 2000 \
    --step-size 0.1 --n-try 2 --sigma-iso 0.05
```

## Open problems

- Add LJ pair-well residuals to `residual.py` so GNDR's preconditioner is
  principled (current `σ_iso = 0.05 nm` is a hyperparameter hack).
- Save and reweight mid-temperature PT-HMC replicas — currently 7/8 of
  the compute is discarded.
- Batch the OpenMM callback in JAX so `vmap(grad)` doesn't serialise
  (currently `vmap_method="sequential"` is the working-but-slow path).
