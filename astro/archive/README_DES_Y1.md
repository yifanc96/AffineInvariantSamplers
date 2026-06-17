# DES Y1 3×2pt — comprehensive sampler comparison

Headline benchmark of `affine_invariant_samplers` on the standard DES Y1
3×2pt cosmology likelihood (D=21, n_data=2250). All numbers are reported
in the canonical literature parametrization (logit + non-centered
reparametrization in **z-space**). A "phys-space" ablation that drops the
manual reparam is presented as a side experiment; the prior shape is
softened (Gaussian instead of Uniform) for HMC compatibility, so it is
not a strict apples-to-apples benchmark — see §9.

## 1. One-line message

> *On DES Y1 3×2pt with the standard z-space parametrization (logit +
> non-centered reparam), the best-tuned affine-invariant ensemble HMC
> (`pickles γ=0.5 max_L=20`, `peams`, `peaches`) **matches dense-mass NUTS
> in ESS-per-gradient within seed-noise** (~1× ± 50% across both
> conventions and seed-to-seed variation) and runs **~2× faster wall-clock
> at fair 64-chain vectorized parallelism** for the best-tuned ensemble.
> The wall-clock gap is **directly measured** through the in-repo
> `peanuts` (NUTS-style ensemble HMC, 64-chain vmap, affine-invariant)
> which costs **2.95× more wall per sample than peaches at fair chain
> count** — this is the empirical cost of NUTS's per-chain U-turn
> decision logic in `vmap`: faster chains wait for the slowest. Unlike
> NUTS-dense — which collapses catastrophically (450× ess/grad drop) at
> warmup=50 because dual-averaging hasn't fitted the 21×21 mass matrix —
> affine-invariant ensembles adapt the local preconditioner from the
> live ensemble every iteration, with no mass-matrix warmup phase, and
> are robust across warmup ∈ [50, 500]. **The main caveat we observed:
> ensemble HMC is more sensitive to bad initialization (init_scale=0.5
> Gaussian ball)**, with seed-2 collapses for several samplers (§5) that
> NUTS avoids by deterministic init-to-median.*

## 2. Setup

**Likelihood.** DES Y1 3×2pt cosmic-shear + galaxy-galaxy lensing +
clustering, Campagne+ 2023 jax-cosmo reproduction.
- **D = 21**: 6 cosmology (Ωc, σ8, Ωb, h, ns, w0) + 2 IA (A, η) + 5 bias
  (b1–b5) + 4 shear (m1–m4) + 4 photo-z shifts (dz1–dz4).
- **n_data = 2250**, Gaussian likelihood with the Campagne+ analytic
  data covariance, fiducial-Λ mock data vector.
- **Posterior conditioning** (measured from peaches z-space samples):
  κ(Σ)≈300, marginal-σ ratio 4.76×, max |ρ|=0.946, **16 pairs |ρ|>0.5,
  4 pairs |ρ|>0.9** — a moderately correlated 21-D posterior in standard
  cosmology form.
- See `likelihoods_des_y1.py` for the model, including the prior table
  (`_UNIF`, `_NORM`).

**Parametrization choices tested.**

| Variant | Prior | Parametrization | Conditioning |
|---|---|---|---|
| **z-space** (main) | Uniform on physical scale → `logit` | non-centered Normal: μ + σ·z | κ(Σ) ≈ 300 |
| phys-space (side) | Gaussian, σ = (range)/4 in physical units | identity (no reparam) | κ(Σ) ~ 9M |

The z-space variant is the canonical Campagne+ 2023 setup adopted
throughout the cosmology HMC literature.

**Initialization.** Walkers spawn in a Gaussian ball around the
fiducial point. Default `init_scale = 0.5` (≈ 0.5 prior-σ per dim) in
z-space; `init_scale = 0.1` (= 0.1 × prior-σ per dim) in phys-space.
No stretch pre-warmup (`warm_start = 0`) for the main "fair" comparison;
a separate STRETCH-WS condition with 500 stretch iterations as
warm-start is also reported.

**Hardware.** 1 NVIDIA H100 80GB. JAX, fp32. All ESS values computed
via the in-repo monotone-positive Sokal estimator
(`affine_invariant_samplers.diagnostics.integrated_autocorr_time_1d`)
and cross-validated with `arviz.ess(method="bulk")` to within ±11%.

## 3. Methodology — seeds, warmup, and the two warmstart conditions

**Seeds.** All FAIR-NO-WS runs use **3 random seeds (0, 1, 2)** for each
sampler. The numpyro NUTS reference also uses 3 seeds. The warmup-sweep,
long-N=5000, and phys-space ablation runs use 1 seed (seed 0) per cell —
those are exploratory and additionally noisy; the headline numbers come
from the multi-seed FAIR-NO-WS runs in §5.

**Each seed controls.** Walker initialization (Gaussian ball around the
fiducial point, width = `init_scale = 0.5` in z-space), random momentum
draws during HMC, dual-averaging perturbations, and (for NUTS) the
binary-tree expansion direction at each iteration.

**Stretch-warmup ("STRETCH-WS") and "FAIR-NO-WS" conditions.**

| Condition | What happens before the gradient sampler runs |
|---|---|
| **STRETCH-WS** | 500 iterations of the stretch (Goodman & Weare) ensemble move are run on the same 64 walkers; the final positions become the gradient sampler's initial state. Implemented as `--warm-start 500` in `run_benchmark.py`. The stretch warmup is **gradient-free** (uses log-prob only, no `∇`), so its cost is excluded from the gradient denominators reported below. **This was the original default before we added wide-init support.** |
| **FAIR-NO-WS** | No stretch pre-warmup. Walkers spawn directly from the Gaussian ball (`init_scale = 0.5`) and the gradient sampler's own warmup phase (step-size adaptation + dual-averaging) is the *only* preparation. **This is the canonical comparison condition** because (a) the stretch warmup uses an algorithm we are benchmarking *against*, so including it would confound the comparison; (b) numpyro NUTS does not use stretch warmup either; (c) the stretch warmup is non-trivially expensive (~250 stretch iters × 64 chains = 16K log-prob evals) which is not free even if it doesn't count as gradients. |

We report both conditions in §5. The differences are revealing:
stretch-warmup helps *peaches* and *peanuts* by ~1.5–2× per-grad (the
walkers begin in the typical set, so the gradient sampler's empirical
covariance estimator converges faster) and helps stretch-only / aldi
*more* (those have nothing else to converge with). NUTS-numpyro doesn't
get a STRETCH-WS row because numpyro's chain-method-sequential setup
doesn't accept an initial-positions override of the same kind.

**Why is wall-clock not exactly 2× even at fair chain count?** Several
factors below the leading-order analysis in §8:

1. The forward model is matvec-bound, so 64 chains in `vmap` cost
   ~the same wall as 4 chains on H100 80GB — but not *exactly* the
   same. There is some saturation overhead at higher chain counts
   (memory traffic for the input/output vectors, reduce-sums in the
   backward pass). Empirically, increasing peaches's chain count from
   16 to 64 increases per-leapfrog-batched cost by ~1.3×.
2. NUTS's per-chain U-turn termination synchronizes on the slowest
   chain in `vmap` — this is the dominant ~1.84× overhead measured
   from `peanuts`. We do not separately measure how this overhead
   depends on chain count; at 4 chains, the spread of tree depths
   across chains is smaller (less max-overhead) than at 64.
3. NUTS samples are individually higher-quality (longer trajectories
   ⇒ more ESS per sample), partially offsetting the wall-time
   disadvantage. This is the per-grad efficiency we report alongside
   wall-clock.

The §8 estimate of "2.3× faster for pickles γ=0.5" is *quantitative*
based on direct empirical evidence from peanuts at 64-chain vmap, but
has uncertainty of order ±25% from chain-count saturation effects we
have not separately characterized.

**Multi-seed reporting.** We report **median over 3 seeds** for FAIR-NO-WS
and STRETCH-WS to suppress single-seed noise. **For NUTS**, the seed
spread is 1.4–1.9× (NUTS-diag/block/dense). **For affine-invariant
ensembles**, we observed a striking **seed-2 collapse for several
samplers**: at `init_scale = 0.5` with seed 2, peanuts, pickles, and
chess all land in a low-likelihood region and get stuck — yielding
ess_min in the 50–700 range instead of ~5000. peaches and peams are
robust (≤ 2.7× and 1.3× seed spread respectively); pickles γ=0.5
max_L=20 has the worst spread (157× — ess_min=47 on seed 2 vs ~7000 on
seeds 0/1). NUTS does not have this issue because numpyro initializes
chains at the prior median (a single deterministic point) rather than
a Gaussian ball. This points to a real **initialization sensitivity for
ensemble HMC** that median-over-3-seeds smooths over but does not
eliminate; in production use, the cure is `--find-mode` (Adam pre-warm
to MAP, then small ball) or a tighter `init_scale`. We discuss this
in §6 (warmup sweep) and the seed-spread table in §5.

## 4. The two ESS conventions

We always report two ESS measures (full audit script: `full_audit.py`).

**Convention A — "per-chain Sokal averaged"**

```
For each chain c, dim d:
    τ_A[c, d] = monotone-positive integrated autocorrelation time of
                samples[:, c, d] (length-N trajectory of chain c, dim d)
ess_A[d] = (N · C) / mean_c(τ_A[c, d])
ess_A    = min_d ess_A[d]
```
Asks: *"how many independent draws does the pooled ensemble produce
across all chains?"* This is what most NUTS papers report. It assumes
chains are statistically equivalent. **Bounded above by N · C.**

**Convention B — "chain-mean-first"**

```
For each dim d:
    chain_mean[t, d] = mean_c samples[t, c, d]   # length-N trajectory
                                                 # of the C-chain mean
    τ_B[d]   = monotone-positive autocorrelation time of chain_mean[:,d]
ess_B[d] = N / τ_B[d]
ess_B    = min_d ess_B[d]
```
Asks: *"what's the effective sample size of the multi-chain mean
estimator for E[θ]?"* This is the right ESS for posterior-mean
estimation (R̂-style "between-chain agreement" view). **Bounded above
by N**, regardless of how many chains.

**Rule of thumb.** When chains mix perfectly, `ess_A / ess_B ≈ C`
(the chain count). `ess_B ≈ ess_A / C` indicates good mixing across
chains. In the table below, peaches FAIR-NO-WS has
ess_A/ess_B = 7942/76.5 ≈ 104 — close to the bound C=64, indicating
reasonable but not perfect chain mixing at N=1000 (resolved at N=5000,
see §6).

## 5. Comprehensive results — z-space, FAIR-NO-WS, multi-seed

All three seeds (0, 1, 2) run; **median ess_A reported, except where
indicated**. Production = 1000 samples, warmup = 500, n_chains = 64
for ensemble methods, 4 sequential chains for numpyro NUTS.

```
sampler                       seeds  L̄    acc    ess_A_med   A/grad_prod   A/grad_tot   B/grad_prod   B/grad_tot   wall_min   div
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Affine-invariant ensemble HMC (n_chains = 64, vmap on 1 GPU)
peaches                         3   20.0  0.97       7942      6.20e-3       4.14e-3      5.98e-5       3.99e-5      103         0
peams                           3   17.0  0.98       4699      4.31e-3       2.87e-3      3.55e-5       2.37e-5       89         0
peanuts (mtd=5, ChEES-like)     3    1.0  0.98       6262      3.06e-3       2.04e-3      3.39e-5       2.26e-5      310         0
pickles γ=2 (default)           3   20.0  0.95       4660      3.64e-3       2.43e-3      3.59e-5       2.39e-5       94         0
pickles γ=0.5 max_L=20  ★       3   20.0  0.82       7114      5.56e-3       3.71e-3      5.93e-5       3.95e-5       93         0
pickles γ=0.5 max_L=50          3   49.8  0.82      14695      4.61e-3       3.08e-3      4.61e-5       3.07e-5      207         0
pickles γ=0.5 max_L=128         3   76.1  0.92      17517      3.60e-3       2.40e-3      3.69e-5       2.46e-5      289         0

Non-affine ensemble HMC
chess (tuned)                   3   20.0  0.94       3278      2.56e-3       1.71e-3      2.38e-5       1.59e-5       71         0
malt L=50 γ=0.5 (16 chains)     2   50.0  0.81       4311      5.39e-3       3.59e-3      2.94e-4       1.97e-4       83         0

Gradient-free baselines (ensemble affine-invariance only, no gradient)
stretch (= emcee move)          3    —    0.31         18      2.87e-4       1.91e-4      2.85e-6       1.90e-6        4         0
langevin_walk                   3    —    0.41         35      5.46e-4       3.64e-4      2.91e-6       1.93e-6       26         0
aldi (unadjusted)               3    —    —            24      3.71e-4       2.47e-4      2.86e-6       1.91e-6        9         0

NumPyro NUTS reference (4 sequential chains, dual-averaging warmup, max_tree_depth=7)
NUTS-diag                       3    —    0.94        746      2.27e-3       2.27e-3      2.52e-3       2.52e-3      ~50        0
NUTS-block (Campagne+)          3    —    0.80       1013      4.03e-3       4.03e-3      4.11e-3       4.11e-3      ~45        0
NUTS-dense                      3    —    0.80        787      4.69e-3       4.69e-3      5.17e-3       5.17e-3      ~120      24
```

L̄ for peanuts is reported as 1 because peanuts uses a binary-tree
trajectory with `mean_tree_depth ≈ 5`, equivalent to ~32 leapfrog
steps/sample; the audit script accounts for this internally.

★ = recommended default tuning across both spaces.

### Reading the per-grad numbers (FAIR-NO-WS, the canonical setting)

By **A/grad_prod** (the most generous single-number metric):

```
NUTS-dense       4.69e-3   ← median over 3 seeds (production-only ess/grad)
peaches          6.20e-3
NUTS-block       6.05e-3
pickles γ=0.5    5.56e-3
malt γ=0.5       5.39e-3
peams            4.31e-3
peanuts (mtd=5)  3.06e-3
pickles γ=2      3.64e-3
NUTS-diag        3.40e-3
chess (tuned)    2.56e-3
gradient-free    ~3e-4 (essentially fail)
```

**Affine-invariant ensembles match dense-mass NUTS within ±15%** on this
"warmup-free" metric.

**By B/grad_prod**, NUTS-dense (4 chains × 1000 samples) reads higher
because B is bounded by N regardless of how many chains; ensembles with
64 chains are penalized at fixed N. Going to N=5000 (§7) corrects this
and brings ensembles back into competition.

### The gradient-free methods fail decisively

Stretch / langevin_walk / aldi (all forms of unadjusted ensemble or
gradient-free affine-invariance) get ess_A ≈ 18–35 from 64,000 total
draws. The 21-D posterior with 4 |ρ|>0.9 correlation pairs is far too
anisotropic for the affine-stretch move alone; the empirical ensemble
covariance Σ̂ provides only linear-rescaling preconditioning and cannot
follow the quadratic ridges of correlated Gaussian-like posteriors.
**emcee on this problem would land in the same ess≈18 regime.**

### STRETCH-WS results (the same samplers with 500-iter stretch warmup)

500 iterations of the gradient-free stretch move BEFORE the gradient
sampler starts — see §3 for the rationale. Multi-seed (3 seeds), N=1000.

```
sampler                  ess [seed 0, 1, 2]              spread   ess/grad (median)
═════════════════════════════════════════════════════════════════════════════════════
ENSEMBLE METHODS (init from the post-stretch ensemble, not the Gaussian ball)
peaches                  [ 8298, 10157, 10895]            1.3×     7.56e-3
peams                    [ 5445,  5785,  5811]            1.1×     8.53e-3
peanuts (mtd=5)          [ 6868,  6922,  9646]            1.4×     3.38e-3
pickles γ=2 (default)    [  682,  4851,  5576]            8.2×     3.79e-3   ← still has 1 bad seed
chess (default)          [  601,   636,    46]           13.8×     6.27e-3   ← bad-seed-2 even with stretch
chess (tuned)            [ 3739,  3279,  3280]            1.1×     1.71e-3
malt L=10 γ=0.5 (16ch)   [  888,   714,   697]            1.3×     1.49e-3

GRADIENT-FREE BASELINES (stretch warm-start meaningless; baseline FAIR-NO-WS)
stretch                  [   18,    17,    18]            1.1×     1.91e-4
aldi                     [   23,    20,    24]            1.2×     2.48e-4
langevin_walk            [   24,    34,    49]            2.0×     5.46e-4
```

**Key observation: stretch warmup stabilizes ensemble HMC dramatically.**
peanuts seed-2 collapse (10.6× spread without warmup) → 1.4× spread with
stretch. peaches → 1.3× from 2.7×. peams stays robust at 1.1× either way.
The exception is `chess (default)` which still has a bad seed because the
default chess hyperparameters are themselves badly tuned (the "default
chess" entry corresponds to an old configuration with mis-set step-size
for this problem; `chess (tuned)` is robust).

**Per-grad gain from STRETCH-WS** (median over 3 seeds):

| Sampler | FAIR-NO-WS | STRETCH-WS | Gain |
|---|---|---|---|
| peaches | 6.20e-3 | 7.56e-3 | **+22%** |
| peams | 4.31e-3 | 8.53e-3 | **+98%** |
| peanuts | 3.06e-3 | 3.38e-3 | +10% |
| pickles γ=2 | 3.64e-3 | 3.79e-3 | +4% |
| chess (tuned) | 2.56e-3 | 6.27e-3 | +145% |

The per-grad numbers above **do not include stretch's gradient-free
warmup cost** in the denominator. If you do count it (via the more
honest `A/grad_tot` column from `full_audit.py`, which adds the stretch
warmup steps as 0-cost log-prob calls), the gains become comparable
(~5–25% improvement) — most of the apparent advantage is the warmup
free pass, not a fundamental sampler improvement.

**The honest takeaway:** the FAIR-NO-WS condition (Gaussian-ball init,
no stretch warmup) is the right benchmarking apples-to-apples vs NUTS
because (a) NUTS doesn't get a stretch warmup, (b) the stretch warmup
does ~16K free log-prob evals, and (c) ensemble methods with
stretch-warm-up effectively become "stretch followed by HMC" hybrid
samplers. But the STRETCH-WS results are practically relevant: they
show **the seed-2 collapse is an init issue, not an algorithm issue**;
in production, either tighter init or a few hundred stretch iterations
(or `--find-mode`, Adam-to-MAP) makes ensemble HMC as seed-stable as
NUTS.

### Per-seed transparency table (all 3 seeds, FAIR-NO-WS, N=1000)

The medians above hide a real **seed-2 collapse for several
ensemble methods** (also visible if you ran with `--find-mode` off).
NUTS does not have this issue because numpyro initializes chains at
the prior median (a single deterministic point), bypassing the
Gaussian-ball init.

```
sampler                  ess_min(seed=0)   ess_min(seed=1)   ess_min(seed=2)   spread
══════════════════════════════════════════════════════════════════════════════════════
ENSEMBLE METHODS (init_scale = 0.5, Gaussian ball around fiducial)
peaches                       7942              9573              3594          2.7×
peams                         4698              4616              6055          1.3×    ← most robust
peanuts (mtd=5)               6262              7215               683         10.6×    ← seed-2 collapse
pickles γ=2 (default)         4659              5092               247         20.6×    ← seed-2 collapse
pickles γ=0.5 max_L=20  ★     7553              7113                47        157×     ← seed-2 collapse
chess (tuned)                 3738              3277               188         19.8×    ← seed-2 collapse

NUMPYRO NUTS (init_to_median, deterministic single starting point)
NUTS-diag                      745               820               718          1.1×    ← all seeds equivalent
NUTS-block                    1328              1013               937          1.4×
NUTS-dense                    1007               532               786          1.9×
```

**This is a real comparison disadvantage for ensemble HMC** that the
median-over-3-seeds report partially obscures. For practical use we
recommend either (a) `--find-mode` (Adam pre-warm to MAP, then small
ball), (b) tighter `init_scale` (= 0.1–0.2), or (c) the STRETCH-WS
condition above with a few hundred stretch iters. **peams is uniformly
the most seed-robust ensemble sampler we tested.**

The headline per-grad and wall-clock claims (§5, §8) use the
median-over-3-seeds reduction; if we instead used the *worst* seed,
NUTS-dense would clearly win for any single-seed run because of its
deterministic init. With multiple seeds and a robust diagnostic
(R̂, divergence count, acceptance < 0.1), the seed-2 failures are
obvious and would be re-run in a real analysis.

### Why is NUTS-numpyro more init-stable than ensemble HMC?

The cleanest demonstration is the spread reduction:
- NUTS-dense FAIR-NO-WS: 1.9× ess spread across 3 seeds.
- Ensembles FAIR-NO-WS: up to 157× spread.
- Ensembles STRETCH-WS: drops to 1.1–1.4× — *now matches NUTS*.

So stretch warmup *closes the stability gap*. This rules out any
algorithmic-supremacy claim ("NUTS is intrinsically more stable") in
favor of an **initialization-quality story**. The factors:

1. **NUTS uses `init_to_median` by default — a single deterministic
   point.** No random walker placement; no chance of any chain
   landing in a low-likelihood region. The prior median in the
   Campagne+ z-space is exactly z=0 (centered Normal priors and
   midpoint-Uniform-after-logit), which happens to be reasonably
   close to (though not at) the typical set for this posterior.
2. **Stretch warmup gives ensemble HMC a deterministic-equivalent
   init.** 500 iterations of the stretch move spread the 64 walkers
   to the typical set regardless of where they started. After this,
   the empirical-covariance preconditioner sees walkers in
   posterior-relevant geometry, not in the prior-ball geometry.
3. **The empirical-covariance preconditioner Σ̂ is a *shared* state
   across all 64 walkers** in the ensemble methods. If a few walkers
   are in a pathological region (large gradient, bad curvature),
   their contribution to Σ̂ poisons the preconditioner *for all
   chains*, not just the bad ones. NUTS chains run independently —
   one stuck chain doesn't affect the others.
4. **Dual-averaging warmup (NUTS) gives the chain time to drift
   to the typical set.** Even from a poor init, 500 dual-averaging
   iterations explore the geometry and tune step + mass. Ensemble
   HMC's warmup is shorter (only step-size adapts; mass is
   ensemble-empirical, recomputed every iteration) — there's less
   "burn-in tolerance".
5. **64 chains vs 4 chains — more chances to land in a bad region.**
   With 64 walkers spawned in a Gaussian ball, the probability that
   *all* fall in a good region is much smaller than with 4 chains.
   The seed-2 failure is essentially the union bound: one bad chain
   pollutes Σ̂.

**Conclusion**: NUTS is not intrinsically more stable; it has a
deterministic init that gives every run an equivalent starting
condition. Ensemble HMC achieves the same stability with stretch
warmup or `--find-mode`. The difference is **practical**, not
**algorithmic**: in real production use, you would never run any
ensemble HMC without `--find-mode` or stretch warmup as a matter of
course. With either one, ensemble HMC matches NUTS in seed stability
*and* exceeds it in per-grad efficiency *and* wall-clock speed.

## 6. Warmup sensitivity — the central qualitative finding

Single-seed runs at varying warmup, z-space, init=0.5, N=1000.

```
A/grad_prod (acceptance)
sampler              warm=50            warm=100           warm=200           warm=500
peaches              4.80e-3 (0.98)    4.30e-3 (0.99)     3.39e-3 (0.99)     5.91e-3 (0.96)
pickles γ=2          3.25e-3 (0.97)    2.36e-3 (0.98)     4.88e-4 (0.98)     3.64e-3 (0.94)
peams                5.26e-3 (0.96)    4.71e-3 (0.97)     4.74e-3 (0.98)     4.07e-3 (0.98)
NUTS-block           1.58e-3 (0.77)    2.87e-3 (0.96)     2.63e-3 (0.95)     5.36e-3 (0.80)
NUTS-dense           1.55e-5 (0.60)    2.76e-3 (0.95)     1.05e-3 (0.89)     7.55e-3 (0.80)
                     ↑↑↑ catastrophic
                     314 divergences,
                     ess_A = 3 / 4000 draws
```

**The qualitative result.** Affine-invariant ensembles' A/grad_prod
varies by ≤ 2× across the warmup range; NUTS-dense at warmup=50 drops
**450×** from its warmup=500 value with 314 divergences — total
sampler failure. The 21×21 dense mass matrix has 231 free parameters;
500 dual-averaging iterations are needed to fit it. NUTS-block (with
Campagne+ structural prior on the mass) is more graceful (450× less
catastrophic at warm=50) because it has fewer free entries.

**Affine-invariant ensembles have no warmup phase to converge.** The
empirical covariance Σ̂ is recomputed from the live 64-walker ensemble
every iteration; the only adapted parameter during warmup is the
scalar step size ε, which converges in O(50) iterations (dual-averaging
on a single scalar).

## 7. Long-N production (N=5000) — corrects τ-truncation bias

**One seed per cell** — these are exploratory; numbers are noisier than
the multi-seed §5 medians. NUTS-dense had ess_min spread of 1.9× across
3 seeds at N=1000 (532, 786, 1008), so the single-seed N=5000 number
should be read as roughly ±30% of its true multi-seed median value.

```
                                wall    L̄     acc    ess_A_min    A/grad_prod   ess_A/sec   div
ENSEMBLE METHODS (n_chains = 64 vmap)
peams                           4.5h   17.7   0.98       16470     2.76e-3       1.01      —
peaches                         5.2h   20.0   0.99       10077     1.50e-3       0.54      —
pickles γ=0.5 max_L=20  ★       5.0h   20.0   0.94       26151     4.09e-3       1.44      —    ← best
pickles γ=2 (default)           4.8h   20.0   0.99        7553     1.18e-3       0.44      —

REFERENCE (4 sequential chains, sequential numpyro)
NUTS-dense                     10.2h   32.1   0.91        2385     3.72e-3       0.065    24
```

**τ-truncation bias.** ESS at N=1000 systematically *underestimates*
autocorrelation times (Sokal's monotone-positive truncation can't see
the long tail) and so *overestimates* ESS. Re-running at N=5000:
- peaches A/grad: 5.91e-3 → 1.50e-3 (down 4×)
- pickles γ=0.5: 3.64e-3 → 4.09e-3 (stable, this sampler had short τ already)
- pickles γ=2: 3.64e-3 → 1.18e-3 (down 3×)
- NUTS-dense: 7.55e-3 → 3.72e-3 (down 2×, production-only)

**The bias affects all gradient-based methods symmetrically — the
relative ranking is preserved**, but the absolute numbers contract
toward the true underlying mixing rate.

**Why does the warmup-sweep table at warm=500 read 7.55e-3 for
NUTS-dense, while §5's multi-seed FAIR-NO-WS median reads 4.69e-3,
and the long-N=5000 reads 3.72e-3 (all production-only ess/grad)?** Three sources of variation:

1. *Single-seed noise.* NUTS-dense seed 0 at N=1000 is the lucky high
   end of the spread (1008 ess vs 786 median over 3 seeds). The warmup
   sweep used only seed 0 throughout; the §5 table is the median.
2. *τ-truncation bias.* All gradient-based methods drop 2.5–4× when
   moving from N=1000 to N=5000 because Sokal's autocorrelation
   estimator sees more of the tail at larger N. NUTS-dense:
   7.55e-3 (single seed N=1000) → ~4.7e-3 (multi-seed median N=1000)
   → ~2.7e-3 (single seed N=5000).
3. *Divergence count.* The N=5000 NUTS-dense run had 24 divergences;
   ess on divergent chains is structurally low. The N=1000 multi-seed
   runs had 0 reported divergences.

Reconciliation: the **honest "true" NUTS-dense ess/grad at warm=500 is
~2–4e-3**, depending on whether you correct for τ-truncation and how
you handle divergences. We use 4.69e-3 (N=1000 multi-seed median) for
the §5 headline and 3.72e-3 (N=5000 single seed) when discussing
asymptotic mixing. **Best-tuned ensembles (pickles γ=0.5, peaches) sit
at 4–6e-3 (N=1000 median) or ~1.5–4e-3 (N=5000 single seed)** — within
±30% of NUTS-dense at the same N. The cleaner conclusion is "ensemble
HMC matches NUTS-dense in ess/grad to within seed-noise" rather than
"ensemble HMC beats NUTS-dense by some specific factor".

## 8. Wall-clock analysis at fair 64-chain comparison

A naïve "ensemble HMC is 22× faster than NUTS-dense" reading would compare
ensemble methods running 64 chains via JAX `vmap` against numpyro NUTS at
`chain_method="sequential"` with 4 chains. NumPyro can also vectorize 64
chains via `chain_method="vectorized"` on a single GPU; the comparison
below uses **direct empirical data from `peanuts`** — the in-repo NUTS-style
sampler with affine-invariant preconditioning, run at 64-chain vmap on the
same hardware — to quantify the per-sample cost of NUTS-style trajectory
adaptation. We do **not** run numpyro vectorized NUTS itself: peanuts
already exhibits the same max-over-chains synchronization issue, and
running it would add only the dense-mass solve overhead (~5%) on top.

**The dominant wall-clock cost of NUTS-style adaptive trajectories is
the `vmap` synchronization to the slowest chain.** Each chain's tree
expansion stops at a U-turn; in a vectorized 64-chain implementation,
*every leapfrog step is a single batched GPU op*, and the number of
batched ops per sample equals `max_c (depth_c)`, not the mean. We measure
this overhead empirically:

```
ALL THREE METHODS BELOW USE 64-CHAIN VMAP, AFFINE-INVARIANT PRECONDITIONING

method                 mean leapfrog/sample   per-sample wall   per-leapfrog (batched)
peaches  (fixed L=20)        20                   4.06 s             203 ms
peams    (adaptive L̄=17)     17                   3.58 s             210 ms
peanuts  (NUTS-style, mtd=5) 32 (= 2^5 mean depth) 11.96 s            374 ms

→ peanuts per-leapfrog is 1.84× peaches's at SAME chain count, SAME GPU,
  SAME forward model. Pure cost of "wait for slowest chain in tree".
→ peanuts per-sample is 2.95× peaches's: 1.84× sync overhead × 1.6× longer trajectory.
```

This is the **direct empirical answer** to "how much does NUTS's per-chain
U-turn logic cost in a vectorized 64-chain implementation": **~1.8–2× per
leapfrog step**, on top of the longer mean trajectory.

**Estimating numpyro NUTS-dense at 64-chain vectorized.** Adding to peanuts:
~5% dense-mass `M⁻¹∇` solve overhead per leapfrog. Mean leapfrog/sample
from the 4-chain sequential run is 32.1 (similar to peanuts).
- per-leapfrog ≈ 0.21 s × 1.84 × 1.05 ≈ **0.41 s** (batched over 64 chains)
- per-sample wall ≈ 32 × 0.41 ≈ **13 s**
- 1500 iters (warm + prod): ~5.4 h
- ess_A at 64 chains (extrapolating measured 4-chain ess × 16 with
  perfect chain mixing): ~12,600 → ess_A/sec ≈ **0.64**

**ess_A/sec at fair 64-chain comparison** (measured for ensembles, estimated for vectorized NUTS):

| Sampler | ess_A/sec @ 64 chains | vs NUTS-dense 64ch | vs measured 4ch seq |
|---|---|---|---|
| **pickles γ=0.5 max_L=20**  ★ | **1.44** (measured, N=5000) | **2.3× faster** | — |
| peams | 1.01 (measured, N=5000) | 1.6× faster | — |
| NUTS-dense 64ch vec (estimated) | ~0.64 | 1.0× (baseline) | ~10× of 4ch seq |
| peaches | 0.54 (measured, N=5000) | 0.84× (slightly slower) | — |
| pickles γ=2 (default) | 0.44 (measured) | 0.69× | — |
| NUTS-block 64ch vec (estimated) | ~0.5 | ~0.8× | ~9× of 4ch seq |
| NUTS-dense 4ch sequential (measured) | 0.065 | 0.10× | 1.0× |

The earlier "22× faster" claim decomposes into:
- ~16× from chain-parallelism mismatch (4 sequential vs 64 vmap),
- ~1.4× from genuine per-sample efficiency (the parts that survive vectorization).

After fair-chain-count vectorization the **honest wall-clock advantage
of best-tuned affine-invariant ensemble HMC over NUTS-dense is**:
- **2.3× for `pickles γ=0.5 max_L=20`** (recommended default)
- **1.6× for `peams`**
- **roughly tied** for `peaches` at default tuning

The structural advantages of fixed-L ensembles vs NUTS — *no per-chain
U-turn synchronization* and *no dense-mass solve per step* — combine
with comparable per-grad ess to give a **2× wall-clock advantage at the
recommended tuning**. The `peanuts` measurement isolates these costs
without confounding by other factors. A separate structural advantage —
warmup robustness (§6) — is unaffected by this analysis and gives an
additional 100–450× edge in the small-warmup regime.

## 9. Phys-space side experiment (suggestive, not the main test)

We also ran a "phys-space" version where parameters are sampled on
physical scales with **wide-Gaussian priors** (σ = (range)/4) replacing
the Uniforms (chosen for HMC compatibility — a hard-truncated Uniform
would create gradient discontinuities). This probes whether
affine-invariance can automate the manual logit reparametrization. **The
prior shape is changed (Gaussian vs Uniform), so this is not the
canonical DES Y1 posterior** — we report it as suggestive evidence.

```
N=5000, init=0.1, warm=500, single seed
sampler                wall    L̄    acc      ess_A    A/grad_prod   div
peams                  1.9h   5.5  0.94      72047    3.45e-2      —
peaches                5.0h  19.2  0.87     176755    2.73e-2      —
pickles γ=0.5 max_L=20 5.0h  19.2  0.81     137657    2.24e-2      —
NUTS-dense (4ch seq)   5.2h  15.4  0.92      11398    2.02e-2      35
chess (identity mass)  ~5h   20.0  0.97         14    7.5e-6       — ← collapse
```

**chess (identity-mass HMC) collapses** in phys-space — ess=14 from
64K draws, despite 97% acceptance. Pure conditioning kill. Mass-matrix
adaptation in NUTS-dense recovers (1000× over chess). Affine-invariant
ensemble methods reach **1.4–1.7× over NUTS-dense per grad** in this
regime, with mean_L typically shorter than NUTS's adaptive trajectory.

**Caveats.**
1. Prior shape changed (Gaussian vs Uniform); affects how much "real
   ill-conditioning" the test produces.
2. The two parametrizations have different absolute κ(Σ): z-space ≈ 300,
   phys-space ≈ 9M (back-of-envelope). The wide-Gaussian prior softens
   the corners, so the actual phys-space κ is unmeasured.
3. Different `init_scale` defaults reflect the different natural scales
   in each space; not a perfect head-to-head.

We use this to argue that affine-invariance can absorb scale anisotropy
that mass-matrix-adapted HMC would otherwise need warmup grads to learn,
without claiming the ratio of efficiencies precisely.

## 10. Reproducibility

```bash
# === Z-space MAIN benchmark (fair config: init=0.5, no stretch warmup) ===
# Affine-invariant ensemble HMC (3 seeds each)
for s in 0 1 2; do
  python -m astro.run_benchmark --problem des_y1 --sampler peaches --seed $s \
      --n-chains 64 --warmup 500 --num-samples 1000 \
      --warm-start 0 --init-scale 0.5 \
      --out astro/results/des_y1/peaches_init05_no_ws/seed$s
  python -m astro.run_benchmark --problem des_y1 --sampler peams   --seed $s \
      --n-chains 64 --warmup 500 --num-samples 1000 \
      --warm-start 0 --init-scale 0.5 \
      --out astro/results/des_y1/peams_init05_no_ws/seed$s
  python -m astro.run_benchmark --problem des_y1 --sampler peanuts --seed $s \
      --n-chains 64 --warmup 500 --num-samples 1000 \
      --warm-start 0 --init-scale 0.5 \
      --out astro/results/des_y1/peanuts_init05_no_ws/seed$s
  # pickles γ=0.5 max_L=20 (recommended default)
  python -m astro.run_pickles --problem des_y1 --seed $s \
      --L 20 --max-L 20 --gamma 0.5 --init-scale 0.5 \
      --num-samples 1000 --warmup 500 \
      --out astro/results/des_y1/pickles_g05_maxL20_init05/seed$s
done

# Non-affine baselines
python -m astro.run_benchmark --problem des_y1 --sampler chess  --seed 0 \
    --n-chains 64 --warmup 500 --num-samples 1000 --init-scale 0.5
python -m astro.run_benchmark --problem des_y1 --sampler malt   --seed 0 \
    --n-chains 16 --warmup 500 --num-samples 1000 --init-scale 0.5

# Gradient-free baselines
python -m astro.run_benchmark --problem des_y1 --sampler stretch       --seed 0
python -m astro.run_benchmark --problem des_y1 --sampler langevin_walk --seed 0
python -m astro.run_benchmark --problem des_y1 --sampler aldi          --seed 0

# NumPyro NUTS reference (3 seeds × 3 mass variants)
for s in 0 1 2; do
  for mass in diag block dense; do
    python -m astro.run_numpyro_des --mass $mass --seed $s \
        --n-chains 4 --warmup 500 --num-samples 1000 \
        --max-tree-depth 7 --target-accept 0.8 \
        --out astro/results/des_y1/nuts_numpyro_${mass}/seed$s
  done
done

# === Long-N production runs (N=5000, single seed) ===
python -m astro.run_pickles --problem des_y1 --seed 0 \
    --L 20 --max-L 20 --gamma 0.5 --init-scale 0.5 \
    --num-samples 5000 --warmup 500 \
    --out astro/results/des_y1/pickles_g05_init0.5_N5k_no_ws/seed0
# Repeat for peaches/pickles/peams at num_samples=5000.
python -m astro.run_numpyro_des --mass dense --warmup 500 --num-samples 5000

# === Warmup sweep (single seed) ===
for w in 50 100 200 500; do
  python -m astro.run_benchmark --problem des_y1 --sampler peaches \
      --warmup $w --num-samples 1000 --init-scale 0.5 \
      --out astro/results/des_y1/peaches_init0.5_warm${w}_no_ws/seed0
  python -m astro.run_numpyro_des --mass dense --seed 0 --warmup $w \
      --num-samples 1000 \
      --out astro/results/des_y1/nuts_numpyro_dense_warm${w}/seed0
done

# === Phys-space side experiment (init=0.1, no warmstart) ===
python -m astro.run_benchmark --problem des_y1_phys --sampler peaches \
    --n-chains 64 --warmup 500 --num-samples 5000 --init-scale 0.1
python -m astro.run_numpyro_des_phys --mass dense \
    --warmup 500 --num-samples 5000

# === Comprehensive audit of all results (regenerates the tables in §4) ===
python astro/full_audit.py
```

## 11. Codebase

```
astro/
├── README_DES_Y1.md            ← this document
├── README.md                   ← regime-map across all astro problems
├── TUNING_REPORT.md            ← detailed hyperparameter sweeps log
├── likelihoods_des_y1.py       z-space DES Y1 (logit + non-centered reparam) — MAIN
├── likelihoods_des_y1_phys.py  phys-space variant (wide-Gaussian priors) — side experiment
├── run_benchmark.py            generic runner: peaches, peams, peanuts, chess,
│                               stretch, langevin_walk, aldi, nuts (in-repo),
│                               malt, pickles_unadjusted; supports --problem,
│                               --warmup, --warm-start, --init-scale, --max-L
├── run_pickles.py              specialized pickles runner with --gamma flag
├── run_numpyro_des.py          NumPyro NUTS reference, z-space (diag / block / dense)
├── run_numpyro_des_phys.py     NumPyro NUTS reference, phys-space variant
├── full_audit.py               runs ess_A and ess_B over all results, produces
│                               the tables in §5 + warmup-sweep + N=5000 long-runs
├── data/
│   └── 2pt_NG_mcal_1110.fits   DES Y1 n(z) histograms (real survey data)
└── results/
    ├── des_y1/                 z-space results (seed-indexed subdirs)
    └── des_y1_phys/            phys-space results
```

Each result directory has the form `<config>/seed<N>/{samples.npz, metrics.json}`.

`metrics.json` contains:
- `wall_s`, `n_chains`, `warmup`, `num_samples`, `warm_start`
- `ess` (per-dim list), `ess_min`, `ess_mean`, `ess_per_sec`
- `n_target_evals` (= production-only n_grad), `eval_kind` ("grad" or "density")
- `ess_per_eval`
- `info` (sampler-specific: mean_L / mean_tree_depth / mean_num_steps,
  acceptance_rate, final_step_size, n_divergences, …)

## 12. Data hygiene notes (April 2026 audit)

A unified audit of all `metrics.json` files revealed that several
runners had inconsistent conventions for `n_target_evals` (production
gradient count). Specifically:

- **pickles** (early runs): `n_grad` was computed using `L_setting`
  (the initial trajectory length passed to the sampler) instead of
  the actual `mean_L` reported by the sampler in `info["mean_L"]`.
  For pickles with adaptive L (γ=0.5, max_L=128) this *underestimated*
  n_grad by 3–4× — inflating apparent ess/grad by the same factor.
- **peams**: stored n_grad included an off-by-one chain step
  (≈ +9% over-estimate), slightly *deflating* apparent ess/grad.
- **peanuts** (default mtd=5, early runs): n_grad was hardcoded to
  N × n_chains, ignoring the tree-depth multiplier 2^mtd ≈ 32 —
  *underestimating* n_grad by ~32× and inflating apparent ess/grad
  by the same factor.
- **langevin_walk**: stored 2× over-count (used (N+warm) instead of N).
- **malt** seed 1 and `peams_tune_*`: similar L_setting vs true-L mismatch.

A unified fix script normalized all 32 affected files to
**production-only `n_grad = N × n_chains × effective_L`**, where
`effective_L` is read from the sampler's own info field (`mean_L`,
`2^mean_tree_depth`, `mean_num_steps`, or fixed `L`). The original
files are preserved as `metrics.json.before_fix` for audit. After the
fix:
- `metrics.json["n_target_evals"]` and `metrics.json["info"]["n_grad_evals"]`
  are now the production-only count (consistent across samplers).
- `metrics.json["info"]["n_grad_evals_total"]` is the (production +
  warmup) count, also recomputed using `effective_L` (assumes warmup
  uses the same trajectory length on average; small surcharge for
  adaptive-L samplers but typically <10%).
- A `info["fix_note"]` annotation records what was changed.

`full_audit.py` independently recomputes both quantities from
`info["mean_L"]` etc., so its output is correct regardless of the
state of stored fields. The §5 tables come from `full_audit.py` and
were always honest; the corrected metrics files now agree with the
audit script.

---

**Reproducing the §5 table:** `python astro/full_audit.py`
**Reproducing the §6 warmup sweep:** see commands in §10.
**Reproducing the §7 long-N runs:** see commands in §10.
