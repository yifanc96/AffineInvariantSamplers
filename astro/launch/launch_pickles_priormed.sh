#!/bin/bash
# PICKLES with NUTS-matched init (prior_median = z=0) center.
# Sweep: γ ∈ {0.5, 2} × ws ∈ {0, 200} × 3 seeds = 12 runs.
# Spread (init_scale=0.5) is the extra ensemble-method hyperparameter that
# NUTS does not need; chosen to match numpyro init_to_median's effective
# per-chain noise of std≈0.5 in z-space (15-sample median estimator).
# Config-outer / seed-inner so JAX persistent compile cache amortizes.

set -e
cd /home/yifanchen/research/AffineInvariantSamplers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSIST_CACHE=/home/yifanchen/.cache/jax_pickles_des
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python
LOG=astro/logs/pickles_priormed_$(date +%H%M%S).log

run() {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag ===" | tee -a $LOG
  $PY -m astro.run_pickles "$@" 2>&1 | tee -a $LOG
}

# === γ=0.5, ws=0 ===
for SEED in 0 1 2; do
  run "g05_ws0_seed$SEED" \
    --problem des_y1 --seed $SEED --gamma 0.5 --L 20 --max-L 20 \
    --init-scale 0.5 --init-mode prior_median --n-chains 64 \
    --num-samples 1000 --warmup 200 --warm-start 0 \
    --out astro/results/des_y1/pickles_priormed_g05_ws0/seed$SEED
done

# === γ=0.5, ws=200 ===
for SEED in 0 1 2; do
  run "g05_ws200_seed$SEED" \
    --problem des_y1 --seed $SEED --gamma 0.5 --L 20 --max-L 20 \
    --init-scale 0.5 --init-mode prior_median --n-chains 64 \
    --num-samples 1000 --warmup 200 --warm-start 200 \
    --out astro/results/des_y1/pickles_priormed_g05_ws200/seed$SEED
done

# === γ=2, ws=0 ===
for SEED in 0 1 2; do
  run "g2_ws0_seed$SEED" \
    --problem des_y1 --seed $SEED --gamma 2.0 --L 20 --max-L 20 \
    --init-scale 0.5 --init-mode prior_median --n-chains 64 \
    --num-samples 1000 --warmup 200 --warm-start 0 \
    --out astro/results/des_y1/pickles_priormed_g2_ws0/seed$SEED
done

# === γ=2, ws=200 ===
for SEED in 0 1 2; do
  run "g2_ws200_seed$SEED" \
    --problem des_y1 --seed $SEED --gamma 2.0 --L 20 --max-L 20 \
    --init-scale 0.5 --init-mode prior_median --n-chains 64 \
    --num-samples 1000 --warmup 200 --warm-start 200 \
    --out astro/results/des_y1/pickles_priormed_g2_ws200/seed$SEED
done

echo "=== $(date +%H:%M:%S) ALL DONE ===" | tee -a $LOG
