#!/bin/bash
# warm=150 sweep: probe the dual-averaging overshoot pattern between
# warm=100 (peak) and warm=200 (degraded).  Run 3 seeds each for:
#   - PICKLES γ=0.5 w_s=0 (worst degradation, 9× drop 100→200)
#   - PICKLES γ=0.5 w_s=200 (sanity-check stable case)
#   - PICKLES γ=2 w_s=0 (monotone degradation)
#   - NUTS-dense (2× drop 100→200)
# All other settings match Table 3 (prior_median init, init_scale=0.5,
# N_prod=1000, L_max=20 for PICKLES).

set -e
cd /home/yifanchen/research/AffineInvariantSamplers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSIST_CACHE=/home/yifanchen/.cache/jax_pickles_des
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python
LOG=astro/logs/warm150_$(date +%H%M%S).log
mkdir -p astro/logs

run() {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag ===" | tee -a $LOG
  $PY "$@" 2>&1 | tee -a $LOG
}

# === PICKLES γ=0.5 w_s=0 warm=150 ===
for SEED in 0 1 2; do
  run "pkl_g05_ws0_w150_s$SEED" -m astro.run_pickles \
    --problem des_y1 --seed $SEED --gamma 0.5 --L 20 --max-L 20 \
    --init-scale 0.5 --init-mode prior_median --n-chains 64 \
    --num-samples 1000 --warmup 150 --warm-start 0 \
    --out astro/results/des_y1/pickles_priormed_g05_ws0_w150/seed$SEED
done

# === PICKLES γ=0.5 w_s=200 warm=150 ===
for SEED in 0 1 2; do
  run "pkl_g05_ws200_w150_s$SEED" -m astro.run_pickles \
    --problem des_y1 --seed $SEED --gamma 0.5 --L 20 --max-L 20 \
    --init-scale 0.5 --init-mode prior_median --n-chains 64 \
    --num-samples 1000 --warmup 150 --warm-start 200 \
    --out astro/results/des_y1/pickles_priormed_g05_ws200_w150/seed$SEED
done

# === PICKLES γ=2 w_s=0 warm=150 ===
for SEED in 0 1 2; do
  run "pkl_g2_ws0_w150_s$SEED" -m astro.run_pickles \
    --problem des_y1 --seed $SEED --gamma 2.0 --L 20 --max-L 20 \
    --init-scale 0.5 --init-mode prior_median --n-chains 64 \
    --num-samples 1000 --warmup 150 --warm-start 0 \
    --out astro/results/des_y1/pickles_priormed_g2_ws0_w150/seed$SEED
done

# === NUTS-dense warm=150 ===
for SEED in 0 1 2; do
  run "nuts_dense_w150_s$SEED" -m astro.run_numpyro_des \
    --mass dense --seed $SEED --n-chains 4 --warmup 150 \
    --num-samples 1000 --max-tree-depth 7 --target-accept 0.8 \
    --chain-method sequential \
    --out astro/results/des_y1/nuts_numpyro_dense_w150/seed$SEED
done

echo "=== $(date +%H:%M:%S) ALL DONE ===" | tee -a $LOG
