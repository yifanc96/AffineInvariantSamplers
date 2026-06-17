#!/bin/bash
# Parallel warm=150 sweep across 6 GPUs (0, 1, 2, 4, 5, 6).
# Each GPU runs its own sequential batch of 1-3 runs.
# Wall ETA: ~3.7h (NUTS-dense bottleneck on each NUTS GPU).

set -e
cd /home/yifanchen/research/AffineInvariantSamplers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSIST_CACHE=/home/yifanchen/.cache/jax_pickles_des
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python
mkdir -p astro/logs

launch_pickles_batch() {
  local gpu=$1; local cfg=$2; local gamma=$3; local ws=$4
  local log=astro/logs/w150_${cfg}_gpu${gpu}_$(date +%H%M%S).log
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    for SEED in 0 1 2; do
      echo "=== $(date +%H:%M:%S) GPU$gpu $cfg seed$SEED ===" >> $log
      $PY -m astro.run_pickles \
        --problem des_y1 --seed $SEED --gamma $gamma --L 20 --max-L 20 \
        --init-scale 0.5 --init-mode prior_median --n-chains 64 \
        --num-samples 1000 --warmup 150 --warm-start $ws \
        --out astro/results/des_y1/pickles_priormed_${cfg}_w150/seed$SEED \
        >> $log 2>&1
    done
    echo "=== $(date +%H:%M:%S) GPU$gpu $cfg DONE ===" >> $log
  ) &
}

launch_nuts_one() {
  local gpu=$1; local seed=$2
  local log=astro/logs/w150_nutsdense_seed${seed}_gpu${gpu}_$(date +%H%M%S).log
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "=== $(date +%H:%M:%S) GPU$gpu NUTS-dense seed$seed ===" >> $log
    $PY -m astro.run_numpyro_des \
      --mass dense --seed $seed --n-chains 4 --warmup 150 \
      --num-samples 1000 --max-tree-depth 7 --target-accept 0.8 \
      --chain-method sequential \
      --out astro/results/des_y1/nuts_numpyro_dense_w150/seed$seed \
      >> $log 2>&1
    echo "=== $(date +%H:%M:%S) GPU$gpu NUTS-dense seed$seed DONE ===" >> $log
  ) &
}

# Launch 6 parallel batches:
launch_pickles_batch 0  g05_ws0    0.5  0     # GPU0: PICKLES γ=0.5 ws=0   × 3 seeds
launch_pickles_batch 1  g05_ws200  0.5  200   # GPU1: PICKLES γ=0.5 ws=200 × 3 seeds
launch_pickles_batch 2  g2_ws0     2.0  0     # GPU2: PICKLES γ=2   ws=0   × 3 seeds
launch_nuts_one      4  0                      # GPU4: NUTS-dense seed0
launch_nuts_one      5  1                      # GPU5: NUTS-dense seed1
launch_nuts_one      6  2                      # GPU6: NUTS-dense seed2

wait
echo "=== $(date +%H:%M:%S) ALL PARALLEL BATCHES DONE ==="
