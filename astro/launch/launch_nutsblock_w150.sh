#!/bin/bash
# Fill in NUTS-block warmup=150 for tab:desWarmupSweep — 3 seeds, parallel on GPUs 2, 4, 5.

set -e
cd /home/yifanchen/research/AffineInvariantSamplers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python
mkdir -p astro/logs

run_one() {
  local gpu=$1; local seed=$2
  local log=astro/logs/nutsblock_w150_s${seed}_gpu${gpu}_$(date +%H%M%S).log
  CUDA_VISIBLE_DEVICES=$gpu $PY -m astro.run_numpyro_des \
    --mass block --seed $seed --n-chains 4 --warmup 150 \
    --num-samples 1000 --max-tree-depth 7 --target-accept 0.8 \
    --chain-method sequential \
    --out astro/results/des_y1/nuts_numpyro_block_w150/seed$seed \
    > $log 2>&1
}

( run_one 2 0 ) &
( run_one 4 1 ) &
( run_one 5 2 ) &

wait
echo "=== $(date +%H:%M:%S) NUTS-block w=150 DONE ==="
