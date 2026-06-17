#!/bin/bash
# Fill in seeds 1, 2 for NUTS warmup sweep (already have seed0 + warm150).
# 12 runs (6 NUTS-dense + 6 NUTS-block), parallel across 6 GPUs (1-6).
# Balanced pairing to minimize max wall.

set -e
cd /home/yifanchen/research/AffineInvariantSamplers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python
mkdir -p astro/logs

# One run launcher
run_nuts() {
  local gpu=$1; local mass=$2; local warm=$3; local seed=$4
  local log=astro/logs/nuts_${mass}_w${warm}_s${seed}_gpu${gpu}_$(date +%H%M%S).log
  CUDA_VISIBLE_DEVICES=$gpu $PY -m astro.run_numpyro_des \
    --mass $mass --seed $seed --n-chains 4 --warmup $warm \
    --num-samples 1000 --max-tree-depth 7 --target-accept 0.8 \
    --chain-method sequential \
    --out astro/results/des_y1/nuts_numpyro_${mass}_warm${warm}/seed${seed} \
    > $log 2>&1
}

# GPU 1: dense w50 s1 (2.7h) + block w100 s2 (5.1h) = 7.8h
( run_nuts 1 dense 50 1 && run_nuts 1 block 100 2 ) &
# GPU 2: dense w50 s2 (2.7h) + block w200 s2 (4.4h) = 7.0h
( run_nuts 2 dense 50 2 && run_nuts 2 block 200 2 ) &
# GPU 3: dense w100 s1 (3.7h) + block w200 s1 (4.4h) = 8.0h
( run_nuts 3 dense 100 1 && run_nuts 3 block 200 1 ) &
# GPU 4: dense w100 s2 (3.7h) + block w100 s1 (5.1h) = 8.8h
( run_nuts 4 dense 100 2 && run_nuts 4 block 100 1 ) &
# GPU 5: dense w200 s1 (3.7h) + block w50 s2 (3.6h) = 7.3h
( run_nuts 5 dense 200 1 && run_nuts 5 block 50 2 ) &
# GPU 6: dense w200 s2 (3.7h) + block w50 s1 (3.6h) = 7.3h
( run_nuts 6 dense 200 2 && run_nuts 6 block 50 1 ) &

wait
echo "=== $(date +%H:%M:%S) ALL 12 NUTS MULTI-SEED RUNS DONE ==="
