#!/bin/bash
# Fill in γ=2 ws=0 warm=50 and warm=100 multi-seed cells.
# 6 runs across 4 GPUs (0,1,2,3). Wall ETA ~2.2h.

set -e
cd /home/yifanchen/research/AffineInvariantSamplers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSIST_CACHE=/home/yifanchen/.cache/jax_pickles_des
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python
mkdir -p astro/logs

run_pkl() {
  local gpu=$1; local warm=$2; local seed=$3
  local log=astro/logs/pkl_g2_ws0_w${warm}_s${seed}_gpu${gpu}_$(date +%H%M%S).log
  CUDA_VISIBLE_DEVICES=$gpu $PY -m astro.run_pickles \
    --problem des_y1 --seed $seed --gamma 2.0 --L 20 --max-L 20 \
    --init-scale 0.5 --init-mode prior_median --n-chains 64 \
    --num-samples 1000 --warmup $warm --warm-start 0 \
    --out astro/results/des_y1/pickles_priormed_g2_ws0_w${warm}/seed${seed} \
    > $log 2>&1
}

# GPU 0: warm=50 seed0, warm=100 seed0
( run_pkl 0 50 0 && run_pkl 0 100 0 ) &
# GPU 1: warm=50 seed1, warm=100 seed1
( run_pkl 1 50 1 && run_pkl 1 100 1 ) &
# GPU 2: warm=50 seed2
( run_pkl 2 50 2 ) &
# GPU 3: warm=100 seed2
( run_pkl 3 100 2 ) &

wait
echo "=== $(date +%H:%M:%S) ALL DONE ==="
