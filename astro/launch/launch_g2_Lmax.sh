#!/bin/bash
# PICKLES γ=2 L_max=50 and L_max=128 multi-seed (3 seeds each).
# Matches tab:desMainN1k config: fid_ball init, ws=0, warm=500, N=1000.
# 6 runs across 4 GPUs. Wall ETA ~3-4h.

set -e
cd /home/yifanchen/research/AffineInvariantSamplers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PERSIST_CACHE=/home/yifanchen/.cache/jax_pickles_des
PY=/home/yifanchen/miniconda3/envs/gpu/bin/python
mkdir -p astro/logs

# Wait for current γ=2 ws=0 warm=50/100 runs to finish before starting
while pgrep -f "run_pickles.*pickles_priormed_g2_ws0_w" > /dev/null; do
  sleep 60
done
echo "=== $(date +%H:%M:%S) prior runs done, launching γ=2 L_max sweep ==="

run_pkl() {
  local gpu=$1; local Lmax=$2; local seed=$3
  local log=astro/logs/pkl_g2_L${Lmax}_s${seed}_gpu${gpu}_$(date +%H%M%S).log
  CUDA_VISIBLE_DEVICES=$gpu $PY -m astro.run_pickles \
    --problem des_y1 --seed $seed --gamma 2.0 --L $Lmax --max-L $Lmax \
    --init-scale 0.5 --init-mode fid_ball --n-chains 64 \
    --num-samples 1000 --warmup 500 --warm-start 0 \
    --out astro/results/des_y1/pickles_g2_maxL${Lmax}_init05/seed${seed} \
    > $log 2>&1
}

# 6 runs across 4 GPUs:
# GPU 0: L=50 s0, L=128 s0
( run_pkl 0 50 0 && run_pkl 0 128 0 ) &
# GPU 1: L=50 s1, L=128 s1
( run_pkl 1 50 1 && run_pkl 1 128 1 ) &
# GPU 2: L=50 s2
( run_pkl 2 50 2 ) &
# GPU 3: L=128 s2
( run_pkl 3 128 2 ) &

wait
echo "=== $(date +%H:%M:%S) γ=2 L_max sweep done ==="
