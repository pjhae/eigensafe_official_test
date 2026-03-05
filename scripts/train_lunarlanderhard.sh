#!/usr/bin/env bash
set -euo pipefail

# HalfCheetah run-low variant focused on safety
python3 train.py \
  --exp_name "exp0305-halfcheetah-run-low-v5-lambda800" \
  --env_name "Halfcheetah-run-low-v5" \
  --seed 101 \
  --lambda_value 800 \
  --epsilon 5e-4 \
  --gamma_target 0.95 \
  --episodes_per_epoch 5 \
  --gradient_steps_per_epoch 32 \
  --start_steps 5000 \
  --eval_epoch_ratio 50 \
  --save_epoch_ratio 200 \
  --num_episodes 200000 "$@"
