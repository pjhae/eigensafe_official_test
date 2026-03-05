#!/usr/bin/env bash
set -euo pipefail

# Hopper run-high benchmark with slightly smaller batch
python3 train.py \
  --exp_name "exp0305-hopper-run-high-v5-bsz256" \
  --env_name "Hopper-run-high-v5" \
  --seed 202 \
  --batch_size 256 \
  --replay_size 100000 \
  --hidden_size 256 \
  --lambda_value 600 \
  --episodes_per_epoch 8 \
  --gradient_steps_per_epoch 48 \
  --start_steps 2000 \
  --eval_epoch_ratio 40 \
  --save_epoch_ratio 200 \
  --num_episodes 150000 "$@"
