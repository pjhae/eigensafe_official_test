#!/usr/bin/env bash
set -euo pipefail

# Ant-ball control with higher entropy temperature start
python3 train.py \
  --exp_name "exp0305-ant-ball-v5-alpha03" \
  --env_name "Ant-ball-v5" \
  --seed 303 \
  --alpha 0.03 \
  --automatic_entropy_tuning True \
  --gamma 0.995 \
  --lr 0.0003 \
  --tau 0.01 \
  --batch_size 256 \
  --replay_size 200000 \
  --hidden_size 512 \
  --lambda_value 1000 \
  --episodes_per_epoch 6 \
  --gradient_steps_per_epoch 96 \
  --start_steps 10000 \
  --eval_epoch_ratio 40 \
  --save_epoch_ratio 150 \
  --num_episodes 120000 "$@"
