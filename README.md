# EigenSafe (unofficial)

EigenSafe is a PyTorch implementation of a safety-aware Soft Actor-Critic (SAC) variant that augments the policy with an eigenfunction-based Lagrangian term (`Psi`). It trains on custom Gymnasium environments (HalfCheetah/Hopper/Ant variants and a continuous LunarLander-safety task) registered in `envs/`.

## Repository layout
- `train.py` — main training entrypoint using args from `algo/arguments.py`.
- `evaluate.py` — loads saved checkpoints and renders rollouts with Psi traces.
- `algo/` — SAC agent, replay buffer, eigen Psi model, and argument parser.
- `envs/` — custom Gymnasium registrations (Mujoco + Box2D safety).
- `scripts/` — ready-to-run bash launchers (`train_001.sh` … `train_004.sh`).
- `results/` — default output root (tensorboard runs, videos, checkpoints).

## Setup
1) Create an environment with PyTorch + Gymnasium + Mujoco/Box2D stack, e.g.:
```bash
conda create -n eigensafe python=3.10 -y
conda activate eigensafe
pip install torch gymnasium[mujoco] box2d-py tensorboard imageio matplotlib
```
2) Ensure Mujoco drivers are available (use `MUJOCO_GL=egl` for headless; already set in scripts).

## Training
Use one of the curated launchers from `scripts/`:
```bash
cd eigensafe_official_test/scripts
./train_001.sh          # LunarLander-safety baseline (defaults)
./train_002.sh          # HalfCheetah-run-low-v5 safety-focused
./train_003.sh          # Hopper-run-high-v5 with smaller batches
./train_004.sh          # Ant-ball-v5 with higher entropy temperature
```
All launchers forward extra CLI flags to `train.py`, so you can override any argument, e.g.:
```bash
./train_001.sh --seed 42 --lambda_value 900
```

Artifacts per run land in `results/<exp_name>/`:
- `runs/` TensorBoard logs,
- `video/` episode videos,
- `checkpoints/` SAC and Psi weights,
- a snapshot of key source folders for reproducibility.

## Evaluation
1) Set `env_name`, `exp_name`, and `num_episode` near the top of `evaluate.py`.
2) Run:
```bash
python evaluate.py --cuda    # drop the flag to force CPU
```
This renders side-by-side frames with Psi curves and writes an MP4 plus per-step PNGs to `results/<exp_name>/plots/`.

## Notes
- Default hyperparameters live in `algo/arguments.py`; anything not specified on the CLI falls back to those values.
- Environments are registered in `envs/register.py`; add new IDs there if you introduce more tasks.
