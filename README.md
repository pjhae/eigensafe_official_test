# EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment



- Paper: under review — arXiv:2509.17750
- Envs: custom MuJoCo/Box2D tasks registered in `envs/register.py`. XML assets load from `envs/mujoco/assets/` by default (no hard-coded absolute paths).
- Scripts: quick launchers in `scripts/` (CheetahLow, HopperHigh, AntBall, LunarLanderHard).

## Setup
```bash
conda env create -f environment.yml
conda activate eigensafe
pip install -e .
```

## Train
```bash
cd scripts
./train_cheetahlow.sh          # or train_hopperhigh.sh, train_antball.sh, train_lunarlanderhard.sh
```
Pass extra args to override defaults, e.g. `./train_cheetahlow.sh --seed 42`.

## Evaluate
```bash
python evaluate.py --env Halfcheetah-run-low-v5 --checkpoint results/<exp>/checkpoints/latest.pt --cuda
```

## Repo map (short)
- `envs/`: task definitions and assets
- `algo/`: SAC + eigenpair safety critic
- `train.py` / `evaluate.py`: entry points
- `results/`: logs, videos, checkpoints

## Citation
```bibtex
@article{jang2026eigensafe,
  title={EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment},
  author={Jang, Inkyu and Park, Jonghae and Cho, Sihyun and Mballo, Chams E and Tomlin, Claire J and Kim, H Jin},
  journal={arXiv preprint arXiv:2509.17750},
  year={2026}
}
```
