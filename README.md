# EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment

This repository provides the official implementation of **EigenSafe** for Safe Reinforcement Learning,
a spectral framework for learning-based probabilistic safety assessment.

For more details, please refer to the [paper](https://arxiv.org/pdf/2509.17750) and the README below.
 
- **Project Page:** [https://eigen-safe.github.io](https://eigen-safe.github.io)  
- **Envs:** MuJoCo/Box2D tasks registered in `envs/register.py`. 
- **Scripts:** quick launchers in `scripts/`.

## Setup
```bash
conda env create -f environment.yml
conda activate eigensafe
pip install -e .
```

## Train

Run the training scripts from the repository root:

```bash
./scripts/train_{cheetahlow,hopperhigh,antball,lunarlanderhard}.sh
```
Pass extra args to override defaults, e.g. `./train_cheetahlow.sh --seed 42`.

## Evaluate

Select an experiment in `results/`, choose the desired episode checkpoint, and run:

```bash
python evaluate.py
```

## Repo map
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
