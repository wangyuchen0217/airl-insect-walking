# AIRL-Insect-Walking

This project implements Adversarial Inverse Reinforcement Learning (AIRL) and Soft Actor-Critic (SAC) in MuJoCo-based continuous control environments. The framework supports both imitation learning and standard reinforcement learning pipelines. It is designed for investigating locomotion strategies, including future applications to a custom stick insect simulation model.

This repository implements Adversarial Inverse Reinforcement Learning (AIRL)
for insect-inspired hexapod locomotion using MuJoCo.
The goal is to learn transferable reward functions from biological gait data.

## Project Structure
airl-insect-walking/
├── algorithms/               # modules of airl/ppo etc.
├── common/               # 
├── irl/                # AIRL / MaxEnt IRL algorithms
├── policies/           # Policy networks (PPO, SAC)
├── data/               # Demonstration trajectories
├── configs/            # YAML configuration files
├── scripts/            # Training and evaluation scripts
└── README.md


## Installation

```bash
conda create -n airl python=3.10
conda activate airl
pip install -r requirements.txt

```md
- MuJoCo >= 2.3
- CUDA 12.3 (optional, for GPU training)


## Quick Start

Train AIRL on Ant-v4:
```bash
python scripts/train_airl.py --env Ant-v4 --config configs/airl.yaml

python scripts/eval_policy.py --checkpoint logs/ant/airl/model.pt
