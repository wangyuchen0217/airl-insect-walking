# AIRL-Insect-Walking

This repository implements Adversarial Inverse Reinforcement Learning (AIRL)
for insect-inspired hexapod locomotion using MuJoCo.
The goal is to learn transferable reward functions from biological gait data.

## Project Structure
airl-insect-walking/
├── algorithms/               # modules of airl/ppo etc.
├── common/               # base modules and CoppeliaSim environment interfaces 
├── env/               # CoppeliaSim main scripts
├── env_legloss/               # CoppeliaSim main scripts for leg loss scenarios
├── evaluation/               # evaluation records
├── expert/               # expert demonstrations
├── logs/               # training logs
├── networks/               # modules for building Actor, Critic, Discriminator neural networks
├── ros2_ws/               # ros2 interfaces and logs
├── .gitignore
├── eval_plot.ipynb               # plots the evaluation results of foot trajectories etc. (part of the trail)
├── eval_plot.py               # plots the evaluation results of body pose, velocity and gaits etc. (the whole trail)
├── expert.py               # load the expert demonstration (add contact columns, create a symmetric action bounds)
├── gait_analysis.ipynb               # plots the duty factor, phase difference, synchronization matrix etc. (part of the trail)
├── intra-limb analysis.ipynb               # r
├── README.md
├── test_airl.py               # r
├── train_airl.py               # r
├── train_ppo.py               # r
└── visualize_reward.py               # r


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
