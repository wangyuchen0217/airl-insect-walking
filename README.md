# AIRL-Insect-Walking

This repository implements Adversarial Inverse Reinforcement Learning (AIRL)
for insect-inspired hexapod locomotion using MuJoCo.
The goal is to learn transferable reward functions from biological gait data.

## Project Structure
```bash
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
├── environment.yml
├── eval_plot.ipynb               # plots the evaluation results of foot trajectories etc. (part of the trail)
├── eval_plot.py               # plots the evaluation results of body pose, velocity and gaits etc. (the whole trail)
├── expert.py               # load the expert demonstration (add contact columns, create a symmetric action bounds)
├── gait_analysis.ipynb               # plots the duty factor, phase difference, synchronization matrix etc. (part of the trail)
├── intra-limb analysis.ipynb               # plots the cyclograms etc. (part of the trail)
├── README.md
├── test_airl.py               # tests a trained AIRL/PPO Actor model in a CoppeliaSim environment
├── train_airl.py               # trains an AIRL agent in a CoppeliaSim environment using expert demonstrations
└── train_ppo.py               # trains a PPO agent in a CoppeliaSim environment
```

## Installation

1. Install CoppeliaSim v4.10.0
2. Enable ZMQ remote API
3. Navigate to the project root directory, and create conda environment:

```bash
cd airl-insect-walking
conda env export > environment.yml
conda activate coppeliasim
```

## Quick Start
#### 1. Start CoppeliaSim (in a new terminal)

```bash
conda activate coppeliasim
cd /path/to/CoppeliaSim
./coppeliaSim
```

#### 2. Run the Code (in another terminal)

```bash
conda activate coppeliasim
cd airl-insect-walking
python train_airl.py
```

