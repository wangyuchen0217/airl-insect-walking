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
load the created scene `medauroidea_stick_insect.ttt` in CoppeliaSim.

#### 2. Run the AIRL Training Code (in another terminal)

```bash
conda activate coppeliasim
cd airl-insect-walking
python train_airl.py
```

## Experiments

This project includes five main experimental directions:

---
### 1. Expert Demonstration Generation

Expert demonstration are generated in CoppeliaSim based on the real stick insect, *Medauroidea extradentata*, walking trajectories. The demonstrations include:
- States: body-z, rpy, joint positions, and binary foot contact information.
- Actions: joint commands. 

Note: Demonstration generation is done with the CoppeliaSim UI with a defined `main_script.py` in folder `env` or `env_legloss`. The`expert.py` module provides the data preparation of the complete expert demonstration for the AIRL training. It is integrated in the `trian_airl.py`, thus no need to run it separately.

---
### 2. AIRL Learning from Expert Demonstrations

Adversarial Inverse Reinforcement Learning (AIRL) is applied to infer the underlying reward structure and the policy from expert data. The Discriminator learns to distinguish expert trajectories from policy-generated ones, while the policy is optimized using the recovered reward.

**(1) Run the AIRL training code:**
```bash
python train_airl.py
```

**(2) Test the trained Actor (policy) network:**
```
python test_airl.py
```
Note: Import the required `normalized_env` module and change the parameters based on the different tasks and needs.

**(3) Evaluation:**
Use `eval_plot.py`, `eval_plot.ipynb`, `gait_analysis.ipynb`, and `intra_limb analysis.ipynb` for quantitative analysis.

---
### 3. Policy Learning Generalization

We evaluate whether the learned reward enables policy generalization
to unseen velocity commands and environmental perturbations.


---
### 4. Cross-Dynamic Transfer

The learned reward is tested under modified dynamics,
including mass variations and friction changes,
to examine its robustness across dynamic conditions.


#### 5. Sim-to-Real Calibration

A sim-to-real calibration procedure is implemented to bridge
the gap between simulation and the physical hexapod robot.
Policy transfer performance is evaluated under real hardware constraints.
