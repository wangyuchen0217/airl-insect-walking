# AIRL-Insect-Walking

This project implements Adversarial Inverse Reinforcement Learning (AIRL) and Soft Actor-Critic (SAC) in MuJoCo-based continuous control environments. The framework supports both imitation learning and standard reinforcement learning pipelines. It is designed for investigating locomotion strategies, including future applications to a custom stick insect simulation model.


## Overview

- **Algorithms**: 
  - [x] AIRL (Adversarial Inverse Reinforcement Learning)
  - [x] SAC (Soft Actor-Critic)

- **Simulation Environment**: [MuJoCo](https://mujoco.org/)
- **Verified Environments for AIRL**:
  - Ant-v4: `logs/Ant-v4/airl/20250314-2354` and `logs/Ant-v4/airl/20250315-1418`
  - Hopper-v4: `logs/Hopper-v4/airl/20250313-1539`
- **Verified Environments for SAC**:
  - Ant-v4: `logs/Ant-v4/sac/20250314-1353` and `logs/Ant-v4/sac/20250314-1354`
  - Hopper-v4: `logs/Hopper-v4/sac/20250314-1353`
- **In Progress**:
  - Custom stick insect locomotion model (currently under development and debugging)
    **xml**：
    - StickInsect-v1 integrator="implicitfast", actuator: pos & vel
    - StickInsect-v4 integrator="RK4", actuator: pos
    - StickInsect-v5 integrator="RK4", actuator: pos, ctrlrange revised
    **env**:
    - StickInsect-v4
    - StickInsect-v5 action=data.ctrl - qpos[:N]
    **expert data**：
    - SrickInsect_states/actions.pt: all 3 insects, v4 env
    - SrickInsect_states/actions_v1.pt: w/o 1st insect, v4 env
    - SrickInsect_states/actions_v2.pt: w/o 1st insect, v5 env, v5 xml, with xy
    - SrickInsect_states/actions_v3.pt: w/o 1st insect, v5 env, v5 xml, without xy