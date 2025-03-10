import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"
import sys
import torch
import envs
import gymnasium as gym
import numpy as np
from datetime import datetime
from algorithms.airl import AIRL
from expert import load_expert_data, ExpertBuffer
from common.trainer import Trainer
import logging
from common.base import LoggerWriter

# ======== Parameters (modify these as needed) =========
STATE_FILE = "HalfCheetah_states.pt"
ACTION_FILE = "HalfCheetah_actions.pt"
ENV_ID = "HalfCheetah-v4"
CUDA = 0
ROLLOUT_LENGTH = 2048
NUM_STEPS = 10**7
EVAL_INTERVAL = 10**5
GAMMA = 0.995
MIX_BUFFER = 1
BATCH_SIZE = 64
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_DISC = 3e-4
UNITS_ACTOR = (64, 64)
UNITS_CRITIC = (64, 64)
UNITS_DISC_R = (100, 100)
UNITS_DISC_V = (100, 100)
EPOCH_PPO = 10
EPOCH_DISC = 5
CLIP_EPS = 0.2
LAMBDA = 0.97
COEF_ENT = 0.0
MAX_GRAD_NORM = 10.0
SEED = 123
# ========================================================

def main():
    # Create log directory.
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", ENV_ID, "airl", f"seed{SEED}-{current_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, "training_process.log")
    logging.basicConfig(
        filename=log_filename,    
        level=logging.INFO,
        format='%(message)s',
        filemode='w'
        )
    sys.stdout = LoggerWriter(logging.info)
    print(f"Logging started at {current_time}")

    # Create training and testing environments.
    env = gym.make(ENV_ID)
    env_test = gym.make(ENV_ID)
    device = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() and CUDA >= 0 else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(CUDA))
    else:
        print("Running on CPU")
    print(f"Process ID: {os.getpid()}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load expert data from .pt files and wrap into an ExpertBuffer.
    expert_data = load_expert_data(STATE_FILE, ACTION_FILE, save_npz=False)
    expert_buffer = ExpertBuffer(expert_data, device)
    print(f"Expert buffer size: {expert_buffer.size}")

    # Create AIRL agent.
    algo = AIRL(
        buffer_exp=expert_buffer,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        seed=SEED,
        gamma=GAMMA,
        rollout_length=ROLLOUT_LENGTH,
        mix_buffer=MIX_BUFFER,
        batch_size=BATCH_SIZE,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        lr_disc=LR_DISC,
        units_actor=UNITS_ACTOR,
        units_critic=UNITS_CRITIC,
        units_disc_r=UNITS_DISC_R,
        units_disc_v=UNITS_DISC_V,
        epoch_ppo=EPOCH_PPO,
        epoch_disc=EPOCH_DISC,
        clip_eps=CLIP_EPS,
        lambd=LAMBDA,
        coef_ent=COEF_ENT,
        max_grad_norm=MAX_GRAD_NORM
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=NUM_STEPS,
        eval_interval=EVAL_INTERVAL,
        seed=SEED
    )
    trainer.train()

    # Save the final model.
    algo.save_models(trainer.model_dir)

if __name__ == "__main__":
    main()
