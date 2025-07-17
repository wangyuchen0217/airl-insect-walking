import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"
import sys
import torch
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from datetime import datetime
from algorithms.airl import AIRL
from expert import load_expert_data, ExpertBuffer
from common.trainer import Trainer
import logging
from common.base import LoggerWriter
from common.base import log_parameters
# from common.env import CoppeliaSimEnv
from common.normalized_a_env import CoppeliaSimEnv
from common.buffer import SerializedBuffer
import torch.utils.tensorboard

# ======== Parameters (modify these as needed) =========
NAME = "StickInsect"
EXPERT_FILE = "expert.csv"
ENV_ID = "Medauroidea"
CUDA = 0
ROLLOUT_LENGTH = 1000 # 3000
NUM_STEPS = 17*10**4 
EVAL_INTERVAL = 5000
GAMMA = 0.995
MIX_BUFFER = 1
BATCH_SIZE = 64
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
LR_DISC = 1e-4
UNITS_ACTOR = (64, 64)
UNITS_CRITIC = (64, 64)
UNITS_DISC_R = (100, 100)
UNITS_DISC_V = (100, 100)
EPOCH_PPO = 50
EPOCH_DISC = 3
CLIP_EPS = 0.2
LAMBDA = 0.97
COEF_ENT = 0.05
MAX_GRAD_NORM = 10.0
SEED = 0
# ========================================================

def main():
    # Create log directory.
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", ENV_ID, "airl", f"{current_time}")
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

    # set up the environment and communication
    OnTimeStep=True
    env = CoppeliaSimEnv(port=23000, OnTimeStep=OnTimeStep)
    env_test = CoppeliaSimEnv(port=23001, OnTimeStep=OnTimeStep)

    device = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() and CUDA >= 0 else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(CUDA))
    else:
        print("Running on CPU")
    print(f"Process ID: {os.getpid()}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log_parameters(ENV_ID, EXPERT_FILE, ROLLOUT_LENGTH, NUM_STEPS, EVAL_INTERVAL, 
                   GAMMA, MIX_BUFFER, BATCH_SIZE, LR_ACTOR, LR_CRITIC, LR_DISC, 
                   UNITS_ACTOR, UNITS_CRITIC, UNITS_DISC_R, UNITS_DISC_V, 
                   EPOCH_PPO, EPOCH_DISC, CLIP_EPS, LAMBDA, COEF_ENT, MAX_GRAD_NORM, SEED)

    # Load expert data from .pt files and wrap into an ExpertBuffer.
    expert_data = load_expert_data(EXPERT_FILE, save_npz=False)
    expert_data = env.normalize_expert_data(expert_data)
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
        eval_interval=EVAL_INTERVAL
    )
    trainer.train()

    # Save the final model.
    algo.save_models(trainer.model_dir)

if __name__ == "__main__":
    main()
