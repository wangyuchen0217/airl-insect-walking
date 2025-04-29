import os
import argparse
from datetime import datetime
import torch
import logging
import sys
from common.base import LoggerWriter

import envs
from common.env import make_env
from algorithms.sac import SAC
from common.trainer import Trainer

def run(args):
    CUDA = 0

    # Create log directory.
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, "sac", f"{current_time}")
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

    env = make_env(args.env_id)
    env_test = make_env(args.env_id)
    device = torch.device(f"cuda:{CUDA}" if torch.cuda.is_available() and CUDA >= 0 else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(CUDA))
    else:
        print("Running on CPU")
    print(f"Process ID: {os.getpid()}")

    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        lr_actor=1e-4,
        lr_critic=1e-4,
        lr_alpha=1e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        seed=args.seed
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()

    # Save the final model.
    algo.save_models(trainer.model_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=10**6)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)