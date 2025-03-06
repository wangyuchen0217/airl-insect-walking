import os
import torch
import gymnasium as gym
import numpy as np
from datetime import datetime, timedelta
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter

from algorithms.airl import AIRL
from expert import load_expert_data, ExpertBuffer

# ======== Parameters (modify these as needed) =========
STATE_FILE = "Ant_states.pt"
ACTION_FILE = "Ant_actions.pt"
ENV_ID = "Ant-v4"
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
CUDA = True
SEED = 123
# ========================================================

class Trainer:
    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**7,
                 eval_interval=10**5, num_eval_episodes=5):
        self.env = env
        # self.env.seed(seed)

        self.env_test = env_test
        # self.env_test.seed(2**31 - seed)

        self.algo = algo
        self.log_dir = log_dir

        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        self.start_time = time()
        t = 0

        # Reset the environment properly.
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            state, _ = reset_out
        else:
            state = reset_out
        if isinstance(state, dict):
            state = state.get("observation", state)
        print("Initial state shape:", state.shape)

        for step in range(1, self.num_steps + 1):
            state, t = self.algo.step(self.env, state, t, step)
            if self.algo.is_update(step):
                self.algo.update(self.writer)
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(os.path.join(self.model_dir, f'step{step}'))
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0
        for _ in range(self.num_eval_episodes):
            reset_out = self.env_test.reset()
            if isinstance(reset_out, tuple):
                state, _ = reset_out
            else:
                state = reset_out
            if isinstance(state, dict):
                state = state.get("observation", state)
            episode_return = 0.0
            done = False
            while not done:
                action = self.algo.exploit(state)
                next_step = self.env_test.step(action)
                state, reward, terminated, truncated, _ = next_step
                done = terminated or truncated
                episode_return += reward
            mean_return += episode_return / self.num_eval_episodes
        self.writer.add_scalar('return/test', mean_return, step)
        print(f"Num steps: {step:<6}   Return: {mean_return:<5.1f}   Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))

def main():
    # Create training and testing environments.
    env = gym.make(ENV_ID)
    env_test = gym.make(ENV_ID)
    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")
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

    # Create log directory.
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", ENV_ID, "airl", f"seed{SEED}-{current_time}")

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

if __name__ == "__main__":
    main()
