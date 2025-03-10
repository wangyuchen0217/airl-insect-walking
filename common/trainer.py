import os
from datetime import timedelta
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**7,
                 eval_interval=10**5, num_eval_episodes=5):
        self.env = env
        self.seed = seed

        self.env_test = env_test

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
        reset_out = self.env.reset(seed=self.seed)
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
                # self.algo.save_models(os.path.join(self.model_dir, f'step{step}'))
        sleep(10)


    def evaluate(self, step):
        mean_return = 0.0
        for _ in range(self.num_eval_episodes):
            reset_out = self.env_test.reset(seed=2**31 - self.seed)
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