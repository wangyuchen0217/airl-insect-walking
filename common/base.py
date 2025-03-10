from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


# Redirect stdout to the log file
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # Only log non-empty messages
            self.level(message)

    def flush(self):
        pass  # Required for compatibility

def log_parameters(ENV_ID, STATE_FILE, ACTION_FILE, ROLLOUT_LENGTH, NUM_STEPS, EVAL_INTERVAL, 
                   GAMMA, MIX_BUFFER, BATCH_SIZE, LR_ACTOR, LR_CRITIC, LR_DISC, 
                   UNITS_ACTOR, UNITS_CRITIC, UNITS_DISC_R, UNITS_DISC_V, 
                   EPOCH_PPO, EPOCH_DISC, CLIP_EPS, LAMBDA, COEF_ENT, MAX_GRAD_NORM, SEED):
    print(f"Env: {ENV_ID}, {STATE_FILE}, {ACTION_FILE}, Seed: {SEED}") 
    print(f"Rollout length: {ROLLOUT_LENGTH}")
    print(f"Num steps: {NUM_STEPS}, Eval interval: {EVAL_INTERVAL}")
    print(f"Gamma: {GAMMA}, Mix buffer: {MIX_BUFFER}, Batch size: {BATCH_SIZE}")
    print(f"lr_actor: {LR_ACTOR}, lr_critic: {LR_CRITIC}, lr_disc: {LR_DISC}")
    print(f"Units Actor: {UNITS_ACTOR}, Units Critic: {UNITS_CRITIC}")
    print(f"Units Disc R: {UNITS_DISC_R}, Units Disc V: {UNITS_DISC_V}")
    print(f"Epoch ppo: {EPOCH_PPO}, Epoch disc: {EPOCH_DISC}")
    print(f"Clip Epsc: {CLIP_EPS}, Lambda: {LAMBDA}, Coef Ent: {COEF_ENT}, Max Grad Norm: {MAX_GRAD_NORM}") 