import gymnasium as gym
import numpy as np

gym.logger.set_level(40)


def make_env(env_id, test=False):
    if env_id == "Ant-v4" and not test:
        return NormalizedEnv(gym.make(env_id, 
                                      healthy_z_range=(0.26, 1.0)))
    elif env_id == "Ant-v4" and test:
        return NormalizedEnv(gym.make(env_id, 
                                      healthy_z_range=(0.26, 1.0), 
                                      render_mode="human"))
    elif not test:
        return NormalizedEnv(gym.make(env_id))
    else:
        return NormalizedEnv(gym.make(env_id, render_mode="human"))
    

# class NormalizedEnv(gym.Wrapper):

#     def __init__(self, env):
#         gym.Wrapper.__init__(self, env)
#         self._max_episode_steps = env._max_episode_steps

#         self.scale = env.action_space.high.copy()
#         self.action_space.high /= self.scale
#         self.action_space.low /= self.scale
#         # print to check the scale
#         print(f"scale: {self.scale}")
#         print(f"action space high: {self.action_space.high}")
#         print(f"action space low: {self.action_space.low}")

#     def step(self, action):
#         return self.env.step(action * self.scale)
    
#     def normalize_expert_data(self, expert_data):
#         expert_data['action'] = expert_data['action'] / self.scale
#         # print to check the scale
#         print(f"data scale: {self.scale}")
#         print(f"data action space high: {expert_data['action'].max()}")
#         print(f"data action space low: {expert_data['action'].min()}")
#         return expert_data 
    
class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps

        self.high = env.action_space.high
        self.low = env.action_space.low
        self.mid = (self.high + self.low) / 2
        self.scale = (self.high - self.low) / 2

        # Update the action space to normalized range
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self.scale.shape, dtype=np.float32
        )

        print(f"scale: {self.scale}")
        print(f"mid: {self.mid}")
        print(f"action space high: {self.action_space.high}")
        print(f"action space low: {self.action_space.low}")

    def step(self, action):
        real_action = action * self.scale + self.mid
        return self.env.step(real_action)

    def normalize_expert_data(self, expert_data):
        expert_data['action'] = (expert_data['action'] - self.mid) / self.scale
        expert_data['action'] = np.clip(expert_data['action'], -0.999, 0.999)
        print(f"normalized expert action high: {expert_data['action'].max()}")
        print(f"normalized expert action low: {expert_data['action'].min()}")
        return expert_data
