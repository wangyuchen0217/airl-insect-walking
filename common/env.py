import gymnasium as gym

gym.logger.set_level(40)


def make_env(env_id, healthy_z_range=(0.26, 1.0), healthy_reward=0, use_contact_forces=False, render_mode="human"):
    return NormalizedEnv(gym.make(env_id, 
                                  healthy_z_range=healthy_z_range, 
                                  healthy_reward=healthy_reward, 
                                  use_contact_forces=use_contact_forces,
                                  render_mode=render_mode))

# def make_env(env_id, render_mode="human"):
#     return NormalizedEnv(gym.make(env_id, 
#                                   render_mode=render_mode))


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        self.scale = env.action_space.high
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale

    def step(self, action):
        return self.env.step(action * self.scale)
    

def normalize_expert_data(expert_data, env):
    scale = env.action_space.high
    expert_data['action'] = expert_data['action'] / scale
    return expert_data