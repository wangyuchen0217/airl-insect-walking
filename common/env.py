import gymnasium as gym

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
    

class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        # super().__init__(env)
        # # Get the maximum number of steps from env.spec instead of accessing private properties
        # self._max_episode_steps = getattr(env.spec, "max_episode_steps", None)

        # # If also want to enable TimeLimit (to ensure that it is automatically done when the step count is reached)
        # if self._max_episode_steps is not None:
        #     self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=self._max_episode_steps)

        self.scale = env.action_space.high
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale

    def step(self, action):
        return self.env.step(action * self.scale)
    

def normalize_expert_data(expert_data, env):
    scale = env.action_space.high
    expert_data['action'] = expert_data['action'] / scale
    return expert_data