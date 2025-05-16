from gymnasium.envs.registration import register

# integrator="RK4" actuator: pos
register(
    id='StickInsect-v4',
    entry_point='envs.StickInsectEnv_v4:StickInsectEnv',
    max_episode_steps=3000,
)

# integrator="RK4" actuator: pos
register(
    id='StickInsect-v5',
    entry_point='envs.StickInsectEnv_v5:StickInsectEnv',
    max_episode_steps=3000,
)