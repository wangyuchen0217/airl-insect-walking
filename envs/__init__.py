from gymnasium.envs.registration import register

# integrator="RK4" actuator: pos
register(
    id='StickInsect-v0',
    entry_point='envs.StickInsectEnv_v0:StickInsectEnv',
    max_episode_steps=1000,
)