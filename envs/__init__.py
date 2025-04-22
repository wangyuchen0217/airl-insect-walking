from gymnasium.envs.registration import register

register(
    id='StickInsectEnv-v0',
    entry_point='envs.StickInsectEnv:StickInsectEnv',
)