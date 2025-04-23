from gymnasium.envs.registration import register

register(
    id='StickInsectEnv-v4',
    entry_point='envs.StickInsectEnv:StickInsectEnv',
)