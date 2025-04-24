from gymnasium.envs.registration import register

register(
    id='StickInsect-v4',
    entry_point='envs.StickInsectEnv:StickInsectEnv',
    max_episode_steps=3000,
)