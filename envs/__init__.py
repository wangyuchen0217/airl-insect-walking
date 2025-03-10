from gymnasium.envs.registration import register

register(
    id='AntEnv-v0',
    entry_point='envs.ant_v4:AntEnv',
)