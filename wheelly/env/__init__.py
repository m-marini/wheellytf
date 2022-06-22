from gym.envs.registration import register

register(
    id='wheelly/WheellyEnv-v0',
    entry_point='wheelly.env.wheelly_env:WheellyEnv',
    max_episode_steps=300,
)