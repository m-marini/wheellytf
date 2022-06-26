from gym.envs.registration import register

register(
    id='wheelly/WheellyEnv-v0',
    entry_point='wheelly.envs.wheelly_env:WheellyEnv',
    max_episode_steps=300,
)

register(
    id='wheelly/RobotEnv-v0',
    entry_point='wheelly.envs.robot_env:RobotEnv',
)