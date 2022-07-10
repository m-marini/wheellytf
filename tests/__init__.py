from gym.envs.registration import register

register(
    id='tests/MockRobot-v0',
    entry_point='tests.mocks:MockRobotEnv',
    max_episode_steps=10000,
)
