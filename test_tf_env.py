import logging

from tensorforce import Environment

from wheelly.envs import MockRobotEnv


def test_env1():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)

    environment = Environment.create(
        environment=MockRobotEnv
    )
    states_space = environment.states()
    assert isinstance(states_space, dict)
    actions_space = environment.actions()
    assert isinstance(actions_space, dict)
