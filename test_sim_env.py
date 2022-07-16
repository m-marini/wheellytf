import logging

from tensorforce import Environment

from wheelly.envs import SimRobotEnv
from numpy.testing import assert_allclose, assert_array_equal, assert_approx_equal


def test_env1():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)

    environment:SimRobotEnv = Environment.create(
        environment=SimRobotEnv
    )
    states_space = environment.states()
    assert isinstance(states_space, dict)
    actions_space = environment.actions()
    assert isinstance(actions_space, dict)
    assert_allclose(environment.robot.position, (0,0))
    assert_allclose(environment.robot.linearVelocity, (0,0))
    environment.reset()
    actions = {
        "halt": 0,
        "direction": 0,
        "speed": 1,
        "sensor": 0
    }
    environment.execute(actions)
