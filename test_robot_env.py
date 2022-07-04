import numpy as np
from numpy import int8, ndarray, zeros,array
from tests.mocks import MockRobotEnv
from wheelly.envs.robot_wrapper import RobotWrapper
import gym.spaces as spaces
import numpy.testing as tnp

NO_TILINIG = 8
NO_TILES = 8 * 31
NO_SENSOR_FEATURES = NO_TILINIG * NO_TILES
NO_CONTACTS_FEATURES = 16 * 2
NO_FEATURES = NO_SENSOR_FEATURES + NO_CONTACTS_FEATURES

def test_space():
    base_env = MockRobotEnv()
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    wrapper = RobotWrapper(base_env)
    assert type(wrapper.observation_space) is spaces.MultiBinary
    assert wrapper.observation_space.n == NO_FEATURES

def test_observation():
    base_env = MockRobotEnv()
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    base_env.set_can_move_forward(0)
    wrapper = RobotWrapper(base_env)
    obs0 = base_env.observation()
    obs1 = wrapper.observation(obs0)
    assert type(obs1) is ndarray
    exp = zeros((NO_FEATURES), dtype=int8)
    exp[0] = 1
    exp[NO_TILES + 0] = 1
    exp[2 * NO_TILES + 0] = 1
    exp[3 * NO_TILES + 0] = 1
    exp[4 * NO_TILES + 0] = 1
    exp[5 * NO_TILES + 0] = 1
    exp[6 * NO_TILES + 0] = 1
    exp[7 * NO_TILES + 0] = 1
    exp[NO_SENSOR_FEATURES + 0] = 1
    tnp.assert_array_equal(obs1, exp)

def test_reset():
    base_env = MockRobotEnv()
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    base_env.set_can_move_forward(0)
    wrapper = RobotWrapper(base_env)
    obs1 = wrapper.reset()
    assert type(obs1) is ndarray
    exp = zeros((NO_FEATURES), dtype=int8)
    exp[0] = 1
    exp[NO_TILES + 0] = 1
    exp[2 * NO_TILES + 0] = 1
    exp[3 * NO_TILES + 0] = 1
    exp[4 * NO_TILES + 0] = 1
    exp[5 * NO_TILES + 0] = 1
    exp[6 * NO_TILES + 0] = 1
    exp[7 * NO_TILES + 0] = 1
    exp[NO_SENSOR_FEATURES + 0] = 1
    tnp.assert_array_equal(obs1, exp)

def test_action_space():
    base_env = MockRobotEnv()
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    base_env.set_can_move_forward(0)
    wrapper = RobotWrapper(base_env)
    space = wrapper.action_space
    assert type(space) is spaces.Dict
    assert type(space["halt"]) is spaces.Discrete
    assert type(space["direction"]) is spaces.Discrete
    assert type(space["speed"]) is spaces.Discrete
    assert type(space["sensor"]) is spaces.Discrete
    assert space["halt"].n == 2
    assert space["direction"].n == 25
    assert space["speed"].n == 9
    assert space["sensor"].n == 7

def test_action1():
    base_env = MockRobotEnv()
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    base_env.set_can_move_forward(0)
    wrapper = RobotWrapper(base_env)
    act0 = {
        "halt": np.array([1]),
        "direction": np.array([0]),
        "speed": np.array([0]),
        "sensor": np.array([0]),
    }
    act1 = wrapper.action(act0)
    tnp.assert_equal(act1, {
        "halt": np.array([1]),
        "direction": np.array([-180]),
        "speed": np.array([-1]),
        "sensor": np.array([-90])
        })

def test_action2():
    base_env = MockRobotEnv()
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    base_env.set_can_move_forward(0)
    wrapper = RobotWrapper(base_env)
    act0 = {
        "halt": np.array([0]),
        "direction": np.array([1]),
        "speed": np.array([1]),
        "sensor": np.array([1]),
    }
    act1 = wrapper.action(act0)
    tnp.assert_equal(act1, {
        "halt": np.array([0]),
        "direction": np.array([-165]),
        "speed": np.array([-0.75]),
        "sensor": np.array([-60])
        })
