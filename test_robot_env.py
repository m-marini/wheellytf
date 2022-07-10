import numpy as np
from numpy import int8, ndarray, zeros
from tensorforce import Environment
from wheelly.envs import EncodedRobotEnv, MockRobotEnv
from numpy.testing import assert_equal

NO_TILINIG = 8
NO_TILES = 8 * 31
NO_SENSOR_FEATURES = NO_TILINIG * NO_TILES
NO_CONTACTS_FEATURES = 16 * 2
NO_FEATURES = NO_SENSOR_FEATURES + NO_CONTACTS_FEATURES

def test_space():
    base_env = Environment.create(
            environment=MockRobotEnv
    )
    wrapper:EncodedRobotEnv = Environment.create(
            environment=EncodedRobotEnv,
            env=base_env
    )
    s = wrapper.states()
    assert s == {
        "type": "bool",
        "shape": NO_FEATURES
    }

def test_reset():
    base_env: MockRobotEnv= Environment.create(
            environment=MockRobotEnv
    )
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    base_env.set_can_move_forward(0)
    wrapper:EncodedRobotEnv = Environment.create(
            environment=EncodedRobotEnv,
            env=base_env
    )

    obs1 = wrapper.reset()
    assert isinstance(obs1, ndarray)
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
    assert_equal(obs1, exp)

def test_step1():
    base_env: MockRobotEnv= Environment.create(
            environment=MockRobotEnv
    )
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    base_env.set_can_move_forward(0)
    wrapper:EncodedRobotEnv = Environment.create(
            environment=EncodedRobotEnv,
            env=base_env
    )

    wrapper.reset()
    action = {
        "halt": np.array([0]),
        "direction": np.array([1]),
        "speed": np.array([1]),
        "sensorAction": np.array([1]),
    }
    obs, done, reward = wrapper.execute(action)
    assert isinstance(obs, ndarray)
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
    assert_equal(obs, exp)

    act1 = base_env.act()
    assert_equal(act1, {
        "halt": np.array([0]),
        "direction": np.array([-165]),
        "speed": np.array([-0.75]),
        "sensorAction": np.array([-60])
    })

def test_step2():
    base_env: MockRobotEnv= Environment.create(
            environment=MockRobotEnv
    )
    base_env.set_distance(0)
    base_env.set_sensor(-90)
    base_env.set_can_move_forward(0)
    wrapper:EncodedRobotEnv = Environment.create(
            environment=EncodedRobotEnv,
            env=base_env
    )

    wrapper.reset()
    action = {
        "halt": np.array([1]),
        "direction": np.array([0]),
        "speed": np.array([0]),
        "sensorAction": np.array([0]),
    }
    obs, done, reward = wrapper.execute(action)
    assert isinstance(obs, ndarray)
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
    assert_equal(obs, exp)

    act1 = base_env.act()
    assert_equal(act1, {
        "halt": np.array([1]),
        "direction": np.array([-180]),
        "speed": np.array([-1]),
        "sensorAction": np.array([-90])
        })
