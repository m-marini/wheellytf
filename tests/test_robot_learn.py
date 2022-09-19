from numpy import ndarray
from tensorforce import Agent, Environment

from wheelly.envs import EncodedRobotEnv, MockRobotEnv

NO_TILINIG = 8
NO_TILES = 8 * 31
NO_SENSOR_FEATURES = NO_TILINIG * NO_TILES
NO_CONTACTS_FEATURES = 16 * 2
NO_FEATURES = NO_SENSOR_FEATURES + NO_CONTACTS_FEATURES

def test_learn():
    base_env = Environment.create(
            environment=MockRobotEnv
    )
    wrapper:EncodedRobotEnv = Environment.create(
            environment=EncodedRobotEnv,
            env=base_env
    )
    agent:Agent  = Agent.create(
        agent='random',
        environment=wrapper)
    states = wrapper.reset()
    assert isinstance(states, dict)
    assert isinstance(states['obs'], ndarray)
    assert states['obs'].shape == (NO_FEATURES, )

    actions = agent.act(states)
    
    assert isinstance(actions, dict)
    assert isinstance(actions['halt'], ndarray)
    assert isinstance(actions['direction'], ndarray)
    assert isinstance(actions['speed'], ndarray)
    assert isinstance(actions['sensorAction'], ndarray)

    assert actions['halt'].shape == (1,)
    assert actions['direction'].shape == (1,)
    assert actions['speed'].shape == (1,)
    assert actions['sensorAction'].shape == (1,)

    states, terminal, reward = wrapper.execute(actions=actions)

    agent.observe(terminal=terminal, reward=reward)

    wrapper.close()
