import tensorflow as tf
from wheelly.tdagents import TDAgent
from wheelly.tdlayers import TDDense, TDRelu, TDSoftmax, TDTanh


def test_create():
    spec = {
        "reward_alpha": 0.1,
        "critic": {
            "alpha": 1e-3,
            "lambda": 0.8,
            "network": {
                "output": {
                    "layers": [
                        {"type": "dense", "size": 1},
                        {"type": "relu"},
                        {"type": "tanh"}
                    ]
                }
            }
        },
        "policy": {
            "alpha": 1e-3,
            "lambda": 0.8,
            "network": {
                "output": {
                    "layers": [
                        {"type": "dense", "size": 2},
                        {"type": "tanh"},
                        {"type": "softmax", "temperature": 0.8}
                    ]
                }
            }
        }
    }

    state_spec = dict(type="int",
                      shape=(1,),
                      num_values=1)
    actions_spec = dict(type="int",
                        shape=(1,),
                        num_values=2)
    random = tf.random.Generator.from_seed(1234)

    agent = TDAgent.create(state_spec=state_spec,
                           actions_spec=actions_spec,
                           agent_spec=spec,
                           random=random)

    assert agent is not None

    assert agent.spec() == {
        "state_spec": dict(type="int",
                           shape=(1,),
                           num_values=1),
        "actions_spec": dict(type="int",
                             shape=(1,),
                             num_values=2),
        "critic": [{
            "name": "output[0]",
            "type": "dense",
            "num_inputs": 1,
            "num_outputs": 1,
            "inputs": ["input"]
        }, {
            "name": "output[1]",
            "type": "relu",
            "inputs": ["output[0]"]
        }, {
            "name": "output",
            "type": "tanh",
            "inputs": ["output[1]"]
        }],
        "policy": [{
            "type": "dense",
            "name": "output[0]",
            "num_inputs": 1,
            "num_outputs": 2,
            "inputs": ["input"]
        }, {
            "name": "output[1]",
            "type": "tanh",
            "inputs": ["output[0]"]
        }, {
            "name": "output",
            "type": "softmax",
            "inputs": ["output[1]"]
        }]
    }


def test_by_spec():
    spec = {
        "state_spec": dict(type="int",
                           shape=(1,),
                           num_values=1),
        "actions_spec": dict(type="int",
                             shape=(1,),
                             num_values=2),
        "critic": [{
            "name": "output[0]",
            "type": "dense",
            "num_inputs": 1,
            "num_outputs": 1,
            "inputs": ["input"]
        }, {
            "name": "output[1]",
            "type": "relu",
            "inputs": ["output[0]"]
        }, {
            "name": "output",
            "type": "tanh",
            "inputs": ["output[1]"]
        }],
        "policy": [{
            "type": "dense",
            "name": "output[0]",
            "num_inputs": 1,
            "num_outputs": 2,
            "inputs": ["input"]
        }, {
            "name": "output[1]",
            "type": "tanh",
            "inputs": ["output[0]"]
        }, {
            "name": "output",
            "type": "softmax",
            "inputs": ["output[1]"]
        }]
    }
    random = tf.random.Generator.from_seed(1234)
    agent = TDAgent.by_spec(spec=spec,
                            random=random)

    assert agent is not None
    assert isinstance(agent.critic.graph.forward[0], TDDense)
    assert isinstance(agent.critic.graph.forward[1], TDRelu)
    assert isinstance(agent.critic.graph.forward[2], TDTanh)
    assert isinstance(agent.policy.graph.forward[0], TDDense)
    assert isinstance(agent.policy.graph.forward[1], TDTanh)
    assert isinstance(agent.policy.graph.forward[2], TDSoftmax)
    assert agent.states_spec == dict(type="int",
                                     shape=(1,),
                                     num_values=1)
    assert agent.actions_spec == dict(type="int",
                                      shape=(1,),
                                      num_values=2)
