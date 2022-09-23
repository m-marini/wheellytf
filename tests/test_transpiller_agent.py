import tensorflow as tf
from wheelly.transpiller import AgentTranspiller


def agent_spec():
    return {
        "state_spec": dict(type="int",
                           shape=(1,),
                           num_values=1),
        "actions_spec": dict(type="int",
                             shape=(1,),
                             num_values=2),
        "reward_alpha": 0.1,
        "critic": {
            "alpha": 1e-3,
            "tdlambda": 0.8,
            "network": {
                "output": {
                    "layers": [
                        {"type": "dense", "output_size": 1},
                        {"type": "relu"},
                        {"type": "tanh"}
                    ]
                }
            }
        },
        "policy": {
            "alpha": 1e-3,
            "tdlambda": 0.8,
            "network": {
                "output": {
                    "layers": [
                        {"type": "dense", "output_size": 2},
                        {"type": "tanh"},
                        {"type": "softmax", "temperature": 0.8}
                    ]
                }
            }
        }
    }


def test_create():
    random = tf.random.Generator.from_seed(1234)
    ts = AgentTranspiller(agent_spec())

    spec = ts.parse()

    assert spec == {
        "state_spec": dict(type="int",
                           shape=(1,),
                           num_values=1),
        "actions_spec": dict(type="int",
                             shape=(1,),
                             num_values=2),
        "reward_alpha": 0.1,
        "critic": {
            "alpha": 1e-3,
            "tdlambda": 0.8,
            "layers": [{
                "name": "output[0]",
                "type": "dense",
                "input_size": 1,
                "output_size": 1,
                "inputs": ["input"]
            }, {
                "name": "output[1]",
                "type": "relu",
                "inputs": ["output[0]"]
            }, {
                "name": "output",
                "type": "tanh",
                "inputs": ["output[1]"]
            }]},
        "policy": {
            "alpha": 1e-3,
            "tdlambda": 0.8,
            "layers": [{
                "type": "dense",
                "name": "output[0]",
                "input_size": 1,
                "output_size": 2,
                "inputs": ["input"]
            }, {
                "name": "output[1]",
                "type": "tanh",
                "inputs": ["output[0]"]
            }, {
                "name": "output",
                "type": "softmax",
                "temperature": 0.8,
                "inputs": ["output[1]"]
            }]}
    }
    agent = ts.build(random=random)
    assert agent is not None
