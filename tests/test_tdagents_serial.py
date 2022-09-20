import tensorflow as tf
from wheelly.tdagents import TDAgent

from numpy.testing import assert_equal


def agent_spec():
    return {
        "reward_alpha": 0.3,
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


def test_save_load():
    state_spec = dict(type="int",
                      shape=(1,),
                      num_values=1)
    actions_spec = dict(type="int",
                        shape=(1,),
                        num_values=2)
    random = tf.random.Generator.from_seed(1234)

    agent = TDAgent.create(state_spec=state_spec,
                           actions_spec=actions_spec,
                           agent_spec=agent_spec(),
                           random=random)

    assert agent is not None

    agent.save_model(path="./models/td-test")

    restored = TDAgent.load(path="./models//td-test")
    assert_equal(restored.reward_alpha.numpy(), agent.reward_alpha.numpy())
    assert_equal(restored.reward_alpha.numpy(), agent.reward_alpha.numpy())
    assert_equal(restored.policy.node_prop("output", "temperature").numpy(),
                 agent.policy.node_prop("output", "temperature").numpy())
