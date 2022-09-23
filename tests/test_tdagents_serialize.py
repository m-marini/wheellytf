import tensorflow as tf
from wheelly.tdagents import TDAgent

from numpy.testing import assert_equal


def agent_spec():
    return dict(
        state_spec=dict(type="int",
                        shape=(1,),
                        num_values=1),
        actions_spec=dict(type="int",
                          shape=(1,),
                          num_values=2),
        reward_alpha=0.3,
        critic=dict(
            alpha=1e-3,
            tdlambda=0.8,
            layers=[
                dict(name="output[0]",
                     type="dense",
                     input_size=1,
                     output_size=1,
                     inputs=["input"]),
                dict(name="output[1]",
                     type="relu",
                     inputs=["output[0]"]),
                dict(name="output",
                     type="tanh",
                     inputs=["output[1]"])
            ]),
        policy=dict(
            alpha=1e-3,
            tdlambda=0.8,
            layers=[
                dict(name="output[0]",
                     type="dense",
                     input_size=1,
                     output_size=2,
                     inputs=["input"]),
                dict(name="output[1]",
                     type="tanh",
                     inputs=["output[0]"]),
                dict(name="output",
                     type="softmax",
                     temperature=0.5,
                     inputs=["output[1]"])
            ])
    )


def test_save_load():
    random = tf.random.Generator.from_seed(1234)

    agent = TDAgent.create(spec=agent_spec(), random=random)
    assert agent is not None

    agent.save_model(path="./models/td-test")

    last = tf.train.latest_checkpoint("./models/td-test")
    reader = tf.train.load_checkpoint(last)
    var_shape = reader.get_variable_to_shape_map()

    restored = TDAgent.load(path="./models/td-test")
    assert_equal(restored.reward_alpha, agent.reward_alpha)
    assert_equal(restored.policy.layers["output"].temperature,
                 restored.policy.layers["output"].temperature)
    assert_equal(restored.policy.layers["output[0]"].w.numpy(),
                 agent.policy.layers["output[0]"].w.numpy(), )
