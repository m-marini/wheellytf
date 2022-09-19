from math import log

import tensorflow as tf
from numpy.testing import assert_almost_equal
from wheelly.tdagents import (TDAgent, choose_actions, flat_states, log_pi,
                            unflat_actions)
from wheelly.tdlayers import (TDDense, TDGraph, TDRelu, TDSoftmax, TDState,
                            TDTanh)

import tests.fixtures as wfix
from tests.mocks1 import ContinuousMockEnv
from wheelly.tdlisteners import DataCollectorConsumer, KpiListenerBuilder


def cases(num_cases: int = 30):
    spec = dict(
        alpha=dict(type="exp",
                   shape=(1,),
                   minval=1e-3,
                   maxval=10e-3),
        tdlambda=dict(type="uniform",
                      shape=(1,),
                      minval=0,
                      maxval=0.9),
        seed=dict(type="uniform",
                  shape=(1,),
                  minval=1,
                  maxval=1234),
        t=dict(type="exp",
               shape=(1,),
               minval=0.16,
               maxval=1.6),
        states=dict(type='uniform',
                    shape=(1,),
                    minval=0,
                    maxval=1),
        reward_alpha=dict(type="exp",
                          shape=(1,),
                          minval=0.1,
                          maxval=0.5)
    )
    return wfix.random_cases(spec=spec, num_test=num_cases)


def test_flat_states1():
    states = [0, 1]
    flatten = flat_states(states)
    assert "input" in flatten
    assert_almost_equal(flatten["input"].numpy(), [
        [0, 1]
    ])


def test_flat_states2():
    states = dict(a=[0, 1],
                  b=dict(a=[1, 2, 3],
                         b=[1, 2, 3, 4]))
    flatten = flat_states(states)
    assert "input.a" in flatten
    assert "input.b.a" in flatten
    assert "input.b.b" in flatten
    assert_almost_equal(flatten["input.a"].numpy(), [
        [0, 1]
    ])
    assert_almost_equal(flatten["input.b.a"].numpy(), [
        [1, 2, 3]
    ])
    assert_almost_equal(flatten["input.b.b"].numpy(), [
        [1, 2, 3, 4]
    ])


def test_unflat_actions1():
    spec = dict(type="int",
                shape=(1,),
                num_values=2)
    flatten = dict(output=0)
    result = unflat_actions(actions_spec=spec,
                            flatten=flatten)
    assert result == 0


def test_unflat_actions2():
    spec = dict(type="int",
                shape=(1,),
                num_values=2)
    flatten = dict(output=1)
    result = unflat_actions(actions_spec=spec,
                            flatten=flatten)
    assert result == 1


def test_unflat_actions3():
    spec = dict(
        a=dict(
            a=dict(type="int",
                   shape=(1,),
                   num_values=2)),
        b=dict(type="int",
               shape=(1,),
               num_values=3))
    flatten = {
        "output.a.a": 0,
        "output.b": 1
    }
    result = unflat_actions(actions_spec=spec,
                            flatten=flatten)
    assert result == {
        "a": {
            "a": 0
        },
        "b": 1
    }


def test_choose_actions():
    random = tf.random.Generator.from_seed(1234)
    flatten = {
        "output.a.a": tf.constant([[1, 0]], dtype=tf.float32),
        "output.b": tf.constant([[0, 1, 0]], dtype=tf.float32),
        "output.c": tf.constant([[0, 0, 1]], dtype=tf.float32)
    }
    result = choose_actions(flatten_pis=flatten, random=random)
    assert result == {
        "output.a.a": 0,
        "output.b": 1,
        "output.c": 2
    }


def test_log_pi0():
    spec = dict(type="int")
    pi = dict(output=tf.constant([[0.2, 0.3, 0.5]]))
    action = 0
    result = log_pi(pi=pi, action=action, actions_spec=spec)
    assert "output" in result
    assert_almost_equal(result["output"].numpy(), [[5, 0, 0]])


def test_log_pi1():
    spec = dict(type="int")
    pi = dict(output=tf.constant([[0.2, 0.3, 0.5]]))
    action = 2
    result = log_pi(pi=pi, action=action, actions_spec=spec)
    assert "output" in result
    assert_almost_equal(result["output"].numpy(), [[0, 0, 2]])


def test_log_pi2():
    spec = dict(a=dict(type="int"),
                b=dict(type="int"))
    pi = {
        "output.a": tf.constant([[0.2, 0.3, 0.5]]),
        "output.b": tf.constant([[0.5, 0.3, 0.2]])
    }
    action = dict(a=0, b=2)
    result = log_pi(pi=pi, action=action, actions_spec=spec)
    assert "output.a" in result
    assert "output.b" in result
    assert_almost_equal(result["output.a"].numpy(), [[5, 0, 0]])
    assert_almost_equal(result["output.b"].numpy(), [[0, 0, 5]])


def create_agent(case):
    alpha = 30e-3
    tdlambda = 0.5
    temperature = 2 / log(10)
    reward_alpha = 0.1

    state_spec = dict(type="int",
                      num_values=1)
    actions_spec = dict(type="int",
                        num_values=2)

    random = tf.random.Generator.from_seed(int(case["seed"][0]))
    props = {
        "alpha":  tf.constant(alpha),
        "lambda": tf.constant(tdlambda)
    }
    # input(1)-> hidden0=dense(1,2) -> act0=relu -> hidden1=dense(2,1) -> output=tanh
    critic_graph = TDGraph(forward=[TDDense(name="hidden0"),
                                     TDRelu(name="act0"),
                                     TDDense(name="hidden1"),
                                     TDTanh(name="output")],
                            inputs=dict(output=["hidden1"],
                                        hidden1=["act0"],
                                        act0=["hidden0"],
                                        hidden0=["input"]),
                            outputs=dict(hidden0=["act0"],
                                         act0=["hidden1"],
                                         hidden1=["output"]))
    critic_props = dict(hidden0=TDDense.initProps(num_inputs=1,
                                                   num_outputs=2,
                                                   random=random),
                        hidden1=TDDense.initProps(num_inputs=2,
                                                   num_outputs=1,
                                                   random=random))
    critic = TDState(props=props,
                      graph=critic_graph,
                      node_props=critic_props)

    # input(2)-> hidden0=dense(1,2) -> act0=tanh -> output=softmax(t)
    policy_graph = TDGraph(forward=[TDDense(name="hidden0"),
                                     TDTanh(name="act0"),
                                     TDSoftmax(name="output")],
                            inputs=dict(output=["act0"],
                                        act0=["hidden0"],
                                        hidden0=["input"]),
                            outputs=dict(hidden0=["act0"],
                                         act0=["output"]))
    policy_props = dict(hidden0=TDDense.initProps(num_inputs=1,
                                                   num_outputs=2,
                                                   random=random),
                        output=TDSoftmax.initProps(
                            temperature=tf.constant(temperature))
                        )
    policy = TDState(props=props,
                      graph=policy_graph,
                      node_props=policy_props)
    agent = TDAgent(states_spec=state_spec,
                     actions_spec=actions_spec,
                     policy=policy,
                     critic=critic,
                     reward_alpha=tf.constant(reward_alpha),
                     random=random)
    return agent


def create_states(case):
    i = round(float(case["states"]))
    return [i]


def test_act():
    for case in cases():
        agent = create_agent(case)
        states = create_states(case)
        action = agent.act(states=states)
        assert isinstance(action, int)


def test_train():
    case = cases(1)[0]
    env = ContinuousMockEnv()
    agent = create_agent(case)
    assert agent is not None
    reward_listener: DataCollectorConsumer = KpiListenerBuilder.getter("reward") \
        .register(agent, DataCollectorConsumer())
    avg_reward_listener: DataCollectorConsumer = KpiListenerBuilder.getter("avg_reward") \
        .register(agent, DataCollectorConsumer())
    delta_listener: DataCollectorConsumer = KpiListenerBuilder.getter("delta") \
        .register(agent, DataCollectorConsumer())
    pi_listener: DataCollectorConsumer = KpiListenerBuilder.getter("pi") \
        .get("output") \
        .register(agent, DataCollectorConsumer())
    v0_listener: DataCollectorConsumer = KpiListenerBuilder.getter("c0") \
        .get("output") \
        .register(agent, DataCollectorConsumer())
    v1_listener: DataCollectorConsumer = KpiListenerBuilder.getter("c1") \
        .get("output") \
        .register(agent, DataCollectorConsumer())
    s0_listener: DataCollectorConsumer = KpiListenerBuilder.getter("c0") \
        .get("input") \
        .register(agent, DataCollectorConsumer())
    s1_listener: DataCollectorConsumer = KpiListenerBuilder.getter("c1") \
        .get("input") \
        .register(agent, DataCollectorConsumer())
    actions_listener: DataCollectorConsumer = KpiListenerBuilder.getter("actions") \
        .map(lambda x: float(x)) \
        .register(agent, DataCollectorConsumer())
    trained_v0_listener: DataCollectorConsumer = KpiListenerBuilder.getter("trained_c0") \
        .get("output") \
        .register(agent, DataCollectorConsumer())
    trained_v1_listener: DataCollectorConsumer = KpiListenerBuilder.getter("trained_c1") \
        .get("output") \
        .register(agent, DataCollectorConsumer())
    trained_pi_listener: DataCollectorConsumer = KpiListenerBuilder.getter("trained_pi") \
        .get("output") \
        .register(agent, DataCollectorConsumer())
    trained_avg_listener: DataCollectorConsumer = KpiListenerBuilder.getter("trained_avg_reward") \
        .register(agent, DataCollectorConsumer())
    grad_pi_listener: DataCollectorConsumer = KpiListenerBuilder.getter("dp") \
        .get("output") \
        .register(agent, DataCollectorConsumer())

    steps = 0
    num_steps = 300
    states = env.reset()
    while steps <= num_steps:
        steps += 1
        actions = agent.act(states=[states])
        states, terminal, reward = env.execute(actions)
        agent.observe(reward=reward, terminal=terminal)

    s0_listener.to_csv("data/s0.csv")
    reward_listener.to_csv("data/reward.csv")
    actions_listener.to_csv("data/actions.csv")
    s1_listener.to_csv("data/s1.csv")
    avg_reward_listener.to_csv("data/avg_reward.csv")
    v0_listener.to_csv("data/v0.csv")
    v1_listener.to_csv("data/v1.csv")
    delta_listener.to_csv("data/delta.csv")
    pi_listener.to_csv("data/pi.csv")
    grad_pi_listener.to_csv("data/grad_pi.csv")
    trained_v0_listener.to_csv("data/trained_v0.csv")
    trained_v1_listener.to_csv("data/trained_v1.csv")
    trained_pi_listener.to_csv("data/trained_pi.csv")
    trained_avg_listener.to_csv("data/trained_avg_reward.csv")
    assert reward_listener.kpi().lin_poly.coef[1] > 0
