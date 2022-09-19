import pytest
import tensorflow as tf
from tensorflow import Tensor
from wheelly.tdlayers import (TDDense, TDGraph, TDState, TDTanh, td_forward,
                            td_train)

import tests.fixtures as wfix

# The test network is:
#    a=in(2) -> b=dense0(2,2) -> c=tanh

alpha = 1e-1
tdlambda = 0


def create_node_props(random: tf.random.Generator):
    props = dict(
        b=TDDense.initProps(num_inputs=2, num_outputs=1, random=random)
    )
    return props


def create_graph():
    nodes = [
        TDDense(name="b"),
        TDTanh(name="c"),
    ]
    inputs = {"b": ["a"],
              "c": ["b"]}
    outputs = {"b": ["c"]}
    gr = TDGraph(forward=nodes,
                  inputs=inputs,
                  outputs=outputs)
    return gr


def create_state(random: tf.random.Generator):
    state = TDState(props={
        "lambda": tdlambda,
        "alpha": alpha
    },
        graph=create_graph(),
        node_props=create_node_props(random=random)
    )
    return state


def loss(x: Tensor) -> Tensor:
    return -(x * x) / 2


def grad_loss(x: Tensor) -> Tensor:
    return -x


def create_case(i: int, random: tf.random.Generator):
    inp = dict(a=random.normal(shape=[1, 2]))
    state = create_state(random)
    return dict(
        case=i,
        inp=inp,
        state=state)


@pytest.fixture
def cases():
    spec = dict(
        inp=dict(type="uniform", shape=(1, 2), minval=-1, maxval=1),
        state=dict(type="func", func=create_state)
    )
    return wfix.random_cases(spec=spec)


def test_train(cases):
    for case in cases:
        inputs = dict(a=case["inp"])
        state = case["state"]

        outputs1 = td_forward(inputs=inputs, state=state)
        loss1 = loss(x=outputs1["c"])
        grad_loss1 = grad_loss(x=outputs1["c"])

        state2, _ = td_train(state=state,
                            nodes=outputs1,
                            grad_loss=dict(c=grad_loss1),
                            delta=loss1)

        outputs2 = td_forward(inputs=inputs, state=state2)
        loss2 = loss(x=outputs2["c"])

        assert float(loss2[0, 0]) < float(loss1[0, 0]), str(cases)
