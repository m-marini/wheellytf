import pytest
import tensorflow as tf
from tensorflow import Tensor
from wheelly.tdlayers import TDDense, TDTanh, TDNetwork

import tests.fixtures as wfix

# The test network is:
#    a=in(2) -> b=dense0(2,2) -> c=tanh


def create_net(random: tf.random.Generator):
    alpha = 1e-1
    tdlambda = 0.0

    spec = dict(
        alpha=alpha,
        tdlambda=tdlambda,
        layers=[
            dict(name="b",
                 type="dense",
                 input_size=2,
                 output_size=1,
                 inputs=["a"]),
            dict(name="c",
                 type="tanh",
                 inputs=["b"])
        ])
    return TDNetwork.create(spec=spec, random=random)


def loss(x: Tensor) -> Tensor:
    return -(x * x) / 2


def grad_loss(x: Tensor) -> Tensor:
    return -x


def create_case(i: int, random: tf.random.Generator):
    inp = dict(a=random.normal(shape=[1, 2]))
    state = create_net(random)
    return dict(
        case=i,
        inp=inp,
        state=state)


def cases():
    spec = dict(
        inp=dict(type="uniform", shape=(1, 2), minval=-1, maxval=1),
        net=dict(type="func", func=create_net)
    )
    return wfix.random_cases(spec=spec)


def test_train():
    for case in cases():
        inputs = dict(a=case["inp"])
        net: TDNetwork = case["net"]

        outputs1 = net.forward(inputs)
        loss1 = loss(x=outputs1["c"])
        grad_out = dict(c=grad_loss(x=outputs1["c"]))

        net.train(outputs=outputs1,
                  grad_outputs=grad_out,
                  delta=loss1)

        outputs2 = net.forward(inputs)
        loss2 = loss(x=outputs2["c"])

        assert float(loss2[0, 0]) < float(loss1[0, 0]), str(cases)
