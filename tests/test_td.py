from math import exp, tanh
from typing import Any

import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal
from wheelly.tdlayers import (TDDense, TDGraph, TDSoftmax, TDState, TDTanh,
                            td_forward, td_train)

import tests.fixtures as wfix

# The test network is:
#    a=in(2) -> b=dense0(2,2) -> c=tanh -> d=softmax(t=0.869)
#                                       -> e=dense1(2,1)


def create_node_props(case: dict[str, Any]):
    b_eb = tf.Variable(initial_value=case["b_eb"], trainable=False)
    b_ew = tf.Variable(initial_value=case["b_ew"], trainable=False)
    b_b = tf.Variable(initial_value=case["b_b"])
    b_w = tf.Variable(initial_value=case["b_w"])

    e_eb = tf.Variable(initial_value=case["e_eb"], trainable=False)
    e_ew = tf.Variable(initial_value=case["e_ew"], trainable=False)
    e_b = tf.Variable(initial_value=case["e_b"])
    e_w = tf.Variable(initial_value=case["e_w"])

    props = dict(
        b=TDDense.createProps(eb=b_eb, ew=b_ew, b=b_b, w=b_w),
        e=TDDense.createProps(eb=e_eb, ew=e_ew, b=e_b, w=e_w),
        d=TDSoftmax.initProps(temperature=case["t"])
    )
    return props


def create_graph():
    nodes = [
        TDDense(name="b"),
        TDTanh(name="c"),
        TDSoftmax(name="d"),
        TDDense(name="e"),
    ]
    inputs = {"b": ["a"],
              "c": ["b"],
              "d": ["c"],
              "e": ["c"]}
    outputs = {"b": ["c"],
               "c": ["d", "e"]}
    gr = TDGraph(forward=nodes,
                  inputs=inputs,
                  outputs=outputs)
    return gr


def create_state(case: dict[str, Any]):
    state = TDState(props={
        "lambda": case["tdlambda"],
        "alpha": case["alpha"]
    },
        graph=create_graph(),
        node_props=create_node_props(case)
    )
    return state


@ pytest.fixture
def cases():
    rangeval = 0.3
    rangeinp = 1
    spec = dict(
        t=dict(type="exp", shape=(1,), minval=0.4, maxval=1),
        alpha=dict(type="exp", shape=(1,), minval=1e-3, maxval=10e-3),
        tdlambda=dict(type="uniform", shape=(1,), minval=0, maxval=0.9),
        a=dict(type="uniform",
                    shape=(1, 2),
                    minval=-rangeinp,
                    maxval=rangeinp),
        b_eb=dict(type="uniform",
                  shape=(1, 2),
                  minval=-rangeval,
                  maxval=rangeval),
        b_ew=dict(type="uniform",
                  shape=(2, 2),
                  minval=-rangeval,
                  maxval=rangeval),
        b_b=dict(type="uniform",
                 shape=(1, 2),
                 minval=-rangeval,
                 maxval=rangeval),
        b_w=dict(type="uniform",
                 shape=(2, 2),
                 minval=-rangeval,
                 maxval=rangeval),
        e_eb=dict(type="uniform",
                  shape=(1, 1),
                  minval=-rangeval,
                  maxval=rangeval),
        e_ew=dict(type="uniform",
                  shape=(2, 1),
                  minval=-rangeval,
                  maxval=rangeval),
        e_b=dict(type="uniform",
                 shape=(1, 1),
                 minval=-rangeval,
                 maxval=rangeval),
        e_w=dict(type="uniform",
                 shape=(2, 1),
                 minval=-rangeval,
                 maxval=rangeval),
        delta=dict(type="uniform",
                   shape=(1,),
                   minval=-rangeval,
                   maxval=rangeval),
        dl_dd=dict(type="uniform",
                   shape=(1, 2),
                   minval=-rangeval,
                   maxval=rangeval),
        dl_de=dict(type="uniform",
                   shape=(1, 1),
                   minval=-rangeval,
                   maxval=rangeval)
    )
    return wfix.random_cases(spec=spec)


def test_forward(cases: list[dict[str, Any]]):
    for case in cases:
        a = case["a"]
        state = create_state(case)
        a0 = float(a[0, 0])
        a1 = float(a[0, 1])

        b_b0 = float(case["b_b"][0, 0])
        b_b1 = float(case["b_b"][0, 1])
        b_w00 = float(case["b_w"][0, 0])
        b_w01 = float(case["b_w"][0, 1])
        b_w10 = float(case["b_w"][1, 0])
        b_w11 = float(case["b_w"][1, 1])

        e_b0 = float(case["e_b"][0, 0])
        e_w0 = float(case["e_w"][0, 0])
        e_w1 = float(case["e_w"][1, 0])

        t = case["t"]

        b0 = a0 * b_w00 + a1 * b_w10 + b_b0
        b1 = a0 * b_w01 + a1 * b_w11 + b_b1
        c0 = tanh(b0)
        c1 = tanh(b1)
        z0 = exp(c0 / t)
        z1 = exp(c1 / t)
        z = z0 + z1
        d0 = z0 / z
        d1 = z1 / z
        e1 = c0 * e_w0 + c1 * e_w1 + e_b0

        outputs = td_forward(inputs=dict(a=a), state=state)

        assert isinstance(outputs, dict)
        assert_almost_equal(outputs["b"].numpy(), [
            [b0, b1]
        ])
        assert_almost_equal(outputs["c"].numpy(), [
            [c0, c1]
        ])
        assert_almost_equal(outputs["d"].numpy(), [
            [d0, d1]
        ])
        assert_almost_equal(outputs["e"].numpy(), [
            [e1]
        ])


def test_train(cases: list[dict[str, Any]]):
    for case in cases:
        a = case["a"]
        state = create_state(case)
        a0 = float(a[0, 0])
        a1 = float(a[0, 1])

        b_eb0 = float(case["b_eb"][0, 0])
        b_eb1 = float(case["b_eb"][0, 1])
        b_ew00 = float(case["b_ew"][0, 0])
        b_ew01 = float(case["b_ew"][0, 1])
        b_ew10 = float(case["b_ew"][1, 0])
        b_ew11 = float(case["b_ew"][1, 1])

        b_b0 = float(case["b_b"][0, 0])
        b_b1 = float(case["b_b"][0, 1])
        b_w00 = float(case["b_w"][0, 0])
        b_w01 = float(case["b_w"][0, 1])
        b_w10 = float(case["b_w"][1, 0])
        b_w11 = float(case["b_w"][1, 1])

        e_eb0 = float(case["e_eb"][0, 0])
        e_ew0 = float(case["e_ew"][0, 0])
        e_ew1 = float(case["e_ew"][1, 0])

        e_b0 = float(case["e_b"][0, 0])
        e_w0 = float(case["e_w"][0, 0])
        e_w1 = float(case["e_w"][1, 0])

        t = float(case["t"][0])
        alpha = float(case["alpha"][0])
        tdlambda = float(case["tdlambda"][0])

        b0 = a0 * b_w00 + a1 * b_w10 + b_b0
        b1 = a0 * b_w01 + a1 * b_w11 + b_b1
        c0 = tanh(b0)
        c1 = tanh(b1)
        z0 = exp(c0 / t)
        z1 = exp(c1 / t)
        z = z0 + z1
        d0 = z0 / z
        d1 = z1 / z
        e1 = c0 * e_w0 + c1 * e_w1 + e_b0
        delta = float(case["delta"])

        dl_dd0 = float(case["dl_dd"][0, 0])
        dl_dd1 = float(case["dl_dd"][0, 1])
        dl_de0 = float(case["dl_de"][0])

# The test network is:
#    a=in(2) -> b=dense0(2,2) -> c=tanh -> d=softmax(t=0.869)
#                                       -> e=dense1(2,1)

        b0 = a0 * b_w00 + a1 * b_w10 + b_b0
        b1 = a0 * b_w01 + a1 * b_w11 + b_b1
        c0 = tanh(b0)
        c1 = tanh(b1)
        z0 = exp(c0 / t)
        z1 = exp(c1 / t)
        z = z0 + z1
        d0 = z0 / z
        d1 = z1 / z
        e0 = c0 * e_w0 + c1 * e_w1 + e_b0

        dl_de0_in = dl_de0 * e_w0
        dl_de1_in = dl_de0 * e_w1

        dl_dd0_in = (dl_dd0 * d0 * (1 - d0) - dl_dd1 * d1 * d0) / t
        dl_dd1_in = (-dl_dd0 * d0 * d1 + dl_dd1 * d1 * (1 - d1)) / t

        dl_dc0 = dl_de0_in + dl_dd0_in
        dl_dc1 = dl_de1_in + dl_dd1_in

        dl_db0 = dl_dc0 * (1 - c0 * c0)
        dl_db1 = dl_dc1 * (1 - c1 * c1)

        dl_db_b0 = dl_db0
        dl_db_b1 = dl_db1
        dl_db_w00 = dl_db0 * a0
        dl_db_w01 = dl_db1 * a0
        dl_db_w10 = dl_db0 * a1
        dl_db_w11 = dl_db1 * a1

        dl_de_b0 = dl_de0
        dl_de_w0 = dl_de0 * c0
        dl_de_w1 = dl_de0 * c1

        post_b_eb0 = b_eb0 * tdlambda + dl_db_b0
        post_b_eb1 = b_eb1 * tdlambda + dl_db_b1
        post_b_ew00 = b_ew00 * tdlambda + dl_db_w00
        post_b_ew01 = b_ew01 * tdlambda + dl_db_w01
        post_b_ew10 = b_ew10 * tdlambda + dl_db_w10
        post_b_ew11 = b_ew11 * tdlambda + dl_db_w11
        post_b_b0 = b_b0 + post_b_eb0 * delta * alpha
        post_b_b1 = b_b1 + post_b_eb1 * delta * alpha
        post_b_w00 = b_w00 + post_b_ew00 * delta * alpha
        post_b_w01 = b_w01 + post_b_ew01 * delta * alpha
        post_b_w10 = b_w10 + post_b_ew10 * delta * alpha
        post_b_w11 = b_w11 + post_b_ew11 * delta * alpha

        post_e_eb0 = e_eb0 * tdlambda + dl_de_b0
        post_e_ew0 = e_ew0 * tdlambda + dl_de_w0
        post_e_ew1 = e_ew1 * tdlambda + dl_de_w1
        post_e_b0 = e_b0 + post_e_eb0 * delta * alpha
        post_e_w0 = e_w0 + post_e_ew0 * delta * alpha
        post_e_w1 = e_w1 + post_e_ew1 * delta * alpha

        nodes = td_forward(inputs=dict(a=a), state=state)
        state1, ctx = td_train(state=state,
                                nodes=nodes,
                                grad_loss=dict(
                                    d=case["dl_dd"], e=case["dl_de"]),
                                delta=case["delta"])

        assert state1 == state

        assert_almost_equal(state1.node_prop("e", "eb"), [
            [post_e_eb0]
        ])
        assert_almost_equal(state1.node_prop("e", "ew"), [
            [post_e_ew0], [post_e_ew1]
        ])
        assert_almost_equal(state1.node_prop("e", "b"), [
            [post_e_b0]
        ])
        assert_almost_equal(state1.node_prop("e", "w"), [
            [post_e_w0], [post_e_w1]
        ])

        assert_almost_equal(state1.node_prop("b", "eb"), [
            [post_b_eb0, post_b_eb1]
        ])
        assert_almost_equal(state1.node_prop("b", "ew"), [
            [post_b_ew00, post_b_ew01],
            [post_b_ew10, post_b_ew11]
        ])
        assert_almost_equal(state1.node_prop("b", "b"), [
            [post_b_b0, post_b_b1]
        ])
        assert_almost_equal(state1.node_prop("b", "w"), [
            [post_b_w00, post_b_w01],
            [post_b_w10, post_b_w11]
        ])
