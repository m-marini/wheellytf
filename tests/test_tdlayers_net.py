from math import exp, tanh
from typing import Any

import tensorflow as tf
from numpy.testing import assert_almost_equal
from tensorflow import Tensor
from wheelly.tdlayers import (TDConcat, TDDense, TDLinear, TDNetwork, TDRelu,
                               TDSoftmax, TDSum, TDTanh)

from tests.fixtures import random_cases


def net_cases():
    return random_cases(spec=dict(
        input0=dict(
            type="uniform",
            shape=(1,),
            minval=-1
        ),
        input1=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        alpha=dict(
            type="exp",
            shape=(1,),
            minval=1e-3,
            maxval=100e-3
        ),
        tdlambda=dict(
            type="uniform",
            shape=(1,),
            maxval=0.9
        ),
        b=dict(
            type="uniform",
            shape=(1,),
            minvalue=-1
        ),
        w=dict(
            type="exp",
            shape=(1,),
            minval=0.1,
            maxval=10
        ),
        temperature=dict(
            type="exp",
            shape=(1,),
            minval=0.5,
            maxval=2
        ),
        grad_layer7=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        grad_layer8=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        delta=dict(
            type="uniform",
            shape=(1,),
            minval=-1
        )
    ))


def create_net_spec(case: dict[str, Tensor]) -> dict[str, Any]:
    """
        input1(1) -> layer1(concat) -> layer2(dense,2)  -> layer3(relu) -> layer4(dense, 2) -> layer5(sum) -> layer6(tanh)-> layer7(linear)
        input2(2) ->                                                    --------------------->                            -> layer8(softmax)
    """
    spec = dict(
        alpha=float(case["alpha"]),
        tdlambda=float(case["tdlambda"]),
        layers=[
            dict(
                name="layer1",
                type="concat",
                inputs=["input0", "input1"]
            ),
            dict(
                name="layer2",
                type="dense",
                input_size=3,
                output_size=2,
                inputs=["layer1"]
            ),
            dict(
                name="layer3",
                type="relu",
                inputs=["layer2"]
            ),
            dict(
                name="layer4",
                type="dense",
                input_size=2,
                output_size=2,
                inputs=["layer3"]
            ),
            dict(
                name="layer5",
                type="sum",
                inputs=["layer3", "layer4"]
            ),
            dict(
                name="layer6",
                type="tanh",
                inputs=["layer5"]
            ),
            dict(
                name="layer7",
                type="lin",
                b=float(case["b"]),
                w=float(case["w"]),
                inputs=["layer6"]
            ),
            dict(
                name="layer8",
                type="softmax",
                temperature=float(case["temperature"]),
                inputs=["layer6"]
            )
        ]
    )
    return spec


def test_net_create():
    for case in net_cases():
        spec = create_net_spec(case)
        net = TDNetwork.create(
            spec=spec, random=tf.random.Generator.from_seed(1234))
        assert net.alpha == float(case["alpha"])
        assert net.tdlambda == float(case["tdlambda"])
        assert len(net.layers) == 8
        for key, node in net.layers.items():
            assert key == node.name
        assert isinstance(net.layers["layer1"], TDConcat)
        assert isinstance(net.layers["layer2"], TDDense)
        assert isinstance(net.layers["layer3"], TDRelu)
        assert isinstance(net.layers["layer4"], TDDense)
        assert isinstance(net.layers["layer5"], TDSum)
        assert isinstance(net.layers["layer6"], TDTanh)
        assert isinstance(net.layers["layer7"], TDLinear)
        assert isinstance(net.layers["layer8"], TDSoftmax)

        assert net.layers["layer2"].eb.shape == (1, 2)
        assert net.layers["layer2"].b.shape == (1, 2)
        assert net.layers["layer2"].ew.shape == (3, 2)
        assert net.layers["layer2"].w.shape == (3, 2)

        assert net.layers["layer4"].eb.shape == (1, 2)
        assert net.layers["layer4"].b.shape == (1, 2)
        assert net.layers["layer4"].ew.shape == (2, 2)
        assert net.layers["layer4"].w.shape == (2, 2)

        assert net.layers["layer7"].b == float(case["b"])
        assert net.layers["layer7"].w == float(case["w"])

        assert net.layers["layer8"].temperature == float(case["temperature"])

        assert net.forward_seq == [
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "layer5",
            "layer6",
            "layer7",
            "layer8",
        ]

        assert net.backward_seq == [
            "layer8",
            "layer7",
            "layer6",
            "layer5",
            "layer4",
            "layer3",
            "layer2",
            "layer1",
        ]
    assert net.inputs == {
        "layer1": ["input0", "input1"],
        "layer2": ["layer1"],
        "layer3": ["layer2"],
        "layer4": ["layer3"],
        "layer5": ["layer3", "layer4"],
        "layer6": ["layer5"],
        "layer7": ["layer6"],
        "layer8": ["layer6"]
    }
    assert net.outputs == {
        "layer1": ["layer2"],
        "layer2": ["layer3"],
        "layer3": ["layer4", "layer5"],
        "layer4": ["layer5"],
        "layer5": ["layer6"],
        "layer6": ["layer7", "layer8"],
        "layer7": [],
        "layer8": []
    }


def test_net_forward():
    for case in net_cases():
        spec = create_net_spec(case)
        net = TDNetwork.create(
            spec=spec, random=tf.random.Generator.from_seed(1234))
        input0 = tf.reshape(tensor=case["input0"], shape=(1, -1))
        input1 = tf.reshape(tensor=case["input1"], shape=(1, -1))
        in00 = float(input0[0, 0])
        in10 = float(input1[0, 0])
        in11 = float(input1[0, 1])
        inputs = dict(input0=input0, input1=input1)

        b20 = float(net.layers["layer2"].b[0, 0])
        b21 = float(net.layers["layer2"].b[0, 1])

        w200 = float(net.layers["layer2"].w[0, 0])
        w201 = float(net.layers["layer2"].w[0, 1])

        w210 = float(net.layers["layer2"].w[1, 0])
        w211 = float(net.layers["layer2"].w[1, 1])

        w220 = float(net.layers["layer2"].w[2, 0])
        w221 = float(net.layers["layer2"].w[2, 1])

        b40 = float(net.layers["layer4"].b[0, 0])
        b41 = float(net.layers["layer4"].b[0, 1])

        w400 = float(net.layers["layer4"].w[0, 0])
        w401 = float(net.layers["layer4"].w[0, 1])

        w410 = float(net.layers["layer4"].w[1, 0])
        w411 = float(net.layers["layer4"].w[1, 1])

        b7 = float(case["b"])
        w7 = float(case["w"])

        t = float(case["temperature"])

        l20 = in00 * w200 + in10 * w210 + in11 * w220 + b20
        l21 = in00 * w201 + in10 * w211 + in11 * w221 + b21

        l30 = l20 if l20 > 0 else 0.0
        l31 = l21 if l21 > 0 else 0.0

        l40 = l30 * w400 + l31 * w410 + b40
        l41 = l30 * w401 + l31 * w411 + b41

        l50 = l30 + l40
        l51 = l31 + l41

        l60 = tanh(l50)
        l61 = tanh(l51)

        l70 = l60 * w7 + b7
        l71 = l61 * w7 + b7

        ez80 = exp(l60 / t)
        ez81 = exp(l61 / t)
        ez8 = ez80 + ez81

        l80 = ez80 / ez8
        l81 = ez81 / ez8

        output = net.forward(inputs)

        assert_almost_equal(output["layer1"].numpy(), [[
            in00, in10, in11
        ]])
        assert_almost_equal(output["layer2"].numpy(), [[
            l20, l21
        ]])
        assert_almost_equal(output["layer3"].numpy(), [[
            l30, l31
        ]])
        assert_almost_equal(output["layer4"].numpy(), [[
            l40, l41
        ]])
        assert_almost_equal(output["layer5"].numpy(), [[
            l50, l51
        ]])
        assert_almost_equal(output["layer6"].numpy(), [[
            l60, l61
        ]])
        assert_almost_equal(output["layer7"].numpy(), [[
            l70, l71
        ]], decimal=6)
        assert_almost_equal(output["layer8"].numpy(), [[
            l80, l81
        ]], err_msg=f'Case #{case["case_num"]}')


def test_net_train():
    for case in net_cases():
        spec = create_net_spec(case)
        net = TDNetwork.create(
            spec=spec, random=tf.random.Generator.from_seed(1234))
        input0 = tf.reshape(tensor=case["input0"], shape=(1, -1))
        input1 = tf.reshape(tensor=case["input1"], shape=(1, -1))
        in00 = float(input0[0, 0])
        in10 = float(input1[0, 0])
        in11 = float(input1[0, 1])
        inputs = dict(input0=input0, input1=input1)

        b20 = float(net.layers["layer2"].b[0, 0])
        b21 = float(net.layers["layer2"].b[0, 1])

        w200 = float(net.layers["layer2"].w[0, 0])
        w201 = float(net.layers["layer2"].w[0, 1])

        w210 = float(net.layers["layer2"].w[1, 0])
        w211 = float(net.layers["layer2"].w[1, 1])

        w220 = float(net.layers["layer2"].w[2, 0])
        w221 = float(net.layers["layer2"].w[2, 1])

        b40 = float(net.layers["layer4"].b[0, 0])
        b41 = float(net.layers["layer4"].b[0, 1])

        w400 = float(net.layers["layer4"].w[0, 0])
        w401 = float(net.layers["layer4"].w[0, 1])

        w410 = float(net.layers["layer4"].w[1, 0])
        w411 = float(net.layers["layer4"].w[1, 1])

        b7 = float(case["b"])
        w7 = float(case["w"])

        t = float(case["temperature"])
        grad_layer7 = tf.reshape(tensor=case["grad_layer7"], shape=(1, -1))
        grad_layer8 = tf.reshape(tensor=case["grad_layer8"], shape=(1, -1))
        delta = tf.reshape(tensor=case["delta"], shape=(1, -1))
        fdelta = float(delta)
        alpha = float(case["alpha"])
        tdlambda = float(case["tdlambda"])
        eb20 = eb21 = 0
        ew200 = ew201 = ew210 = ew211 = ew220 = ew221 = 0
        eb40 = eb41 = 0
        ew400 = ew401 = ew410 = ew411 = 0

        l20 = in00 * w200 + in10 * w210 + in11 * w220 + b20
        l21 = in00 * w201 + in10 * w211 + in11 * w221 + b21

        l30 = l20 if l20 > 0 else 0.0
        l31 = l21 if l21 > 0 else 0.0

        l40 = l30 * w400 + l31 * w410 + b40
        l41 = l30 * w401 + l31 * w411 + b41

        l50 = l30 + l40
        l51 = l31 + l41

        l60 = tanh(l50)
        l61 = tanh(l51)

        l70 = l60 * w7 + b7
        l71 = l61 * w7 + b7

        ez80 = exp(l60 / t)
        ez81 = exp(l61 / t)
        ez8 = ez80 + ez81

        l80 = ez80 / ez8
        l81 = ez81 / ez8

        g70 = float(grad_layer7[0, 0])
        g71 = float(grad_layer7[0, 1])

        g80 = float(grad_layer8[0, 0])
        g81 = float(grad_layer8[0, 1])

        gi70 = g70 * w7
        gi71 = g71 * w7

        gi80 = (g80 * l80 * (1 - l80) - g81 * l81 * l80) / t
        gi81 = (-g80 * l80 * l81 + g81 * l81 * (1 - l81)) / t

        g60 = gi70 + gi80
        g61 = gi71 + gi81

        g50 = g60 * (1 - l60 * l60)
        g51 = g61 * (1 - l61 * l61)

        g40 = g50
        g41 = g51

        gi40 = g40 * w400 + g41 * w401
        gi41 = g40 * w410 + g41 * w411

        g30 = gi40 + g50
        g31 = gi41 + g51

        g20 = g30 if l20 > 0 else 0.0
        g21 = g31 if l21 > 0 else 0.0

        g10 = g20 * w200 + g21 * w201
        g11 = g20 * w210 + g21 * w211
        g12 = g20 * w220 + g21 * w221

        post_eb20 = eb20 * tdlambda + g20
        post_eb21 = eb21 * tdlambda + g21

        post_b20 = b20 + alpha * fdelta * post_eb20
        post_b21 = b21 + alpha * fdelta * post_eb21

        post_ew200 = ew200 * tdlambda + g20 * in00
        post_ew201 = ew201 * tdlambda + g21 * in00
        post_ew210 = ew210 * tdlambda + g20 * in10
        post_ew211 = ew211 * tdlambda + g21 * in10
        post_ew220 = ew220 * tdlambda + g20 * in11
        post_ew221 = ew221 * tdlambda + g21 * in11

        post_w200 = w200 + alpha * fdelta * post_ew200
        post_w201 = w201 + alpha * fdelta * post_ew201
        post_w210 = w210 + alpha * fdelta * post_ew210
        post_w211 = w211 + alpha * fdelta * post_ew211
        post_w220 = w220 + alpha * fdelta * post_ew220
        post_w221 = w221 + alpha * fdelta * post_ew221

        post_eb40 = eb40 * tdlambda + g40
        post_eb41 = eb41 * tdlambda + g41

        post_b40 = b40 + alpha * fdelta * post_eb40
        post_b41 = b41 + alpha * fdelta * post_eb41

        post_ew400 = ew400 * tdlambda + g40 * l30
        post_ew401 = ew401 * tdlambda + g41 * l30
        post_ew410 = ew410 * tdlambda + g40 * l31
        post_ew411 = ew411 * tdlambda + g41 * l31

        post_w400 = w400 + alpha * fdelta * post_ew400
        post_w401 = w401 + alpha * fdelta * post_ew401
        post_w410 = w410 + alpha * fdelta * post_ew410
        post_w411 = w411 + alpha * fdelta * post_ew411

        output = net.forward(inputs)
        grads = dict(layer7=grad_layer7,
                     layer8=grad_layer8)
        grads1 = net.train(outputs=output,
                           grad_outputs=grads,
                           delta=delta)

        assert_almost_equal(grads1["layer6"].numpy(), [[
            g60, g61
        ]], decimal=6)
        assert_almost_equal(grads1["layer5"].numpy(), [[
            g50, g51
        ]], decimal=6)
        assert_almost_equal(grads1["layer4"].numpy(), [[
            g40, g41
        ]], decimal=6)
        assert_almost_equal(grads1["layer3"].numpy(), [[
            g30, g31
        ]], decimal=6)
        assert_almost_equal(grads1["layer2"].numpy(), [[
            g20, g21
        ]], decimal=6)
        assert_almost_equal(grads1["layer1"].numpy(), [[
            g10, g11, g12
        ]], decimal=6)
        assert_almost_equal(grads1["input1"].numpy(), [[
            g11, g12
        ]], decimal=6)
        assert_almost_equal(grads1["input0"].numpy(), [[
            g10
        ]], decimal=6)

        assert_almost_equal(net.layers["layer2"].eb.numpy(), [[
            post_eb20, post_eb21
        ]], decimal=6)
        assert_almost_equal(net.layers["layer2"].b.numpy(), [[
            post_b20, post_b21
        ]], decimal=6)
        assert_almost_equal(net.layers["layer2"].ew.numpy(), [
            [post_ew200, post_ew201],
            [post_ew210, post_ew211],
            [post_ew220, post_ew221]
        ], decimal=6)
        assert_almost_equal(net.layers["layer2"].w.numpy(), [
            [post_w200, post_w201],
            [post_w210, post_w211],
            [post_w220, post_w221]
        ], decimal=6)

        assert_almost_equal(net.layers["layer4"].eb.numpy(), [[
            post_eb40, post_eb41
        ]], decimal=6)
        assert_almost_equal(net.layers["layer4"].b.numpy(), [[
            post_b40, post_b41
        ]], decimal=6)
        assert_almost_equal(net.layers["layer4"].ew.numpy(), [
            [post_ew400, post_ew401],
            [post_ew410, post_ew411],
        ], decimal=6)
        assert_almost_equal(net.layers["layer4"].w.numpy(), [
            [post_w400, post_w401],
            [post_w410, post_w411],
        ], decimal=6)
