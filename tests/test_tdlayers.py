from math import exp, tanh

import tensorflow as tf
from numpy.testing import assert_almost_equal
from tensorflow import Tensor, Variable
from wheelly.tdlayers import (TDConcat, TDDense, TDLayer, TDLinear, TDRelu,
                               TDSoftmax, TDNetwork, TDSum, TDTanh)

from tests.fixtures import random_cases


def mock_network(case: dict[str, Tensor]):
    return TDNetwork(alpha=float(case["alpha"]),
                   tdlambda=float(case["tdlambda"]),
                   layers={},
                   forward_seq=[],
                   inputs={})

##########################################################################


def dense_cases():
    return random_cases(spec=dict(
        inputs=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        eb=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        b=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        ew=dict(
            type="uniform",
            shape=(2, 3),
            minval=-1
        ),
        w=dict(
            type="uniform",
            shape=(2, 3),
            minval=-1
        ),
        grad=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        delta=dict(
            type="uniform",
            shape=(1,),
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
        )
    ))


def create_dense(case: dict[str, Tensor]):
    eb = Variable(initial_value=tf.reshape(tensor=case["eb"], shape=(1, -1)))
    b = Variable(initial_value=tf.reshape(tensor=case["b"], shape=(1, -1)))
    ew = Variable(initial_value=case["ew"])
    w = Variable(initial_value=case["w"])
    return TDDense(name="name", eb=eb, ew=ew, b=b, w=w)


def test_dense_create():
    spec = dict(
        name="name",
        type="dense",
        input_size=2,
        output_size=3
    )
    layer = TDLayer.create(
        spec=spec, random=tf.random.Generator.from_seed(1234))
    assert isinstance(layer, TDDense)
    assert layer.name == "name"
    assert layer.eb.shape == (1, 3)
    assert layer.ew.shape == (2, 3)
    assert layer.b.shape == (1, 3)
    assert layer.w.shape == (2, 3)
    assert_almost_equal(layer.eb.numpy(), [[0, 0, 0]])
    assert_almost_equal(layer.ew.numpy(), [[0, 0, 0], [0, 0, 0]])
    assert_almost_equal(layer.b.numpy(), [[0, 0, 0]])
    w = layer.w.numpy()
    # probability of test failure = 0.003
    sigma3 = 3 / (2 + 3)
    assert w[0, 0] != 0.0
    assert w[0, 1] != 0.0
    assert w[0, 1] != 0.0
    assert w[1, 0] != 0.0
    assert w[1, 1] != 0.0
    assert w[1, 2] != 0.0
    assert abs(w[0, 0]) < sigma3
    assert abs(w[0, 1]) < sigma3
    assert abs(w[0, 2]) < sigma3
    assert abs(w[1, 0]) < sigma3
    assert abs(w[1, 1]) < sigma3
    assert abs(w[1, 2]) < sigma3


def test_dense_forward():
    for case in dense_cases():
        layer = create_dense(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        in0 = float(inp[0, 0])
        in1 = float(inp[0, 1])
        b0 = float(case ["b"][0])
        b1 = float(case ["b"][1])
        b2 = float(case ["b"][2])
        w00 = float(case ["w"][0, 0])
        w01 = float(case ["w"][0, 1])
        w02 = float(case ["w"][0, 2])
        w10 = float(case ["w"][1, 0])
        w11 = float(case ["w"][1, 1])
        w12 = float(case ["w"][1, 2])
        out = layer.forward(inputs=[inp], net_status=None)
        assert_almost_equal(out.numpy(), [[
            in0 * w00 + in1 * w10 + b0,
            in0 * w01 + in1 * w11 + b1,
            in0 * w02 + in1 * w12 + b2
        ]])


def test_dense_train():
    for case in dense_cases():
        layer = create_dense(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        in0 = float(inp[0, 0])
        in1 = float(inp[0, 1])
        b0 = float(case ["b"][0])
        b1 = float(case ["b"][1])
        b2 = float(case ["b"][2])
        w00 = float(case ["w"][0, 0])
        w01 = float(case ["w"][0, 1])
        w02 = float(case ["w"][0, 2])
        w10 = float(case ["w"][1, 0])
        w11 = float(case ["w"][1, 1])
        w12 = float(case ["w"][1, 2])
        eb0 = float(case ["eb"][0])
        eb1 = float(case ["eb"][1])
        eb2 = float(case ["eb"][2])
        ew00 = float(case ["ew"][0, 0])
        ew01 = float(case ["ew"][0, 1])
        ew02 = float(case ["ew"][0, 2])
        ew10 = float(case ["ew"][1, 0])
        ew11 = float(case ["ew"][1, 1])
        ew12 = float(case ["ew"][1, 2])
        alpha = float(case["alpha"])
        tdlambda = float(case["tdlambda"])
        grad = tf.reshape(tensor=case["grad"], shape=(1, -1))
        grad0 = float(grad[0, 0])
        grad1 = float(grad[0, 1])
        grad2 = float(grad[0, 2])
        delta = tf.reshape(tensor=case["delta"], shape=(1, -1))
        fdelta = float(delta)
        out = layer.forward(inputs=[inp], net_status=mock_network(case))
        post_grad = layer.train(inputs=[inp],
                                output=out,
                                grad=grad,
                                delta=delta,
                                net_status=mock_network(case))

        assert len(post_grad) == 1
        post_eb0 = eb0 * tdlambda + grad0
        post_eb1 = eb1 * tdlambda + grad1
        post_eb2 = eb2 * tdlambda + grad2
        post_ew00 = ew00 * tdlambda + in0 * grad0
        post_ew01 = ew01 * tdlambda + in0 * grad1
        post_ew02 = ew02 * tdlambda + in0 * grad2
        post_ew10 = ew10 * tdlambda + in1 * grad0
        post_ew11 = ew11 * tdlambda + in1 * grad1
        post_ew12 = ew12 * tdlambda + in1 * grad2
        post_b0 = b0 + fdelta * alpha * post_eb0
        post_b1 = b1 + fdelta * alpha * post_eb1
        post_b2 = b2 + fdelta * alpha * post_eb2
        post_w00 = w00 + fdelta * alpha * post_ew00
        post_w01 = w01 + fdelta * alpha * post_ew01
        post_w02 = w02 + fdelta * alpha * post_ew02
        post_w10 = w10 + fdelta * alpha * post_ew10
        post_w11 = w11 + fdelta * alpha * post_ew11
        post_w12 = w12 + fdelta * alpha * post_ew12
        post_grad0 = w00 * grad0 + w01 * grad1 + w02 * grad2
        post_grad1 = w10 * grad0 + w11 * grad1 + w12 * grad2

        assert_almost_equal(post_grad[0].numpy(),
                            [[post_grad0, post_grad1]])

        assert_almost_equal(layer.eb.numpy(),
                            [[post_eb0, post_eb1, post_eb2]])
        assert_almost_equal(layer.ew.numpy(),
                            [[post_ew00, post_ew01, post_ew02],
                            [post_ew10, post_ew11, post_ew12]])
        assert_almost_equal(layer.b.numpy(),
                            [[post_b0, post_b1, post_b2]])
        assert_almost_equal(layer.w.numpy(),
                            [[post_w00, post_w01, post_w02],
                            [post_w10, post_w11, post_w12]])

##########################################################################


def relu_cases():
    return random_cases(spec=dict(
        inputs=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        grad=dict(
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


def create_relu(case: dict[str, Tensor]):
    return TDRelu(name="test")


def test_relu_create():
    spec = dict(
        name="name",
        type="relu"
    )
    layer = TDLayer.create(spec=spec, random=None)
    assert isinstance(layer, TDRelu)
    assert layer.name == "name"


def test_relu_forward():
    for case in relu_cases():
        layer = create_relu(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        in0 = inp[0, 0]
        in1 = inp[0, 1]
        out = layer.forward(inputs=[inp], net_status=None)
        assert_almost_equal(out.numpy(), [[
            in0 if in0 > 0 else 0.0,
            in1 if in1 > 0 else 0.0,
        ]])


def test_relu_train():
    for case in relu_cases():
        layer = create_relu(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        grad = tf.reshape(tensor=case["grad"], shape=(1, -1))
        delta = tf.reshape(tensor=case["delta"], shape=(1, -1))
        in0 = inp[0, 0]
        in1 = inp[0, 1]
        out = layer.forward(inputs=[inp], net_status=None)
        grad0 = grad[0, 0]
        grad1 = grad[0, 1]

        grad_post = layer.train(inputs=[inp],
                                output=out,
                                grad=grad,
                                delta=delta,
                                net_status=None)

        assert len(grad_post) == 1
        assert_almost_equal(grad_post[0].numpy(), [[
            grad0 if in0 > 0 else 0.0,
            grad1 if in1 > 0 else 0.0
        ]])

##########################################################################


def tanh_cases():
    return random_cases(spec=dict(
        inputs=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        grad=dict(
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


def create_tanh(case: dict[str, Tensor]):
    return TDTanh(name="test")


def test_tanh_create():
    spec = dict(
        name="name",
        type="tanh"
    )
    layer = TDLayer.create(spec=spec, random=None)
    assert isinstance(layer, TDTanh)
    assert layer.name == "name"


def test_tanh_forward():
    for case in tanh_cases():
        layer = create_tanh(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        in0 = inp[0, 0]
        in1 = inp[0, 1]
        out = layer.forward(inputs=[inp], net_status=None)
        assert_almost_equal(out.numpy(), [[
            tanh(in0), tanh(in1)
        ]])


def test_tanh_train():
    for case in tanh_cases():
        layer = create_tanh(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        grad = tf.reshape(tensor=case["grad"], shape=(1, -1))
        delta = tf.reshape(tensor=case["delta"], shape=(1, -1))
        in0 = inp[0, 0]
        in1 = inp[0, 1]
        out = layer.forward(inputs=[inp], net_status=None)
        grad0 = grad[0, 0]
        grad1 = grad[0, 1]

        grad_post = layer.train(inputs=[inp],
                                output=out,
                                grad=grad,
                                delta=delta,
                                net_status=None)

        assert len(grad_post) == 1
        assert_almost_equal(grad_post[0].numpy(), [[
            (1 - pow(tanh(in0), 2)) * grad0, (1 - pow(tanh(in1), 2)) * grad1
        ]])

##########################################################################


def linear_cases():
    return random_cases(spec=dict(
        inputs=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        b=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        w=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        grad=dict(
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


def create_linear(case: dict[str, Tensor]):
    return TDLinear(name="test",
                    b=float(case["b"][0]),
                    w=float(case["w"][0]))


def test_linear_create():
    spec = dict(
        name="name",
        type="lin",
        b=1.0,
        w=2.0
    )
    layer = TDLayer.create(spec=spec, random=None)
    assert isinstance(layer, TDLinear)
    assert layer.name == "name"
    assert layer.b == 1
    assert layer.w == 2


def test_linear_forward():
    for case in linear_cases():
        layer = create_linear(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        in0 = inp[0, 0]
        in1 = inp[0, 1]
        b = float(case ["b"][0])
        w = float(case ["w"][0])
        out = layer.forward(inputs=[inp], net_status=None)
        assert_almost_equal(out.numpy(), [[
            in0 * w + b, in1 * w + b
        ]])


def test_linear_train():
    for case in linear_cases():
        layer = create_linear(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        grad = tf.reshape(tensor=case["grad"], shape=(1, -1))
        delta = tf.reshape(tensor=case["delta"], shape=(1, -1))
        w = float(case ["w"][0])
        out = layer.forward(inputs=[inp], net_status=None)
        grad0 = grad[0, 0]
        grad1 = grad[0, 1]

        grad_post = layer.train(inputs=[inp],
                                output=out,
                                grad=grad,
                                delta=delta,
                                net_status=None)

        assert len(grad_post) == 1
        assert_almost_equal(grad_post[0].numpy(),
                            [[grad0 * w,
                              grad1 * w]])

##########################################################################


def softmax_cases():
    return random_cases(spec=dict(
        inputs=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        grad=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        delta=dict(
            type="uniform",
            shape=(1,),
            minval=-1
        ),
        temperature=dict(
            type="exp",
            shape=(1,),
            minval=0.1,
            maxval=1
        )
    ))


def create_softmax(case: dict[str, Tensor]):
    temperature = float(case["temperature"])
    return TDSoftmax(name="test", temperature=temperature)


def test_softmax_create():
    spec = dict(
        name="name",
        type="softmax",
        temperature=0.5
    )
    layer = TDLayer.create(spec=spec, random=None)
    assert isinstance(layer, TDSoftmax)
    assert layer.name == "name"
    assert layer.temperature == 0.5


def test_softmax_forward():
    for case in softmax_cases():
        layer = create_softmax(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        temperature = float(case["temperature"])
        in0 = inp[0, 0]
        in1 = inp[0, 1]
        in2 = inp[0, 2]
        ez0 = exp(in0 / temperature)
        ez1 = exp(in1 / temperature)
        ez2 = exp(in2 / temperature)
        ez = ez0 + ez1 + ez2
        pi0 = ez0 / ez
        pi1 = ez1 / ez
        pi2 = ez2 / ez

        out = layer.forward(inputs=[inp], net_status=None)

        assert_almost_equal(out.numpy(), [[
            pi0, pi1, pi2
        ]])


def test_softmax_train():
    for case in softmax_cases():

        layer = create_softmax(case)
        inp = tf.reshape(tensor=case["inputs"], shape=(1, -1))
        grad = tf.reshape(tensor=case["grad"], shape=(1, -1))
        delta = tf.reshape(tensor=case["delta"], shape=(1, -1))
        temperature = float(case["temperature"])
        in0 = inp[0, 0]
        in1 = inp[0, 1]
        in2 = inp[0, 2]
        ez0 = exp(in0 / temperature)
        ez1 = exp(in1 / temperature)
        ez2 = exp(in2 / temperature)
        ez = ez0 + ez1 + ez2
        pi0 = ez0 / ez
        pi1 = ez1 / ez
        pi2 = ez2 / ez
        grad0 = grad[0, 0]
        grad1 = grad[0, 1]
        grad2 = grad[0, 2]

        post_grad0 = (grad0 * pi0 * (1 - pi0) - grad1 * pi1 *
                      pi0 - grad2 * pi2 * pi0) / temperature
        post_grad1 = (-grad0 * pi0 * pi1 + grad1 * pi1 *
                      (1 - pi1) - grad2 * pi2 * pi1) / temperature
        post_grad2 = (-grad0 * pi0 * pi2 - grad1 * pi1 * pi2 +
                      grad2 * pi2 * (1 - pi2)) / temperature

        out = layer.forward(inputs=[inp], net_status=None)
        grad_post = layer.train(inputs=[inp],
                                output=out,
                                grad=grad,
                                delta=delta,
                                net_status=None)

        assert len(grad_post) == 1
        assert_almost_equal(grad_post[0].numpy(), [[
            post_grad0, post_grad1, post_grad2
        ]], decimal=6, err_msg=f'Case #{case["case_num"]}')

##########################################################################


def sum_cases():
    return random_cases(spec=dict(
        a=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        b=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        grad=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        delta=dict(
            type="uniform",
            shape=(1,),
            minval=-1
        )
    ))


def create_sum(case: dict[str, Tensor]):
    return TDSum(name="test")


def test_sum_create():
    spec = dict(
        name="name",
        type="sum"
    )
    layer = TDLayer.create(spec=spec, random=None)
    assert isinstance(layer, TDSum)
    assert layer.name == "name"


def test_sum_forward():
    for case in sum_cases():
        layer = create_sum(case)
        a = tf.reshape(tensor=case["a"], shape=(1, -1))
        b = tf.reshape(tensor=case["b"], shape=(1, -1))
        a0 = a[0, 0]
        a1 = a[0, 1]
        a2 = a[0, 2]
        b0 = b[0, 0]
        b1 = b[0, 1]
        b2 = b[0, 2]
        inputs = [a, b]

        out = layer.forward(inputs=inputs, net_status=None)

        assert_almost_equal(out.numpy(), [[
            a0 + b0, a1 + b1, a2 + b2
        ]])


def test_sum_train():
    for case in sum_cases():
        layer = create_sum(case)
        layer = create_sum(case)
        a = tf.reshape(tensor=case["a"], shape=(1, -1))
        b = tf.reshape(tensor=case["b"], shape=(1, -1))
        grad = tf.reshape(tensor=case["grad"], shape=(1, -1))
        delta = tf.reshape(tensor=case["delta"], shape=(1, -1))
        grad0 = grad[0, 0]
        grad1 = grad[0, 1]
        grad2 = grad[0, 2]
        inputs = [a, b]

        out = layer.forward(inputs=inputs, net_status=None)

        grad_post = layer.train(inputs=inputs,
                                output=out,
                                grad=grad,
                                delta=delta,
                                net_status=None)

        assert len(grad_post) == 2
        assert_almost_equal(grad_post[0].numpy(), [[
            grad0, grad1, grad2
        ]])
        assert_almost_equal(grad_post[1].numpy(), [[
            grad0, grad1, grad2
        ]])


##########################################################################


def concat_cases():
    return random_cases(spec=dict(
        a=dict(
            type="uniform",
            shape=(2,),
            minval=-1
        ),
        b=dict(
            type="uniform",
            shape=(3,),
            minval=-1
        ),
        grad=dict(
            type="uniform",
            shape=(5,),
            minval=-1
        ),
        delta=dict(
            type="uniform",
            shape=(1,),
            minval=-1
        )
    ))


def create_concat(case: dict[str, Tensor]):
    return TDConcat(name="test")


def test_concat_create():
    spec = dict(
        name="name",
        type="concat"
    )
    layer = TDLayer.create(spec=spec, random=None)
    assert isinstance(layer, TDConcat)
    assert layer.name == "name"


def test_concat_forward():
    for case in concat_cases():
        layer = create_concat(case)
        a = tf.reshape(tensor=case["a"], shape=(1, -1))
        b = tf.reshape(tensor=case["b"], shape=(1, -1))
        a0 = a[0, 0]
        a1 = a[0, 1]
        b0 = b[0, 0]
        b1 = b[0, 1]
        b2 = b[0, 2]
        inputs = [a, b]

        out = layer.forward(inputs=inputs, net_status=None)

        assert_almost_equal(out.numpy(), [[
            a0, a1, b0, b1, b2
        ]])


def test_concat_train():
    for case in concat_cases():
        layer = create_concat(case)
        a = tf.reshape(tensor=case["a"], shape=(1, -1))
        b = tf.reshape(tensor=case["b"], shape=(1, -1))
        grad = tf.reshape(tensor=case["grad"], shape=(1, -1))
        delta = tf.reshape(tensor=case["delta"], shape=(1, -1))
        grad0 = grad[0, 0]
        grad1 = grad[0, 1]
        grad2 = grad[0, 2]
        grad3 = grad[0, 3]
        grad4 = grad[0, 4]
        inputs = [a, b]

        out = layer.forward(inputs=inputs, net_status=None)

        grad_post = layer.train(inputs=inputs,
                                output=out,
                                grad=grad,
                                delta=delta,
                                net_status=None)

        assert len(grad_post) == 2
        assert_almost_equal(grad_post[0].numpy(), [[
            grad0, grad1
        ]])
        assert_almost_equal(grad_post[1].numpy(), [[
            grad2, grad3, grad4
        ]])
