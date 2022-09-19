import random
from math import exp, tanh

import tensorflow as tf
from numpy.testing import assert_almost_equal
from tensorflow import Tensor, Variable
from this import d
from wheelly.tdlayers import (TDDense, TDLinear, TDRelu, TDSoftmax, TDState,
                            TDTanh)

random.seed = 1234

lin_b0 = random.gauss(mu=0, sigma=0.5)
lin_b1 = random.gauss(mu=0, sigma=0.5)
lin_w0 = random.gauss(mu=0, sigma=0.5)
lin_w1 = random.gauss(mu=0, sigma=0.5)

dense_eb0 = random.gauss(mu=0, sigma=0.5)
dense_eb1 = random.gauss(mu=0, sigma=0.5)
dense_ew01 = random.gauss(mu=0, sigma=0.5)
dense_ew00 = random.gauss(mu=0, sigma=0.5)
dense_ew01 = random.gauss(mu=0, sigma=0.5)
dense_ew10 = random.gauss(mu=0, sigma=0.5)
dense_ew11 = random.gauss(mu=0, sigma=0.5)
dense_ew20 = random.gauss(mu=0, sigma=0.5)
dense_ew21 = random.gauss(mu=0, sigma=0.5)

dense_w00 = random.gauss(mu=0, sigma=0.5)
dense_w01 = random.gauss(mu=0, sigma=0.5)
dense_w10 = random.gauss(mu=0, sigma=0.5)
dense_w11 = random.gauss(mu=0, sigma=0.5)
dense_w20 = random.gauss(mu=0, sigma=0.5)
dense_w21 = random.gauss(mu=0, sigma=0.5)

temperature = 0.869

tdlambda = 0.8
alpha = 1e-3


def create_linear():
    return TDLinear.createProps(b=Variable(initial_value=[[lin_b0, lin_b1]],
                                            dtype=tf.float32),
                                 w=Variable(initial_value=[[lin_w0, lin_w1]], dtype=tf.float32))


def create_dense():
    return TDDense.createProps(eb=Variable(initial_value=[[dense_eb0, dense_eb1]], dtype=tf.float32, trainable=False),
                                ew=Variable(initial_value=[[
                                    dense_ew00, dense_ew01], [
                                    dense_ew10, dense_ew11], [
                                    dense_ew20, dense_ew21
                                ]], dtype=tf.float32, trainable=False),
                                b=Variable(initial_value=[
                                           [lin_b0, lin_b1]], dtype=tf.float32),
                                w=Variable(initial_value=[[
                                    dense_w00, dense_w01], [
                                    dense_w10, dense_w11], [
                                    dense_w20, dense_w21
                                ]], dtype=tf.float32))


def create_softmax():
    return TDSoftmax.initProps(temperature=tf.constant(temperature, dtype=tf.float32))


def create_state(name: str, node_props: dict[str, Tensor]):
    props = {
        "lambda": tf.constant(value=tdlambda, dtype=tf.float32),
        "alpha": tf.constant(value=alpha, dtype=tf.float32),
    }
    nodes = {
        name: node_props
    }
    return TDState(props=props,
                    graph=None,
                    node_props=nodes)


def test_state():
    props = {"b": tf.zeros(shape=0)}
    state = create_state(name="a", node_props=props)
    assert_almost_equal(state.prop(key="lambda").numpy(), tdlambda)
    assert_almost_equal(state.prop(key="alpha").numpy(), alpha)
    assert_almost_equal(state.node_prop(node="a", key="b").numpy(), 0)


def test_create_linear():
    props = create_linear()
    assert isinstance(props, dict)
    assert_almost_equal(props["b"].numpy(), [[lin_b0, lin_b1]])
    assert_almost_equal(props["w"].numpy(), [[lin_w0, lin_w1]])


def test_forward_linear():
    props = create_linear()
    layer = TDLinear(name="layer")
    in0 = 0.1
    in1 = 0.2
    inp = tf.constant(value=[[in0, in1]], dtype=tf.float32)
    state = create_state(name="layer", node_props=props)
    out = layer.forward(inputs=[inp], net_status=state)
    assert_almost_equal(out.numpy(), [[
        in0 * lin_w0 + lin_b0, in1 * lin_w1 + lin_b1
    ]])


def test_train_linear():
    props = create_linear()
    layer = TDLinear(name="layer")
    in0 = 0.1
    in1 = 0.2
    inp = tf.constant(value=[[in0, in1]], dtype=tf.float32)
    state = create_state(name="layer", node_props=props)
    out = layer.forward(inputs=[inp], net_status=state)
    grad0 = 0.3
    grad1 = 0.5
    delta = 0.2

    state1, grad_post = layer.train(inputs=[inp],
                                  output=out,
                                  grad=tf.constant(
        [[grad0, grad1]], dtype=tf.float32),
        delta=tf.constant(
        [[delta]], dtype=tf.float32),
        net_status=state)

    assert len(grad_post) == 1
    assert_almost_equal(grad_post[0].numpy(),
                        [[grad0 * lin_w0,
                          grad1 * lin_w1]])


def test_forward_relu():
    layer = TDRelu("layer")
    in0 = 0.1
    in1 = -0.2
    inp = tf.constant(value=[[in0, in1]], dtype=tf.float32)
    state = create_state(name="layer", node_props={})
    out = layer.forward(inputs=[inp], net_status=state)
    assert_almost_equal(out.numpy(), [[
        0.1, 0
    ]])


def test_train_relu():
    layer = TDRelu(name="layer")
    in0 = 0.1
    in1 = -0.2
    inp = tf.constant(value=[[in0, in1]], dtype=tf.float32)
    state = create_state(name="layer", node_props={})
    out = layer.forward(inputs=[inp], net_status=state)
    grad0 = 0.3
    grad1 = 0.5
    delta = 0.2

    state1, grad_post = layer.train(inputs=[inp],
                                  output=out,
                                  grad=tf.constant(
        [[grad0, grad1]], dtype=tf.float32),
        delta=tf.constant(
        [[delta]], dtype=tf.float32),
        net_status=state)

    assert len(grad_post) == 1
    assert_almost_equal(grad_post[0].numpy(),
                        [[grad0,
                          0]])
    assert state1 == state


def test_forward_tanh():
    layer = TDTanh("layer")
    in0 = -1
    in1 = 0
    in2 = 1
    inp = tf.constant(value=[[in0, in1, in2]], dtype=tf.float32)
    state = create_state(name="layer", node_props={})
    out = layer.forward(inputs=[inp], net_status=state)
    assert_almost_equal(out.numpy(), [[
        tanh(in0), 0, tanh(in2)
    ]])


def test_train_tanh():
    layer = TDTanh("layer")
    in0 = -1
    in1 = 0
    in2 = 1
    inp = tf.constant(value=[[in0, in1, in2]], dtype=tf.float32)
    state = create_state(name="layer", node_props={})
    out = layer.forward(inputs=[inp], net_status=state)
    grad0 = 0.3
    grad1 = 0.1
    grad2 = -0.3
    delta = 0.2

    state1, grad_post = layer.train(inputs=[inp],
                                  output=out,
                                  grad=tf.constant(
        [[grad0, grad1, grad2]], dtype=tf.float32),
        delta=tf.constant(
        [[delta]], dtype=tf.float32),
        net_status=state)

    assert len(grad_post) == 1
    assert_almost_equal(grad_post[0].numpy(),
                        [[(1 - pow(tanh(in0), 2)) * grad0,
                          (1 - pow(tanh(in1), 2)) * grad1,
                            (1 - pow(tanh(in2), 2)) * grad2]])
    assert state1 == state


def test_create_dense():
    props = create_dense()
    assert isinstance(props, dict)
    assert_almost_equal(props["eb"].numpy(), [[dense_eb0, dense_eb1]])
    assert_almost_equal(props["ew"].numpy(), [[
        dense_ew00, dense_ew01], [
        dense_ew10, dense_ew11], [
        dense_ew20, dense_ew21]])
    assert_almost_equal(props["b"].numpy(), [[lin_b0, lin_b1]])
    assert_almost_equal(props["w"].numpy(), [[
        dense_w00, dense_w01], [
        dense_w10, dense_w11], [
        dense_w20, dense_w21]])


def test_init_dense():
    random = tf.random.Generator.from_seed(seed=1234)
    props = TDDense.initProps(num_inputs=3, num_outputs=2, random=random)
    assert isinstance(props, dict)
    assert_almost_equal(props["eb"].numpy(), [[0, 0]])
    assert_almost_equal(props["ew"].numpy(), [[0, 0], [0, 0], [0, 0]])
    assert_almost_equal(props["b"].numpy(), [[0, 0]])
    w = props["w"].numpy()
    # probability of test failure = 0.003
    sigma3 = 3 / (2 + 2)
    assert w[0, 0] != 0.0
    assert w[0, 1] != 0.0
    assert w[1, 0] != 0.0
    assert w[1, 1] != 0.0
    assert w[2, 0] != 0.0
    assert w[2, 1] != 0.0
    assert abs(w[0, 0]) < sigma3
    assert abs(w[0, 1]) < sigma3
    assert abs(w[1, 0]) < sigma3
    assert abs(w[1, 1]) < sigma3
    assert abs(w[2, 0]) < sigma3
    assert abs(w[2, 1]) < sigma3


def test_forward_dense():
    props = create_dense()
    layer = TDDense(name="layer")
    in0 = 0.1
    in1 = 0.2
    in2 = -0.2
    inp = tf.constant(value=[[in0, in1, in2]], dtype=tf.float32)
    state = create_state(name="layer", node_props=props)
    out = layer.forward(inputs=[inp], net_status=state)
    assert_almost_equal(out.numpy(), [[
        in0 * dense_w00 + in1 * dense_w10 + in2 * dense_w20 + lin_b0,
        in0 * dense_w01 + in1 * dense_w11 + in2 * dense_w21 + lin_b1,
    ]])


def test_train_dense():
    props = create_dense()
    layer = TDDense(name="layer")
    in0 = 0.1
    in1 = 0.2
    in2 = -0.2
    inp = tf.constant(value=[[in0, in1, in2]], dtype=tf.float32)
    state = create_state(name="layer", node_props=props)
    out = layer.forward(inputs=[inp], net_status=state)
    grad0 = 0.3
    grad1 = 0.5
    delta = 0.2

    state1, grad_post = layer.train(inputs=[inp],
                                  output=out,
                                  grad=tf.constant(
        [[grad0, grad1]], dtype=tf.float32),
        delta=tf.constant(
        [[delta]], dtype=tf.float32),
        net_status=state)

    assert len(grad_post) == 1
    post_eb0 = dense_eb0 * tdlambda + grad0
    post_eb1 = dense_eb1 * tdlambda + grad1
    post_ew00 = dense_ew00 * tdlambda + in0 * grad0
    post_ew01 = dense_ew01 * tdlambda + in0 * grad1
    post_ew10 = dense_ew10 * tdlambda + in1 * grad0
    post_ew11 = dense_ew11 * tdlambda + in1 * grad1
    post_ew20 = dense_ew20 * tdlambda + in2 * grad0
    post_ew21 = dense_ew21 * tdlambda + in2 * grad1

    assert_almost_equal(grad_post[0].numpy(),
                        [[dense_w00 * grad0 + dense_w01 * grad1,
                          dense_w10 * grad0 + dense_w11 * grad1,
                          dense_w20 * grad0 + dense_w21 * grad1]])

    assert_almost_equal(state1.node_prop(node="layer", key="eb").numpy(),
                        [[post_eb0, post_eb1]])
    assert_almost_equal(state1.node_prop(node="layer", key="ew").numpy(),
                        [[post_ew00, post_ew01],
                         [post_ew10, post_ew11],
                         [post_ew20, post_ew21]])
    assert_almost_equal(state1.node_prop(node="layer", key="b").numpy(),
                        [[lin_b0 + delta * alpha * post_eb0,
                          lin_b1 + delta * alpha * post_eb1]])
    assert_almost_equal(state1.node_prop(node="layer", key="w").numpy(),
                        [[
                            dense_w00 + delta * alpha * post_ew00,
                            dense_w01 + delta * alpha * post_ew01], [
                            dense_w10 + delta * alpha * post_ew10,
                            dense_w11 + delta * alpha * post_ew11], [
                            dense_w20 + delta * alpha * post_ew20,
                            dense_w21 + delta * alpha * post_ew21]])


def test_create_softmax():
    props = create_softmax()
    assert isinstance(props, dict)
    assert_almost_equal(props["temperature"].numpy(), temperature)


def test_forward_softmax():
    props = create_softmax()
    layer = TDSoftmax(name="layer")
    in0 = -1
    in1 = 0
    in2 = 1
    inp = tf.constant(value=[[in0, in1, in2]], dtype=tf.float32)
    state = create_state(name="layer", node_props=props)
    out = layer.forward(inputs=[inp], net_status=state)

    ez0 = exp(in0 / temperature)
    ez1 = exp(in1 / temperature)
    ez2 = exp(in2 / temperature)
    ez = ez0 + ez1 + ez2
    pi0 = ez0 / ez
    pi1 = ez1 / ez
    pi2 = ez2 / ez
    assert_almost_equal(out.numpy(), [[
        pi0, pi1, pi2
    ]])


def test_train_softmax():
    props = create_softmax()
    layer = TDSoftmax(name="layer")
    in0 = -1
    in1 = 0
    in2 = 1
    inp = tf.constant(value=[[in0, in1, in2]], dtype=tf.float32)
    state = create_state(name="layer", node_props=props)
    out = layer.forward(inputs=[inp], net_status=state)
    grad0 = 0.3
    grad1 = 0.5
    grad2 = 0.1
    delta = 0.2

    state1, grad_post = layer.train(inputs=[inp],
                                  output=out,
                                  grad=tf.constant(
        [[grad0, grad1, grad2]], dtype=tf.float32),
        delta=tf.constant(
        [[delta]], dtype=tf.float32),
        net_status=state)

    ez0 = exp(in0 / temperature)
    ez1 = exp(in1 / temperature)
    ez2 = exp(in2 / temperature)
    ez = ez0 + ez1 + ez2
    pi0 = ez0 / ez
    pi1 = ez1 / ez
    pi2 = ez2 / ez

    assert len(grad_post) == 1

    assert_almost_equal(grad_post[0].numpy(),
                        [[(grad0 * pi0 * (1 - pi0) - grad1 * pi1 * pi0 - grad2 * pi2 * pi0) / temperature,
                            (-grad0 * pi0 * pi1 + grad1 * pi1 *
                             (1 - pi1) - grad2 * pi2 * pi1) / temperature,
                            (-grad0 * pi0 * pi2 - grad1 * pi1 * pi2 + grad2 * pi2 * (1 - pi2)) / temperature]])
    assert state1 == state
