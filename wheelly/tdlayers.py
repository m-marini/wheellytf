from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor, Variable


class TDGraph:
    def __init__(self,
                 forward: list[TDLayer],
                 inputs: dict[str, list[str]],
                 outputs: dict[str, list[str]]) -> None:
        self._forward = forward
        self._backward = list(reversed(forward))
        self._inputs = inputs
        self._outputs = outputs

    @property
    def forward(self) -> list[TDLayer]:
        """Returns the sorted list of node to perform a forward pass (from sources to sinks)"""
        return self._forward

    @property
    def backward(self) -> list[TDLayer]:
        """Returns the sorted list of node to perform a backward pass (from sinks to sources"""
        return self._backward

    def inputs(self, node: TDLayer) -> list[str]:
        """Returns the list of input tensors of a node

        Arguments:
        node -- the node"""
        return self._inputs[node.name]

    def outputs(self, node: TDLayer) -> list[str]:
        """Returns the list of outpus tensors of a node

        Arguments:
        node -- the node"""
        return self._outputs.get(node.name, [])


class TDState:

    def __init__(self,
                 props: dict[str, Tensor],
                 graph: TDGraph,
                 node_props: dict[str, dict[str, Tensor]]) -> None:
        self._props = props
        self._graph = graph
        self._node_props = node_props

    @property
    def props(self) -> dict[str, Tensor]:
        return self._props

    @property
    def graph(self) -> TDGraph:
        return self._graph

    @property
    def node_props(self) -> dict[str, dict[str, Tensor]]:
        """Returns the nodes' properties"""
        return self._node_props

    def prop(self, key: str) -> Tensor:
        """Returns the value of a property of a node

        Arguments:
        key -- the property name"""
        return self.props[key]

    def node_prop(self, node: str, key: str) -> Tensor:
        """Returns the value of a property of a node
        Arguments:

        node -- the node name
        key -- the property name"""
        return self.node_props[node][key]

    def set_prop(self, key: str, value: Tensor):
        """Sets the value of a property

        Arguments:
        key -- the property name
        value -- the value"""
        self.props[key] = value

    def set_node_prop(self, node: str, key: str, value: Tensor):
        """Sets the value of a node property

        Arguments:
        node -- the node name
        key -- the property name
        value -- the value"""
        self.node_props[node][key] = value


class TDLayer():

    def __init__(self, name: str):
        """Creates a DNNLayer

        Arguments:
        names -- name of node"""
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of inputs"""
        return self._name

    def forward(self, inputs: list[Tensor], net_status: TDState) -> Tensor:
        """Performs a forward pass of layer returning the modified context

        Arguments:
        inputs -- the list of inputs
        net_status -- the status of network"""
        raise NotImplementedError()

    def train(self,
              inputs: list[Tensor],
              output: Tensor,
              grad: Tensor,
              delta: Tensor,
              net_status: TDState) -> Tuple[TDState, list[Tensor]]:
        """Performs a backward pass of layer returning the modified network status and the gradients at inputs

        Arguments:
        inputs -- the list of inputs
        output -- the output
        grad -- the gradient of loss function
        delta -- the loss error
        net_status -- the status of network"""
        raise NotImplementedError()


class TDLinear(TDLayer):

    @staticmethod
    def createProps(b: Tensor, w: Tensor) -> dict[str, Tensor]:
        """Returns the initial properties of a linear layer

        Arguments:
        b -- the bias
        w -- the weights"""
        return dict(b=b, w=w)

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDState) -> Tensor:
        w = net_status.node_prop(self.name, "w")
        b = net_status.node_prop(self.name, "b")
        result = inputs[0] * w + b
        return result

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDState) -> Tuple[TDState, list[Tensor]]:
        w = net_status.node_prop(self.name, "w")

        grad1 = grad * w

        return net_status, [grad1]


class TDRelu(TDLayer):
    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDState) -> Tensor:
        return tf.raw_ops.Relu(features=inputs[0])

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDState) -> Tuple[TDState, list[Tensor]]:
        x = inputs[0]
        grad1 = grad * tf.cast(x=x > 0, dtype=tf.float32)
        return net_status, [grad1]


class TDTanh(TDLayer):
    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDState) -> Tensor:
        return tf.math.tanh(x=inputs[0])

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDState) -> Tuple[TDState, list[Tensor]]:
        grad1 = grad * (1 - output * output)
        return net_status, [grad1]


class TDDense(TDLayer):

    @staticmethod
    def createProps(eb: Variable, ew: Variable, b: Variable, w: Variable) -> dict[str, Tensor]:
        """Returns the initial properties of a linear layer

        Arguments:
        eb -- the initial bias eligibility trace
        ew -- the initial weight eligibility trace
        eb -- the initial bias
        eb -- the initial weights"""
        return dict(eb=eb, ew=ew, b=b, w=w)

    @staticmethod
    def initProps(num_inputs: int, num_outputs: int, random: tf.random.Generator) -> dict[str, Tensor]:
        """Returns the initial properties of a linear layer.
        The traces and the bias are set to zeros, the weights are set to ones

        Arguments:
        num_inputs -- the number of inputs
        num_outputs -- the number of outputs
        random -- the random generator"""
        eb = b = tf.zeros(shape=(1, num_outputs), dtype=tf.float32)
        ew = tf.zeros(shape=(num_inputs, num_outputs), dtype=tf.float32)
        # Xavier initialization
        w = random.normal(shape=(num_inputs, num_outputs),
                          stddev=1.0 / (num_inputs + num_outputs))
        return TDDense.createProps(eb=Variable(initial_value=eb, trainable=False),
                                    ew=Variable(initial_value=ew,
                                                trainable=False),
                                    b=Variable(initial_value=b),
                                    w=Variable(initial_value=w))

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDState) -> Tensor:
        w = net_status.node_prop(self.name, "w")
        b = net_status.node_prop(self.name, "b")
        result = inputs[0] @ w + b
        return result

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDState) -> Tuple[TDState, list[Tensor]]:
        ew: Variable = net_status.node_prop(self.name, "ew")
        eb: Variable = net_status.node_prop(self.name, "eb")
        w: Variable = net_status.node_prop(self.name, "w")
        b: Variable = net_status.node_prop(self.name, "b")
        tdlambda = net_status.prop("lambda")
        alpha = net_status.prop("alpha")

        grad1 = grad @ tf.transpose(w)

        eb.assign(tdlambda * eb + grad)

        ws = w.shape
        bgrad = tf.broadcast_to(input=grad, shape=ws)
        bin = tf.broadcast_to(input=tf.transpose(inputs[0]), shape=ws)
        grad_dw = bin * bgrad
        ew.assign(tdlambda * ew + grad_dw)

        b.assign_add(alpha * delta * eb)
        w.assign_add(alpha * delta * ew)
        return net_status, [grad1]


class TDSoftmax(TDLayer):

    @staticmethod
    def initProps(temperature: Tensor):
        """Returns the softmax layer properties

        Arguments:
        temperature -- the temerature of softmax layer"""
        return dict(temperature=temperature)

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDState) -> Tensor:
        t = net_status.node_prop(self.name, "temperature")
        result = tf.nn.softmax(logits=inputs[0] / t)
        return result

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDState) -> Tuple[TDState, list[Tensor]]:
        t = net_status.node_prop(self.name, "temperature")
        lo = grad * output / t
        n = output.shape[1]
        m1 = tf.eye(num_rows=n, num_columns=n)
        yit = m1 - output
        grad1 = lo @ yit
        return net_status, [grad1]


def td_forward(inputs: dict[str, Tensor], state: TDState) -> dict[str, Tensor]:
    """Returns the outputs of each layer

    Arguments:
    inputs -- the input values
    state -- the network state"""
    outs = inputs.copy()
    graph = state.graph
    for node in graph.forward:
        # creates inputs
        inputs = [outs[name] for name in graph.inputs(node)]
        output = node.forward(inputs=inputs, net_status=state)
        outs[node.name] = output
    return outs


def td_train(state: TDState,
              nodes: dict[str, Tensor],
              grad_loss: dict[str, Tensor],
              delta: Tensor) -> Tuple[TDState, dict[str, Tensor]]:
    """Returns the updated network state

    Arguments:
    state -- the network state
    grad_loss -- the gradients of loss function for each output components
    delta -- the error"""
    ctx = grad_loss.copy()
    graph = state.graph
    for node in graph.backward:
        # creates outputs
        input_names = graph.inputs(node)
        inputs = [nodes[name] for name in input_names]
        output = nodes[node.name]
        out_grad = ctx[node.name]
        state, in_grad = node.train(inputs=inputs,
                                    output=output,
                                    grad=out_grad,
                                    delta=delta,
                                    net_status=state)
        for i in range(len(input_names)):
            name = input_names[i]
            value = in_grad[i]
            if name in ctx:
                ctx[name] = ctx[name] + value
            else:
                ctx[name] = value
    return state, ctx
