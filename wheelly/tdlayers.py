from __future__ import annotations

from typing import Any, Tuple

import tensorflow as tf
from tensorflow import Tensor, Variable


class TDLayer():
    @staticmethod
    def create(name: str, spec: dict[str, Any]) -> TDLayer:
        """Creates a layer

        Arguments:
        spec -- the layer specification"""
        type = spec["type"]
        if type == "dense":
            return TDDense(name)
        elif type == "relu":
            return TDRelu(name)
        elif type == "tanh":
            return TDTanh(name)
        elif type == "softmax":
            return TDSoftmax(name)
        else:
            raise Exception(f'Layer type "{type}" not supported')

    @staticmethod
    def by_spec(spec: dict[str, Any], random: tf.random.Generator):
        """Creates a layer

        Arguments:
        spec -- the layer specification"""
        type = spec["type"]
        if type == "dense":
            return TDDense.by_spec(spec=spec, random=random)
        elif type == "relu":
            return TDRelu.by_spec(spec)
        elif type == "linear":
            return TDLinear.by_spec(spec)
        elif type == "tanh":
            return TDTanh.by_spec(spec)
        elif type == "softmax":
            return TDSoftmax.by_spec(spec)
        else:
            raise Exception(f'Layer type "{type}" not supported')

    def __init__(self, name: str):
        """Creates a DNNLayer

        Arguments:
        names -- name of node"""
        self.name = name

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

    def spec(self, props: dict[str, Tensor]) -> dict[str, Any]:
        """Returns the layer specification

        Arguments:
        props -- the layer properties"""
        raise NotImplementedError()


class TDLinear(TDLayer):
    @staticmethod
    def by_spec(spec: dict[str, Any]):
        return TDLinear(spec["name"]), TDLinear.createProps(0.0, 1.0)

    @staticmethod
    def createProps(b: float, w: float) -> dict[str, Tensor]:
        """Returns the initial properties of a linear layer

        Arguments:
        b -- the bias
        w -- the weights"""
        return dict(b=Variable(initial_value=b, trainable=False),
                    w=Variable(initial_value=w, trainable=False))

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

    def spec(self, props: dict[str, Tensor]) -> dict[str, Any]:
        return dict(name=self.name, type="linear")


class TDRelu(TDLayer):
    @staticmethod
    def by_spec(spec: dict[str, Any]):
        return TDRelu(spec["name"]), None

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDState) -> Tensor:
        return tf.raw_ops.Relu(features=inputs[0])

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDState) -> Tuple[TDState, list[Tensor]]:
        x = inputs[0]
        grad1 = grad * tf.cast(x=x > 0, dtype=tf.float32)
        return net_status, [grad1]

    def spec(self, props: dict[str, Tensor]) -> dict[str, Any]:
        return dict(name=self.name, type="relu")


class TDTanh(TDLayer):
    @staticmethod
    def by_spec(spec: dict[str, Any]):
        return TDTanh(spec["name"]), None

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDState) -> Tensor:
        return tf.math.tanh(x=inputs[0])

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDState) -> Tuple[TDState, list[Tensor]]:
        grad1 = grad * (1 - output * output)
        return net_status, [grad1]

    def spec(self, props: dict[str, Tensor]) -> dict[str, Any]:
        return dict(name=self.name, type="tanh")


class TDDense(TDLayer):
    @staticmethod
    def by_spec(spec: dict[str, Any], random: tf.random.Generator):
        return TDDense(spec["name"]), TDDense.initProps(num_inputs=spec["num_inputs"],
                                                        num_outputs=spec["num_outputs"],
                                                        random=random)

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

    def spec(self, props: dict[str, Tensor]) -> dict[str, Any]:
        ew = props["ew"]
        return dict(name=self.name,
                    type="dense",
                    num_inputs=ew.shape[0],
                    num_outputs=ew.shape[1])


class TDSoftmax(TDLayer):
    @staticmethod
    def by_spec(spec: dict[str, Any]):
        return TDSoftmax(spec["name"]), TDSoftmax.initProps(1.0)

    @staticmethod
    def initProps(temperature: float):
        """Returns the softmax layer properties

        Arguments:
        temperature -- the temerature of softmax layer"""
        return dict(temperature=Variable(initial_value=temperature, trainable=False))

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

    def spec(self, props: dict[str, Tensor]) -> dict[str, Any]:
        return dict(name=self.name, type="softmax")


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


class TDState(tf.train.Checkpoint):
    @staticmethod
    def by_spec(spec: list[dict[str, Any]],
                random: tf.random.Generator) -> TDState:
        props = {"alpha": Variable(initial_value=1e-3, trainable=False),
                 "lambda": Variable(initial_value=0.0, trainable=False)}
        from_spec = [TDLayer.by_spec(spec=s, random=random) for s in spec]
        forward = [layer for layer, _ in from_spec]
        inputs = {s["name"]: s["inputs"] for s in spec}
        outputs = {node["name"]: [inp["name"]
         for inp in spec if node["name"] in inp["inputs"]]
         for node in spec}
        graph = TDGraph(forward=forward,
                        inputs=inputs,
                        outputs=outputs)
        node_props = {layer.name: props for layer, props in from_spec}
        return TDState(props=props,
                       graph=graph,
                       node_props=node_props)

    @staticmethod
    def create(net_spec: dict[str, Any],
               state_spec: dict[str, Any],
               actions_spec: dict[str, Any],
               random: tf.random.Generator) -> TDState:
        """Returns the network state from network specification

        Arguments:
        net_spec -- network specification
        state_spec -- flatten state specification
        actions_spec -- flatten actions specification
        random -- the random generator"""
        assert "alpha" in net_spec, f'Missing "alpha" parameter'
        assert "lambda" in net_spec, f'Missing "lambda" parameter'
        assert "network" in net_spec, f'Missing "network" parameter'
        props = {"alpha": tf.Variable(initial_value=net_spec["alpha"], trainable=False),
                 "lambda": tf.Variable(initial_value=net_spec["lambda"], trainable=False)}
        net_spec1 = net_spec["network"]
        layers_spec = parse_for_inputs(
            net_spec=net_spec1, state_spec=state_spec)
        layers_spec = parse_for_size(
            layers_spec=layers_spec, state_spec=state_spec)
        validate_output(layers_spec=layers_spec, output_spec=actions_spec)

        graph = create_graph(layers_spec)
        node_props = create_layers_props(spec=layers_spec, random=random)

        return TDState(props=props, graph=graph, node_props=node_props)

    def __init__(self,
                 props: dict[str, Tensor],
                 graph: TDGraph,
                 node_props: dict[str, dict[str, Tensor]]) -> None:
        super().__init__(None)
        self.props = props
        self.graph = graph
        self.node_props = node_props

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

    def spec(self) -> dict[str, Any]:
        def full_layer_spec(layer: TDLayer):
            result = layer.spec(self.node_props.get(layer.name, None))
            result["inputs"] = self.graph.inputs(layer)
            return result

        layers = [full_layer_spec(layer) for layer in self.graph.forward]
        return layers


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


def create_graph(spec: dict[str, Any]) -> TDGraph:
    """Returns the graph for a lyaer specification"

    Arguments:
    spec -- the full layer specification"""
    names = sort_layers(spec)
    forward = [TDLayer.create(name=name, spec=spec[name]) for name in names]
    inputs = {key: [layer_spec["input"]] for key, layer_spec in spec.items()}
    outputs = {key: find_outputs(spec=spec, layer_name=key) for key in spec}

    return TDGraph(forward=forward, inputs=inputs, outputs=outputs)


def create_layer_props(spec: dict[str, Any], random: tf.random.Generator) -> dict[str, Tensor] | None:
    """Returns the initial properties of a layer or None if not available

    Arguments:
    spec -- layer specification
    random -- the random generator"""
    type = spec["type"]
    if type == "dense":
        return TDDense.initProps(num_inputs=spec["inp_size"], num_outputs=spec["size"], random=random)
    elif type == "softmax":
        return TDSoftmax.initProps(temperature=tf.constant(spec.get("temperature", 1), dtype=tf.float32))
    else:
        return None


def create_layers_props(spec: dict[str, Any], random: tf.random.Generator) -> dict[str, dict[str, Tensor]]:
    """Returns the initial properties of the layers

    Arguments:
    spec -- full layers specification
    random -- the random generator"""
    result = {}
    for key, layer_spec in spec.items():
        props = create_layer_props(spec=layer_spec, random=random)
        if props is not None:
            result[key] = props
    return result


def find_outputs(spec: dict[str, Any], layer_name: str) -> list[str]:
    """Returns the outputs of a layer

    Arguments:
    spec -- the full ayers specification
    layer_name -- the layer name to find the output for"""

    result = [key for key, layer_spec in spec.items() if key !=
              layer_name and layer_spec["input"] == layer_name]

    """
    result = []
    for key in spec:
        if key != layer_name and spec["input"] == layer_name:
            result.append(key)
    """
    return result


def flat_type_spec(spec: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Returns the flat type specification

    Arguments:
    spec -- the type specification
    prefix -- the prefix of the flat type"""
    result = {}
    def flat(prefix: str,
             spec: dict[str, Any]):
        if "type" in spec:
            result[prefix] = spec
        else:
            for key, sub_spec in spec.items():
                flat(prefix=f"{prefix}.{key}",
                     spec=sub_spec)

    flat(spec=spec, prefix=prefix)
    return result


def get_output_size(spec: dict[str, Any]) -> int:
    """Returns the size of output of a layer specification

    Argument:
    spec -- the single layer specification"""
    assert "type" in spec, "Missing type"
    type = spec["type"]
    if type == "dense":
        assert "size" in spec, "Missing size"
        return spec["size"]
    else:
        return spec["inp_size"]


def parse_for_size(layers_spec: dict[str, Any], state_spec: dict[str, Any]):
    """Returns the layer specification agumented by infered inp_size and size

    Arguments:
    layers_spec -- the full layer specification
    state_spec -- the flatten state type specification"""
    sorted = sort_layers(layers_spec)
    for name in sorted:
        spec = layers_spec[name]
        input = spec["input"]
        if input in layers_spec:
            spec["inp_size"] = layers_spec[input]["size"]
        else:
            inp_spec = state_spec[input]
            assert "shape" in inp_spec, f'Missing shape for "{input}"'
            shape = inp_spec["shape"]
            assert len(shape) == 1, f'Shape rank must be 1 ({len(shape)})'
            spec["inp_size"] = shape[0]
        spec["size"] = get_output_size(spec)
    return layers_spec


def sort_layers(layers_spec: dict[str, Any]) -> list[str]:
    """Returns the sorted layers for forward pass

    Arguments:
    layers_spec -- the full layer specification"""
    inputs = set()
    for spec in layers_spec.values():
        inputs.add(spec["input"])

    sinks = [name for name in layers_spec if name not in inputs]

    ordered = []

    def sort(node: str):
        if node not in ordered and node in layers_spec:
            sort(layers_spec[node]["input"])
            ordered.append(node)

    for node in sinks:
        sort(node)
    return ordered


def parse_for_inputs(net_spec: dict[str, Any], state_spec: dict[str, Any]) -> dict[str, Any]:
    """Returns the expanded full layers specification

    Arguments:
    net_spec -- the network specification
    state_spec -- the flatten state type specification """
    layers_spec = {}
    for seq_name, seq_spec in net_spec.items():
        input = seq_spec.get("input", "input")
        layers = seq_spec["layers"]
        n = len(layers)
        for i in range(n):
            layer_spec = layers[i].copy()
            layer_name = seq_name if i >= n - 1 else f"{seq_name}[{i}]"
            layer_spec["input"] = input
            layers_spec[layer_name] = layer_spec
            input = layer_name
    for spec in layers_spec.values():
        input = spec["input"]
        assert (input in layers_spec) or (
            input in state_spec), f'Input "{input}" undefined'
    return layers_spec


def validate_output(layers_spec: dict[str, Any], output_spec=dict[str, Any]):
    """Validate the layers specification for output definition

    Arguments:
    layers_spec -- the full layer specification
    output_spec -- the flatten output specification"""
    for name, spec in output_spec.items():
        assert name in layers_spec, f'Missing output for "{name}"'
        assert "type" in spec, f'Missing "type" spec for "{name}"'
        assert "shape" in spec, f'Missing "shape" spec for "{name}"'
        type = spec["type"]
        shape = spec["shape"]
        assert len(
            shape) == 1, f'Shape rank must be = 1 for "{name} ({len(shape)})'
        if type == "int":
            assert shape[0] == 1, f'Shape must be = 1 for "{name} ({shape[0]})'
            assert "num_values" in spec, f'Missing "num_values" spec for "{name}"'
            num_values = spec["num_values"]
            assert layers_spec[name]["size"] == num_values, \
                f'Unmatched size for "{name}" ({layers_spec[name]["size"]}) != ({num_values})'
        else:
            assert layers_spec[name]["size"] == shape[0], \
                f'Unmatched size for "{name}" ({layers_spec[name]["size"]}) != ({shape[0]})'
