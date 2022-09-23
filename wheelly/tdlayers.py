from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow import Tensor, Variable


class TDNetwork(tf.train.Checkpoint):
    @staticmethod
    def create(spec: dict[str, Any], random: tf.random.Generator) -> TDNetwork:
        layers = {ls["name"]: TDLayer.create(
            spec=ls, random=random) for ls in spec["layers"]}
        forward_seq = [ls["name"] for ls in spec["layers"]]
        inputs = {ls["name"]: ls["inputs"] for ls in spec["layers"]}
        net = TDNetwork(alpha=spec["alpha"],
                        tdlambda=spec["tdlambda"],
                        layers=layers,
                        forward_seq=forward_seq,
                        inputs=inputs
                        )
        return net

    def __init__(self,
                 alpha: float,
                 tdlambda: float,
                 layers: dict[str, TDLayer],
                 forward_seq: list[str],
                 inputs: dict[str, list[str]]):
        super().__init__()
        assert isinstance(alpha, float)
        assert isinstance(tdlambda, float)
        for id, node in layers.items():
            assert id == node.name
            assert id in inputs

        self.alpha = alpha
        self.tdlambda = tdlambda
        self.layers = layers
        self.forward_seq = forward_seq
        self.backward_seq = list(reversed(forward_seq))
        self.inputs = inputs
        self.outputs = {key: [out
                              for out in inputs
                              if key in inputs[out]
                              ]
                        for key in layers
                        }

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        outs = inputs.copy()
        for id in self.forward_seq:
            # creates inputs
            layer = self.layers[id]
            inputs = [outs[name] for name in self.inputs[id]]
            output = layer.forward(inputs=inputs, net_status=self)
            outs[id] = output
        return outs

    def train(self,
              outputs: dict[str, Tensor],
              grad_outputs: dict[str, Tensor],
              delta: Tensor) -> dict[str, Tensor]:
        """Updates the network and returns the layer gradients

        Arguments:
        outputs -- the layers outputs
        grad_outputs -- the gradients of each output components
        delta -- the error"""
        grads = grad_outputs.copy()
        for id in self.backward_seq:
            node = self.layers[id]
            # creates outputs
            input_names = self.inputs[id]
            inputs = [outputs[name] for name in input_names]
            output = outputs[id]
            out_grad = grads[id]
            in_grad = node.train(inputs=inputs,
                                 output=output,
                                 grad=out_grad,
                                 delta=delta,
                                 net_status=self)
            for i in range(len(input_names)):
                name = input_names[i]
                value = in_grad[i]
                if name in grads:
                    grads[name] = grads[name] + value
                else:
                    grads[name] = value
        return grads

    def spec(self) -> dict[str, Any]:
        def layer_spec(layer: TDLayer):
            spec = layer.spec()
            spec["inputs"] = self.inputs[layer.name]
            return spec

        spec = dict(alpha=self.alpha,
                    tdlambda=self.tdlambda,
                    layers=[layer_spec(self.layers[id]) for id in self.forward_seq])
        return spec


class TDLayer(tf.train.Checkpoint):
    @ staticmethod
    def create(spec: dict[str, Any], random: tf.random.Generator) -> TDLayer:
        type = spec["type"]
        if type == "dense":
            return TDDense.create(spec=spec, random=random)
        elif type == "relu":
            return TDRelu.create(spec)
        elif type == "tanh":
            return TDTanh.create(spec)
        elif type == "lin":
            return TDLinear.create(spec)
        elif type == "softmax":
            return TDSoftmax.create(spec)
        elif type == "sum":
            return TDSum.create(spec)
        elif type == "concat":
            return TDConcat.create(spec)
        else:
            raise Exception(f'Type "{type}" unknown')

    def __init__(self, name: str):
        """Creates a DNNLayer

        Arguments:
        names -- name of node"""
        super().__init__()
        assert isinstance(name, str)
        self.name = name

    def forward(self, inputs: list[Tensor], net_status: TDNetwork) -> Tensor:
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
              net_status: TDNetwork) -> list[Tensor]:
        """Performs a backward pass of layer returning the modified network status and the gradients at inputs

        Arguments:
        inputs -- the list of inputs
        output -- the output
        grad -- the gradient of loss function
        delta -- the loss error
        net_status -- the status of network"""
        raise NotImplementedError()

    def spec(self) -> dict[str, Any]:
        raise NotImplementedError()


class TDDense(TDLayer):
    @staticmethod
    def create(spec: dict[str, Any], random: tf.random.Generator):
        """Returns a dense layer.

        Arguments:
        spec -- the layer specification
        random -- the random generator"""
        output_size = spec["output_size"]
        input_size = spec["input_size"]
        eb = b = tf.zeros(shape=(1, output_size), dtype=tf.float32)
        ew = tf.zeros(shape=(input_size, output_size), dtype=tf.float32)
        # Xavier initialization
        w = random.normal(shape=(input_size, output_size),
                          stddev=1.0 / (input_size + output_size))
        return TDDense(name=spec["name"],
                       eb=Variable(initial_value=eb, trainable=False),
                       ew=Variable(initial_value=ew, trainable=False),
                       b=Variable(initial_value=b),
                       w=Variable(initial_value=w))

    def __init__(self, name: str, eb: Variable, ew: Variable, b: Variable, w: Variable):
        super().__init__(name)
        assert isinstance(eb, Variable)
        assert isinstance(ew, Variable)
        assert isinstance(b, Variable)
        assert isinstance(w, Variable)
        assert len(eb.shape) == 2
        assert len(ew.shape) == 2
        assert len(b.shape) == 2
        assert len(w.shape) == 2
        assert eb.shape[0] == 1
        assert eb.shape == b.shape
        assert eb.shape[1] == ew.shape[1]
        assert ew.shape == w.shape
        self.eb = eb
        self.ew = ew
        self.b = b
        self.w = w

    def forward(self, inputs: list[Tensor], net_status: TDNetwork) -> Tensor:
        result = inputs[0] @ self.w + self.b
        return result

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDNetwork) -> list[Tensor]:
        tdlambda = net_status.tdlambda
        alpha = net_status.alpha

        grad1 = grad @ tf.transpose(self.w)

        self.eb.assign(tdlambda * self.eb + grad)

        ws = self.w.shape
        bgrad = tf.broadcast_to(input=grad, shape=ws)
        bin = tf.broadcast_to(input=tf.transpose(inputs[0]), shape=ws)
        grad_dw = bin * bgrad
        self.ew.assign(tdlambda * self.ew + grad_dw)

        self.b.assign_add(alpha * delta * self.eb)
        self.w.assign_add(alpha * delta * self.ew)
        return [grad1]

    def spec(self) -> dict[str, Any]:
        return dict(name=self.name,
                    type="dense",
                    input_size=self.w.shape[0],
                    output_size=self.w.shape[1])


class TDRelu(TDLayer):
    @staticmethod
    def create(spec: dict[str, Any]):
        return TDRelu(name=spec["name"])

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDNetwork) -> Tensor:
        return tf.raw_ops.Relu(features=inputs[0])

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDNetwork) -> list[Tensor]:
        x = inputs[0]
        grad1 = grad * tf.cast(x=x > 0, dtype=tf.float32)
        return [grad1]

    def spec(self) -> dict[str, Any]:
        return dict(name=self.name,
                    type="relu")


class TDTanh(TDLayer):
    @staticmethod
    def create(spec: dict[str, Any]):
        return TDTanh(name=spec["name"])

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDNetwork) -> Tensor:
        return tf.math.tanh(inputs[0])

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDNetwork) -> list[Tensor]:
        grad1 = grad * (1 - output * output)
        return [grad1]

    def spec(self) -> dict[str, Any]:
        return dict(name=self.name,
                    type="tanh")


class TDLinear(TDLayer):

    @staticmethod
    def create(spec: dict[str, Any]):
        return TDLinear(name=spec["name"],
                        b=spec["b"],
                        w=spec["w"])

    def __init__(self, name: str, b: float, w: float):
        super().__init__(name)
        assert isinstance(b, float)
        assert isinstance(w, float)
        self.b = b
        self.w = w

    def forward(self, inputs: list[Tensor], net_status: TDNetwork) -> Tensor:
        result = inputs[0] * self.w + self.b
        return result

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDNetwork) -> list[Tensor]:
        grad1 = grad * self.w
        return [grad1]

    def spec(self) -> dict[str, Any]:
        return dict(name=self.name,
                    type="lin",
                    b=self.b,
                    w=self.w)


class TDSoftmax(TDLayer):
    @staticmethod
    def create(spec: dict[str, Any]):
        return TDSoftmax(name=spec["name"],
                         temperature=spec["temperature"])

    def __init__(self, name: str, temperature: float):
        super().__init__(name)
        assert isinstance(temperature, float)
        self.temperature = temperature

    def forward(self, inputs: list[Tensor], net_status: TDNetwork) -> Tensor:
        result = tf.nn.softmax(logits=inputs[0] / self.temperature)
        return result

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDNetwork) -> list[Tensor]:
        lo = grad * output / self.temperature
        n = output.shape[1]
        m1 = tf.eye(num_rows=n, num_columns=n)
        yit = m1 - output
        grad1 = lo @ yit
        return [grad1]

    def spec(self) -> dict[str, Any]:
        return dict(name=self.name,
                    type="softmax",
                    temperature=self.temperature)


class TDSum(TDLayer):
    @staticmethod
    def create(spec: dict[str, Any]):
        return TDSum(name=spec["name"])

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDNetwork) -> Tensor:
        result = tf.reduce_sum(input_tensor=tf.stack(values=inputs), axis=0)
        return result

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDNetwork) -> list[Tensor]:
        return [grad for _ in inputs]

    def spec(self) -> dict[str, Any]:
        return dict(name=self.name,
                    type="sum")


class TDConcat(TDLayer):
    @staticmethod
    def create(spec: dict[str, Any]):
        return TDConcat(name=spec["name"])

    def __init__(self, name: str):
        super().__init__(name)

    def forward(self, inputs: list[Tensor], net_status: TDNetwork) -> Tensor:
        result = tf.concat(values=inputs, axis=1)
        return result

    def train(self, inputs: list[Tensor], output: Tensor, grad: Tensor, delta: Tensor, net_status: TDNetwork) -> list[Tensor]:
        result = []
        idx = 0
        for inp in inputs:
            shape = inp.shape
            last = idx + shape[1]
            result.append(grad[:, idx: last])
            idx = last
        return result

    def spec(self) -> dict[str, Any]:
        return dict(name=self.name,
                    type="concat")


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
