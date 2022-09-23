from typing import Any, Tuple

import tensorflow as tf

from wheelly.tdlayers import flat_type_spec
from wheelly.tdagents import TDAgent
from tensorforce import Environment


def _validate_layer(spec: dict[str, Any]):
    assert "type" in spec, "Missing type"
    type = spec["type"]
    assert isinstance(type, str), "Type must be str"
    inputs = spec["inputs"]
    if type == "dense":
        assert "output_size" in spec, \
            "Missing output_size"
        assert len(inputs) == 1, \
            f'inputs must have only one item ({len(inputs)})'
        assert isinstance(spec["output_size"], int), \
            "output_size must be int"
        assert spec["output_size"] > 0, \
            f'output_size must be > 0 ({spec["output_size"]})'
    elif type in ["relu", "tanh"]:
        assert len(inputs) == 1, \
            f'inputs must have only one item ({len(inputs)})'
    elif type == "lin":
        assert "b" in spec, "Missing b spec"
        assert "w" in spec, "Missing w spec"
        assert len(inputs) == 1, \
            f'inputs must have only one item ({len(inputs)})'
        assert isinstance(spec["b"], float), \
            "b must be a float"
        assert isinstance(spec["w"], float), \
            "w must be a float"
    elif type == "softmax":
        assert "temperature" in spec, "Missing temperature"
        assert len(inputs) == 1, \
            f'inputs must have only one item ({len(inputs)})'
        assert isinstance(spec["temperature"], (float | int))
    elif type in ["sum", "concat"]:
        assert len(inputs) > 0, \
            f'inputs must have one or more items ({len(inputs)})'
    else:
        raise Exception(f'Wrong type "{type}"')


class NetworkTranspiller:
    def __init__(self,
                 spec: dict[str, Any]) -> None:
        """Creates a network transpiller

        Arguments:
        spec -- general network spec"""
        self.spec = spec

    def parse(self) -> dict[str, Any]:
        """Parses the specifications and returns the internal network specification"""
        self.validate_spec()
        self.parse_for_inputs()
        self.validate_layers()
        self.sort_layers()
        out_spec = dict(alpha=self.spec["alpha"],
                        tdlambda=self.spec["tdlambda"],
                        layers=[self.build_layer_spec(id)for id in self.forward_order])
        return out_spec

    def validate_spec(self):
        """Validates the network specification"""
        assert "input_spec" in self.spec
        assert "output_spec" in self.spec
        assert "alpha" in self.spec
        assert "tdlambda" in self.spec
        assert "network" in self.spec

        assert isinstance(self.spec["alpha"], float)
        assert isinstance(self.spec["tdlambda"], float)
        assert isinstance(self.spec["input_spec"], dict)
        assert isinstance(self.spec["output_spec"], dict)
        assert isinstance(self.spec["network"], dict)

        assert len(self.spec["input_spec"]) > 0
        assert len(self.spec["output_spec"]) > 0
        assert len(self.spec["network"]) > 0

    def parse_for_inputs(self):
        """Generates the inputs for each network sequence"""
        layers = []
        for seq_name, seq_spec in self.spec["network"].items():
            # Parse for inputs specification
            inputes = seq_spec.get("inputs", "input")
            offset = 0
            if isinstance(inputes, str):
                inputes = [inputes]
            else:
                # Parse for composite inputs specification
                assert isinstance(inputes, dict)
                assert "type" in inputes
                type = inputes["type"]
                assert isinstance(type, str)

                assert "inputs" in inputes
                inputes = inputes["inputs"]
                assert isinstance(inputes, list)
                assert len(inputes) > 0
                for s in inputes:
                    assert isinstance(s, str)
                layer0 = dict(name=f"{seq_name}[0]",
                              type=type,
                              inputs=inputes)
                layers.append(layer0)
                inputes = [layer0["name"]]
                offset = 1

            # Parse for layers
            assert "layers" in seq_spec
            seq_layers = seq_spec["layers"]
            assert isinstance(seq_layers, list)
            n = len(seq_layers)
            assert n > 0
            for i in range(n):
                layer_spec = seq_layers[i].copy()
                layer_name = seq_name if i >= n - 1 \
                    else f"{seq_name}[{i + offset}]"
                layer_spec["name"] = layer_name
                layer_spec["inputs"] = inputes
                layers.append(layer_spec)
                inputes = [layer_name]
        self.layers: dict[str, dict[str, Any]] = {
            layer["name"]: layer for layer in layers}

        # Extracts not sinks
        inputes = set[str]([input
                            for spec in layers
                            for input in spec["inputs"]
                            ])

        # Extracts sinks
        self.sinks = [name for name in self.layers if name not in inputes]

        # Extracts input names
        self.inputs = [name for name in inputes if name not in self.layers]

        # Validate inputs
        for input in self.inputs:
            assert input in self.spec["input_spec"], f'Input "{input}" undefined'

    def sort_layers(self):
        """Returns the sorted layers for forward pass

        Arguments:
        layers_spec -- the full layer specification"""
        ordered = list[str]()

        def sort(name: str):
            if name not in ordered and name in self.layers:
                for input in self.layers[name]["inputs"]:
                    sort(input)
                ordered.append(name)

        for node in self.sinks:
            sort(node)
        self.forward_order = ordered

    def output_size(self, id: str):
        """Returns the output size of a layer

        Arguments:
        id -- the layer id"""
        if id in self.spec["input_spec"]:
            assert "shape" in self.spec["input_spec"][id], \
                f'Missing shape for "{id}"'
            shape = self.spec["input_spec"][id]["shape"]
            assert isinstance(shape, (list, Tuple)), \
                f'shape of "{id}" must be a list or a tuple'
            return shape[0]
        else:
            spec = self.layers[id]
            type = spec["type"]
            inputs = spec["inputs"]
            if type == "dense":
                return spec["output_size"]
            elif type == "concat":
                size = 0
                for inp in inputs:
                    size = size + self.output_size(inp)
                return size
            elif type == "sum":
                size = self.output_size(inputs[0])
                for inp in inputs:
                    assert self.output_size(inp) == size
                return size
            elif type in ["relu", "lin", "tanh", "softmax"]:
                return self.output_size(inputs[0])
            else:
                raise Exception(f'wrong type "{type}"')

    def validate_layers(self):
        for layer in self.layers.values():
            _validate_layer(layer)

    def build_layer_spec(self, id: str) -> dict[str, Any]:
        spec = self.layers[id]
        type = spec["type"]
        result = dict(name=spec["name"],
                      type=type,
                      inputs=spec["inputs"])
        if type == "dense":
            result["input_size"] = self.output_size(spec["inputs"][0])
            result["output_size"] = spec["output_size"]
        elif type == "lin":
            result["b"] = spec["b"]
            result["w"] = spec["w"]
        elif type == "softmax":
            result["temperature"] = spec["temperature"]
        return result


class AgentTranspiller:
    @staticmethod
    def byEnv(env: Environment,
              spec=dict[str, Any]):
        agent_spec = spec.copy()
        agent_spec["state_spec"] = env.states()
        agent_spec["actions_spec"] = env.actions()
        return AgentTranspiller(agent_spec)

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    def parse(self) -> dict[str, Any]:
        assert "state_spec" in self.spec
        assert "actions_spec" in self.spec
        assert "reward_alpha" in self.spec
        assert "policy" in self.spec
        assert "critic" in self.spec

        assert isinstance(self.spec["state_spec"], dict)
        assert isinstance(self.spec["actions_spec"], dict)
        assert isinstance(self.spec["reward_alpha"], float)
        assert isinstance(self.spec["policy"], dict)
        assert isinstance(self.spec["critic"], dict)

        self.flatten_state = flat_type_spec(
            spec=self.spec["state_spec"], prefix="input")
        self.flatten_actions = flat_type_spec(
            spec=self.spec["actions_spec"], prefix="output")

        policy_spec = self.spec["policy"].copy()
        policy_spec["input_spec"] = self.flatten_state
        policy_spec["output_spec"] = self.flatten_actions

        policy = NetworkTranspiller(policy_spec).parse()

        critic_output = flat_type_spec(spec=dict(type="float",
                                                 shape=(1,)), prefix="output")
        critic_spec = self.spec["critic"].copy()
        critic_spec["input_spec"] = self.flatten_state
        critic_spec["output_spec"] = critic_output
        critic = NetworkTranspiller(critic_spec).parse()

        out_spec = dict(
            state_spec=self.spec["state_spec"],
            actions_spec=self.spec["actions_spec"],
            reward_alpha=self.spec["reward_alpha"],
            policy=policy,
            critic=critic
        )
        return out_spec

    def build(self, random=tf.random.Generator):
        return TDAgent.create(spec=self.parse(), random=random)
