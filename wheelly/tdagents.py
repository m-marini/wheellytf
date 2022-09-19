from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from wheelly.tdlayers import (TDDense, TDGraph, TDLayer, TDRelu, TDSoftmax,
                            TDState, TDTanh, td_forward, td_train)


def create_layer(name: str, spec: dict[str, Any]) -> TDLayer:
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


def create_graph(spec: dict[str, Any]) -> TDGraph:
    """Returns the graph for a lyaer specification"

    Arguments:
    spec -- the full layer specification"""
    names = sort_layers(spec)
    forward = [create_layer(name=name, spec=spec[name]) for name in names]
    inputs = {key: [layer_spec["input"]] for key, layer_spec in spec.items()}
    outputs = {key: find_outputs(spec=spec, layer_name=key) for key in spec}

    return TDGraph(forward=forward, inputs=inputs, outputs=outputs)


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


def create_network(net_spec: dict[str, Any],
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
    props = {"alpha": net_spec["alpha"],
             "lambda": net_spec["lambda"]}
    net_spec1 = net_spec["network"]
    layers_spec = parse_for_inputs(net_spec=net_spec1, state_spec=state_spec)
    layers_spec = parse_for_size(
        layers_spec=layers_spec, state_spec=state_spec)
    validate_output(layers_spec=layers_spec, output_spec=actions_spec)
    graph = create_graph(layers_spec)
    node_props = create_layers_props(spec=layers_spec, random=random)

    return TDState(props=props, graph=graph, node_props=node_props)


class TDAgent:
    @ staticmethod
    def create(state_spec: dict[str, Any],
               actions_spec: dict[str, Any],
               agent_spec: dict[str, Any],
               random: tf.random.Generator):
        """Returns the agent by agent specification

        Arguments:
        state_spec -- the state specification
        actions_spec -- the on specification
        agent_spec -- the agent specification
        random -- random generator"""
        flatten_state = flat_type_spec(spec=state_spec, prefix="input")
        flatten_actions = flat_type_spec(spec=actions_spec, prefix="output")
        reward_alpha = agent_spec.get("reward_alfa", 0.1)
        assert "critic" in agent_spec, f'Missing "critic" specification'
        critic = create_network(net_spec=agent_spec["critic"],
                                state_spec=flatten_state,
                                actions_spec=dict(output=dict(
                                    type="float",
                                    shape=(1,)
                                )),
                                random=random)
        assert "policy" in agent_spec, f'Missing "policy" specification'
        policy = create_network(net_spec=agent_spec["policy"],
                                state_spec=flatten_state,
                                actions_spec=flatten_actions,
                                random=random)
        agent = TDAgent(states_spec=state_spec,
                         actions_spec=actions_spec,
                         policy=policy,
                         critic=critic,
                         reward_alpha=reward_alpha,
                         random=random)
        return agent

    def __init__(self,
                 states_spec: dict[str, Any],
                 actions_spec: dict[str, Any],
                 policy: TDState,
                 critic: TDState,
                 reward_alpha: float,
                 random: tf.random.Generator):
        self._states_spec = states_spec
        self._actions_spec = actions_spec
        self._flatten_states_spec = flat_type_spec(
            spec=states_spec, prefix="input")
        self._flatten_actions_spec = flat_type_spec(
            spec=actions_spec, prefix="output")
        self._policy = policy
        self._random = random
        self._critic = critic
        self._reward_alpha = reward_alpha
        self._avg_reward = tf.Variable(initial_value=[0],
                                       dtype=tf.float32,
                                       trainable=False)
        self._prev_inputs = None
        self._kpi_listeners: list[Callable[[dict[str, Any]], None]] = []

    @ property
    def states_spec(self):
        return self._states_spec

    @ property
    def actions_spec(self):
        return self._actions_spec

    def act(self, states: Any) -> dict[str, Any] | int | float:
        """Returns the action for a states based on current policy

        Argument:
        states -- the states"""
        policy_status, inputs = self._policy_status(states=states)
        pis = {key: value for key,
               value in policy_status.items() if key in self._flatten_actions_spec}
        flatten_action = choose_actions(flatten_pis=pis, random=self._random)
        actions = unflat_actions(
            flatten=flatten_action, actions_spec=self._actions_spec,)
        self._last_actions = actions
        self._last_inputs = inputs
        self._last_policy = policy_status
        return actions

    def observe(self, reward: float = 0.0, terminal: bool = False):
        """Observe the result of environment interaction and train the agent

        Arguments:
        reward -- reward
        terminal -- true if terminal state"""
        if self._prev_inputs is not None:
            self._train(s0=self._prev_inputs,
                        actions=self._prev_actions,
                        reward=self._prev_reward,
                        terminal=self._prev_terminal,
                        s1=self._last_inputs)
        if terminal:
            self._prev_inputs = None
            self._prev_actions = None
            self._prev_reward = None
            self._prev_terminal = None
            self._prev_policy = None
        else:
            self._prev_inputs = self._last_inputs
            self._prev_actions = self._last_actions
            self._prev_reward = reward
            self._prev_terminal = terminal
            self._prev_policy = self._last_policy

    def close(self):
        pass

    def _policy_status(self, states: Any):
        """Returns the status of policy network and the states inputs

        Argument:
        states -- the states"""
        inputs = flat_states(states)
        policy_status = td_forward(inputs=inputs,
                                    state=self._policy)
        return policy_status, inputs

    def _train(self, s0: dict[str, Tensor],
               actions: dict[str, Any] | int | float,
               reward: float,
               terminal: bool,
               s1: dict[str, Tensor]):

        c0 = td_forward(state=self._critic, inputs=s0)
        c1 = td_forward(state=self._critic, inputs=s1)
        v0 = c0["output"]
        v1 = c1["output"]
        delta = reward - self._avg_reward + v1 - v0
        pi = self._prev_policy
        dc = dict(output=tf.ones(shape=(1, 1)))
        dp = log_pi(pi=pi, action=actions, actions_spec=self._actions_spec)

        avg_reward = tf.constant(self._avg_reward)
        self._avg_reward.assign_add(delta=delta[0] * self._reward_alpha)
        _, grad_c = td_train(state=self._critic,
                              nodes=c0,
                              grad_loss=dc,
                              delta=delta)
        _, grad_pi = td_train(state=self._policy,
                               nodes=pi,
                               grad_loss=dp,
                               delta=delta)

        if len(self._kpi_listeners) > 0:
            trained_c0 = td_forward(state=self._critic, inputs=s0)
            trained_c1 = td_forward(state=self._critic, inputs=s1)
            trained_pi = td_forward(state=self._policy, inputs=s0)

            kpis = dict(
                s0=s0,
                reward=reward,
                terminal=terminal,
                actions=actions,
                s1=s1,
                avg_reward=avg_reward,
                trained_avg_reward=tf.constant(self._avg_reward),
                c0=c0,
                c1=c1,
                delta=delta,
                pi=pi,
                dc=grad_c,
                dp=grad_pi,
                trained_c0=trained_c0,
                trained_c1=trained_c1,
                trained_pi=trained_pi
            )
            self._notify_kpi(kpis)

    def _notify_kpi(self, kpis: dict[str, Any]):
        for listener in self._kpi_listeners:
            listener(kpis)

    def add_kpi_listener(self, callback: Callable[[dict[str, Any]], None]):
        """Adds a kpi listener

        Arguments:
        callback -- the call back listener"""
        self._kpi_listeners.append(callback)


def log_pi(pi: dict[str, Tensor],
           action: dict[str, Any] | int | float,
           actions_spec: dict[str, Any]) -> dict[str, Tensor]:
    """Returns the gradient of log pi for a given pi and action

    Arguments:
    pi -- the probabilities of each action
    action -- the action applied
    actions_spec -- the action specification"""

    def log_pi1(values: dict[str, Tensor],
                prefix: str,
                action: dict[str, Any] | int | float,
                spec: dict[str, Any]) -> dict[str, Tensor]:

        if "type" in spec:
            assert spec["type"] == "int"
            pi_a = pi[prefix]
            x = np.zeros(shape=pi_a.shape)
            x[0, action] = 1
            dl = (1 / pi_a) * x
            values[prefix] = dl
            return values
        else:
            for key, sub_spec in spec.items():
                log_pi1(values=values,
                        prefix=f"{prefix}.{key}",
                        action=action[key],
                        spec=sub_spec)
            return values

    return log_pi1(values={}, prefix="output", action=action, spec=actions_spec)


def flat_states(states: Any) -> dict[str, Tensor]:
    """Returns the flatten states

    Arguments:
    states_spec -- the state specification
    states -- the state"""
    result = dict()

    def flat(prefix: str, value: Any):
        if isinstance(value, dict):
            for key, sub_values in value.items():
                flat(prefix=prefix+"." + key,
                     value=sub_values)
        else:
            result[prefix] = tf.constant([value], dtype=tf.float32)

    flat(prefix="input", value=states)
    return result


def choose_actions(flatten_pis: dict[str, Tensor],
                   random: tf.random.Generator) -> dict[str, int]:
    """Returns the actions selecting random by probabilities

    Arguments:
    flatten_pis -- the flatten probabilities
    random -- the random generator"""
    def choose(prob: Tensor) -> int:
        cum = tf.cumsum(x=prob, axis=1)
        sel = random.uniform(shape=(1,))
        action = int(tf.math.count_nonzero(sel >= cum))
        return action

    result = {key: choose(pis) for key, pis in flatten_pis.items()}
    return result


def unflat_actions(actions_spec: dict[str, Any],
                    flatten: dict[str, int]) -> dict[str, Any] | int | float:
    """Returns the unflatten actions

    Arguments:
    actions_spec -- the actions specification
    flatten -- the flatten actions"""
    def unflat(prefix: str,
               spec: dict[str, Any]) -> dict[str, Any] | int | float:

        if "type" in spec:
            assert spec["type"] == "int"
            return flatten[prefix]
        else:
            result = {key: unflat(prefix=f"{prefix}.{key}",
                                  spec=sub_spec)
                      for key, sub_spec in spec.items()}
            return result

    return unflat(prefix="output", spec=actions_spec)
