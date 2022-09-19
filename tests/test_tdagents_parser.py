import tensorflow as tf
from wheelly.tdagents import (TDAgent, create_graph, create_layers_props,
                            create_network, flat_type_spec, parse_for_inputs,
                            parse_for_size, sort_layers, validate_output)
from wheelly.tdlayers import TDDense, TDRelu, TDSoftmax, TDTanh


def test_flat_type_spec1():
    spec = dict(type="int",
                shape=(1,),
                num_values=1)
    result = flat_type_spec(spec=spec, prefix="input")
    assert result == {
        "input": {"type": "int", "shape": (1,), "num_values": 1}
    }


def test_flat_type_spec2():
    spec = dict(
        a=dict(
            a=dict(type="int", shape=(1,), num_values=1),
            b=dict(type="int", shape=(2,), num_values=2),
        ),
        b=dict(type="int", shape=(3,), num_values=4)
    )

    result = flat_type_spec(spec=spec, prefix="input")
    assert result == {
        "input.a.a": {"type": "int", "shape": (1,), "num_values": 1},
        "input.a.b": {"type": "int", "shape": (2,), "num_values": 2},
        "input.b": {"type": "int", "shape": (3,), "num_values": 4}
    }


def net0_spec():
    return {
        "output": {
            "layers": [
                {"type": "dense", "size": 1},
                {"type": "relu"},
                {"type": "tanh"}
            ]}
    }


def test_parse_for_inputs1():
    state_spec = {
        "input": {"shape": [1]}
    }
    result = parse_for_inputs(net_spec=net0_spec(), state_spec=state_spec)
    assert result == {
        "output[0]": {"type": "dense", "size": 1, "input": "input"},
        "output[1]": {"type": "relu", "input": "output[0]"},
        "output": {"type": "tanh", "input": "output[1]"}
    }


def net1_spec():
    return {
        "output": {
            "layers": [
                {"type": "dense", "size": 2},
                {"type": "relu"},
                {"type": "softmax", "temperature": 0.8}
            ]
        },
        "output.a": {
            "input": "output",
            "layers": [
                {"type": "dense", "size": 3},
                {"type": "tanh"}
            ]
        },
        "output.b": {
            "input": "output",
            "layers": [
                {"type": "dense", "size": 4},
                {"type": "tanh"}
            ]
        },
        "output.c": {
            "input": "aaaa",
            "layers": [
                {"type": "dense", "size": 5},
                {"type": "tanh"}
            ]
        }
    }


def test_parse_for_inputs2():
    state_spec = {
        "input": {"shape": [10]},
        "aaaa": {"shape": [12]}
    }
    result = parse_for_inputs(net_spec=net1_spec(), state_spec=state_spec)
    assert result == {
        "output[0]": {"type": "dense", "size": 2, "input": "input"},
        "output[1]": {"type": "relu", "input": "output[0]"},
        "output": {"type": "softmax", "temperature": 0.8, "input": "output[1]"},
        "output.a[0]": {"type": "dense", "size": 3, "input": "output"},
        "output.a": {"type": "tanh", "input": "output.a[0]"},
        "output.b[0]": {"type": "dense", "size": 4, "input": "output"},
        "output.b": {"type": "tanh", "input": "output.b[0]"},
        "output.c[0]": {"type": "dense", "size": 5, "input": "aaaa"},
        "output.c": {"type": "tanh", "input": "output.c[0]"}
    }


def test_sort():
    state_spec = {
        "input": {"shape": [10]},
        "aaaa": {"shape": [12]}
    }
    result = parse_for_inputs(net_spec=net1_spec(), state_spec=state_spec)
    sorted = sort_layers(result)
    assert sorted == [
        "output[0]",
        "output[1]",
        "output",
        "output.a[0]",
        "output.a",
        "output.b[0]",
        "output.b",
        "output.c[0]",
        "output.c"
    ]


def test_parse_size():
    state_spec = {
        "input": {"shape": [10]},
        "aaaa": {"shape": [12]}
    }
    result = parse_for_inputs(net_spec=net1_spec(), state_spec=state_spec)
    result = parse_for_size(layers_spec=result, state_spec=state_spec)
    assert result == {
        "output[0]": {"type": "dense", "size": 2, "input": "input", "inp_size": 10},
        "output[1]": {"type": "relu", "input": "output[0]", "inp_size": 2, "size": 2},
        "output": {"type": "softmax", "temperature": 0.8, "input": "output[1]", "inp_size": 2, "size": 2},
        "output.a[0]": {"type": "dense", "size": 3, "input": "output", "inp_size": 2},
        "output.a": {"type": "tanh", "input": "output.a[0]", "inp_size": 3, "size": 3},
        "output.b[0]": {"type": "dense", "size": 4, "input": "output", "inp_size": 2},
        "output.b": {"type": "tanh", "input": "output.b[0]", "inp_size": 4, "size": 4},
        "output.c[0]": {"type": "dense", "size": 5, "input": "aaaa", "inp_size": 12},
        "output.c": {"type": "tanh", "input": "output.c[0]", "inp_size": 5, "size": 5}
    }


def test_validate_outputs():
    state_spec = {
        "input": {"shape": [10]},
        "aaaa": {"shape": [12]}
    }
    result = parse_for_inputs(net_spec=net1_spec(), state_spec=state_spec)
    result = parse_for_size(layers_spec=result, state_spec=state_spec)
    out_spec = {
        "output.a": {"type": "int", "shape": (1,), "num_values": 3},
        "output.b": {"type": "int", "shape": (1,), "num_values": 4},
        "output.c": {"type": "int", "shape": (1,), "num_values": 5}
    }
    validate_output(layers_spec=result, output_spec=out_spec)


def test_create_graph():
    state_spec = {
        "input": {"shape": [10]},
        "aaaa": {"shape": [12]}
    }
    result = parse_for_inputs(net_spec=net1_spec(), state_spec=state_spec)
    result = parse_for_size(layers_spec=result, state_spec=state_spec)
    graph = create_graph(result)

    names = [layer.name for layer in graph.forward]
    assert names == [
        "output[0]",
        "output[1]",
        "output",
        "output.a[0]",
        "output.a",
        "output.b[0]",
        "output.b",
        "output.c[0]",
        "output.c"
    ]
    assert isinstance(graph.forward[0], TDDense)
    assert isinstance(graph.forward[1], TDRelu)
    assert isinstance(graph.forward[2], TDSoftmax)
    assert isinstance(graph.forward[3], TDDense)
    assert isinstance(graph.forward[4], TDTanh)
    assert isinstance(graph.forward[5], TDDense)
    assert isinstance(graph.forward[6], TDTanh)
    assert isinstance(graph.forward[7], TDDense)
    assert isinstance(graph.forward[8], TDTanh)

    assert graph._inputs["output[0]"] == ["input"]
    assert graph._inputs["output[1]"] == ["output[0]"]
    assert graph._inputs["output"] == ["output[1]"]
    assert graph._inputs["output.a[0]"] == ["output"]
    assert graph._inputs["output.a"] == ["output.a[0]"]
    assert graph._inputs["output.b[0]"] == ["output"]
    assert graph._inputs["output.b"] == ["output.b[0]"]
    assert graph._inputs["output.c[0]"] == ["aaaa"]
    assert graph._inputs["output.c"] == ["output.c[0]"]

    assert graph._outputs["output[0]"] == ["output[1]"]
    assert graph._outputs["output[1]"] == ["output"]
    assert graph._outputs["output"] == ["output.a[0]", "output.b[0]"]
    assert graph._outputs["output.a[0]"] == ["output.a"]
    assert graph._outputs["output.a"] == []
    assert graph._outputs["output.b[0]"] == ["output.b"]
    assert graph._outputs["output.b"] == []
    assert graph._outputs["output.c[0]"] == ["output.c"]
    assert graph._outputs["output.c"] == []


def test_create_layers_props():
    state_spec = {
        "input": {"shape": [10]},
        "aaaa": {"shape": [12]}
    }
    random = tf.random.Generator.from_seed(1234)
    result = parse_for_inputs(net_spec=net1_spec(), state_spec=state_spec)
    result = parse_for_size(layers_spec=result, state_spec=state_spec)
    result = create_layers_props(result, random=random)

    assert isinstance(result["output[0]"]["w"], tf.Variable)
    assert isinstance(result["output.a[0]"]["w"], tf.Variable)
    assert isinstance(result["output.b[0]"]["w"], tf.Variable)
    assert isinstance(result["output.c[0]"]["w"], tf.Variable)
    assert isinstance(result["output"]["temperature"], tf.Tensor)


def test_create_network():
    state_spec = {
        "input": {"shape": [10]},
        "aaaa": {"shape": [12]}
    }
    out_spec = {
        "output.a": {"type": "int", "shape": (1,), "num_values": 3},
        "output.b": {"type": "int", "shape": (1,), "num_values": 4},
        "output.c": {"type": "int", "shape": (1,), "num_values": 5}
    }
    random = tf.random.Generator.from_seed(1234)
    net_spec = {
        "alpha": 0.1,
        "lambda": 0.5,
        "network": net1_spec()
    }
    result = create_network(net_spec=net_spec,
                            state_spec=state_spec,
                            actions_spec=out_spec,
                            random=random)

    assert result.graph is not None
    assert len(result.node_props) == 5
    assert len(result.props) == 2


def test_create():
    spec = {
        "reward_alpha": 0.1,
        "critic": {
            "alpha": 1e-3,
            "lambda": 0.8,
            "network": {
                "output": {
                    "layers": [
                        {"type": "dense", "size": 1},
                        {"type": "relu"},
                        {"type": "tanh"}
                    ]
                }
            }
        },
        "policy": {
            "alpha": 1e-3,
            "lambda": 0.8,
            "network": {
                "output": {
                    "layers": [
                        {"type": "dense", "size": 2},
                        {"type": "tanh"},
                        {"type": "softmax", "temperature": 0.8}
                    ]
                }
            }
        }
    }

    state_spec = dict(type="int",
                      shape=(1,),
                      num_values=1)
    actions_spec = dict(type="int",
                        shape=(1,),
                        num_values=2)
    random = tf.random.Generator.from_seed(1234)

    agent = TDAgent.create(state_spec=state_spec,
                            actions_spec=actions_spec,
                            agent_spec=spec,
                            random=random)

    assert agent is not None
