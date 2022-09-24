import tensorflow as tf
from wheelly.transpiller import NetworkTranspiller


def net0_spec():
    return {
        "input_spec": {
            "input": {"shape": [10]}
        },
        "output_spec": {
            "output": {"type": "int", "shape": (1,), "num_values": 2}
        },
        "alpha": 0.1,
        "tdlambda": 0.5,
        "network": {
            "output": {
                "layers": [
                    {"type": "dense", "output_size": 2},
                    {"type": "relu"},
                    {"type": "tanh"},
                    {"type": "lin", "b": 1.0, "w": 2.0},
                    {"type": "softmax", "temperature": 0.5}
                ]}
        }
    }


def net1_spec():
    return {
        "input_spec": {
            "input.a": {"shape": [10]}
        },
        "output_spec": {
            "output": {"type": "int", "shape": (1,), "num_values": 2}
        },
        "alpha": 0.1,
        "tdlambda": 0.5,
        "network": {
            "output": {
                "inputs": "input.a",
                "layers": [
                    {"type": "dense", "output_size": 2},
                    {"type": "tanh"},
                ]}
        }
    }


def net2_spec():
    return {
        "input_spec": {
            "input.a": {"shape": [4]},
            "input.b": {"shape": [6]}
        },
        "output_spec": {
            "output": {"type": "int", "shape": (1,), "num_values": 2}
        },
        "alpha": 0.1,
        "tdlambda": 0.5,
        "network": {
            "output": {
                "inputs": {"type": "concat", "inputs": ["input.a", "input.b"]},
                "layers": [
                    {"type": "dense", "output_size": 2},
                    {"type": "tanh"},
                ]}
        }
    }


def net3_spec():
    return {
        "input_spec": {
            "input": {"shape": [4]},
        },
        "output_spec": {
            "output.a": {"type": "int", "shape": (1,), "num_values": 2},
            "output.b": {"type": "int", "shape": (1,), "num_values": 2}
        },
        "alpha": 0.1,
        "tdlambda": 0.5,
        "network": {
            "output.a": {
                "inputs": "layer3",
                "layers": [
                    {"type": "tanh"}
                ]},
            "output.b": {
                "inputs": "layer3",
                "layers": [
                    {"type": "relu"}
                ]},
            "layer3": {
                "inputs": {"type": "sum", "inputs": ["layer1", "layer2"]},
                "layers": [
                    {"type": "dense", "output_size": 2}
                ]},
            "layer2": {
                "inputs": "layer1",
                "layers": [
                    {"type": "dense", "output_size": 2}
                ]},
            "layer1": {
                "layers": [
                    {"type": "dense", "output_size": 2}
                ]},
        }
    }


def net4_spec():
    return dict(
        input_spec=dict(
            input=dict(shape=[4])
        ),
        output_spec=dict(
            output=dict(type="int",
                        shape=(1,),
                        num_values=2)
        ),
        alpha=0.1,
        tdlambda=0.5,
        network=dict(
            hidden0=dict(
                layers=[dict(type="dense", output_size=2)]
            ),
            hidden1=dict(
                inputs="hidden0",
                layers=[dict(type="dense", output_size=2)]
            ),
            output=dict(
                inputs=dict(
                    type="sum",
                    inputs=["hidden0", "hidden1"]
                ),
                layers=[]
            )
        )
    )


def test_parse0():
    ts = NetworkTranspiller(spec=net0_spec())
    ts.validate_spec()
    ts.parse_for_inputs()
    assert ts.layers == {
        "output[0]": dict(name="output[0]",
                          type="dense",
                          output_size=2,
                          inputs=["input"]),
        "output[1]": dict(name="output[1]",
                          type="relu",
                          inputs=["output[0]"]),
        "output[2]": dict(name="output[2]",
                          type="tanh",
                          inputs=["output[1]"]),
        "output[3]": dict(name="output[3]",
                          type="lin",
                          b=1.0,
                          w=2.0,
                          inputs=["output[2]"]),
        "output": dict(name="output",
                       type="softmax",
                       temperature=0.5,
                       inputs=["output[3]"])
    }
    assert ts.inputs == ["input"]
    assert ts.sinks == ["output"]

    ts.validate_layers()
    assert ts.output_size("output[0]") == 2
    assert ts.output_size("output[1]") == 2
    assert ts.output_size("output[2]") == 2
    assert ts.output_size("output[3]") == 2
    assert ts.output_size("output") == 2

    ts.sort_layers()
    assert ts.forward_order == [
        "output[0]",
        "output[1]",
        "output[2]",
        "output[3]",
        "output"
    ]
    spec = ts.parse()
    assert spec == dict(
        alpha=0.1,
        tdlambda=0.5,
        layers=[
            dict(name="output[0]",
                 type="dense",
                 input_size=10,
                 output_size=2,
                 inputs=["input"]),
            dict(name="output[1]",
                 type="relu",
                 inputs=["output[0]"]),
            dict(name="output[2]",
                 type="tanh",
                 inputs=["output[1]"]),
            dict(name="output[3]",
                 type="lin",
                 b=1.0,
                 w=2.0,
                 inputs=["output[2]"]),
            dict(name="output",
                 type="softmax",
                 temperature=0.5,
                 inputs=["output[3]"])
        ]
    )


def test_parse1():
    ts = NetworkTranspiller(spec=net1_spec())
    ts.validate_spec()
    ts.parse_for_inputs()
    assert ts.layers == {
        "output[0]": dict(name="output[0]",
                          type="dense",
                          output_size=2,
                          inputs=["input.a"]),
        "output": dict(name="output",
                       type="tanh",
                       inputs=["output[0]"])
    }
    assert ts.inputs == ["input.a"]
    assert ts.sinks == ["output"]

    ts.validate_layers()
    assert ts.output_size("output[0]") == 2
    assert ts.output_size("output") == 2

    ts.sort_layers()
    assert ts.forward_order == [
        "output[0]",
        "output"
    ]
    spec = ts.parse()
    assert spec == dict(
        alpha=0.1,
        tdlambda=0.5,
        layers=[
            dict(
                name="output[0]",
                type="dense",
                input_size=10,
                output_size=2,
                inputs=["input.a"]),
            dict(name="output",
                 type="tanh",
                 inputs=["output[0]"])
        ])


def test_parse2():
    ts = NetworkTranspiller(spec=net2_spec())
    ts.validate_spec()
    ts.parse_for_inputs()
    assert ts.layers == {
        "output[0]": dict(name="output[0]",
                          type="concat",
                          inputs=["input.a", "input.b"]),
        "output[1]": dict(name="output[1]",
                          type="dense",
                          output_size=2,
                          inputs=["output[0]"]),
        "output": dict(name="output",
                       type="tanh",
                       inputs=["output[1]"])
    }
    assert len(ts.inputs) == 2
    assert "input.a" in ts.inputs
    assert "input.b" in ts.inputs
    assert ts.sinks == ["output"]

    ts.validate_layers()
    assert ts.output_size("output[0]") == 10
    assert ts.output_size("output[1]") == 2
    assert ts.output_size("output") == 2

    ts.sort_layers()
    assert ts.forward_order == [
        "output[0]",
        "output[1]",
        "output"
    ]

    spec = ts.parse()
    assert spec == dict(
        alpha=0.1,
        tdlambda=0.5,
        layers=[
            dict(name="output[0]",
                 type="concat",
                 inputs=["input.a", "input.b"]),
            dict(name="output[1]",
                 type="dense",
                 input_size=10,
                 output_size=2,
                 inputs=["output[0]"]),
            dict(name="output",
                 type="tanh",
                 inputs=["output[1]"])
        ])


def test_parse3():
    ts = NetworkTranspiller(spec=net3_spec())
    ts.validate_spec()
    ts.parse_for_inputs()
    assert ts.layers == {
        "output.a": dict(name="output.a",
                         type="tanh",
                         inputs=["layer3"]),
        "output.b": dict(name="output.b",
                         type="relu",
                         inputs=["layer3"]),
        "layer3": dict(name="layer3",
                       type="dense",
                       output_size=2,
                       inputs=["layer3[0]"]),
        "layer3[0]": dict(name="layer3[0]",
                          type="sum",
                          inputs=["layer1", "layer2"]),
        "layer2": dict(name="layer2",
                       type="dense",
                       output_size=2,
                       inputs=["layer1"]),
        "layer1": dict(name="layer1",
                       type="dense",
                       output_size=2,
                       inputs=["input"]),
    }
    assert ts.inputs == ["input"]
    assert ts.sinks == ["output.a", "output.b"]

    ts.validate_layers()
    assert ts.output_size("layer1") == 2
    assert ts.output_size("layer2") == 2
    assert ts.output_size("layer3[0]") == 2
    assert ts.output_size("layer3") == 2
    assert ts.output_size("output.a") == 2
    assert ts.output_size("output.b") == 2

    ts.sort_layers()
    assert ts.forward_order == [
        "layer1",
        "layer2",
        "layer3[0]",
        "layer3",
        "output.a",
        "output.b"
    ]

    spec = ts.parse()
    assert spec == dict(
        alpha=0.1,
        tdlambda=0.5,
        layers=[
            dict(name="layer1",
                 type="dense",
                 input_size=4,
                 output_size=2,
                 inputs=["input"]),
            dict(name="layer2",
                 type="dense",
                 input_size=2,
                 output_size=2,
                 inputs=["layer1"]),
            dict(name="layer3[0]",
                 type="sum",
                 inputs=["layer1", "layer2"]),
            dict(name="layer3",
                 type="dense",
                 input_size=2,
                 output_size=2,
                 inputs=["layer3[0]"]),
            dict(name="output.a",
                 type="tanh",
                 inputs=["layer3"]),
            dict(name="output.b",
                 type="relu",
                 inputs=["layer3"]),
        ])


def test_parse4():
    ts = NetworkTranspiller(spec=net4_spec())
    ts.validate_spec()
    ts.parse_for_inputs()
    assert ts.layers == {
        "hidden0": dict(name="hidden0", type="dense", output_size=2, inputs=["input"]),
        "hidden1": dict(name="hidden1", type="dense", output_size=2, inputs=["hidden0"]),
        "output": dict(name="output", type="sum", inputs=["hidden0", "hidden1"])
    }
    assert ts.inputs == ["input"]
    assert ts.sinks == ["output"]

    ts.validate_layers()
    assert ts.output_size("hidden0") == 2
    assert ts.output_size("hidden1") == 2
    assert ts.output_size("output") == 2

    ts.sort_layers()
    assert ts.forward_order == [
        "hidden0",
        "hidden1",
        "output"
    ]

    spec = ts.parse()
    assert spec == dict(
        alpha=0.1,
        tdlambda=0.5,
        layers=[
            dict(name="hidden0",
                 type="dense",
                 input_size=4,
                 output_size=2,
                 inputs=["input"]),
            dict(name="hidden1",
                 type="dense",
                 input_size=2,
                 output_size=2,
                 inputs=["hidden0"]),
            dict(name="output",
                 type="sum",
                 inputs=["hidden0", "hidden1"]),
        ])
