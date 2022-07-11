import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from numpy.testing import assert_equal

from wheelly.encoders import createSpace


def test_create_binary():
    space = createSpace({
        "type": "bool",
        "shape": 1
    })
    assert isinstance(space, MultiBinary)
    assert space.shape == (1,)

def test_create_discrete():
    space = createSpace({
        "type": "int",
        "shape": 1,
        "num_values": 3
    })
    assert isinstance(space, Discrete)
    assert space.n == 3

def test_create_multidiscrete():
    space = createSpace({
        "type": "int",
        "shape": 3,
        "num_values": 3
    })
    assert isinstance(space, MultiDiscrete)
    assert space.shape == (3,)
    assert_equal(space.nvec, np.array([3,3,3]))

def test_create_box():
    space = createSpace({
        "type": "float",
        "shape": 3,
        "min_value": 1,
        "max_value": 2
    })
    assert isinstance(space, Box)
    assert_equal(space.shape, (3,))
    assert_equal(space.low, np.array([1,1,1]))
    assert_equal(space.high, np.array([2,2,2]))


def test_create_dict():
    space = createSpace({
        "a": {
            "type": "bool",
            "shape": 1
        },
        'b': {
            'c': {
                "type": "float",
                "shape": 3,
                "min_value": 1,
                "max_value": 2 }
        }
    })
    assert isinstance(space, Dict)
    assert isinstance(space['a'], MultiBinary)
    assert space['a'].shape == (1,)
    assert isinstance(space['b'], Dict)
    assert isinstance(space['b']['c'], Box)
    assert_equal(space['b']['c'].low, [1,1,1])
    assert_equal(space['b']['c'].high, [2,2,2])
