import numpy as np
import pytest
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from numpy.testing import assert_equal

from wheelly.encoders import SupplyEncoder


def test_space1():
    space = Discrete(n=2)
    encoder = SupplyEncoder(space, lambda: np.array([1]))
    result = encoder.space
    assert result == space
    assert encoder.spec() == {
        'type': 'int',
        'shape': 1,
        'num_values': 2
    }

def test_encode1():
    space = Discrete(n=2)
    encoder = SupplyEncoder(space, lambda: np.array([1]))
    result = encoder.encode()
    assert_equal(result, [1])

def test_space2():
    space = MultiDiscrete((3,3))
    value= np.array([1, 2])
    encoder = SupplyEncoder(space, lambda: value)
    result = encoder.space
    assert result == space
    assert_equal(encoder.spec(),{
        'type': 'int',
        'shape': 2,
        'num_values': 3
    })

def test_encode2():
    space = MultiDiscrete((2,3))
    value= np.array([1, 2]),
    encoder = SupplyEncoder(space, lambda: value)
    result = encoder.encode()
    assert_equal(result, value)

def test_space_multidiscrete1():
    space = MultiDiscrete((3,3))
    value= np.array([1, 2]),
    encoder = SupplyEncoder(space, lambda: value)
    result = encoder.space
    assert result == space
    assert_equal(encoder.spec(),{
        'type': 'int',
        'shape': 2,
        'num_values': 3
    })

def test_spec_err1():
    space = MultiDiscrete((2,3))
    value= np.array([1, 2]),
    with pytest.raises(Exception):
        SupplyEncoder(space, lambda: value).spec()

def test_space_bin():
    space = MultiBinary(n=3)
    value= np.array([0, 1, 0]),
    encoder = SupplyEncoder(space, lambda: value)
    result = encoder.space
    assert result == space
    assert_equal(encoder.spec(),{
        'type': 'bool',
        'shape': 3
    })

def test_space_box1():
    space = Box(low = -1, high = 1, shape=(2,))
    value= np.array([-0.5, 0.5]),
    encoder = SupplyEncoder(space, lambda: value)
    result = encoder.space
    assert result == space
    assert_equal(encoder.spec(),{
        'type': 'float',
        'shape': (2,),
        'min_value': -1.0,
        'max_value': 1.0,
    })

def test_space_box_err1():
    space = Box(low=np.array([-1, 0]), high=np.array([1, 1]))
    value= np.array([-0.5, 0.5]),
    with pytest.raises(Exception):
        SupplyEncoder(space, lambda: value).spec()

def test_space_box_err2():
    space = Box(low=np.array([-1, -1]), high=np.array([0, 1]))
    value= np.array([-0.5, 0.5]),
    with pytest.raises(Exception):
        SupplyEncoder(space, lambda: value).spec()

def test_space_dict1():
    space = Dict(
        a=MultiBinary(2),
        b=Dict(
            c=Discrete(3)
        )
    )
    encoder = SupplyEncoder(space, None)
    result = encoder.space
    assert result == space
    assert_equal(encoder.spec(), {
        'a': {
            'type': 'bool',
            'shape': 2 },
        'b': {
            'c': {
                'type': 'int',
                'shape': 1,
                'num_values': 3
                },
            }
        })
