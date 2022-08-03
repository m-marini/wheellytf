import numpy as np
import pytest
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numpy.testing import assert_equal

from wheelly.encoders import Binary2BoxEncoder, SupplyEncoder


def test_space():
    space = MultiBinary(3)
    encoder = Binary2BoxEncoder(SupplyEncoder(space, lambda:np.array([0, 1, 0])))
    encoded_space = encoder.space()
    assert isinstance(encoded_space, Box)
    assert_equal(encoded_space.shape, (3,))
    assert_equal(encoded_space.low, 0)
    assert_equal(encoded_space.high, 1)

def test_encode():
    space = MultiBinary(3)
    encoder = Binary2BoxEncoder(SupplyEncoder(space, lambda:np.array([0, 1, 0])))
    y = encoder.encode()
    assert_equal(y, np.array([0, 1, 0]))
