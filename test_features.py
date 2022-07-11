import numpy as np
import pytest
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numpy.testing import assert_equal

from wheelly.encoders import FeaturesEncoder, SupplyEncoder


def test_features_space_float():
    space = Box(0, 1, (1,))
    with pytest.raises(Exception) as e_info:
        FeaturesEncoder.create(SupplyEncoder(space, None))

def test_features_space_discrete():
    space = Discrete(3)
    encoder = FeaturesEncoder.create(SupplyEncoder(space, None))
    bin_space = encoder.space()
    assert isinstance(bin_space, MultiBinary)
    assert bin_space.n == 3

def test_features_space_multidiscrete():
    space = MultiDiscrete(np.array([2,3]))
    encoder = FeaturesEncoder.create(SupplyEncoder(space, None))
    bin_space = encoder.space()
    assert isinstance(bin_space, MultiBinary)
    assert bin_space.n == 6

def test_features_discrete():
    space = Discrete(3)
    x = np.array([0])
    encoder = FeaturesEncoder.create(SupplyEncoder(space, lambda: x))
    y = encoder.encode()
    assert_equal(y, np.array([1,0,0]))

def test_features_multidiscrete1():
    space = MultiDiscrete(np.array([2,3]))
    x = np.array([0,0])
    encoder = FeaturesEncoder.create(SupplyEncoder(space, lambda: x))
    y = encoder.encode()
    assert_equal(y, np.array([1,0,0,0,0,0]))

def test_features_multidiscrete2():
    space = MultiDiscrete(np.array([2,3]))
    x = np.array([1,1])
    encoder = FeaturesEncoder.create(SupplyEncoder(space, lambda: x))
    y = encoder.encode()
    assert_equal(y, np.array([0,0,0,0,1,0]))
