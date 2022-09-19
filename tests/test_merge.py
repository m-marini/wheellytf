import numpy as np
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numpy.testing import assert_equal

from wheelly.encoders import MergeEncoder, SupplyEncoder


def test_merge_binary_space1():
    space1 = MultiBinary(2)
    space2 = MultiBinary(3)
    encoder0 = SupplyEncoder(space1, lambda: np.array([0, 1], dtype=np.int8))
    encoder1 = SupplyEncoder(space2, lambda: np.array([0,0,1],dtype=np.int8))
    encoder = MergeEncoder.create(encoder0, encoder1)
    clip_space = encoder.space
    assert isinstance(clip_space, MultiBinary)
    assert clip_space.n == 5

def test_merge_binary():
    space1 = MultiBinary(2)
    space2 = MultiBinary(3)
    encoder0 = SupplyEncoder(space1, lambda: np.array([0, 1], dtype=np.int8))
    encoder1 = SupplyEncoder(space2, lambda: np.array([0,0,1],dtype=np.int8))
    encoder = MergeEncoder.create(encoder0, encoder1)
    y = encoder.encode()
    assert_equal(y, np.array([0,1,0,0,1]))

def test_merge_discrete_space1():
    space1 = Discrete(2)
    space2 = MultiDiscrete(np.array([3, 4]))
    encoder0 = SupplyEncoder(space1, lambda: np.array([0], dtype=np.int8))
    encoder1 = SupplyEncoder(space2, lambda: np.array([1,2],dtype=np.int8))
    encoder = MergeEncoder.create(encoder0, encoder1)
    clip_space = encoder.space
    assert isinstance(clip_space, MultiDiscrete)
    assert_equal(clip_space.nvec, np.array([2,3,4]))

def test_merge_discrete():
    space1 = Discrete(2)
    space2 = MultiDiscrete(np.array([3, 4]))
    encoder0 = SupplyEncoder(space1, lambda: np.array([0], dtype=np.int8))
    encoder1 = SupplyEncoder(space2, lambda: np.array([1,2],dtype=np.int8))
    encoder = MergeEncoder.create(encoder0, encoder1)
    y = encoder.encode()
    assert_equal(y, np.array([0, 1, 2]))

def test_merge_float_space1():
    space1 = Box(np.array([-1.0,-2.0]), np.array([1.0,2.0]))
    space2 = Box(np.array([-10.0,-10.0,-10.0]), np.array([10.0,10.0,10.0]))
    encoder0 = SupplyEncoder(space1, lambda: np.array([0, 1], dtype=np.float32))
    encoder1 = SupplyEncoder(space2, lambda: np.array([-2, 0, 2],dtype=np.float32))
    encoder = MergeEncoder.create(encoder0, encoder1)
    clip_space = encoder.space
    assert isinstance(clip_space, Box)
    assert_equal(clip_space.low, np.array([-1,-2,-10,-10,-10]))
    assert_equal(clip_space.high, np.array([1,2,10,10,10]))

def test_merge_float1():
    space1 = Box(np.array([-1.0,-2.0]), np.array([1.0,2.0]))
    space2 = Box(np.array([-10.0,-10.0,-10.0]), np.array([10.0,10.0,10.0]))
    encoder0 = SupplyEncoder(space1, lambda: np.array([0, 1], dtype=np.float32))
    encoder1 = SupplyEncoder(space2, lambda: np.array([-2, 0, 2],dtype=np.float32))
    encoder = MergeEncoder.create(encoder0, encoder1)
    y = encoder.encode()
    assert_equal(y, np.array([0,1,-2,0,2]))
