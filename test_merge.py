import numpy as np
import numpy.testing as tnp
from gym import spaces
from wheelly.encoders import IdentityEncoder, MergeBinaryEncoder, MergeBoxEncoder, MergeDiscreteEncoder
import numpy.testing as tnp
import pytest

def test_merge_binary_space1():
    encoder0 = IdentityEncoder(spaces.MultiBinary(2), lambda: np.array([0, 1], dtype=np.int8))
    encoder1 = IdentityEncoder(spaces.MultiBinary(3), lambda: np.array([0,0,1],dtype=np.int8))
    encoder = MergeBinaryEncoder(encoder0, encoder1)
    clip_space = encoder.space()
    assert isinstance(clip_space, spaces.MultiBinary)
    tnp.assert_array_equal(clip_space.n, np.array([5]))

def test_merge_binary():
    encoder0 = IdentityEncoder(spaces.MultiBinary(2), lambda: np.array([0, 1], dtype=np.int8))
    encoder1 = IdentityEncoder(spaces.MultiBinary(3), lambda: np.array([0,0,1],dtype=np.int8))
    encoder = MergeBinaryEncoder(encoder0, encoder1)
    y = encoder.encode()
    tnp.assert_array_equal(y, np.array([0,1,0,0,1]))

def test_merge_discrete_space1():
    encoder0 = IdentityEncoder(spaces.Discrete(2, start=10), lambda: np.array([10]))
    encoder1 = IdentityEncoder(spaces.MultiDiscrete((3, 4)), lambda: np.array([1, 2]))
    encoder = MergeDiscreteEncoder(encoder0, encoder1)
    clip_space = encoder.space()
    assert isinstance(clip_space, spaces.MultiDiscrete)
    tnp.assert_array_equal(clip_space.nvec, np.array([2,3,4]))

def test_merge_discrete():
    encoder0 = IdentityEncoder(spaces.Discrete(2, start=10), lambda: np.array([10]))
    encoder1 = IdentityEncoder(spaces.MultiDiscrete((3, 4)), lambda: np.array([1, 2]))
    encoder = MergeDiscreteEncoder(encoder0, encoder1)
    encoder = MergeDiscreteEncoder(encoder0, encoder1)
    y = encoder.encode()
    tnp.assert_array_equal(y, np.array([0, 1, 2]))

def test_merge_box_space1():
    a=IdentityEncoder(spaces.Box(low=np.array([-1.0,-2.0]),
        high=np.array([1.0,2.0]),
        shape=[2]),
        lambda:np.array([0.0,1.0]))
    b=IdentityEncoder(spaces.Box(-10,10,shape=[3]), lambda:np.array([-2.0,0.0, 2.0]))
    encoder = MergeBoxEncoder(a,b)
    clip_space = encoder.space()
    assert isinstance(clip_space, spaces.Box)
    tnp.assert_array_equal(clip_space.shape, np.array([5]))
    tnp.assert_array_equal(clip_space.low, np.array([-1,-2,-10,-10,-10]))
    tnp.assert_array_equal(clip_space.high, np.array([1,2,10,10,10]))

def test_merge_box1():
    a=IdentityEncoder(spaces.Box(low=np.array([-1.0,-2.0]),
        high=np.array([1.0,2.0]),
        shape=[2]),
        lambda:np.array([0.0,1.0]))
    b=IdentityEncoder(spaces.Box(-10,10,shape=[3]), lambda:np.array([-2.0,0.0, 2.0]))
    encoder = MergeBoxEncoder(a,b)
    y = encoder.encode()
    tnp.assert_array_equal(y, np.array([0,1,-2,0,2]))
