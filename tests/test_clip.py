import numpy as np
import pytest
from gym.spaces import Box, Discrete
from numpy.testing import assert_equal

from wheelly.encoders import ClipEncoder, SupplyEncoder


def test_clip_space1():
    space = Box(np.array([-10,-10]), np.array([10,20]))
    low = np.array([-1, 0])
    high = np.array([0, 1])
    encoder = ClipEncoder(SupplyEncoder(space, lambda:np.array([-10, -10])), low, high)
    clip_space = encoder.space()
    assert isinstance(clip_space, Box)
    assert_equal(clip_space.low, low)
    assert_equal(clip_space.high, high)

def test_clip_space10():
    space = Discrete(2)
    low = np.array([-1, 0])
    high = np.array([0, 1])
    with pytest.raises(Exception) as e_info:
        ClipEncoder(SupplyEncoder(space, lambda:np.array([-10, -10])), low, high)

def test_clip_1():
    space = Box(np.array([-10,-10]), np.array([10,20]))
    low = np.array([-1, 0])
    high = np.array([0, 1])
    encoder = ClipEncoder(SupplyEncoder(space, lambda:np.array([-10, -10])), low, high)
    y = encoder.encode()
    assert_equal(y, np.array([-1,0]))
