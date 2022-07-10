import numpy as np
from wheelly.encoders import SupplyEncoder, ScaleEncoder
from numpy.testing import assert_equal
from gym.spaces import Discrete, Box

def test_scale_space1():
    space = Discrete(2)
    encoder = ScaleEncoder(SupplyEncoder(space, lambda: np.array([0])),
        low = np.array([10]),
        high = np.array([20])
    )
    result = encoder.space()
    assert isinstance(result, Box)

    assert_equal(result, Box(np.array([10]), np.array([20])))

def test_scale1():
    space = Discrete(2)
    encoder = ScaleEncoder(SupplyEncoder(space, lambda: np.array([0])),
        low = np.array([10]),
        high = np.array([20])
    )
    result = encoder.encode()
    assert_equal(result, [10])

def test_scale2():
    space = Discrete(2)
    encoder = ScaleEncoder(SupplyEncoder(space, lambda: np.array([1])),
        low = np.array([10]),
        high = np.array([20])
    )
    result = encoder.encode()
    assert_equal(result, [20])

def test_scale3():
    space = Discrete(6)
    encoder = ScaleEncoder(SupplyEncoder(space, lambda: np.array([0])),
        low = np.array([-10]),
        high = np.array([10])
    )
    result = encoder.encode()
    assert_equal(result, [-10])

def test_scale4():
    space = Discrete(5)
    encoder = ScaleEncoder(SupplyEncoder(space, lambda: np.array([2])),
        low = np.array([-10]),
        high = np.array([10])
    )
    result = encoder.encode()
    assert_equal(result, [0])

def test_scale5():
    space = Discrete(5)
    encoder = ScaleEncoder(SupplyEncoder(space, lambda: np.array([3])),
        low = np.array([-10]),
        high = np.array([10])
    )
    result = encoder.encode()
    assert_equal(result, [5])
