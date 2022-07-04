import numpy
import numpy.testing as tnp
from gym.spaces import Discrete, Box
from wheelly.encoders import IdentityEncoder, ScaleEncoder
import numpy as np
import numpy.testing as tnp

def test_scale_space1():
    space = Discrete(2)
    encoder = ScaleEncoder(IdentityEncoder(space, lambda: np.array([0])),
        low = np.array([10]),
        high = np.array([20])
    )
    result = encoder.space()
    assert result == Box(low=10, high=20, shape=(1,))

def test_scale_space2():
    space = Discrete(5, start= -2)
    encoder = ScaleEncoder(IdentityEncoder(space, lambda: np.array([0])),
        low = np.array([-10]),
        high = np.array([10])
    )
    result = encoder.space()
    assert result == Box(low=-10, high=10, shape=(1,))

def test_scale1():
    space = Discrete(2)
    encoder = ScaleEncoder(IdentityEncoder(space, lambda: np.array([0])),
        low = np.array([10]),
        high = np.array([20])
    )
    result = encoder.encode()
    tnp.assert_equal(result, [10])

def test_scale2():
    space = Discrete(2)
    encoder = ScaleEncoder(IdentityEncoder(space, lambda: np.array([1])),
        low = np.array([10]),
        high = np.array([20])
    )
    result = encoder.encode()
    tnp.assert_equal(result, [20])

def test_scale3():
    space = Discrete(5, start= -2)
    encoder = ScaleEncoder(IdentityEncoder(space, lambda: np.array([-2])),
        low = np.array([-10]),
        high = np.array([10])
    )
    result = encoder.encode()
    tnp.assert_equal(result, [-10])

def test_scale4():
    space = Discrete(5, start= -2)
    encoder = ScaleEncoder(IdentityEncoder(space, lambda: np.array([0])),
        low = np.array([-10]),
        high = np.array([10])
    )
    result = encoder.encode()
    tnp.assert_equal(result, [0])

def test_scale5():
    space = Discrete(5, start= -2)
    encoder = ScaleEncoder(IdentityEncoder(space, lambda: np.array([1])),
        low = np.array([-10]),
        high = np.array([10])
    )
    result = encoder.encode()
    tnp.assert_equal(result, [5])
