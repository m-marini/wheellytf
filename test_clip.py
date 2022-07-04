import numpy as np
import numpy.testing as tnp
from gym import spaces
import wheelly.encoders as en
import numpy.testing as tnp
import pytest

def test_clip_space1():
    space = spaces.Box(low=np.array([-10,-10]), high=np.array([10,20]))
    low=np.array([-1, 0])
    high=np.array([0, 1])
    encoder = en.ClipEncoder(en.IdentityEncoder(space, None), low, high)
    clip_space = encoder.space()
    assert isinstance(clip_space, spaces.Box)
    tnp.assert_array_equal(clip_space.low, low)
    tnp.assert_array_equal(clip_space.high, high)

def test_clip_space10():
    space = spaces.Discrete(2)
    with pytest.raises(Exception) as e_info:
        en.ClipEncoder(space, np.array([5]), np.array([5]))

def test_clip_1():
    space = spaces.Box(low=np.array([-10,-10]), high=np.array([10,20]))
    low=np.array([-1, 0])
    high=np.array([0, 1])
    encoder = en.ClipEncoder(en.IdentityEncoder(space, lambda:np.array([-10, -10])), low, high)
    y = encoder.encode()
    tnp.assert_array_equal(y, np.array([-1,0]))
