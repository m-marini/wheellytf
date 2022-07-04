import numpy.testing as tnp
from gym.spaces import Dict,Discrete
import scipy as sp
from wheelly.encoders import GetEncoder, IdentityEncoder
import numpy.testing as tnp
import pytest

def test_get_space():
    space = Dict({
        "a": Discrete(2),
        "b": Discrete(3),
        "c": Discrete(4)
    })
    encoder = GetEncoder(IdentityEncoder(space, None), "a")
    result = encoder.space()
    assert result == space["a"]

def test_get_space10():
    space = Dict({
        "a": Discrete(2),
        "b": Discrete(3),
        "c": Discrete(4)
    })
    with pytest.raises(Exception) as e_info:
        GetEncoder(IdentityEncoder(space, None), "d")

def test_get1():
    space = Dict({
        "a": Discrete(2),
        "b": Discrete(3),
        "c": Discrete(4)
    })
    x = space.sample()
    encoder = GetEncoder(IdentityEncoder(space, lambda:x), "a")
    result = encoder.encode()
    tnp.assert_array_equal(result, x["a"])
