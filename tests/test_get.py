import numpy as np
import pytest
from gym.spaces import Dict, Discrete
from numpy.testing import assert_equal

from wheelly.encoders import GetEncoder, SupplyEncoder


def test_get_space():
    space = Dict(
        a=Discrete(2),
        b=Discrete(3),
        c=Discrete(4)
    )
    encoder = GetEncoder(SupplyEncoder(space, None), "a")
    result = encoder.space
    assert result == space["a"]

def test_get_space10():
    space = Dict(
        a=Discrete(2),
        b=Discrete(3),
        c=Discrete(4)
    )
    with pytest.raises(Exception) as e_info:
        GetEncoder(SupplyEncoder(space, None), "d")

def test_get1():
    space = Dict(
        a=Discrete(2),
        b=Discrete(3),
        c=Discrete(4)
    )
    x = {
        "a": np.array([1]),
        "b": np.array([2]),
        "c": np.array([3]),
    }
    encoder = GetEncoder(SupplyEncoder(space, lambda:x), "a")
    result = encoder.encode()
    assert_equal(result, x["a"])
