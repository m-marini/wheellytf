import numpy as np
from gym.spaces import Dict, Discrete

from wheelly.encoders import DictEncoder, SupplyEncoder


def test_dict_space():
    space1 = Discrete(2)
    space2 = Discrete(3)
    a = np.array([1])
    b = np.array([2])
    encoder = DictEncoder(
        a=SupplyEncoder(space1, lambda: a),
        b=SupplyEncoder(space2, lambda: b))
    result = encoder.space
    assert isinstance(result, Dict)
    assert result.get("a") == space1
    assert result.get("b") == space2

def test_dict():
    space1 = Discrete(2)
    space2 = Discrete(3)
    a = np.array([1])
    b = np.array([2])
    encoder = DictEncoder(
        a=SupplyEncoder(space1, lambda: a),
        b=SupplyEncoder(space2, lambda: b))
    result = encoder.encode()
    assert result.get("a") == a
    assert result.get("b") == b
