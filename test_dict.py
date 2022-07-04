from gym.spaces import Dict,Discrete
import scipy as sp
from wheelly.encoders import DictEncoder, IdentityEncoder
import numpy.testing as tnp

def test_dict_space():
    space1 = Discrete(2)
    space2 = Discrete(3)
    encoder = DictEncoder(
        a=IdentityEncoder(space1, None),
        b=IdentityEncoder(space2, None))
    result = encoder.space()
    assert isinstance(result, Dict)
    assert result.get("a") == space1
    assert result.get("b") == space2

def test_dict():
    space1 = Discrete(2)
    space2 = Discrete(3)
    a = space1.sample()
    b = space2.sample()
    encoder = DictEncoder(
        a=IdentityEncoder(space1, lambda:a),
        b=IdentityEncoder(space2, lambda:b))
    result = encoder.encode()
    assert result.get("a") == a
    assert result.get("b") == b
