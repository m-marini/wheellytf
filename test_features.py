import numpy as np
import numpy.testing as tnp
from gym import spaces
import wheelly.encoders as en
import numpy.testing as tnp
import pytest

def test_features_space_binary():
    space = spaces.MultiBinary((10, 10))
    with pytest.raises(Exception) as e_info:
        en.FeaturesEncoder(space)

def test_features_space_box():
    space = spaces.Box(low=0, high=2, shape=(2,2), dtype=int)
    with pytest.raises(Exception) as e_info:
        en.FeaturesEncoder(space)

def test_features_space_discrete():
    space = spaces.Discrete(3)
    encoder = en.FeaturesEncoder(en.IdentityEncoder(space, None))
    bin_space = encoder.space()
    assert isinstance(bin_space, spaces.MultiBinary)
    assert bin_space.shape == (3,)

def test_features_space_multidiscrete():
    space = spaces.MultiDiscrete((2,3))
    encoder = en.FeaturesEncoder(en.IdentityEncoder(space, None))
    bin_space = encoder.space()
    assert isinstance(bin_space, spaces.MultiBinary)
    assert bin_space.shape == (6,)

def test_features_space_dict():
    space = spaces.Dict({"a": spaces.Discrete(2)})
    with pytest.raises(Exception) as e_info:
        en.FeaturesEncoder(space)

def test_features_discrete():
    space = spaces.Discrete(3, start=-1)
    x = np.array([-1], dtype=int)
    encoder = en.FeaturesEncoder(en.IdentityEncoder(space, lambda:x))
    y = encoder.encode()
    tnp.assert_array_equal(y, np.array([1,0,0]))

def test_features_multidiscrete1():
    space = spaces.MultiDiscrete((2,3))
    x = np.array([0,0], dtype=int)
    encoder = en.FeaturesEncoder(en.IdentityEncoder(space, lambda:x))
    y = encoder.encode()
    tnp.assert_array_equal(y, np.array([1,0,0,0,0,0]))

def test_features_multidiscrete2():
    space = spaces.MultiDiscrete((2,3))
    x = np.array([1,1], dtype=int)
    encoder = en.FeaturesEncoder(en.IdentityEncoder(space, lambda :x))
    y = encoder.encode()
    tnp.assert_array_equal(y, np.array([0,0,0,0,1,0]))
