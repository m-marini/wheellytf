from typing import Any, Callable
import gym.spaces as spaces
import numpy as np
import math

class Encoder:
    def space(self)->spaces.Space:
        raise Exception("Not implemented")

    def encode(self)->Any:
        raise Exception("Not implemented")

class DictEncoder(Encoder):
    def __init__(self, **encoders:Encoder):
        space_dict = {key:encoders[key].space() for key in encoders}
        self._encoders = encoders
        self._encoded_space = spaces.Dict(space_dict)

    def space(self):
        return self._encoded_space

    def encode(self):
        return {key:self._encoders[key].encode() for key in self._encoders}

class IdentityEncoder(Encoder):
    def __init__(self, space:spaces.Space, value: Callable[[], Any]):
        self._space = space;
        self._value = value

    def space(self):
        return self._space
    
    def encode(self):
        return self._value()

class GetEncoder(Encoder):
    def __init__(self, encoder: Encoder, key: str):
        space = encoder.space()
        assert isinstance(space, spaces.Dict)
        assert space.get(key) != None, f"missing key '{key}'"
        self._encoded_space = space[key]
        self._key = key
        self._encoder = encoder

    def space(self):
        return self._encoded_space

    def encode(self):
        return self._encoder.encode()[self._key]

class MergeBoxEncoder(Encoder):
    def __init__(self, *encoders: Encoder):
        for encoder in encoders:
            assert type(encoder.space()) is spaces.Box \
             and len(encoder.space().shape) == 1

        k = 0
        for encoder in encoders:
            k += encoder.space().shape[0]
        low = np.empty(k, np.float32)
        high = np.empty(k, np.float32)
        index = 0
        for encoder in encoders:
            s = encoder.space()
            n = s.shape[0]
            low[index: index + n] = s.low
            high[index: index + n] = s.high
            index += n

        self._encoders = encoders
        self._space = spaces.Box(low, high)

    def space(self):
        return self._space

    def encode(self):
        n = self._space.shape[0]
        result = np.empty(n, np.float32)
        index = 0
        for encoder in self._encoders:
            s = encoder.space()
            n = s.shape[0]
            result[index: index + n] = encoder.encode()
            index += n
        return result

class MergeDiscreteEncoder(Encoder):
    def __init__(self, *encoders: Encoder):
        for encoder in encoders:
            assert type(encoder.space()) is spaces.Discrete \
                or type(encoder.space()) is spaces.MultiDiscrete

        k = 0
        for encoder in encoders:
            s = encoder.space()
            if isinstance(s, spaces.Discrete):
                k += 1
            elif isinstance(s, spaces.MultiDiscrete):
                k += len(s.nvec)
        nvec = np.zeros(k,  dtype=np.int32)
        index = 0
        for encoder in encoders:
            s = encoder.space()
            if isinstance(s, spaces.Discrete):
                nvec[index] = s.n
                index += 1
            elif isinstance(s, spaces.MultiDiscrete):
                snvec = s.nvec
                n = len(snvec)
                nvec[index:index + n] = snvec
                index += n

        self._encoders = encoders
        self._space = spaces.MultiDiscrete(nvec)

    def space(self):
        return self._space

    def encode(self):
        nvec = self._space.nvec
        n = len(nvec)
        result = np.zeros((n), dtype=np.int32)
        index = 0
        for encoder in self._encoders:
            x = encoder.encode()
            s = encoder.space()
            if isinstance(s, spaces.Discrete):
                result[index] = x - s.start
                index += 1
            elif isinstance(s, spaces.MultiDiscrete):
                n = len(s.nvec)
                result[index: index + n] = x
                index += n
        return result

class MergeBinaryEncoder(Encoder):
    def __init__(self, *encoders: Encoder):
        for encoder in encoders:
            assert type(encoder.space()) is spaces.MultiBinary

        k = 0
        for encoder in encoders:
            s = encoder.space()
            k += s.n

        self._encoders = encoders
        self._space = spaces.MultiBinary(k)

    def space(self):
        return self._space

    def encode(self):
        result = np.zeros((self._space.n), dtype=np.int8)
        index = 0
        for encoder in self._encoders:
            s = encoder.space()
            result[index: index + s.n] = encoder.encode()
            index += s.n
        return result

class ClipEncoder(Encoder):
    def __init__(self, encoder: Encoder, low:np.ndarray, high:np.ndarray):
        self._encoder = encoder
        space = encoder.space()
        assert isinstance(space, spaces.Box)
        assert low.shape == space.low.shape
        assert high.shape == space.high.shape
        self._encoded_space = spaces.Box(low, high, space.shape, dtype = space.dtype)

    def space(self):
        return self._encoded_space

    def encode(self)->np.ndarray:
        x = self._encoder.encode()
        return np.clip(x, self._encoded_space.low, self._encoded_space.high)

class ScaleEncoder(Encoder):
    def __init__(self, encoder: Encoder, low:np.ndarray, high:np.ndarray):
        space = encoder.space()
        assert type(space) == spaces.Discrete
        assert low.shape == (1,)
        assert high.shape == (1,)
        self._encoder = encoder
        self._low = low
        self._high = high
        self._space = spaces.Box(low, high, shape=(1,), dtype = np.float32)

    def space(self):
        return self._space

    def encode(self)->np.ndarray:
        x = self._encoder.encode()
        s = self._encoder.space()
        return (x - s.start) * (self._high - self._low) / (s.n - 1) + self._low

class FeaturesEncoder(Encoder):
    def __init__(self, encoder: Encoder):
        """ Convert a space to features space
        Discrete space is converted in a MultyBinary of n features
        MultiDiscrete space is converted in n = prod(sizes) features (cartesian product)
        """
        space = encoder.space()
        if isinstance(space, spaces.Discrete):
            noFeatures = space.n
        elif isinstance(space, spaces.MultiDiscrete):
            noFeatures = np.prod(space.nvec)
        else:
            assert False, f"space {space} cannot be converted to binary space"
        self._encoder = encoder
        self._features_space = spaces.MultiBinary(noFeatures)
        self._space = space
    
    def space(self):
        return self._features_space

    def encode(self):
        """ Convert a point in space to features space
        Discrete space is converted in a MultyBinary of n features
        MultiDiscrete space is converted in n = prod(sizes) features (cartesian product)
        """
        x = self._encoder.encode()
        result = np.zeros(self._features_space.n, dtype=np.uint8)
        if isinstance(self._space, spaces.Discrete):
            index = x[0] - self._space.start
        elif isinstance(self._space, spaces.MultiDiscrete):
            nvec = self._space.nvec
            k = 1
            index = 0
            for i in range(nvec.size-1, -1, -1):
                index += x[i] * k
                k *= nvec[i]
        else:
            assert False
        result[index] = 1
        return result

def no_tiling(k: int):
    """Returns the number of tiling
    
    Argument:
    k -- the number of space dimensions
    """
    pow2 = math.ceil(math.log(4 * k) / math.log(2))
    return 1 << pow2

def displacement(k: int):
    """" Returns the displacement vector (np array)
    k -- the number of space dimensions
    """
    assert k >= 1
    return  np.arange(k) * 2 + 1

def offsets(k: int):
    """" Returns the offset vectors (np array (n, k))
    k -- the number of space dimensions
    """
    assert k >= 1
    disp = displacement(k)
    n = no_tiling(k)
    z = np.mod(np.broadcast_to(disp, (n,k)) * np.arange(n).reshape((n,1)), n)
    return z

def tile(x: np.ndarray, offsets: np.ndarray):
    """ Returns the coordinates of tile (np array (n, k))

    Arguments:
    x -- (np array(k)) coordinate of space scaled to 0 ... # tiles - 1
    d -- (np.array(n)) displacement of tiling
    """
    #(n, k) = offsets.shape()
    #assert x.shape() == (k)
    n, _ = offsets.shape
    y = np.floor(offsets / n + x).astype(int)
    return y

def features(x: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """ Returns the features tiles (np array, (1))

    Arguments:
    x -- (np array(k)) coordinate of tile
    sizes -- (np.array(k)) the number of tiles
    """
    n = sizes.shape[0]
    if n <= 1:
        return x
    scale = np.ones(n, dtype=int)
    scale[n - 1] = 1
    for i in range(n - 2, -1, -1):
        scale[i] = scale[i + 1] * sizes[i + 1]
    o = x * scale
    return np.sum(o, 1)

def binary_features(x: np.ndarray, n: int):
    m = x.shape[0]
    k = m * n
    result = np.zeros((k), dtype=np.int8)
    for i in range(m):
        idx = i * n + x[i]
        result[idx] = 1
    return result

class TilesEncoder:
    def __init__(self, encoder: Encoder, sizes: np.ndarray):
        space = encoder.space()
        assert isinstance(space, spaces.Box), f"space {space} cannot be converted to tiles space"
        assert len(space.shape) == 1, f"space must have rank 1 {space.shape}"
        k = space.shape[0]
        assert k >= 1, f"space must have at least 1 dimension ({k})"
        sizes1 = sizes + 1
        no_tiles = np.prod(sizes1)
        _no_tiling = no_tiling(k)
        w = (space.high - space.low) / sizes

        self._encoder = encoder
        self._sizes = sizes
        self._sizes1 = sizes1
        self._no_tiles = no_tiles
        self._no_tiling = _no_tiling
        self._w = w
        self._offsets = offsets(k)
        self._tiles_space = spaces.MultiBinary((_no_tiling * no_tiles))

    def space(self):
        """ Convert a Box space of rank 1 to tiles space
        Return a MultyiBinary space of n tiling x m tiles
        """
        return self._tiles_space

    def encode(self):
        """ Convert a Box space of rank 1 to tiles space
        Return a binary array of n tiling x m tiles features
        Args:
        x: the vector in Box space
        """
        x = self._encoder.encode()
        space = self._encoder.space()
        z = (x - space.low) / self._w
        t = tile(z, self._offsets)
        f = features(t, self._sizes1)
        return binary_features(f, self._no_tiles)
