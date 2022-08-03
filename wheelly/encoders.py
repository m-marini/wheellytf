from typing import Any, Callable
from gym.spaces import Discrete, Space, MultiDiscrete, MultiBinary, Box, Dict
import numpy as np
import math

class Encoder:
    def __init__(self, space: Space):
        self._space = space

    def space(self):
        return self._space

    def encode(self) -> Any:
        raise NotImplementedError()

    def spec(self) -> dict[Any]:
        return spec(self._space)

class SupplyEncoder(Encoder):
    def __init__(self, space: Space, value: Callable[[], Any]):
        super().__init__(space)
        self._value = value

    def space(self):
        return self._space

    def encode(self):
        return self._value()

class DictEncoder(Encoder):
    def __init__(self, **encoders: Encoder):
        space_dict = {key: encoders[key].space() for key in encoders}
        super().__init__(Dict(space_dict))
        self._encoders = encoders

    def encode(self):
        return {key: self._encoders[key].encode() for key in self._encoders}

class GetEncoder(Encoder):
    def __init__(self, encoder: Encoder, key: str):
        space = encoder.space()
        assert isinstance(space, Dict)
        assert space.get(key) != None, f"missing key '{key}'"
        super().__init__(space[key])
        self._key = key
        self._encoder = encoder

    def encode(self):
        return self._encoder.encode()[self._key]

class MergeEncoder(Encoder):
    @staticmethod
    def create(*encoders: Encoder):
        s = encoders[0].space()
        if isinstance(s, MultiBinary):
            return MergeBinaryEncoder(*encoders)
        elif isinstance(s, Box):
            return MergeBoxEncoder(*encoders)
        elif isinstance(s, Discrete) or isinstance(s, MultiDiscrete):
            return MergeDiscreteEncoder(*encoders)
        else:
            raise Exception(f"space ({s}) cannot be merged")

class MergeBoxEncoder(Encoder):
    def __init__(self, *encoders: Encoder):
        for encoder in encoders:
            assert isinstance(encoder.space(), Box) \
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
        super().__init__(Box(low, high))
        self._encoders = encoders

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
            assert isinstance(encoder.space(), Discrete) \
                or isinstance(encoder.space(), MultiDiscrete)
        k = 0
        for encoder in encoders:
            s = encoder.space()
            if isinstance(s, Discrete):
                k += 1
            elif isinstance(s, MultiDiscrete):
                k += len(s.nvec)
        nvec = np.zeros(k,  dtype=np.int32)
        index = 0
        for encoder in encoders:
            s = encoder.space()
            if isinstance(s, Discrete):
                nvec[index] = s.n
                index += 1
            elif isinstance(s, MultiDiscrete):
                snvec = s.nvec
                n = len(snvec)
                nvec[index:index + n] = snvec
                index += n

        super().__init__(MultiDiscrete(nvec))
        self._encoders = encoders

    def encode(self):
        nvec = self._space.nvec
        n = len(nvec)
        result = np.zeros((n), dtype=np.int32)
        index = 0
        for encoder in self._encoders:
            x = encoder.encode()
            s = encoder.space()
            if isinstance(s, Discrete):
                result[index] = x - s.start
                index += 1
            elif isinstance(s, MultiDiscrete):
                n = len(s.nvec)
                result[index: index + n] = x
                index += n
        return result

class MergeBinaryEncoder(Encoder):
    def __init__(self, *encoders: Encoder):
        for encoder in encoders:
            assert isinstance(encoder.space(), MultiBinary)

        k = 0
        for encoder in encoders:
            s = encoder.space()
            k += s.n

        super().__init__(MultiBinary(k))
        self._encoders = encoders

    def encode(self):
        result = np.zeros((self._space.n), dtype=np.int8)
        index = 0
        for encoder in self._encoders:
            s = encoder.space()
            result[index: index + s.n] = encoder.encode()
            index += s.n
        return result

class ClipEncoder(Encoder):
    def __init__(self, encoder: Encoder, low: np.ndarray, high: np.ndarray):
        self._encoder = encoder
        space = encoder.space()
        assert isinstance(space, Box)
        assert low.shape == space.low.shape
        assert high.shape == space.high.shape
        super().__init__(Box(low, high))

    def encode(self) -> np.ndarray:
        x = self._encoder.encode()
        return np.clip(x, self._space.low, self._space.high)

class Binary2BoxEncoder(Encoder):
    def __init__(self, encoder: Encoder):
        self._encoder = encoder
        space = encoder.space()
        assert isinstance(space, MultiBinary)
        super().__init__(Box(shape=(space.n,), low=0, high=1))

    def encode(self) -> np.ndarray:
        return self._encoder.encode()

class ScaleEncoder(Encoder):
    def __init__(self, encoder: Encoder, low: np.ndarray, high: np.ndarray):
        space = encoder.space()
        assert isinstance(space, Discrete)
        assert low.shape == (1,)
        assert high.shape == (1,)
        super().__init__(Box(low, high, (1,)))
        self._encoder = encoder

    def encode(self) -> np.ndarray:
        x = self._encoder.encode()
        s = self._encoder.space()
        return (x - s.start) * (self._space.high - self._space.low) / (s.n - 1) + self._space.low

class FeaturesEncoder():
    @staticmethod
    def create(encoder: Encoder):
        space = encoder.space()
        if isinstance(space, Discrete):
            return FeaturesDiscreteEncoder(encoder)
        elif isinstance(space, MultiDiscrete):
            return FeaturesMultiDiscreteEncoder(encoder)
        else:
            assert False, f"space {space} cannot be converted to binary space"

class FeaturesDiscreteEncoder(Encoder):
    def __init__(self, encoder: Encoder):
        """ Convert a space to features space
        Discrete space is converted in a MultyBinary of n features
        MultiDiscrete space is converted in n = prod(sizes) features (cartesian product)
        """
        space = encoder.space()
        assert isinstance(space, Discrete)
        super().__init__(MultiBinary(space.n))
        self._encoder = encoder

    def encode(self):
        """ Convert a point in space to features space
        Discrete space is converted in a MultyBinary of n features
        MultiDiscrete space is converted in n = prod(sizes) features (cartesian product)
        """
        x = self._encoder.encode()
        s = self._encoder.space()
        result = np.zeros(self._space.n, dtype=np.uint8)
        index = x[0] - s.start
        result[index] = 1
        return result

class FeaturesMultiDiscreteEncoder(Encoder):
    def __init__(self, encoder: Encoder):
        """ Convert a space to features space
        Discrete space is converted in a MultyBinary of n features
        MultiDiscrete space is converted in n = prod(sizes) features (cartesian product)
        """
        space = encoder.space()
        assert isinstance(space, MultiDiscrete)
        noFeatures = np.prod(space.nvec)
        super().__init__(MultiBinary(noFeatures))
        self._encoder = encoder

    def encode(self):
        """ Convert a point in space to features space
        Discrete space is converted in a MultyBinary of n features
        MultiDiscrete space is converted in n = prod(sizes) features (cartesian product)
        """
        x = self._encoder.encode()
        result = np.zeros(self._space.n, dtype=np.uint8)
        nvec = self._encoder.space().nvec
        k = 1
        index = 0
        for i in range(nvec.size-1, -1, -1):
            index += x[i] * k
            k *= nvec[i]
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
    return np.arange(k) * 2 + 1


def offsets(k: int):
    """" Returns the offset vectors (np array (n, k))
    k -- the number of space dimensions
    """
    assert k >= 1
    disp = displacement(k)
    n = no_tiling(k)
    z = np.mod(np.broadcast_to(disp, (n, k)) * np.arange(n).reshape((n, 1)), n)
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
        assert isinstance(space,  Box), f"space {space} cannot be converted to tiles space"
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
        self._tiles_space = MultiBinary((_no_tiling * no_tiles))

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


def spec(space: Space):
    if isinstance(space, Discrete):
        assert space.start == 0, f"space must start with 0 value ({space.start})"
        return {
            'type': 'int',
            'shape': 1,
            'num_values': space.n
        }
    elif isinstance(space, MultiDiscrete):
        n = space.nvec[0]
        for x in space.nvec:
            assert x == n, f"space must have same number of values ({x} != {n})"
        return {
            'type': 'int',
            'shape': len(space.nvec),
            'num_values': n
        }
    elif isinstance(space, MultiBinary):
        return {
            'type': 'bool',
            'shape': space.n
        }
    elif isinstance(space, Box):
        _min = float(space.low[0])
        _max = float(space.high[0])
        for x in space.low:
            assert x == _min, f"space must have same min value ({x} != {_min})"
        for x in space.high:
            assert x == _max, f"space must have same min value ({x} != {_max})"
        return {
            'type': 'float',
            'shape': space.shape,
            'min_value': _min,
            'max_value': _max
        }
    elif isinstance(space, Dict):
        return {key: spec(space[key]) for key in space}
    else:
        raise Exception(f"no spec for space {space}")


def createSpace(spec: dict[Any]) -> Space:
    typ = spec.get('type')
    if typ == None:
        return Dict({key: createSpace(spec[key]) for key in spec})
    else:
        shape = spec['shape']
        shape = (shape,) if type(shape) == int else tuple(shape)
        if typ == 'bool':
            return MultiBinary(shape)
        elif typ == 'int':
            num_values = spec['num_values']
            if len(shape) == 1 and shape[0] == 1:
                return Discrete(num_values)
            else:
                return MultiDiscrete(np.ones(shape) * num_values)
        elif typ == 'float':
            low = spec['min_value']
            high = spec['max_value']
            return Box(low, high, shape)
        else:
            raise Exception(f"no space for spec {spec}")
