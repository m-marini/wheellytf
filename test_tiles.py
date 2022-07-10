import numpy as np
import re
import pytest
from wheelly.encoders import TilesEncoder, SupplyEncoder, displacement, features, no_tiling, offsets, tile
from numpy.testing import assert_equal
from gym.spaces import Box, MultiBinary,Discrete

def test_no_tiles1():
    n = no_tiling(1)
    assert n == 4

def test_no_tiles2():
    n = no_tiling(2)
    assert n == 8

def test_no_tiles3():
    n = no_tiling(3)
    assert n == 16

def test_no_tiles4():
    n = no_tiling(4)
    assert n == 16

def test_no_tiles5():
    n = no_tiling(5)
    assert n == 32

def test_disp1():
    v = displacement(1)
    assert_equal(v, np.array([1], dtype=np.int32))

def test_disp2():
    v = displacement(2)
    assert_equal(v, np.array([1,3], dtype=np.int32))

def test_disp4():
    v = displacement(4)
    assert_equal(v, np.array([1,3,5,7], dtype=np.int32))

def test_offset1():
    v = offsets(1)
    assert_equal(v, np.array([
        [0],
        [1],
        [2],
        [3],
    ], dtype=np.int32))

def test_offset2():
    v = offsets(2)
    assert_equal(v, np.array([
        [0,0],
        [1,3],
        [2,6],
        [3,1],
        [4,4],
        [5,7],
        [6,2],
        [7,5],
    ], dtype=np.int32))

def test_offset4():
    v = offsets(4)
    assert_equal(v, np.array([
        [0,0,0,0],
        [1,3,5,7],
        [2,6,10,14],
        [3,9,15,5],
        [4,12,4,12],
        [5,15,9,3],
        [6,2,14,10],
        [7,5,3,1],
        [8,8,8,8],
        [9,11,13,15],
        [10,14,2,6],
        [11,1,7,13],
        [12,4,12,4],
        [13,7,1,11],
        [14,10,6,2],
        [15,13,11,9],
    ]))

def test_tile11():
    off = offsets(1)
    exp = np.array([
        [0],
        [0],
        [0],
        [0],
    ])

    x1 = np.array([0])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.24])
    assert_equal(tile(x1, off), exp)

def test_tile12():
    off = offsets(1)
    exp = np.array([
        [0],
        [0],
        [0],
        [1],
    ])

    x1 = np.array([0.25])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.49])
    assert_equal(tile(x1, off), exp)

def test_tile12():
    off = offsets(1)
    exp = np.array([
        [0],
        [0],
        [1],
        [1],
    ])

    x1 = np.array([0.5])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.74])
    assert_equal(tile(x1, off), exp)

def test_tile13():
    off = offsets(1)
    exp = np.array([
        [0],
        [1],
        [1],
        [1],
    ])

    x1 = np.array([0.75])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.99])
    assert_equal(tile(x1, off), exp)

def test_tile14():
    off = offsets(1)
    exp = np.array([
        [1],
        [1],
        [1],
        [1],
    ])

    x1 = np.array([1])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([1.24])
    assert_equal(tile(x1, off), exp)

def test_tile21():
    off = offsets(2)
    exp = np.array([
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
    ])

    x1 = np.array([0,0])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.123,0.123])
    assert_equal(tile(x1, off), exp)


def test_tile22():
    off = offsets(2)
    exp = np.array([
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [1,0],
    ])

    x1 = np.array([0.125,0])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.24,0.124])
    assert_equal(tile(x1, off), exp)

def test_tile23():
    off = offsets(2)
    exp = np.array([
        [0,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
    ])

    x1 = np.array([0.875,0])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.99,0.124])
    assert_equal(tile(x1, off), exp)

def test_tile24():
    off = offsets(2)
    exp = np.array([
        [0,0],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
    ])

    x1 = np.array([0,0.875])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.124,0.99])
    assert_equal(tile(x1, off), exp)

def test_tile25():
    off = offsets(2)
    exp = np.array([
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,1],
        [0,0],
        [0,0],
    ])

    x1 = np.array([0,0.125])
    assert_equal(tile(x1, off), exp)

    x2 = np.array([0.124,0.24])
    assert_equal(tile(x1, off), exp)

def test_tile26():
    off = offsets(2)
    exp = np.array([
        [1,1],
        [1,1],
        [1,1],
        [1,1],
        [1,1],
        [1,1],
        [1,1],
        [1,1],
    ])

    x1 = np.array([1,1])
    assert_equal(tile(x1, off), exp)

    x1 = np.array([1.124,1.124])
    assert_equal(tile(x1, off), exp)

def test_features1():
    x = np.array([
        [0],
        [1],
        [2]
        ])
    sizes = np.array([2])
    exp = np.array([
        [0],
        [1],
        [2]
        ])
    assert_equal(features(x, sizes),
        exp)

def test_features2():
    x = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
        [2,2]
        ])
    sizes = np.array([3,3])
    exp = np.array([0, 1, 3, 4, 8])
    assert_equal(features(x, sizes),
        exp)

def test_features3():
    x = np.array([
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [1,0,0],
        [1,1,1],
        [1,2,2]
        ])
    sizes = np.array([2,3,3])
    exp = np.array([0, 1, 3, 9, 13, 17])
    assert_equal(features(x, sizes),
        exp)

def test_tiles_space1():
    space = Box(-1,1,shape=(1,))
    encoder = TilesEncoder(SupplyEncoder(space, None), np.array([5]))
    tiles_space = encoder.space()
    assert type(tiles_space) is MultiBinary
    assert tiles_space.n == 24

def test_tiles_space2():
    space = Box(np.array([-1,-1]), np.array([1,1]))
    encoder = TilesEncoder(SupplyEncoder(space, None), np.array([2,3]))
    tiles_space = encoder.space()
    assert type(tiles_space) is MultiBinary
    assert tiles_space.n == 8 * 12

def test_tiles_space10():
    space = Discrete(2)
    with pytest.raises(Exception) as e_info:
        TilesEncoder(SupplyEncoder(space,None), np.array([5]))
    assert re.search("space .* cannot be converted to tiles space", str(e_info))

def test_tiles_space11():
    space = Box(0, 1, shape=(1,2))
    with pytest.raises(Exception) as e_info:
        TilesEncoder(SupplyEncoder(space, None), np.array([5]))
    assert re.search("space must have rank 1 \(1, 2\)", str(e_info))

def test_tiles_space12():
    space = Box(0,1,shape=(0,))
    with pytest.raises(Exception) as e_info:
        TilesEncoder(SupplyEncoder(space, None), np.array([5]))
    assert re.search("space must have at least 1 dimension \(0\)", str(e_info))

def test_tiles1():
    space = Box(1, 5, shape=(1,))
    sizes = np.array([2])
    exp = np.array([
        1,0,0,
        1,0,0,
        1,0,0,
        1,0,0,
    ])
    encoder = TilesEncoder(SupplyEncoder(space, lambda:np.array([1])), sizes)
    y = encoder.encode()
    assert y.dtype == np.int8
    assert_equal(y, exp)

def test_tiles2():
    space = Box(1, 5, shape=(1,))
    sizes = np.array([2])
    exp = np.array([
        1,0,0,
        1,0,0,
        1,0,0,
        0,1,0,
    ])
    encoder = TilesEncoder(SupplyEncoder(space, lambda:np.array([1.5])), sizes)
    y = encoder.encode()
    assert y.dtype == np.int8
    assert_equal(y, exp)

def test_tiles3():
    space = Box(1, 5, shape=(1,))
    sizes = np.array([2])
    exp = np.array([
        0,1,0,
        0,1,0,
        0,1,0,
        0,1,0,
    ])
    encoder = TilesEncoder(SupplyEncoder(space, lambda:np.array([3])), sizes)
    y = encoder.encode()
    assert y.dtype == np.int8
    assert_equal(y, exp)

def test_tiles4():
    space = Box(1, 5, shape=(1,))
    sizes = np.array([2])
    exp = np.array([
        0,1,0,
        0,0,1,
        0,0,1,
        0,0,1,
    ])
    encoder = TilesEncoder(SupplyEncoder(space,lambda:np.array([4.5])), sizes)
    y = encoder.encode()
    assert y.dtype == np.int8
    assert_equal(y, exp)

def test_tiles5():
    space = Box(1, 5, shape=(1,))
    sizes = np.array([2])
    exp = np.array([
        0,0,1,
        0,0,1,
        0,0,1,
        0,0,1,
    ])
    encoder = TilesEncoder(SupplyEncoder(space,lambda:np.array([5])), sizes)
    y = encoder.encode()
    assert y.dtype == np.int8
    assert_equal(y, exp)

def test_tiles10():
    space = Box(np.array([1,1]), np.array([5,5]))
    sizes = np.array([2,2])
    exp = np.array([
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
    ])
    encoder = TilesEncoder(SupplyEncoder(space,lambda:np.array([1,1])), sizes)
    y = encoder.encode()
    assert y.dtype == np.int8
    assert_equal(y, exp)

def test_tiles11():
    space = Box(np.array([1,1]), np.array([5,5]))
    sizes = np.array([2,2])
    exp = np.array([
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,0,
        0,0,0,1,0,0,0,0,0,
    ])
    encoder = TilesEncoder(SupplyEncoder(space, lambda:np.array([1.25,1.25])),sizes)
    y = encoder.encode()
    assert y.dtype == np.int8
    assert_equal(y, exp)

def test_tiles12():
    space = Box(np.array([1,1]), np.array([5,5]))
    sizes = np.array([2,2])
    exp = np.array([
        0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,1,
    ])
    encoder = TilesEncoder(SupplyEncoder(space,lambda:np.array([5,5])), sizes)
    y = encoder.encode()
    assert y.dtype == np.int8
    assert_equal(y, exp)
