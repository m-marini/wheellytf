from math import pi
from wheelly.sims import ObstacleMap, ObstacleMapBuilder
from numpy.testing import assert_equal

import numpy as np

def test_empty():
    map = ObstacleMap(size=0.2)
    assert map.num_obstacles() == 0

def test_num_obstacles():
    map = ObstacleMap(obstacles=np.array([
        [0.0, 0.0],
        [0.2, 0.2]
    ]), size=0.2)
    assert map.num_obstacles() == 2

def test_get():
    map = ObstacleMap(obstacles=np.array([
        [0.0, 0.0],
        [0.2, 0.2]
    ]), size=0.2)
    assert_equal(map[1], (0.2,0.2))

def test_nearest():
    map = ObstacleMap(obstacles=np.array([
        [0.0, 0.0],
        [4.0, 3.0]
    ]), size=0.2)
    i, dist = map.nearest(np.array([-1, 0]), 0)
    assert i == 0
    assert_equal(dist, 1)
    i, dist = map.nearest(np.array([-1, 0]), pi)
    assert i == None
    i, dist = map.nearest(np.array([8, 0]), pi, range_rad=pi/2)
    assert i == 1
    assert_equal(dist, 5)

def test_builder_mpty():
    builder = ObstacleMapBuilder(size=0.2)
    map = builder.build()
    assert map.num_obstacles() == 0

def test_builder_add0():
    builder = ObstacleMapBuilder(size=0.2)\
        .add((0, 0))
    map = builder.build()
    assert map.num_obstacles() == 1
    assert_equal(map[0], (0, 0))

def test_builder_add1():
    builder = ObstacleMapBuilder(size=0.2)\
        .add((0.09, 0.09))
    map = builder.build()
    assert map.num_obstacles() == 1
    assert_equal(map[0], (0, 0))

def test_builder_add2():
    builder = ObstacleMapBuilder(size=0.2)\
        .add((-0.09, -0.09))
    map = builder.build()
    assert map.num_obstacles() == 1
    assert_equal(map[0], (0, 0))

def test_builder_add3():
    builder = ObstacleMapBuilder(size=0.2)\
        .add((0.101, 0.101))
    map = builder.build()
    assert map.num_obstacles() == 1
    assert_equal(map[0], (0.2, 0.2))

def test_builder_add4():
    builder = ObstacleMapBuilder(size=0.2)\
        .add((-0.101, -0.101))
    map = builder.build()
    assert map.num_obstacles() == 1
    assert_equal(map[0], (-0.2, -0.2))

def test_builder_add_duplicate():
    builder = ObstacleMapBuilder(size=0.2) \
        .add((0, 0)) \
        .add((-0.09, -0.09))
    map = builder.build()
    assert map.num_obstacles() == 1
    assert_equal(map[0], (0, 0))

def test_builder_point_line():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((0.0, 0.0), (0.0,0.0))
    map = builder.build()
    assert map.num_obstacles() == 1
    assert map.contains((0.0, 0))

def test_builder_hline1():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((0., 0.), (1.,0.))
    map = builder.build()
    assert map.num_obstacles() == 6
    assert map.contains((0, 0))
    assert map.contains((0.2, 0))
    assert map.contains((0.4, 0))
    assert map.contains((0.6, 0))
    assert map.contains((0.8, 0))
    assert map.contains((1, 0))

def test_builder_hline2():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((0.09, 0.), (0.91,0.01))
    map = builder.build()
    assert map.num_obstacles() == 6
    assert map.contains((0.0, 0))
    assert map.contains((0.2, 0))
    assert map.contains((0.4, 0))
    assert map.contains((0.6, 0))
    assert map.contains((0.8, 0))
    assert map.contains((1, 0))

def test_builder_hline3():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((0.0, 0.0), (1,0.2))
    map = builder.build()
    assert map.num_obstacles() == 6
    assert map.contains((0.0, 0))
    assert map.contains((0.2, 0))
    assert map.contains((0.4, 0))
    assert map.contains((0.6, 0.2))
    assert map.contains((0.8, 0.2))
    assert map.contains((1, 0.2))

def test_builder_hline4():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((1,0.2), (0.0, 0.0))
    map = builder.build()
    assert map.num_obstacles() == 6
    assert map.contains((0.0, 0))
    assert map.contains((0.2, 0))
    assert map.contains((0.4, 0))
    assert map.contains((0.6, 0.2))
    assert map.contains((0.8, 0.2))
    assert map.contains((1, 0.2))

def test_builder_vline1():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((0., 0.), (0.,1.))
    map = builder.build()
    assert map.num_obstacles() == 6
    assert map.contains((0, 0))
    assert map.contains((0, 0.2))
    assert map.contains((0, 0.4))
    assert map.contains((0, 0.6))
    assert map.contains((0, 0.8))
    assert map.contains((0, 1))

def test_builder_vline2():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((0.09, 0.), (0.01,0.91))
    map = builder.build()
    assert map.num_obstacles() == 6
    assert map.contains((0, 0))
    assert map.contains((0, 0.2))
    assert map.contains((0, 0.4))
    assert map.contains((0, 0.6))
    assert map.contains((0, 0.8))
    assert map.contains((0, 1))

def test_builder_vline3():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((0.0, 0.0), (0.2,1.0))
    map = builder.build()
    assert map.num_obstacles() == 6
    assert map.contains((0., 0))
    assert map.contains((0., 0.2))
    assert map.contains((0., 0.4))
    assert map.contains((0.2, 0.6))
    assert map.contains((0.2, 0.7))
    assert map.contains((0.2, 1.))

def test_builder_vline4():
    builder = ObstacleMapBuilder(size=0.2) \
        .line((0.2,1.0),(0.0, 0.0))
    map = builder.build()
    assert map.num_obstacles() == 6
    assert map.contains((0., 0))
    assert map.contains((0., 0.2))
    assert map.contains((0., 0.4))
    assert map.contains((0.2, 0.6))
    assert map.contains((0.2, 0.7))
    assert map.contains((0.2, 1.))

def test_builder_rect():
    builder = ObstacleMapBuilder(size=0.2) \
        .rect((0.0, 0.0), (0.4,0.4))
    map = builder.build()
    assert map.num_obstacles() == 8
    assert map.contains((0, 0))
    assert map.contains((0.2, 0))
    assert map.contains((0.4, 0))

    assert map.contains((0, 0.4))
    assert map.contains((0.2, 0.4))
    assert map.contains((0.4, 0.4))

    assert map.contains((0, 0.2))

    assert map.contains((0.4, 0.2))
