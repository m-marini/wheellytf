from math import radians
from typing import Tuple

import numpy as np
from numpy import ndarray

from wheelly.utils import direction, normalizeRad, sign

_RAD_15 = radians(15)

class ObstacleMap:
    """Map of obstacles"""
    def __init__(self, obstacles: ndarray=np.zeros((0,2)), size = 0.2):
        """Creates a obstacle maps
        Arguments:
        obstacles -- the coordinate of obstacles [n, 2)
        size -- the size of obstacles (m)"""
        assert len(obstacles.shape) == 2, f"obstacles should have rank 2 ({len(obstacles.shape)})"
        assert obstacles.shape[1] == 2, f"obstacles should have shape(..., 2) ({obstacles.shape})"
        self._obstacles = obstacles
        self._size = size
    
    def num_obstacles(self):
        """Returns the number of obstacles"""
        return self._obstacles.shape[0]
    
    def __getitem__(self, index: int):
        """Returns the coordin of an obstacle"""
        return self._obstacles[index, :]

    def nearest(self, location: ndarray, dir_rad: float, range_rad: float = _RAD_15):
        """Returns the nearest obstacle from a location in a direction within a direction range"""
        index = None
        dist = 1e38
        for i in range(0, self.num_obstacles()):
            d_obs = self[i] - location
            obs_dir = direction(d_obs)
            d_angle = abs(normalizeRad(obs_dir - dir_rad))
            if d_angle <= range_rad:
                d = np.linalg.norm(d_obs)
                if d < dist:
                    dist = d
                    index = i
        return index, dist

    def contains(self, location: Tuple[float, float]):
        """Returns true if map contains obstacle at location
        Argument:
        location -- the location"""
        location = snap(location, self._size)
        location = np.array(location)
        for i in range(0, self.num_obstacles()):
            o = self[i]
            if np.array_equal(o, location):
                return True
        return False
    
    def size(self):
        return self._size

class ObstacleMapBuilder:
    """Obstacle map builder"""
    def __init__(self, size: float) -> None:
        """Create the obstacle map builder
        Argument
        size -- the size of obstacle"""
        self._list: list[Tuple[float, float]] = []
        self._size = size

    def build(self):
        """Builds the osbatcle map"""
        obs = set(self._list)
        if len(obs) > 0:
            ary = np.array(list(obs))
            return ObstacleMap(obstacles=ary, size=self._size)
        else:
            return ObstacleMap(size=self._size)

    def add(self, location: Tuple[float, float]):
        """Adds an obstacle snapping to the map grid size
        Argument
        obstacle -- the coordinate of obstacle"""
        self._list.append(snap(location, self._size))
        return self
    
    def rect(self, point1:Tuple[float, float], point2:Tuple[float, float]):
        """Adds obstacle by creating a rectangle
        Arguments:
        point1 -- the rectangle corner
        point2 -- the opposite rectangle corner"""
        return self.line(
            point1,
            (point2[0], point1[1]),
            point2,
            (point1[0], point2[1]),
            point1
        )

    def line(self, *points: Tuple[float, float]):
        """Adds obstacle by creating an obstacle line
        Arguments:
        points -- the list of points"""
        n = len(points)
        if n == 1:
            self.add(points[0])
        elif n > 1:
            curs = points[0]
            for i in range(1, n):
                self._line(curs, points[i])
                curs = points[i]
        return self

    def _line(self, start: Tuple[float, float], end: Tuple[float, float]):
        """Adds obstacle by creating an obstacle line
        Arguments:
        start -- the start location
        end -- the end location"""
        start = snap(start, self._size)
        end = snap(end, self._size)
        x = start[0]
        y = start[1]
        dx = end[0] - x
        dy = end[1] - y
        if dx == 0 and dy == 0:
            self.add(start)
        elif abs(dx) >= abs(dy):
            n = int(abs(dx) / self._size) + 1
            self._line1(x, y, n, sign(dx) * self._size, dy / abs(dx) * self._size)
        else:
            n = int(abs(dy) / self._size) + 1
            self._line1(x, y, n, dx / abs(dy) * self._size, sign(dy) * self._size)
        return self
    
    def _line1(self, x: float, y: float, n: int, dx:float, dy:float):
        for i in range(0, n):
            self.add((x, y))
            x += dx
            y += dy


def snap(location:Tuple[float, float], size:float):
    x = round(location[0] / size) * size
    y = round(location[1] / size) * size
    return x, y
