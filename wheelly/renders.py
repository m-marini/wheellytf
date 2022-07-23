import math
from typing import Iterable, Tuple
import pygame
import numpy as np
from numpy import ndarray

from wheelly.robots import RobotAPI

WINDOW_SIZE = 800
GRID_SIZE = 0.2
ROBOT_WIDTH = 0.18
ROBOT_LENGTH = 0.26
ROBOT_W_BEVEL = 0.06
ROBOT_L_BEVEL = 0.05
WORLD_SIZE = 10.0
BACKGROUND_COLOR = (0, 0, 0)
ROBOT_COLOR = (255, 255, 0)
ROBOT_SHAPE = np.array([
    (ROBOT_WIDTH / 2 - ROBOT_W_BEVEL, ROBOT_LENGTH / 2),
    (ROBOT_WIDTH / 2, ROBOT_LENGTH / 2 - ROBOT_L_BEVEL),
    (ROBOT_WIDTH / 2, -ROBOT_LENGTH / 2),
    (-ROBOT_WIDTH / 2, -ROBOT_LENGTH / 2),
    (-ROBOT_WIDTH / 2, ROBOT_LENGTH / 2 - ROBOT_L_BEVEL),
    (-ROBOT_WIDTH / 2 + ROBOT_W_BEVEL, ROBOT_LENGTH / 2),
    ])

OBSTACLE_SIZE = 0.2
OBSTACLE_COLOR = (255, 0, 0)
OBSTACLE_SHAPE = np.array([
    (OBSTACLE_SIZE / 2, OBSTACLE_SIZE / 2),
    (-OBSTACLE_SIZE / 2, OBSTACLE_SIZE / 2),
    (-OBSTACLE_SIZE / 2, -OBSTACLE_SIZE / 2),
    (OBSTACLE_SIZE / 2, -OBSTACLE_SIZE / 2),
])

SENSOR_COLOR = (255, 0, 0)
SENSOR_LENGTH = 3.0
SENSOR_SHAPE = np.array([
    (0.0, 0.0),
    (SENSOR_LENGTH, 0.0)
])

_FONT_NAME = 'freesans'
_FONT_SIZE = 16

class RobotWindow:

    def __init__(self):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self._robor_pos= np.zeros(2)
        self._robor_dir= 0
        self._sensor_dir = 0
        self._contacts = 0
        self._font = pygame.font.SysFont(_FONT_NAME, _FONT_SIZE) 
        self._canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        self._trans = _transMatrix()

    def robot_pos(self, robot_location:ndarray):
        self._robot_pos = robot_location
        return self

    def robot_dir(self, robot_dir:int):
        self._robot_dir = robot_dir
        return self

    def sensor_dir(self, sensor_dir:int):
        self._sensor_dir = sensor_dir
        return self

    def set_robot(self, robot: RobotAPI):
        self._robot_pos = robot.robot_pos()
        self._robot_dir = robot.robot_dir()
        self._sensor_dir = robot.sensor_dir()
        self._distance = robot.sensor_distance()
        self._sensor_obstacle = robot.sensor_obstacle()
        self._obstaclesMap = robot.obstaclesMap()
        self._contacts = robot.contacts()
        return self

    def render(self):
        self._canvas.fill(BACKGROUND_COLOR)
        self._drawRobot() \
            ._drawSensor() \
            ._drawObstacles() \
            ._drawHud()

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(self._canvas, self._canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        return self

    def _drawHud(self):
        self.drawText(text=f"sensor {self._distance:.1f} m", location=(0, 0), color=(0, 255, 0))
        self.drawText(text=f"contacts {self._contacts} m", location=(0, _FONT_SIZE), color=(0, 255, 0))
        return self

    def drawText(self, text: str, location: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw a text"""
        graphTxt = self._font.render(text, True, color)
        textRect = graphTxt.get_rect()
        textRect.topleft = location
        self._canvas.blit(graphTxt, textRect)
        return self

    def _drawRobot(self):
        """Draw the robot shape"""
        trans = rotate(-math.radians(self._robot_dir)) @ translate(self._robot_pos) @ self._trans
        shape = transform(trans, ROBOT_SHAPE)
        pygame.draw.polygon(
            surface=self._canvas,
            color=ROBOT_COLOR,
            points=shape)
        return self

    def _drawObstacle(self, location: ndarray, phantom=False):
        """Draw the obstacle shape"""
        trans = translate(location) @ self._trans
        shape = transform(trans, OBSTACLE_SHAPE)
        pygame.draw.polygon(
            surface=self._canvas,
            color=OBSTACLE_COLOR,
            width=1 if phantom else 0,
            points=shape)
        return self

    def _drawObstacles(self):
        """Draw the obstacle maps"""
        if self._obstaclesMap:
            for i in range(0, self._obstaclesMap.num_obstacles()):
                self._drawObstacle(location=self._obstaclesMap[i], phantom=True)
        return self

    def _drawSensor(self):
        """Draw the sensor shape
            
        Arguments:
        canvas -- the canvas
        trans -- the transformation matrix
        robot_location -- the robot location
        robot._dir -- the robot direction (DEG)
        sens_dir -- the sensor direction relative to robot (DEG)
        """
        angle = math.radians(90 - self._robot_dir - self._sensor_dir)
        shape = SENSOR_SHAPE.copy()
        shape = transform(rotate(angle) @ translate(self._robot_pos) @ self._trans,
                shape)
        pygame.draw.line(
            surface=self._canvas,
            color=SENSOR_COLOR,
            start_pos=shape[0, :],
            end_pos=shape[1, :]
        )
        if self._sensor_obstacle:
            loc = self._sensor_obstacle
            self._drawObstacle(np.array([loc.x, loc.y]))
        return self
    
def _transMatrix():
    """Return the transform matrix to render the shapes"""
    #return scale((WINDOW_SIZE / WORLD_SIZE, WINDOW_SIZE / WORLD_SIZE)) @ rotate(-math.pi/2) @ translate((WINDOW_SIZE/2, WINDOW_SIZE/2))
    return scale((WINDOW_SIZE / WORLD_SIZE, -WINDOW_SIZE / WORLD_SIZE)) @ translate((WINDOW_SIZE/2, WINDOW_SIZE/2))


def scale(scale:Iterable[float]):
    return np.diag([scale[0], scale[1], 1])

def translate(vect:ndarray):
    result = np.eye(3)
    result[2, 0] = vect[0]
    result[2, 1] = vect[1]
    return result

def rotate(angle:float):
    result = np.zeros((3,3))
    result[0, 0] = math.cos(angle)
    result[0, 1] = math.sin(angle)
    result[1, 1] = math.cos(angle)
    result[1, 0] = -math.sin(angle)
    result[2, 2] = 1
    return result

def toAffine(points:ndarray):
    shape = points.shape
    return np.hstack((points, np.ones((shape[0], 1))))

def fromAffine(points:ndarray):
    shape = points.shape
    result = points[:, :2].copy()
    w = points[:, 2].reshape((shape[0], 1))
    return result / w

def transform(matrix:ndarray, points:ndarray):
    return fromAffine(toAffine(points) @ matrix)

def toGrid(p:ndarray, size:float):
    return np.round(p.copy() / size)
