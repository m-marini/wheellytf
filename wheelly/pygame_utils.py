import math
from typing import Iterable
import pygame
import numpy as np
from numpy import ndarray
from pygame import Surface

WINDOW_SIZE = 800
GRID_SIZE = 0.2
ROBOT_LENGTH = 0.3
ROBOT_WIDTH = 0.2
WORLD_SIZE = 10.0
BACKGROUND_COLOR = (0, 0, 0)
ROBOT_COLOR = (0, 0, 255)
ROBOT_SHAPE = np.array([
    (ROBOT_LENGTH / 2, 0),
    (-ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
    (-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
    ])
OBSTACLE_SIZE = 0.2
OBSTACLE_COLOR = (255, 0, 0)
OBSTACLE_SHAPE = np.array([
    (OBSTACLE_SIZE / 2, OBSTACLE_SIZE / 2),
    (-OBSTACLE_SIZE / 2, OBSTACLE_SIZE / 2),
    (-OBSTACLE_SIZE / 2, -OBSTACLE_SIZE / 2),
    (OBSTACLE_SIZE / 2, -OBSTACLE_SIZE / 2),
])

SENSOR_COLOR = (255, 255, 0)
SENSOR_LENGTH = 0.8
SENSOR_SHAPE = np.array([
    (0.0, 0.0),
    (SENSOR_LENGTH, 0.0)
])

def render(window:Surface, robot_location:ndarray, robot_dir:int, sensor_dir:int):
    """Render the environment
    """
    if window is None:
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

    canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    canvas.fill(BACKGROUND_COLOR)
    trans = transMatrix()
    drawRobot(canvas, trans, robot_location, robot_dir)
    drawSensor(canvas, trans, robot_location, robot_dir, sensor_dir)

    # The following line copies our drawings from `canvas` to the visible window
    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.update()
    return window
    
def transMatrix():
    """Return the transform matrix to render the shapes"""
    return scale((WINDOW_SIZE / WORLD_SIZE, WINDOW_SIZE / WORLD_SIZE)) @ rotate(-math.pi/2) @ translate((WINDOW_SIZE/2, WINDOW_SIZE/2))

def drawRobot(canvas:Surface, trans:ndarray, robot_location:ndarray, robot_dir:int):
    """Draw the robot shape
        
    Arguments:
    canvas -- the canvas
    trans -- the transformation matrix
    """
    trans = rotate(math.radians(robot_dir)) @ translate(robot_location) @ trans
    shape = transform(trans, ROBOT_SHAPE)
    pygame.draw.polygon(
        canvas,
        ROBOT_COLOR,
        shape
    )

def drawSensor(canvas:Surface, trans:ndarray, robot_location:ndarray, robot_dir:int, sens_dir: int):
    """Draw the sensor shape
        
    Arguments:
    canvas -- the canvas
    trans -- the transformation matrix
    """
    angle = math.radians(robot_dir + sens_dir)
    shape = SENSOR_SHAPE.copy()
    shape = transform(rotate(angle) @ translate(robot_location) @ trans,
            shape)
    pygame.draw.line(
        canvas,
        SENSOR_COLOR,
        shape[0, :],
        shape[1, :]
    )

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
