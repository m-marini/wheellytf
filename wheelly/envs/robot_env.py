from time import time
import gym
from gym import spaces
import numpy as np
import pygame
import math
from . import robot

import logging
logger = logging.getLogger(__name__)

REACTION_INTERVAL = 0.3
COMMAND_INTERVAL = 0.9

MIN_SENSOR = -90
MAX_SENSOR = 90
MIN_DISTANCE = 0.0
MAX_DISTANCE = 10.0
NUM_CONTACT_VALUES = 16

MIN_DIRECTION_ACTION = -180
MAX_DIRECTION_ACTION = 180
MIN_SPEED = -1.0
MAX_SPEED = 1.0

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

class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, params):
        """Create a Robot envinment
        
        Argument:
        params -- (dict) the dictionary with parameters
        """
        self._reaction_interval = params["reactionInterval"] if "rectionInterval" in params else REACTION_INTERVAL
        self._command_interval = params["commandInterval"] if "commandInterval" in params else COMMAND_INTERVAL - self._reaction_interval

        self.observation_space = spaces.Dict(
            {
                "sensor": spaces.Box(MIN_SENSOR, MAX_SENSOR, shape=(1,), dtype=np.int32),
                "distance": spaces.Box(MIN_DISTANCE, MAX_DISTANCE, shape=(1,), dtype=np.float32),
                "canMoveForward": spaces.Discrete(2),
                "contacts": spaces.Discrete(NUM_CONTACT_VALUES)
            }
        )
        self.action_space = spaces.Dict(
            {
                "halt":  spaces.Discrete(2),
                "direction": spaces.Box(-MIN_DIRECTION_ACTION, MAX_DIRECTION_ACTION, shape=(1,), dtype=np.int32),
                "speed": spaces.Box(MIN_SPEED, MAX_SPEED, shape=(1,), dtype=np.float32),
                "sensor": spaces.Box(MIN_SENSOR, MAX_SENSOR, shape=(1,), dtype=np.int32),
            }
        )
        self._robot = robot.Robot(params)
        self._robot.connect()
        self._last_move_cmd = None
        self._last_scan_cmd = None
        self._last_move_timestamp = None
        self._last_scan_timestamp = None
        self.window = None

    def _get_obs(self):
        """Return the observation"""
        return {
            "sensor": self._sens_dir,
            "distance": self._distance,
            "canMoveForward": self._can_move_forward,
            "contacts": self._contacts,
        }

    def _get_info(self):
        """Return the info"""
        return {}

    def reset(self, seed=None, return_info=False, options=None):
        """Reset the environment and return the tuple with observation and info """
        super().reset(seed=seed)

        while True:
            status = self._robot.read_status()
            if status != None:
                break
        self.store_status(status)
        self._status_timeout = status["timestamp"] + self._reaction_interval
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def store_status(self, status):
        """Store the status of robot
        
        Argument:
        status -- the status from robot
        """
        self._robot_location = np.array([status["x"], status["y"]], dtype=np.float32)
        self._robot_dir = np.array([status["dir"]], dtype=np.int32)
        self._sens_dir = np.array([status["sensor"]], dtype=np.int32)
        self._distance = np.array([status["dist"]], dtype=np.float32)
        self._can_move_forward = status["canMoveForward"]
        self._can_move_backward = status["canMoveBackward"]
        self._contacts = status["contacts"]

    def step(self, action):
        """Run a step for the environment and return the tuple with observation, reward, done flag, info

        Argument:
        action -- (Action) the action to perfom
        """
        self.process_action(action)

        while True:
            status = self._robot.read_status()
            if status != None and status["timestamp"] >= self._status_timeout:
                break
        self.store_status(status)
        self._status_timeout += REACTION_INTERVAL

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # An episode is done if the agent has reached the target
        done = False
         # Binary sparse rewards
        logger.debug(f"status={status}")
        reward = -1 if self._can_move_forward == 0 or self._can_move_backward == 0 else \
            1 if status["left"] == 0 and status["right"] == 0 else \
            0 
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def process_action(self, action):
        """Process the action"""
        now = time()
        dir = self._robot_dir + action['direction']
        moveCmd = "al" if action["halt"] == 1 else f"mv {dir[0]} {action['speed'][0]:.1f}"
        if self._last_move_cmd != moveCmd:
            self._last_move_cmd = self._robot.write_cmd(moveCmd)
            self._last_move_timestamp = now
        elif moveCmd != "al" and  now >= self._last_move_timestamp + self._command_interval:
            self._last_move_cmd = self._robot.write_cmd(moveCmd)
            self._last_move_timestamp = now
        
        scanCmd = f"sc {action['sensor'][0]}"
        if self._last_scan_cmd != scanCmd:
            self._last_scan_cmd = self._robot.write_cmd(scanCmd)
            self._last_scan_timestamp = now
        elif scanCmd != "sc 0" and now >= self._last_scan_timestamp + self._command_interval:
            self._last_scan_cmd = self._robot.write_cmd(scanCmd)
            self._last_scan_timestamp = now

    def close(self):
        """Close the environment"""
        if self._robot != None:
            self._robot.close()

    def render(self, mode="human"):
        """Render the environment
        
        Argument
        mode -- (string) render mode (default "human")
        """
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

        canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        canvas.fill(BACKGROUND_COLOR)
        trans = self.transMatrix()
        self.drawRobot(canvas, trans)
        self.drawSensor(canvas, trans)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def transMatrix(self):
        """Return the transform matrix to render the shapes"""
        return scale((WINDOW_SIZE / WORLD_SIZE, WINDOW_SIZE / WORLD_SIZE)) @ rotate(-math.pi/2) @ translate((WINDOW_SIZE/2, WINDOW_SIZE/2))

    def drawRobot(self, canvas, trans):
        """Draw the robot shape
        
        Arguments:
        canvas -- the canvas
        trans -- the transformation matrix
        """
        trans = rotate(math.radians(self._robot_dir)) @ translate(self._robot_location) @ trans
        shape = transform(trans, ROBOT_SHAPE)
        pygame.draw.polygon(
            canvas,
            ROBOT_COLOR,
            shape
        )

    def drawSensor(self, canvas, trans):
        """Draw the sensor shape
        
        Arguments:
        canvas -- the canvas
        trans -- the transformation matrix
        """
        angle = math.radians(self._robot_dir + self._sens_dir)
        shape = SENSOR_SHAPE.copy()
        shape = transform(rotate(angle) @ translate(self._robot_location) @ trans,
            shape)
        pygame.draw.line(
            canvas,
            SENSOR_COLOR,
            shape[0, :],
            shape[1, :]
         )

def scale(scale):
    return np.diag([scale[0], scale[1], 1]);

def translate(vect):
    result = np.eye(3)
    result[2, 0] = vect[0]
    result[2, 1] = vect[1]
    return result

def rotate(angle):
    result = np.zeros((3,3))
    result[0, 0] = math.cos(angle)
    result[0, 1] = math.sin(angle)
    result[1, 1] = math.cos(angle)
    result[1, 0] = -math.sin(angle)
    result[2, 2] = 1
    return result

def toAffine(points):
    shape = points.shape
    return np.hstack((points, np.ones((shape[0], 1))))

def fromAffine(points):
    shape = points.shape
    result = points[:, :2].copy()
    w = points[:, 2].reshape((shape[0], 1))
    return result / w

def transform(matrix, points):
    return fromAffine(toAffine(points) @ matrix)

def toGrid(p, size):
    return np.round(p.copy() / size)
