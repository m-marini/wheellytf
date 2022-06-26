import gym
from gym import spaces
import pygame
import numpy as np
import Box2D
import math

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

class WheellyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60, "tps": 3.3}

    def __init__(self, size=5):
        print(f"size={size}")
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 3D actions, corresponding to direction, speed, sensor
        self.action_space = spaces.Box(low=np.array([-180, -1.0, -90]), high=np.array([180, 1.0, 90]), dtype=np.float32)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.world = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2)

        observation = self._get_obs()
        info = self._get_info()
        self._robotLocation = np.zeros(2)
        self._robotDir = 0
        self._sensDir = 0
        self._obstacles = createObstacles()
        return (observation, info) if return_info else observation

    def step(self, action):

        self._sensDir = np.int32(action[2])
        self._robotDir = np.int32(action[0])

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # An episode is done if the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if done else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()
        if (self.world is None and mode == "human"):
            self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)

        canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        canvas.fill(BACKGROUND_COLOR)
        trans = self.transMatrix()
        self.drawRobot(canvas, trans)
        self.drawObstacles(canvas, trans)
        self.drawSensor(canvas, trans)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def transMatrix(self):
        return scale((WINDOW_SIZE / WORLD_SIZE, WINDOW_SIZE / WORLD_SIZE)) @ rotate(-math.pi/2) @ translate((WINDOW_SIZE/2, WINDOW_SIZE/2))

    def drawRobot(self, canvas, trans):
        trans = rotate(math.radians(self._robotDir)) @ translate(self._robotLocation) @ trans
        shape = transform(trans, ROBOT_SHAPE)
        pygame.draw.polygon(
            canvas,
            ROBOT_COLOR,
            shape
        )

    def drawObstacles(self, canvas, trans):
        n = self._obstacles.shape[0]
        m = OBSTACLE_SHAPE.shape[0]
        shapes = np.zeros((n * 4, 2))
        for i in range(n):
            shapes[i * m : i * m + 4, :] = self._obstacles[i, :]
            for j in range(m):
                shapes[i * m + j, :] += OBSTACLE_SHAPE[j, :]
        shapes = transform(trans, shapes)
        for i in range(n):
            pygame.draw.polygon(
                canvas,
                OBSTACLE_COLOR,
                shapes[i * m : i * m + m, :]
            )

    def drawSensor(self, canvas, trans):
        angle = math.radians(self._robotDir+self._sensDir)
        shape = SENSOR_SHAPE.copy()
        shape = transform(rotate(angle) @ translate(self._robotLocation) @ trans,
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

def line(points, gridSize):
    grid = toGrid(points, gridSize)
    step = grid[1, :] - grid[0, :]
    size = np.abs(step) + 1
    n = np.int32(np.max(size))
    result = np.zeros((n, 2))
    p = grid[0, :].copy()
    if (size[0] > size[1]):
        ds = step / np.abs(step[0])
        for i in range(n):
            result[i, :] = np.round(p.copy())
            p += ds
    else:            
        ds = step / np.abs(step[1])
        for i in range(n):
            result[i, :] = np.round(p.copy())
            p += ds

    result *= gridSize
    return result

def createObstacles():
    return np.vstack((
        line(np.array([
            [-WORLD_SIZE / 2, -WORLD_SIZE / 2],
            [-WORLD_SIZE / 2, WORLD_SIZE / 2]
        ]), GRID_SIZE),
        line(np.array([
            [-WORLD_SIZE / 2, WORLD_SIZE / 2],
            [WORLD_SIZE / 2, WORLD_SIZE / 2]
        ]), GRID_SIZE),
        line(np.array([
            [WORLD_SIZE / 2, WORLD_SIZE / 2],
            [WORLD_SIZE / 2, -WORLD_SIZE / 2]
        ]), GRID_SIZE),
        line(np.array([
            [WORLD_SIZE / 2, -WORLD_SIZE / 2],
            [-WORLD_SIZE / 2, -WORLD_SIZE / 2]
        ]), GRID_SIZE)
    ))
