import gym
from gym import spaces
import pygame
import numpy as np
import math

WINDOW_SIZE = 512
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

class WheellyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, size=5):
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

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
        self._robotLocation = np.array([1,1])
        self._robotDir = -math.pi / 4
        return (observation, info) if return_info else observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
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

        canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        canvas.fill(BACKGROUND_COLOR)
        self.drawRobot(canvas)

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

    def drawRobot(self, canvas):
        trans = rotate(self._robotDir) @ translate(self._robotLocation) @ self.transMatrix()
        shape = transform(trans, ROBOT_SHAPE)
        pygame.draw.polygon(
            canvas,
            ROBOT_COLOR,
            shape
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