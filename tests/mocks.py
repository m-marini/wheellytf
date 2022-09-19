import gym
from gym import spaces
from wheelly.envs import MAX_SENSOR, MAX_DISTANCE, MAX_SPEED, MAX_DIRECTION_ACTION, MIN_DIRECTION_ACTION, MIN_DISTANCE, MIN_SENSOR, MIN_SPEED, NUM_CONTACT_VALUES
import numpy as np

import logging
logger = logging.getLogger(__name__)

class MockRobotEnv(gym.Env):

    def __init__(self):
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
        self._last_move_cmd = None
        self._last_scan_cmd = None
        self._last_move_timestamp = None
        self.window = None
        self._robot_dir = 0
        self._sens_dir = 0
        self._distance = 10.0
        self._can_move_forward = 1
        self._can_move_backward = 1
        self._contacts = 0
        self._reward = 0

    def set_reward(self, value:float):
        self._reward = value

    def set_sensor(self, value:int):
        self._sens_dir = value

    def set_distance(self, value:float):
        self._distance = value
    
    def set_can_move_forward(self, value:int):
        self._can_move_forward = value

    def set_contacts(self, value:int):
        self._contacts = value

    def act(self):
        return self._action

    def observation(self):
        return {
            "direction": np.array(self._robot_dir),
            "sensor": np.array(self._sens_dir),
            "distance": np.array(self._distance),
            "canMoveForward": np.array(self._can_move_forward),
            "contacts": np.array(self._contacts),
        }

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)

        observation = self.observation()
        return (observation, {}) if return_info else observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # An episode is done if the agent has reached the target
        # Binary sparse rewards
        self._action = action
        return self.observation(), self._reward, False, {}

    def render(self, mode=""):
        return
