from time import time
from typing import Any
from typing_extensions import Self
from tensorforce import Environment
import numpy as np
from . import robot
from wheelly.encoders import ClipEncoder, DictEncoder, FeaturesEncoder, GetEncoder, SupplyEncoder, MergeEncoder, ScaleEncoder, TilesEncoder, createSpace

import logging

from wheelly.pygame_utils import render
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

NUM_SENSOR_TILES = 7
NUM_DISTANCE_TILES = 30
MAX_DISTANCE = 10.0
CLIP_DISTANCE = 3.0
NUM_DIRECTION_ACTIONS = 25
NUM_SPEED_ACTIONS = 9
NUM_SENSOR_ACTIONS = 7

_BASIC_STATES = {
    "sensor": {
        "type": 'float',
                "shape": (1,),
                "min_value": float(MIN_SENSOR),
                "max_value": float(MAX_SENSOR)},
    "distance": {
        "type": 'float',
                "shape": (1,),
                "min_value": float(MIN_DISTANCE),
                "max_value": float(MAX_DISTANCE)},
    "canMoveForward": {
        "type": 'int',
                "shape": (1,),
                "num_values": 2},
    "contacts": {
        "type": 'int',
                "shape": (1,),
                "num_values": NUM_CONTACT_VALUES}
}

_BASE_ACTION = {
    "halt": {
        "type": 'int',
        "shape": (1,),
        "num_values": 2},
    "direction": {
        "type": 'float',
        "shape": (1,),
        "min_value": float(MIN_DIRECTION_ACTION),
        "max_value": float(MAX_DIRECTION_ACTION)},
    "speed": {
        "type": 'float',
        "shape": (1,),
        "min_value": float(MIN_SPEED),
        "max_value": float(MAX_SPEED)},
    "sensorAction": {
        "type": 'float',
        "shape": (1,),
        "min_value": float(MIN_SENSOR),
        "max_value": float(MAX_SENSOR)}
}


class MockRobotEnv(Environment):
    def __init__(self):
        """Create a Robot envinment
        
        Argument:
        params -- (dict) the dictionary with parameters
        """
        super().__init__()

        self._sensor = np.zeros((1,))
        self._distance = np.array([MAX_DISTANCE])
        self._can_move_forward = np.array([1])
        self._contacts = np.array([0])
        self._reward = np.zeros((1,))

    def states(self):
        return _BASIC_STATES

    def actions(self):
        return _BASE_ACTION

    def reset(self):
        """Reset the environment and return the tuple with observation and info """
        return self._get_obs()

    def execute(self, actions):
        """Run a step for the environment and return the tuple with observation, reward, done flag, info

        Argument:
        action -- (Action) the action to perfom
        """
        self._action = actions
        observation = self._get_obs()
        return observation, False, self._reward

    def set_distance(self, value: float):
        self._distance = np.array(value)

    def set_sensor(self, value: int):
        self._sensor = np.array(value)

    def set_can_move_forward(self, value: int):
        self._can_move_forward = np.array(value)

    def act(self):
        return self._action

    def _get_obs(self):
        """Return the observation"""
        return {
            "sensor": self._sensor,
            "distance": self._distance,
            "canMoveForward": self._can_move_forward,
            "contacts": self._contacts
        }


class RobotEnv(Environment):
    def __init__(self, **kvargs):
        """Create a Robot envinment
        
        Argument:
        params -- (dict) the dictionary with parameters
        """
        super().__init__()
        self._robot = robot.Robot(**kvargs)
        self._reaction_interval = kvargs["reactionInterval"] if "rectionInterval" in kvargs else REACTION_INTERVAL
        self._command_interval = kvargs["commandInterval"] if "commandInterval" in kvargs else COMMAND_INTERVAL - \
            self._reaction_interval
        self._connected = False
        self._robot_location = np.zeros((2,))
        self._robot_dir = np.zeros((1,))
        self._sensor = np.zeros((1,))
        self._distance = np.array([MAX_DISTANCE])
        self._can_move_forward = np.array([1])
        self._contacts = np.array([0])
        self._last_move_cmd = None
        self._last_scan_cmd = None
        self._last_move_timestamp = None
        self._last_scan_timestamp = None
        self._status_timeout = 0
        self.window = None

    def states(self):
        return _BASIC_STATES

    def actions(self):
        return _BASE_ACTION

    def reset(self):
        """Reset the environment and return the tuple with observation and info """
        if not self._connected:
            self._robot.connect()
            self._connected = True
        status = self._readStatus()
        self._status_timeout = status["timestamp"] + self._reaction_interval
        return self._get_obs()

    def execute(self, actions):
        """Run a step for the environment and return the tuple with observation, reward, done flag, info

        Argument:
        action -- (Action) the action to perfom
        """
        self._process_action(actions)
        status = self._readStatus()
        self._status_timeout += self._reaction_interval

        reward = -1 if self._can_move_forward[0] == 0 or status["canMoveBackward"] == 0 else \
            1 if status["left"] == 0 and status["right"] == 0 else \
            0
        observation = self._get_obs()
        return observation, False, reward

    def close(self):
        """Close the environment"""
        if self._robot != None:
            self._robot.close()

    def render(self, mode="human"):
        """Render the environment
        
        Argument
        mode -- (string) render mode (default "human")
        """
        self.window = render(self.window, self._robot_location,
                             self._robot_dir, self._sensor)

    def _readStatus(self):
        while True:
            status = self._robot.read_status()
            if status != None and status["timestamp"] >= self._status_timeout:
                break
        self._store_status(status)
        return status

    def _get_obs(self):
        """Return the observation"""
        return {
            "sensor": self._sensor,
            "distance": self._distance,
            "canMoveForward": self._can_move_forward,
            "contacts": self._contacts
        }

    def _store_status(self, status):
        """Store the status of robot
        
        Argument:
        status -- the status from robot
        """
        self._robot_location = np.array(
            [status["x"], status["y"]], dtype=np.float32)
        self._robot_dir = np.array([status["dir"]], dtype=np.int16)
        self._sensor = np.array([status["sensor"]], dtype=np.int16)
        self._distance = np.array([status["dist"]], dtype=np.float32)
        self._can_move_forward = np.array(
            [status["canMoveForward"]], dtype=np.uint8)
        self._can_move_backward = np.array(
            [status["canMoveBackward"]], dtype=np.uint8)
        self._contacts = np.array([status["contacts"]], dtype=np.uint8)

    def _process_action(self, action):
        """Process the action"""
        now = time()
        dir = self._robot_dir + action['direction']
        moveCmd = "al" if action["halt"] == 1 else f"mv {dir[0]:.0f} {action['speed'][0]:.1f}"
        if self._last_move_cmd != moveCmd:
            self._last_move_cmd = self._robot.write_cmd(moveCmd)
            self._last_move_timestamp = now
        elif moveCmd != "al" and now >= self._last_move_timestamp + self._command_interval:
            self._last_move_cmd = self._robot.write_cmd(moveCmd)
            self._last_move_timestamp = now

        scanCmd = f"sc {action['sensorAction'][0]:.0f}"
        if self._last_scan_cmd != scanCmd:
            self._last_scan_cmd = self._robot.write_cmd(scanCmd)
            self._last_scan_timestamp = now
        elif scanCmd != "sc 0" and now >= self._last_scan_timestamp + self._command_interval:
            self._last_scan_cmd = self._robot.write_cmd(scanCmd)
            self._last_scan_timestamp = now

_ENCODED_ACTION = {
    "halt": {
        "type": 'int',
        "shape": (1,),
        "num_values": 2},
    "direction": {
        "type": 'int',
        "shape": (1,),
        "num_values": NUM_DIRECTION_ACTIONS},
    "speed": {
        "type": 'int',
        "shape": (1,),
        "num_values": NUM_SPEED_ACTIONS},
    "sensorAction": {
        "type": 'int',
        "shape": (1,),
        "num_values": NUM_SENSOR_ACTIONS},
}

class EncodedRobotEnv(Environment):
    def __init__(self, env: Environment):
        super().__init__()
        states_space = createSpace(env.states())
        obs_encoder = SupplyEncoder(states_space, lambda: self._obs)
        distance_filter = GetEncoder(obs_encoder, "distance")
        distance_clipper = ClipEncoder(
            distance_filter, low=np.array([MIN_DISTANCE]), high=np.array([CLIP_DISTANCE]))
        sensor_filter = GetEncoder(obs_encoder, "sensor")
        sensor_flat = MergeEncoder.create(sensor_filter, distance_clipper)
        tiles_sensor = TilesEncoder(sensor_flat, sizes=np.array(
            [NUM_SENSOR_TILES, NUM_DISTANCE_TILES]))
        contacts_filter = GetEncoder(obs_encoder, "contacts")
        can_move_forward = GetEncoder(obs_encoder, "canMoveForward")
        contacts_flat = MergeEncoder.create(contacts_filter, can_move_forward)
        contacts_features = FeaturesEncoder.create(contacts_flat)
        encoder = MergeEncoder.create(tiles_sensor, contacts_features)
        in_act_encoder = SupplyEncoder(createSpace(_ENCODED_ACTION), lambda: self._act)
        halt = GetEncoder(in_act_encoder, "halt")
        direction = ScaleEncoder(GetEncoder(in_act_encoder, "direction"),
                                 np.array([MIN_DIRECTION_ACTION]),
                                np.array([MAX_DIRECTION_ACTION]))
        speed = ScaleEncoder(GetEncoder(in_act_encoder, "speed"),
                             np.array([MIN_SPEED]),
                             np.array([MAX_SPEED]))
        sensor = ScaleEncoder(GetEncoder(in_act_encoder, "sensorAction"),
                              np.array([MIN_SENSOR]),
                              np.array([MAX_SENSOR]))
        out_act_encoder = DictEncoder(
            halt=halt,
            direction=direction,
            speed=speed,
            sensorAction=sensor)

        self._env = env
        self._encoder = encoder
        self._out_act_encoder = out_act_encoder

    def states(self):
        return self._encoder.spec()

    def actions(self):
        return _ENCODED_ACTION

    def reset(self):
        """Reset the environment and return the tuple with observation and info """
        return self._convert_observation(self._env.reset())

    def execute(self, actions):
        """Run a step for the environment and return the tuple with observation, reward, done flag, info

        Argument:
        action -- (Action) the action to perfom
        """
        obs, done, reward = self._env.execute(self._convert_action(actions))
        return self._convert_observation(obs), done, reward

    def close(self):
        """Close the environment"""
        self._env.close()

    def _convert_observation(self, obs: Any):
        self._obs = obs
        return self._encoder.encode()

    def _convert_action(self, act: Any):
        self._act = act
        return self._out_act_encoder.encode()
