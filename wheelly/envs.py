import logging
from typing import Any

import numpy as np
from tensorforce import Environment

from wheelly.encoders import (ClipEncoder, DictEncoder, FeaturesEncoder,
                              GetEncoder, MergeEncoder, ScaleEncoder,
                              SupplyEncoder, TilesEncoder, createSpace)
from wheelly.robot import RobotAPI

_logger = logging.getLogger(__name__)

_REACTION_INTERVAL = 0.3
_COMMAND_INTERVAL = 0.9
_DEFAULT_INTERVAL = 0.01
_VELOCITY_THRESHOLD = 0.01

MIN_SENSOR_DIR = -90
MAX_SENSOR_DIR = 90
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

_BASE_STATES = {
    "sensor": {
        "type": 'float',
                "shape": (1,),
                "min_value": float(MIN_SENSOR_DIR),
                "max_value": float(MAX_SENSOR_DIR)},
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
        "min_value": float(MIN_SENSOR_DIR),
        "max_value": float(MAX_SENSOR_DIR)}
}

HALT_ACTION = {
    "halt": np.ones((1,)),
    "direction": np.zeros((1,)),
    "speed": np.zeros((1,)),
    "sensorAction": np.zeros((1,))
}
"""The halt base action"""

class MockRobotEnv(Environment):
    """Mock robot environment used for code testing """
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
        return _BASE_STATES

    def actions(self):
        return _BASE_ACTION

    def reset(self):
        return self._get_obs()

    def execute(self, actions):
        self._action = actions
        observation = self._get_obs()
        return observation, False, self._reward

    def set_distance(self, value: float):
        """Set the distance for states"""
        self._distance = np.array(value)

    def set_sensor(self, value: int):
        """Set the sensor direction for states"""
        self._sensor = np.array(value)

    def set_can_move_forward(self, value: int):
        """Set true if robot can move forward"""
        self._can_move_forward = np.array(value)

    def act(self):
        """Returns the action of last execute method invocation"""
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
    """Base robot environment"""
    def __init__(self, robot: RobotAPI,
        interval = _DEFAULT_INTERVAL,
        reactionInterval = _REACTION_INTERVAL,
        commandInterval = _COMMAND_INTERVAL):
        """Creates a Robot envinment
        
        Argument:
        robot -- the robot api interface
        interval -- the interval between robot api ticks
        reactionInterval -- the (action/state) reaction interval (sec)
        commandInterval -- the interval between commands (sec)
        """
        super().__init__()
        self._robot = robot
        self._reaction_interval = reactionInterval
        self._command_interval = commandInterval
        self._interval = interval
        
        self._started = False
        
        self._robot_location = np.zeros((2,))
        self._robot_dir = np.zeros((1,))
        self._sensor = np.zeros((1,))
        self._distance = np.array([MAX_DISTANCE])
        self._can_move_forward = np.array([1])
        self._contacts = np.array([0])

        self._prev_halt = True
        self._prev_dir = 0
        self._prev_speed = 0.0
        self._prev_sensor = 0
        self._last_move_timestamp = None
        self._last_scan_timestamp = None

    def states(self):
        return _BASE_STATES

    def robot_pos(self):
        return np.array(self._robot_location)

    def robot_dir(self):
        return self._robot_dir

    def sensor_dir(self):
        return self._sensor

    def actions(self):
        return _BASE_ACTION

    def reset(self):
        if not self._started:
            self._robot.start()
            self._started = True
        self._readStatus(0)
        return self._get_obs()

    def execute(self, actions):
        self._process_action(actions)
        status = self._readStatus(self._reaction_interval)
        
        reward = self._reward(status)

        observation = self._get_obs()
        return observation, False, reward

    def close(self):
        if self._robot != None:
            self._robot.close()

    def _reward(self, status: dict[str, Any]) -> float:
        """Reward function"""
        if self._can_move_forward[0] == 0 or status["canMoveBackward"] == 0:
            return -1
        elif abs(status["left"]) < _VELOCITY_THRESHOLD and abs(status["right"]) < _VELOCITY_THRESHOLD and self._sensor == 0:
            return 1
        else:
            return 0
        """
        return -1 if self._can_move_forward[0] == 0 \
            or status["canMoveBackward"] == 0 else \
            1 if status["left"] == 0 and status["right"] == 0 and self._sensor == 0 else \
            0
"""
    def _readStatus(self, time: float):
        """Reads the status of robot afetr a time interval
        
        Arguments:
        time -- the time interval (sec)"""
        timeout = self._robot.time() + time
        while True:
            self._robot.tick(self._interval)
            status = self._robot.status()
            if status != None and self._robot.time() >= timeout:
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

    def _store_status(self, status: dict[str, Any]):
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

    def _process_action(self, action: dict[str, np.ndarray]):
        """Process the action
        
        Argument:
        action -- the action from agent
        """
        now = self._robot.time()
        dir = int(self._robot_dir + action['direction'])
        speed = round(action['speed'][0], 1)
        is_halt = action['halt'] == 1
        if is_halt != self._prev_halt:
            self._prev_halt = is_halt
            if is_halt:
                self._robot.halt()
                self._last_move_timestamp = now
            else:
                self._robot.move(dir, speed)
                self._prev_dir = dir
                self._prev_speed = speed
                self._last_move_timestamp = now
        elif not is_halt and now >= self._last_move_timestamp + self._command_interval:
            self._prev_halt = is_halt
            if is_halt:
                self._robot.halt()
                self._last_move_timestamp = now
            else:
                self._robot.move(dir, speed)
                self._prev_dir = dir
                self._prev_speed = speed
                self._last_move_timestamp = now

        sensor = round(action['sensorAction'][0])
        if self._prev_sensor != sensor:
            self._robot.scan(sensor)
            self._prev_sensor = sensor
            self._last_scan_timestamp = now
        elif sensor != 0 and now >= self._last_scan_timestamp + self._command_interval:
            self._robot.scan(sensor)
            self._prev_sensor = sensor
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
    """Encoded environment wrapper to base robot environment
    The environment encodes the state using tiles features for the sensor signals
    and features for contacts and canMoveForward signals
    The actions are discretized
    """
    def __init__(self, env: RobotEnv):
        """Creates the wrapper
        
        Arguments:
        env -- the base robot environment
        """
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
                              np.array([MIN_SENSOR_DIR]),
                              np.array([MAX_SENSOR_DIR]))
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
        return self._convert_observation(self._env.reset())

    def execute(self, actions):
        obs, done, reward = self._env.execute(self._convert_action(actions))
        return self._convert_observation(obs), done, reward

    def close(self):
        self._env.close()

    def _convert_observation(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Returns the encoded states observation
        
        Argument:
        obs -- the base observation
        """
        self._obs = obs
        return self._encoder.encode()

    def _convert_action(self, act: dict[str, np.ndarray]):
        """Returns the decoded action
        
        Argument:
        act -- the base action
        """
        self._act = act
        return self._out_act_encoder.encode()

_logger.debug(f"Module {__name__} loaded.")
