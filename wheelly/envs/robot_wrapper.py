from re import A
from typing import Any
import gym
import numpy as np
from gym import spaces
from wheelly.encoders import DictEncoder, FeaturesEncoder, MergeBinaryEncoder, MergeBoxEncoder, MergeDiscreteEncoder, IdentityEncoder, ClipEncoder, GetEncoder, ScaleEncoder, TilesEncoder

import logging

from wheelly.envs.robot_env import MAX_DIRECTION_ACTION, MAX_SENSOR, MAX_SPEED, MIN_DIRECTION_ACTION, MIN_SENSOR, MIN_SPEED

logger = logging.getLogger(__name__)

NUM_SENSOR_TILES = 7
NUM_DISTANCE_TILES = 30
MAX_DISTANCE = 3.0
NUM_DIRECTION_ACTIONS = 25
NUM_SPEED_ACTIONS = 9
NUM_SENSOR_ACTIONS = 7

class RobotWrapper(gym.ObservationWrapper, gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_encoder = IdentityEncoder(env.observation_space, lambda: self._obs)
        distance_filter = GetEncoder(obs_encoder, "distance")
        distance_clipper = ClipEncoder(distance_filter, low = np.array([0]), high = np.array([MAX_DISTANCE]))
        sensor_filter = GetEncoder(obs_encoder, "sensor")
        sensor_flat = MergeBoxEncoder(sensor_filter, distance_clipper)
        tiles_sensor = TilesEncoder(sensor_flat, sizes=np.array([NUM_SENSOR_TILES, NUM_DISTANCE_TILES]))
        contacts_filter = GetEncoder(obs_encoder, "contacts")
        can_move_forward = GetEncoder(obs_encoder, "canMoveForward")
        contacts_flat = MergeDiscreteEncoder(contacts_filter, can_move_forward)
        contacts_features = FeaturesEncoder(contacts_flat)
        encoder = MergeBinaryEncoder(tiles_sensor, contacts_features)
        action_space = spaces.Dict({
            "halt":  spaces.Discrete(2),
            "direction": spaces.Discrete(NUM_DIRECTION_ACTIONS),
            "speed": spaces.Discrete(NUM_SPEED_ACTIONS),
            "sensor": spaces.Discrete(NUM_SENSOR_ACTIONS),
        })
        in_act_encoder = IdentityEncoder(action_space, lambda: self._act)
        halt = GetEncoder(in_act_encoder, "halt")
        direction = ScaleEncoder(GetEncoder(in_act_encoder, "direction"),
            np.array([MIN_DIRECTION_ACTION]),
            np.array([MAX_DIRECTION_ACTION]),
        )
        speed = ScaleEncoder(GetEncoder(in_act_encoder, "speed"),
            np.array([MIN_SPEED]),
            np.array([MAX_SPEED]),
        )
        sensor = ScaleEncoder(GetEncoder(in_act_encoder, "sensor"),
            np.array([MIN_SENSOR]),
            np.array([MAX_SENSOR]),
        )
        out_act_encoder = DictEncoder(
            halt = halt,
            direction = direction,
            speed = speed,
            sensor = sensor,
        )
        self._encoder = encoder
        self._observation_space = encoder.space()
        self._action_space = action_space
        self._out_act_encoder = out_act_encoder
 
    def observation(self, obs:Any):
        self._obs = obs
        return self._encoder.encode()

    def action(self, act):
        self._act = act
        return self._out_act_encoder.encode()