from __future__ import annotations

from ast import Call
from types import NoneType
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorforce.environments import Environment
from wheelly.tdagents import TDAgent


class ContinuousMockEnv(Environment):
    def __init__(self, num_states: int = 2):
        super().__init__()
        self._current_state = 0
        self._num_states = num_states

    def states(self):
        return {
            "type": "int",
            "num_values": self._num_states
        }

    def actions(self):
        return {
            "type": "int",
            "num_values": self._num_states
        }

    def reset(self, num_parallel=None):
        self._current_state = 0
        return self._current_state

    def execute(self, actions):
        if actions == self._current_state:
            self._current_state = 0 if self._current_state >= self._num_states - \
                1 else self._current_state + 1
            return self._current_state, False, 1
        else:
            return self._current_state, False, 0


class SequenceMockEnv(Environment):
    def __init__(self, num_states: int = 2):
        super().__init__()
        self._current_state = 0
        self._num_states = num_states

    def states(self):
        return {
            "type": "int",
            "num_values": self._num_states
        }

    def actions(self):
        return {
            "type": "int",
            "num_values": 2
        }

    def reset(self, num_parallel=None):
        self._current_state = 0
        return self._current_state

    def execute(self, actions):
        next = self._current_state + 1 if actions == 1 else self._current_state
        terminal = next >= self._num_states - 1
        reward = 1 if terminal else 0
        self._current_state = next
        return self._current_state, terminal, reward
