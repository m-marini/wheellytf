from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from wheelly.tdlayers import TDState, flat_type_spec, td_forward, td_train


class TDAgent(tf.train.Checkpoint):

    @ staticmethod
    def load(path: str) -> TDAgent:
        with open(path + "/agent.json") as json_file:
            spec = json.load(json_file)
        agent = TDAgent.by_spec(spec, random=tf.random.get_global_generator())
        file = tf.train.latest_checkpoint(checkpoint_dir=path)
        agent.restore(file)
        return agent

    @ staticmethod
    def by_spec(spec: dict[str, Any],
                random: tf.random.Generator) -> TDAgent:
        states_spec = spec["state_spec"]
        actions_spec = spec["actions_spec"]
        critic = TDState.by_spec(spec=spec["critic"], random=random)
        policy = TDState.by_spec(spec=spec["policy"], random=random)
        return TDAgent(states_spec=states_spec,
                       actions_spec=actions_spec,
                       critic=critic,
                       policy=policy,
                       reward_alpha=1.0,
                       random=random)

    @ staticmethod
    def create(state_spec: dict[str, Any],
               actions_spec: dict[str, Any],
               agent_spec: dict[str, Any],
               random: tf.random.Generator):
        """Returns the agent by agent specification

        Arguments:
        state_spec -- the state specification
        actions_spec -- the on specification
        agent_spec -- the agent specification
        random -- random generator"""
        flatten_state = flat_type_spec(spec=state_spec, prefix="input")
        flatten_actions = flat_type_spec(spec=actions_spec, prefix="output")
        reward_alpha = agent_spec.get("reward_alfa", 0.1)
        assert "critic" in agent_spec, f'Missing "critic" specification'
        critic = TDState.create(net_spec=agent_spec["critic"],
                                state_spec=flatten_state,
                                actions_spec=dict(output=dict(
                                    type="float",
                                    shape=(1,)
                                )),
                                random=random)
        assert "policy" in agent_spec, f'Missing "policy" specification'
        policy = TDState.create(net_spec=agent_spec["policy"],
                                state_spec=flatten_state,
                                actions_spec=flatten_actions,
                                random=random)
        agent = TDAgent(states_spec=state_spec,
                        actions_spec=actions_spec,
                        policy=policy,
                        critic=critic,
                        reward_alpha=reward_alpha,
                        random=random)
        return agent

    def __init__(self,
                 states_spec: dict[str, Any],
                 actions_spec: dict[str, Any],
                 policy: TDState,
                 critic: TDState,
                 reward_alpha: float,
                 random: tf.random.Generator):
        super().__init__()
        self.states_spec = states_spec
        self.actions_spec = actions_spec
        self._flatten_states_spec = flat_type_spec(
            spec=states_spec, prefix="input")
        self._flatten_actions_spec = flat_type_spec(
            spec=actions_spec, prefix="output")
        self.policy = policy
        self.random = random
        self.critic = critic
        self.reward_alpha = tf.Variable(
            initial_value=reward_alpha, trainable=False)
        self.avg_reward = tf.Variable(initial_value=[0],
                                      dtype=tf.float32,
                                      trainable=False)
        self._prev_inputs = None
        self._kpi_listeners: list[Callable[[dict[str, Any]], None]] = []

    def act(self, states: Any) -> dict[str, Any] | int | float:
        """Returns the action for a states based on current policy

        Argument:
        states -- the states"""
        policy_status, inputs = self._policy_status(states=states)
        pis = {key: value for key,
               value in policy_status.items() if key in self._flatten_actions_spec}
        flatten_action = choose_actions(flatten_pis=pis, random=self.random)
        actions = unflat_actions(
            flatten=flatten_action, actions_spec=self.actions_spec,)
        self._last_actions = actions
        self._last_inputs = inputs
        self._last_policy = policy_status
        return actions

    def observe(self, reward: float = 0.0, terminal: bool = False):
        """Observe the result of environment interaction and train the agent

        Arguments:
        reward -- reward
        terminal -- true if terminal state"""
        if self._prev_inputs is not None:
            self._train(s0=self._prev_inputs,
                        actions=self._prev_actions,
                        reward=self._prev_reward,
                        terminal=self._prev_terminal,
                        s1=self._last_inputs)
        if terminal:
            self._prev_inputs = None
            self._prev_actions = None
            self._prev_reward = None
            self._prev_terminal = None
            self._prev_policy = None
        else:
            self._prev_inputs = self._last_inputs
            self._prev_actions = self._last_actions
            self._prev_reward = reward
            self._prev_terminal = terminal
            self._prev_policy = self._last_policy

    def close(self):
        pass

    def _policy_status(self, states: Any):
        """Returns the status of policy network and the states inputs

        Argument:
        states -- the states"""
        inputs = flat_states(states)
        policy_status = td_forward(inputs=inputs,
                                   state=self.policy)
        return policy_status, inputs

    def _train(self, s0: dict[str, Tensor],
               actions: dict[str, Any] | int | float,
               reward: float,
               terminal: bool,
               s1: dict[str, Tensor]):

        c0 = td_forward(state=self.critic, inputs=s0)
        c1 = td_forward(state=self.critic, inputs=s1)
        v0 = c0["output"]
        v1 = c1["output"]
        delta = reward - self.avg_reward + v1 - v0
        pi = self._prev_policy
        dc = dict(output=tf.ones(shape=(1, 1)))
        dp = log_pi(pi=pi, action=actions, actions_spec=self.actions_spec)

        avg_reward = tf.constant(self.avg_reward)
        self.avg_reward.assign_add(delta=delta[0] * self.reward_alpha)
        _, grad_c = td_train(state=self.critic,
                             nodes=c0,
                             grad_loss=dc,
                             delta=delta)
        _, grad_pi = td_train(state=self.policy,
                              nodes=pi,
                              grad_loss=dp,
                              delta=delta)

        if len(self._kpi_listeners) > 0:
            trained_c0 = td_forward(state=self.critic, inputs=s0)
            trained_c1 = td_forward(state=self.critic, inputs=s1)
            trained_pi = td_forward(state=self.policy, inputs=s0)

            kpis = dict(
                s0=s0,
                reward=reward,
                terminal=terminal,
                actions=actions,
                s1=s1,
                avg_reward=avg_reward,
                trained_avg_reward=tf.constant(self.avg_reward),
                c0=c0,
                c1=c1,
                delta=delta,
                pi=pi,
                dc=grad_c,
                dp=grad_pi,
                trained_c0=trained_c0,
                trained_c1=trained_c1,
                trained_pi=trained_pi
            )
            self._notify_kpi(kpis)

    def _notify_kpi(self, kpis: dict[str, Any]):
        for listener in self._kpi_listeners:
            listener(kpis)

    def add_kpi_listener(self, callback: Callable[[dict[str, Any]], None]):
        """Adds a kpi listener

        Arguments:
        callback -- the call back listener"""
        self._kpi_listeners.append(callback)

    def spec(self):
        result = dict(state_spec=self.states_spec,
                      actions_spec=self.actions_spec,
                      critic=self.critic.spec(),
                      policy=self.policy.spec()
                      )
        return result

    def save_model(self, path: str):
        tf.train.CheckpointManager(
            checkpoint=self, directory=path, max_to_keep=1).save()
        with open(path + "/agent.json", "w", encoding="utf-8") as f:
            agent_spec = self.spec()
            json.dump(agent_spec, f, ensure_ascii=False)


def log_pi(pi: dict[str, Tensor],
           action: dict[str, Any] | int | float,
           actions_spec: dict[str, Any]) -> dict[str, Tensor]:
    """Returns the gradient of log pi for a given pi and action

    Arguments:
    pi -- the probabilities of each action
    action -- the action applied
    actions_spec -- the action specification"""

    def log_pi1(values: dict[str, Tensor],
                prefix: str,
                action: dict[str, Any] | int | float,
                spec: dict[str, Any]) -> dict[str, Tensor]:

        if "type" in spec:
            assert spec["type"] == "int"
            pi_a = pi[prefix]
            x = np.zeros(shape=pi_a.shape)
            x[0, action] = 1
            dl = (1 / pi_a) * x
            values[prefix] = dl
            return values
        else:
            for key, sub_spec in spec.items():
                log_pi1(values=values,
                        prefix=f"{prefix}.{key}",
                        action=action[key],
                        spec=sub_spec)
            return values

    return log_pi1(values={}, prefix="output", action=action, spec=actions_spec)


def flat_states(states: Any) -> dict[str, Tensor]:
    """Returns the flatten states

    Arguments:
    states_spec -- the state specification
    states -- the state"""
    result = dict()

    def flat(prefix: str, value: Any):
        if isinstance(value, dict):
            for key, sub_values in value.items():
                flat(prefix=prefix+"." + key,
                     value=sub_values)
        else:
            result[prefix] = tf.constant([value], dtype=tf.float32)

    flat(prefix="input", value=states)
    return result


def choose_actions(flatten_pis: dict[str, Tensor],
                   random: tf.random.Generator) -> dict[str, int]:
    """Returns the actions selecting random by probabilities

    Arguments:
    flatten_pis -- the flatten probabilities
    random -- the random generator"""
    def choose(prob: Tensor) -> int:
        cum = tf.cumsum(x=prob, axis=1)
        sel = random.uniform(shape=(1,))
        action = int(tf.math.count_nonzero(sel >= cum))
        return action

    result = {key: choose(pis) for key, pis in flatten_pis.items()}
    return result


def unflat_actions(actions_spec: dict[str, Any],
                   flatten: dict[str, int]) -> dict[str, Any] | int | float:
    """Returns the unflatten actions

    Arguments:
    actions_spec -- the actions specification
    flatten -- the flatten actions"""
    def unflat(prefix: str,
               spec: dict[str, Any]) -> dict[str, Any] | int | float:

        if "type" in spec:
            assert spec["type"] == "int"
            return flatten[prefix]
        else:
            result = {key: unflat(prefix=f"{prefix}.{key}",
                                  spec=sub_spec)
                      for key, sub_spec in spec.items()}
            return result

    return unflat(prefix="output", spec=actions_spec)
