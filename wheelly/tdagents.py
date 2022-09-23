from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from wheelly.tdlayers import TDNetwork, flat_type_spec


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


class TDAgent(tf.train.Checkpoint):
    @staticmethod
    def create(spec: dict[str, Any], random=tf.random.Generator) -> TDAgent:
        policy = TDNetwork.create(spec=spec["policy"], random=random)
        critic = TDNetwork.create(spec=spec["critic"], random=random)
        return TDAgent(states_spec=spec["state_spec"],
                       actions_spec=spec["actions_spec"],
                       reward_alpha=spec["reward_alpha"],
                       policy=policy,
                       critic=critic,
                       random=random)

    @ staticmethod
    def load(path: str) -> TDAgent:
        with open(path + "/agent.json") as json_file:
            spec = json.load(json_file)
        agent = TDAgent.create(
            spec=spec, random=tf.random.get_global_generator())
        file = tf.train.latest_checkpoint(checkpoint_dir=path)
        agent.restore(file)
        return agent

    def __init__(self,
                 states_spec: dict[str, Any],
                 actions_spec: dict[str, Any],
                 reward_alpha: float,
                 policy: TDNetwork,
                 critic: TDNetwork,
                 random: tf.random.Generator):
        super().__init__()

        assert isinstance(states_spec, dict)
        assert isinstance(actions_spec, dict)
        assert isinstance(reward_alpha, float)
        assert isinstance(policy, TDNetwork)
        assert isinstance(critic, TDNetwork)

        self.states_spec = states_spec
        self.actions_spec = actions_spec
        self.flatten_states_spec = flat_type_spec(
            spec=states_spec, prefix="input")
        self.flatten_actions_spec = flat_type_spec(
            spec=actions_spec, prefix="output")
        self.policy = policy
        self.random = random
        self.critic = critic
        self.reward_alpha = reward_alpha
        self.avg_reward = tf.Variable(initial_value=[0],
                                      dtype=tf.float32,
                                      trainable=False)
        self.prev_inputs = None
        self.kpi_listeners: list[Callable[[dict[str, Any]], None]] = []

    def act(self, states: Any) -> dict[str, Any] | int | float:
        """Returns the action for a states based on current policy

        Argument:
        states -- the states"""
        policy_status, inputs = self._policy_status(states=states)
        pis = {key: value for key,
               value in policy_status.items() if key in self.flatten_actions_spec}
        flatten_action = choose_actions(flatten_pis=pis, random=self.random)
        actions = unflat_actions(
            flatten=flatten_action, actions_spec=self.actions_spec,)
        self.last_actions = actions
        self.last_inputs = inputs
        self.last_policy = policy_status
        return actions

    def observe(self, reward: float = 0.0, terminal: bool = False):
        """Observe the result of environment interaction and train the agent

        Arguments:
        reward -- reward
        terminal -- true if terminal state"""
        if self.prev_inputs is not None:
            self._train(s0=self.prev_inputs,
                        actions=self.prev_actions,
                        reward=self.prev_reward,
                        terminal=self.prev_terminal,
                        s1=self.last_inputs)
        if terminal:
            self.prev_inputs = None
            self.prev_actions = None
            self.prev_reward = None
            self.prev_terminal = None
            self.prev_policy = None
        else:
            self.prev_inputs = self.last_inputs
            self.prev_actions = self.last_actions
            self.prev_reward = reward
            self.prev_terminal = terminal
            self.prev_policy = self.last_policy

    def close(self):
        pass

    def _policy_status(self, states: Any):
        """Returns the status of policy network and the states inputs

        Argument:
        states -- the states"""
        inputs = flat_states(states)
        policy_status = self.policy.forward(inputs)
        return policy_status, inputs

    def _train(self, s0: dict[str, Tensor],
               actions: dict[str, Any] | int | float,
               reward: float,
               terminal: bool,
               s1: dict[str, Tensor]):

        c0 = self.critic.forward(s0)
        c1 = self.critic.forward(s1)
        v0 = c0["output"]
        v1 = c1["output"]
        delta = reward - self.avg_reward + v1 - v0
        pi = self.prev_policy
        dc = dict(output=tf.ones(shape=(1, 1)))
        dp = log_pi(pi=pi, action=actions, actions_spec=self.actions_spec)

        avg_reward = tf.constant(self.avg_reward)
        self.avg_reward.assign_add(delta=delta[0] * self.reward_alpha)
        grad_c = self.critic.train(outputs=c0,
                                   grad_outputs=dc,
                                   delta=delta)
        grad_pi = self.policy.train(outputs=pi,
                                    grad_outputs=dp,
                                    delta=delta)

        if len(self.kpi_listeners) > 0:
            trained_c0 = self.critic.forward(s0)
            trained_c1 = self.critic.forward(s1)
            trained_pi = self.policy.forward(s0)

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
        for listener in self.kpi_listeners:
            listener(kpis)

    def add_kpi_listener(self, callback: Callable[[dict[str, Any]], None]):
        """Adds a kpi listener

        Arguments:
        callback -- the call back listener"""
        self.kpi_listeners.append(callback)

    def save_model(self, path: str):
        tf.train.CheckpointManager(
            checkpoint=self, directory=path, max_to_keep=3).save()
        with open(path + "/agent.json", "w", encoding="utf-8") as f:
            agent_spec = self.spec()
            json.dump(agent_spec, f, ensure_ascii=False)

    def spec(self) -> dict[str, Any]:
        spec = dict(
            state_spec=self.states_spec,
            actions_spec=self.actions_spec,
            reward_alpha=self.reward_alpha,
            policy=self.policy.spec(),
            critic=self.critic.spec()
        )
        return spec
