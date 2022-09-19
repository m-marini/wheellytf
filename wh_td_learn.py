"""Trains a dnn agent to get stuck to an obstacle in robot encoded environment (environment.json)
of simulated environment"""

import argparse
import logging
import tensorflow as tf
from random import Random

import pygame
from tensorforce import Environment

from wheelly.envs import EncodedRobotEnv, RobotEnv
from wheelly.tdlisteners import CsvConsumer, DiscountConsumer, KpiListenerBuilder
from wheelly.objectives import fuzzy_stuck
from wheelly.renders import RobotWindow
from wheelly.robots import SimRobot
from wheelly.sims import ObstacleMapBuilder
from wheelly.tdagents import TDAgent

_FPS = 60

font: pygame.font.Font | None = None


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # usage="%(prog)s [OPTION] [FILE]...",
        description="Learn in encoded robot environment."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1.0"
    )
    parser.add_argument(
        "-e", "--environment", default='environment.json',
        dest='environment',
        help='the json file with environment descriptor (default=environment.json)'
    )
    parser.add_argument(
        "-t", "--time", default=43200,
        dest='time', type=float,
        help='stop after time (default=43200 sec. = 12 hours)'
    )
    parser.add_argument(
        "-s", "--stats",
        dest='stats',
        help='activate and set the stats folder'
    )
    return parser


def agent_spec():
    alpha = 30e-3
    tdlambda = 0.5
    reward_alpha = 5e-3
    temperature = 0.4
    hidden_size = 100
    return{
        "reward_alpha": reward_alpha,
        "critic": {
            "alpha": alpha,
            "lambda": tdlambda,
            "network": {
                "output": {
                    "input": "input.obs",
                    "layers": [
                        {"type": "dense", "size": hidden_size},
                        {"type": "relu"},
                        {"type": "dense", "size": hidden_size},
                        {"type": "relu"},
                        {"type": "dense", "size": 1},
                        {"type": "tanh"},
                    ]
                }
            }
        },
        "policy": {
            "alpha": alpha,
            "lambda": tdlambda,
            "network": {
                "features": {
                    "input": "input.obs",
                    "layers": [
                        {"type": "dense", "size": hidden_size},
                        {"type": "relu"},
                        {"type": "dense", "size": hidden_size},
                        {"type": "relu"},
                    ]
                },
                "output.halt": {
                    "input": "features",
                    "layers": [
                        {"type": "dense", "size": 2},
                        {"type": "tanh"},
                        {"type": "softmax", "temperature": temperature},
                    ]
                },
                "output.direction": {
                    "input": "features",
                    "layers": [
                        {"type": "dense", "size": 25},
                        {"type": "tanh"},
                        {"type": "softmax", "temperature": temperature},
                    ]
                },
                "output.speed": {
                    "input": "features",
                    "layers": [
                        {"type": "dense", "size": 9},
                        {"type": "tanh"},
                        {"type": "softmax", "temperature": temperature},
                    ]
                },
                "output.sensorAction": {
                    "input": "features",
                    "layers": [
                        {"type": "dense", "size": 7},
                        {"type": "tanh"},
                        {"type": "softmax", "temperature": 0.8},
                    ]
                }
            }
        }
    }


def flush_tracers(tracers: list[CsvConsumer]):
    for t in tracers:
        t.flush()


def reward_tracers(agent: TDAgent, folder: str) -> list[CsvConsumer]:
    return [
        KpiListenerBuilder.getter("reward")
        .register(agent, CsvConsumer(folder + "/reward.csv"))
    ]

def full_tracers(agent: TDAgent, folder: str) -> list[CsvConsumer]:
    return [
        KpiListenerBuilder.getter("c0")
        .get("output")
        .register(agent, CsvConsumer(folder + "/v0.csv")),
        KpiListenerBuilder.getter("actions")
        .get("halt")
        .map(lambda x: float(x))
        .register(agent, CsvConsumer(folder + "/action_halt.csv")),
        KpiListenerBuilder.getter("actions")
        .get("direction")
        .map(lambda x: float(x))
        .register(agent, CsvConsumer(folder + "/action_direction.csv")),
        KpiListenerBuilder.getter("actions")
        .get("speed")
        .map(lambda x: float(x))
        .register(agent, CsvConsumer(folder + "/action_speed.csv")),
        KpiListenerBuilder.getter("actions")
        .get("sensorAction")
        .map(lambda x: float(x))
        .register(agent, CsvConsumer(folder + "/action_sensor.csv")),
        KpiListenerBuilder.getter("reward")
        .register(agent, CsvConsumer(folder + "/reward.csv")),
        KpiListenerBuilder.getter("c1")
        .get("output")
        .register(agent, CsvConsumer(folder + "/v1.csv")),
        KpiListenerBuilder.getter("avg_reward")
        .register(agent, CsvConsumer(folder + "/avg_reward.csv")),
        KpiListenerBuilder.getter("delta")
        .register(agent, CsvConsumer(folder + "/delta.csv")),
        KpiListenerBuilder.getter("pi")
        .get("output.halt")
        .register(agent, CsvConsumer(folder + "/pi_halt.csv")),
        KpiListenerBuilder.getter("pi")
        .get("output.direction")
        .register(agent, CsvConsumer(folder + "/pi_direction.csv")),
        KpiListenerBuilder.getter("pi")
        .get("output.speed")
        .register(agent, CsvConsumer(folder + "/pi_speed.csv")),
        KpiListenerBuilder.getter("pi")
        .get("output.sensorAction")
        .register(agent, CsvConsumer(folder + "/spi_sensor.csv")),
        KpiListenerBuilder.getter("trained_c0")
        .get("output")
        .register(agent, CsvConsumer(folder + "/trained_v0.csv")),
        KpiListenerBuilder.getter("trained_c1")
        .get("output")
        .register(agent, CsvConsumer(folder + "/trained_v1.csv")),
        KpiListenerBuilder.getter("trained_pi")
        .get("output.halt")
        .register(agent, CsvConsumer(folder + "/trained_pi_halt.csv")),
        KpiListenerBuilder.getter("trained_pi")
        .get("output.direction")
        .register(agent, CsvConsumer(folder + "/trained_pi_direction.csv")),
        KpiListenerBuilder.getter("trained_pi")
        .get("output.speed")
        .register(agent, CsvConsumer(folder + "/trained_pi_speed.csv")),
        KpiListenerBuilder.getter("trained_pi")
        .get("output.sensorAction")
        .register(agent, CsvConsumer(folder + "/trained_spi_sensor.csv")),
        KpiListenerBuilder.getter("trained_avg_reward")
        .register(agent, CsvConsumer(folder + "/trained_avg_reward.csv"))
    ]


def main():
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)
#    logging.info(pygame.font.get_fonts())
    parser = init_argparse()
    args = parser.parse_args()

    logging.info("Loading environment ...")

    # Creates the simulated environment
    robot = SimRobot(obstacles=ObstacleMapBuilder(size=0.2)
                     .rect((-5, -5), (5, 5))
                     .rand(10, random=Random(1234), min_distance=1, max_distance=3)
                     .build())
#    robot = Robot(
#        robotHost="192.168.1.11",
#        robotPort=22
#    )
    # Create the simulated robot
    env1: RobotEnv = Environment.create(environment=args.environment,
                                        robot=robot,
                                        reward=fuzzy_stuck(distances=(0.1, 0.3, 0.7, 2.0), sensor_range=90))

    # Create the encoed environment
    environment = EncodedRobotEnv(env=env1)
    # Creates the dnn agent
    random = tf.random.Generator.from_seed(1234)
    state_spec = environment.states()
    actions_spec = environment.actions()
    agent = TDAgent.create(state_spec=state_spec,
                            actions_spec=actions_spec,
                            agent_spec=agent_spec(),
                            random=random)

    tracers = reward_tracers(
        agent=agent, folder=args.stats) if args.stats is not None else None

    logging.info("Starting ...")
    states = environment.reset()
    window = RobotWindow().set_robot(robot).render()

    logging.info("Running ...")
    running = True
    reward_kpi: DiscountConsumer = KpiListenerBuilder.getter("reward") \
        .register(agent, DiscountConsumer(0.99))
    frame_inter = int(1000 / _FPS)
    time_frame = pygame.time.get_ticks()
    while running and robot.time() <= args.time:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        t = pygame.time.get_ticks()
        if t > time_frame and reward_kpi.kpi() is not None:
            window.set_robot(robot).set_reward(
                float(reward_kpi.kpi())).render()
            time_frame += frame_inter

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if terminal:
            states = environment.reset()

    pygame.quit()
    logging.info("Closing agent ...")
    agent.close()
    logging.info("Closing environment ...")
    environment.close()
    logging.info("Completed.")
    if tracers is not None:
        flush_tracers(tracers)


if __name__ == '__main__':
    main()
