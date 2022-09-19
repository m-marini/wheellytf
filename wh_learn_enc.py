"""Trains a tensorforce agent defined by json file for robot encoded environment defined in json file
of robot simulated environment"""

import argparse
import logging
from datetime import datetime
from random import Random
from typing import Tuple

import pygame
from numpy import ndarray, zeros
from tensorforce import Environment
from tensorforce.agents import Agent

from wheelly.envs import EncodedRobotEnv, RobotEnv
from wheelly.objectives import fuzzy_stuck
from wheelly.renders import RobotWindow
from wheelly.robots import SimRobot
from wheelly.sims import ObstacleMapBuilder

_DEFAULT_DISCOUNT = 0.99
_FPS = 60

font:pygame.font.Font | None = None

class StatsFile:
    def __init__(self, prefix:str):
        self._filename = f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        self._buffer = zeros((100, 2))
        self._n = 0
    
    def append(self, data: Tuple[float, float]):
        if self._filename:
            self._buffer[self._n, 0] = data[0]
            self._buffer[self._n, 1] = data[1]
            self._n += 1
            if self._n >= self._buffer.shape[0]:
                self.flush()
        return self

    def flush(self):
        if self._filename:
                f = open(self._filename, "a")
                for i in range(0, self._n):
                    f.write(f'{self._buffer[i, 0]},{self._buffer[i, 1]}\n')
                f.close()
                self._n = 0
        return self

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        #usage="%(prog)s [OPTION] [FILE]...",
        description="Learn in encoded robot environment."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 0.1.0"
    )
    parser.add_argument(
        "-e", "--environment", default='environment.json',
        dest='environment',
        help='the json file with environment descriptor (default=environment.json)'
    )
    parser.add_argument(
        "-m", "--model", default='models/default',
        dest='model',
        help='the path of agent model (default=models/default)'
    )
    parser.add_argument(
        "-s", "--stats",
        dest='stats',
        help='activate and set the stats prefix filename'
    )
    parser.add_argument(
        "-t", "--time", default=43200,
        dest='time', type=float,
        help='stop after time (default=43200 sec. = 12 hours)'
    )
    return parser

def append_data(filename:str | None, data:ndarray):
    if filename:
        f = open(filename, "a")
        for i in range(0, data.shape[0]):
            f.write(f'{data[i, 0]},{data[i, 1]}\n')
        f.close()

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)
#    logging.info(pygame.font.get_fonts())
    parser = init_argparse()
    args = parser.parse_args()

    logging.info("Loading environment ...")
    
    stats = StatsFile(args.stats) if args.stats else None

    robot = SimRobot(obstacles=ObstacleMapBuilder(size=0.2) \
        .rect((-5, -5), (5, 5))
        .rand(10, random=Random(1234), min_distance=1, max_distance=3)
        .build())
#    robot = Robot(
#        robotHost="192.168.1.11",
#        robotPort=22
#    )
    env1:RobotEnv = Environment.create(environment=args.environment,
        robot=robot,
        reward=fuzzy_stuck(distances=(0.1, 0.3, 0.7, 2.0), sensor_range=90))

    environment:EncodedRobotEnv = Environment.create(
        environment=EncodedRobotEnv,
        env=env1
    )
    logging.info(f"Loading agent {args.model} ...")
    agent:Agent = Agent.load(directory=args.model,
        environment=environment,
#        summarizer=dict(
#            directory='tensorboard',
#            labels='all'
#            flush=1
#        ),
    )

    logging.info("Starting ...")
    states = environment.reset()
    window = RobotWindow() \
        .set_robot(robot) \
        .render()

    logging.info("Running ...")
    running = True
    avg_rewards = 0.0
    discount = _DEFAULT_DISCOUNT
    frame_inter = int(1000 / _FPS)
    time_frame = pygame.time.get_ticks()
    while running and robot.time() <= args.time:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        if stats:
            stats.append((robot.time(), reward))
        avg_rewards = discount * (avg_rewards  - reward) + reward
        t = pygame.time.get_ticks()
        if t > time_frame:
            window.set_robot(robot).set_reward(avg_rewards).render()
            time_frame += frame_inter

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if terminal:
            states = environment.reset()

    if stats:
        stats.flush()
    pygame.quit()
    logging.info("Closing agent ...")
    agent.close()
    logging.info("Closing environment ...")
    environment.close()
    logging.info("Completed.")

if __name__ == '__main__':
    main()
