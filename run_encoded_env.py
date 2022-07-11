import argparse
import logging
import pathlib

import pygame
from tensorforce import Environment
from tensorforce.agents import Agent

from wheelly.envs import EncodedRobotEnv, RobotEnv

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        #usage="%(prog)s [OPTION] [FILE]...",
        description="Run encoded robot environment."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 0.1.0"
    )
    parser.add_argument(
        "-e", "--environment", default='robot.json',
        dest='environment',
        help='the json file with environment descriptor (default=robot.json)'
    )
    parser.add_argument(
        "-m", "--model", default='saved-model',
        dest='model',
        help='the output path of agent model (default=saved_model)'
    )
    return parser

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)
    parser = init_argparse()
    args = parser.parse_args()

    env1:RobotEnv = Environment.create(environment=args.environment)

    environment:EncodedRobotEnv = Environment.create(
        environment=EncodedRobotEnv,
        env=env1
    )
    agent = Agent.load(directory=args.model, environment=environment)

    states = environment.reset()
    env1.render()

    running = True
    while running:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        env1.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
    pygame.quit()
    agent.close()
    environment.close()

if __name__ == '__main__':
    main()
