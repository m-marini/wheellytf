import argparse
import logging

import pygame
from tensorforce import Environment
from tensorforce.agents import Agent

from wheelly.envs import EncodedRobotEnv, RobotEnv
from wheelly.renders import RobotWindow
from wheelly.robots import Robot, SimRobot
from wheelly.sims import ObstacleMapBuilder
from wheelly.objectives import stuck

_DEFAULT_DISCOUNT = 0.99
_FPS = 60

font:pygame.font.Font | None = None

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
        "-m", "--model", default='saved-model',
        dest='model',
        help='the output path of agent model (default=saved_model)'
    )
    return parser

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)
#    logging.info(pygame.font.get_fonts())
    parser = init_argparse()
    args = parser.parse_args()

    logging.info("Loading environment ...")

    robot = SimRobot(obstacles=ObstacleMapBuilder(size=0.2) \
        .rect((-5, -5), (5, 5))
        .add((2,2))
        .build())
#    robot = Robot(
#        robotHost="192.168.1.11",
#        robotPort=22
#    )
    env1:RobotEnv = Environment.create(environment=args.environment,
        robot=robot,
        reward=stuck())

    environment:EncodedRobotEnv = Environment.create(
        environment=EncodedRobotEnv,
        env=env1
    )
    logging.info("Loading agent ...")
    agent:Agent = Agent.load(directory=args.model,
        environment=environment
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
    while running:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        avg_rewards = avg_rewards * discount + reward * (1 - discount)
        agent.observe(terminal=terminal, reward=reward)
        t = pygame.time.get_ticks()
        if t > time_frame:
            window.set_robot(robot).set_reward(avg_rewards).render()
            time_frame += frame_inter

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
    pygame.quit()
    logging.info("Closing agent ...")
    agent.close()
    logging.info("Closing environment ...")
    environment.close()
    logging.info("Completed.")

if __name__ == '__main__':
    main()
