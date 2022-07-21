import argparse
import logging

import pygame
from tensorforce import Environment
from tensorforce.agents import Agent

from wheelly.envs import EncodedRobotEnv, RobotEnv
from wheelly.renders import WINDOW_SIZE, RobotWindow
from wheelly.robot import Robot, SimRobot

FONT_NAME = 'freesans'
FONT_SIZE = 20

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

def render_info(window: pygame.Surface, string: str):
    logging.debug(string)

def render_info1(window: pygame.Surface, string: str):
    text = font.render(string, True, (0, 0, 0))
    #get the rect of the text
    textRect = text.get_rect()
    #set the position of the text
    textRect.center = (WINDOW_SIZE / 2, 10)
    #add text to window
    window.blit(text, textRect)
    pygame.display.update()

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)
#    logging.info(pygame.font.get_fonts())
    parser = init_argparse()
    args = parser.parse_args()
    pygame.init()
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE) 

    logging.info("Loading environment ...")
    #robot = SimRobot()
    robot = Robot(
        robotHost="192.168.1.11",
        robotPort=22
    )
    env1:RobotEnv = Environment.create(environment=args.environment,
        robot=robot)

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
    discount = 0.99
    while running:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        avg_rewards = avg_rewards * discount + reward * (1 - discount)
        agent.observe(terminal=terminal, reward=reward)

        window.set_robot(robot).render()
        render_info(window=window, string=f"Average {avg_rewards:.2f}")

         #   render_info(env1.window, f"Average reward {tot_rew / no_step}")
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

def dummy():
    agent:Agent = Agent.load(directory=args.model,
        environment=environment,
        summarizer=dict(
            directory='tensorboard',
            # list of labels, or 'all'
            labels=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
            #labels=['all']
        )
    )


if __name__ == '__main__':
    main()
