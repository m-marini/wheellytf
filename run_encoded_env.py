import argparse
import logging

import pygame
from tensorforce import Environment
from tensorforce.agents import Agent

from wheelly.envs import EncodedRobotEnv, RobotEnv
from wheelly.pygame_utils import WINDOW_SIZE, RobotWindow

FONT_NAME = 'freesans'
FONT_SIZE = 20

font:pygame.font.Font | None = None

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

def render_info(window: pygame.Surface, string: str):
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
    env1:RobotEnv = Environment.create(environment=args.environment)

    environment:EncodedRobotEnv = Environment.create(
        environment=EncodedRobotEnv,
        env=env1
    )
    logging.info("Loading agent ...")
    agent = Agent.load(directory=args.model, environment=environment)

    logging.info("Starting ...")
    states = environment.reset()
    window = RobotWindow()
    window.robot_pos(env1.robot_pos())
    window.robot_dir(env1.robot_dir())
    window.sensor_dir(env1.sensor_dir())
    window.render()

    logging.info("Running ...")
    tot_rew = 0.0
    no_step = 0
    running = True
    while running:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        if no_step == 100:
            tot_rew /= no_step
            no_step = 1
        no_step += 1
        tot_rew += reward
        agent.observe(terminal=terminal, reward=reward)

        window = RobotWindow()
        window.robot_pos(env1.robot_pos())
        window.robot_dir(env1.robot_dir())
        window.sensor_dir(env1.sensor_dir())
        window.render()

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

if __name__ == '__main__':
    main()
