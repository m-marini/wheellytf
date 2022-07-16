import logging
import time
from wheelly.envs import SimRobotEnv
import pygame
import math
from math import degrees, floor,ceil
from wheelly.pygame_utils import RobotWindow
import numpy as np

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s: %(message)s')
    env = SimRobotEnv()

    env.reset()
    window = RobotWindow()
    window.robot_pos(env.robot_pos())
    window.robot_dir(env.robot_dir())
    window.sensor_dir(env.sensor_dir())
    window.render()

    running = True
    clock = pygame.time.Clock()
    while running:
        clock.tick(1/0.3)
        t = pygame.time.get_ticks()
        actions = agent_action(t)
        env.execute(actions=actions)
        window.robot_pos(env.robot_pos())
        window.robot_dir(env.robot_dir())
        window.sensor_dir(env.sensor_dir())
        window.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

    pygame.quit()

HALT_ACTION = {
    "halt": 1,
    "direction": 0,
    "speed": 0,
    "sensorAction": 0
}

def agent_action(t:int):
    return HALT_ACTION

if __name__ == '__main__':
    main()
