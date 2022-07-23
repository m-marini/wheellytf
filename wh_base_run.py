import logging

import pygame
from tensorforce import Environment
from tensorforce.agents import Agent

from wheelly.envs import RobotEnv
from wheelly.renders import RobotWindow
from wheelly.robots import Robot

def constant_agent(environment):
    return Agent.create(
            environment=environment,
            agent='constant',
            action_values={
                'halt': 0,
                'direction': 0,
                'speed': 1,
                'sensorAction': 0
            }
    )

def random_agent(environment):
    return Agent.create(
            environment=environment,
            agent='random',
    )

def a2c_agent(environment):
    return Agent.create(
        environment=environment,
        agent='a2c',
        batch_size=10
    )

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)

    #robot = SimRobot()
    robot = Robot(
        robotHost="192.168.1.11",
        robotPort=22
    )

    environment:RobotEnv = Environment.create(
        environment=RobotEnv,
        robot=robot
    )

    agent:Agent = random_agent(environment)

    states = environment.reset()
    window = RobotWindow() \
        .set_robot(robot) \
        .render()

    running = True
    while running:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
        
        window = RobotWindow() \
            .set_robot(robot) \
            .render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
    environment.close()
    pygame.quit()

if __name__ == '__main__':
    main()
