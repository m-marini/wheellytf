import gym
from wheelly.envs.robot_env import RobotEnv
from wheelly.envs.robot_wrapper import RobotWrapper
import pygame
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logging.getLogger("wheelly.envs.robot").setLevel(logging.DEBUG)

base_env = gym.make('wheelly/RobotEnv-v0', params={
    "host": "192.168.1.11",
    "port": 22,
})

env = RobotWrapper(base_env)

env.action_space.seed(42)
observation = env.reset(seed=42, return_info=False)

logging.debug(f"observation=:{observation}")

running = True
while running:
    action = env.action_space.sample()
    #action["halt"] = 1
    #action["direction"] = np.array([0])
    #action["speed"] = np.array([float(0)])
    #action["sensor"] = np.array([0])
    observation, reward, done, info = env.step(action)

    env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
env.close()
pygame.quit()
