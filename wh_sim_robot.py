import logging
from typing import Any, Callable

import pygame

from wheelly.renders import RobotWindow
from wheelly.robot import Robot, SimRobot

FPS = 60

def _move(t:int, env:SimRobot, start:float, stop:float, dir:float, speed: float):
    if t >= start and t <= stop:
        env.move(dir, speed)
    else:
        env.halt()

def _scan(t:int, env:SimRobot, start:float, stop:float, dir:int):
    if t >= start and t <= stop:
        env.scan(dir)
    else:
        env.scan(0)

def move(env:SimRobot, start:float, stop:float, dir:float, speed: float) -> Callable[[float], Any]:
    return lambda t: _move(t=t, env=env, start=start, stop=stop, dir=dir, speed=speed)

def scan(env:SimRobot, start:float, stop:float, dir:int) -> Callable[[float], Any]:
    return lambda t: _scan(t=t, env=env, start=start, stop=stop, dir=dir)

def _concat(t: float, *behaviors: Callable[[float], Any]):
    for b in behaviors:
        b(t)

def concat(*behaviors: Callable[[int], Any]) -> Callable[[int], Any]:
    return lambda t : _concat(t, *behaviors)

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s: %(message)s')
    env = SimRobot()
    window = RobotWindow().set_robot(env).render()
    behavior = concat(
        move(env, 0, 4, 90, 1),
        scan(env, 1, 3, 90)
    )

    env.start()
    running = True
    true_clock = False
    clock = pygame.time.Clock()
    t = 0
    dt = 0.01
    while running:
        clock.tick(FPS)
        for _ in range(0, 100):
            t += dt
            behavior(t)
            env.tick(dt)

        window.set_robot(env).render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

    pygame.quit()

if __name__ == '__main__':
    main()
