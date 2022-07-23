import logging
from typing import Any, Callable

import pygame

from wheelly.renders import RobotWindow
from wheelly.robots import SimRobot
from wheelly.sims import ObstacleMapBuilder

FPS = 60

def _move(t:int, env:SimRobot, start:float, stop:float, dir:float, speed: float):
    if t >= start and t <= stop:
        env.move(dir, speed)

def _halt(t:int, env:SimRobot, start:float, stop:float):
    if t >= start and t <= stop:
        env.halt()

def _scan(t:int, env:SimRobot, start:float, stop:float, dir:int):
    if t >= start and t <= stop:
        env.scan(dir)

def halt(env:SimRobot, start:float, stop:float) -> Callable[[float], Any]:
    return lambda t: _halt(t=t, env=env, start=start, stop=stop)

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
    obs = ObstacleMapBuilder(size=0.2) \
        .rect((-5., -5.), (5., 5.)) \
        .add((1, 0)) \
        .build()
    robot = SimRobot(obs)
    window = RobotWindow().set_robot(robot).render()
    behavior = concat(
        move(robot, 0, 10, 90, 1),
        move(robot, 10, 15, 90, -1),
        halt(robot, 15, 16),
        scan(robot, 1, 9, 60),
        scan(robot, 9, 10, 0)
    )

    robot.start()
    running = True
    t = 0
    dt = 0.01
    tf = int(1000 / FPS)
    next_frame = tf
    sync_wait = 2
    while running:
        t += dt
        behavior(t)
        robot.tick(dt)
        if int(t * 1000) >= next_frame - sync_wait:
            tt = pygame.time.get_ticks()
            pygame.time.wait(next_frame - tt)
            window.set_robot(robot).render()
            next_frame += tf

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

    pygame.quit()

if __name__ == '__main__':
    main()
