import logging
from math import floor

import pygame

from wheelly.behaviors import concat, fscan, halt, move, scan
from wheelly.renders import RobotWindow
from wheelly.robots import SimRobot
from wheelly.sims import ObstacleMapBuilder

FPS = 60

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s: %(message)s')
    obs = ObstacleMapBuilder(size=0.2) \
        .rect((-2., -2.), (2., 2.)) \
        .build()
#        .rect((-5., -5.), (5., 5.)) \
#        .add((0.4, 0)) \
    robot = SimRobot(obs)
    window = RobotWindow().set_robot(robot).render()

    """    behavior = concat(
        move(robot, 0, 10, 90, 1),
        move(robot, 10, 15, -90, 1),
        halt(robot, 15, 16),
        fscan(robot, 1, 30, lambda t: (floor(t / 0.1) % 19) * 10 - 90),
        scan(robot, 30, 100, -90)
    )

    behavior = concat(
        move(robot, 0, 10, -90, 0),
        halt(robot, 10, 100),
        fscan(robot, 1, 30, lambda t: (floor(t / 0.1) % 19) * 10 - 90),
        scan(robot, 30, 100, 0)
    )
"""
    behavior = concat(
        move(robot, 0, 12, 140, 1),
        move(robot, 12, 14, -90, 0),
        halt(robot, 14, 100),
        fscan(robot, 1, 30, lambda t: (floor(t / 0.1) % 19) * 10 - 90),
        scan(robot, 30, 100, 0)
    )

    robot.start()
    running = True
    dt = 0.01
    tf = int(1000 / FPS)
    next_frame = tf
    sync_wait = 2
    cmd_time = robot.time() + 0.3
    while running:
        if robot.time() > cmd_time:
            behavior(robot.time())
            cmd_time += 0.3
        robot.tick(dt)
        t = robot.time()
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
