import logging

import pygame

from wheelly.behaviors import concat, fscan, halt, move, scan
from wheelly.renders import RobotWindow
from wheelly.robots import SimRobot
from wheelly.sims import ObstacleMapBuilder

FPS = 60

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s: %(message)s')
    obs = ObstacleMapBuilder(size=0.2) \
        .rect((-5., -5.), (5., 5.)) \
        .add((0.4, 0)) \
        .build()
    robot = SimRobot(obs)
    window = RobotWindow().set_robot(robot).render()
    behavior = concat(
        move(robot, 0, 10, 90, 1),
        move(robot, 10, 15, 90, -1),
        halt(robot, 15, 16),
        fscan(robot, 1, 9, lambda t: round(t * 18 / 8) * 10 - 90),
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
