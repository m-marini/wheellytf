"""Tests the robot motion by moving it forward for for 2 seconds."""
import logging
import time

import wheelly.robots as robots

TEST_DURATION = 2
CMD_INTERVAL = 0.8

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s: %(message)s')

    r = robots.Robot(
        robotHost="192.168.1.11",
        robotPort=22
    )

    r.start()
    dt = 0.1
    r.tick(dt)

    test_timeout = time.time() + TEST_DURATION
    cmd_timeout = 0
    prev = time.time()
    while time.time() < test_timeout:
        t = time.time()
        if t >= cmd_timeout:
            r.move(0, 1)
            cmd_timeout = t + CMD_INTERVAL
        r.tick(dt)
        s = r.status()
        #logging.debug(f"status={s}")

    r.halt()
    r.close()

if __name__ == '__main__':
    main()
