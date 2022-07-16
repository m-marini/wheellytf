import logging
import time

import wheelly.robot as robot

TEST_DURATION = 2
CMD_INTERVAL = 0.8

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s: %(message)s')

    r = robot.Robot(
        robotHost="192.168.1.11",
        robotPort=22
    )

    r.connect()
    r.read_status()

    test_timeout = time.time() + TEST_DURATION
    cmd_timeout = 0
    while time.time() < test_timeout:
        t = time.time()
        if t >= cmd_timeout:
            r.move(0, 1)
            cmd_timeout = t + CMD_INTERVAL
        s = r.read_status()
        #logging.debug(f"status={s}")

    r.halt()
    r.close()

if __name__ == '__main__':
    main()
