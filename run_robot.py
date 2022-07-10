import wheelly.robot as robot
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s: %(message)s')

r = robot.Robot(
    host= "192.168.1.11",
    port= 22
)

r.connect()
s = r.read_status()
logging.debug(f"status={s}")
r.move(90, 1)
time.sleep(0.5)
r.halt()
r.scan(90)
to = time.time() + 3
while time.time() < to:
    s = r.read_status()
    logging.debug(f"status={s}")
r.close()
