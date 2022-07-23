import re

import wheelly.robots as robots


def test_re():

    text = "sc -1.2"
    exp = r"sc (-?\d*\.\d)"

    x = re.search(exp, text)

    assert x
    assert x.group(1) == "-1.2"

#s = robot.parse_status("st 36517 0.000 0.000 1 0 0.22 0.000 0.000 0 13.12 1 1 0 1 0 0.00 0")
#print (f"status={s}")
