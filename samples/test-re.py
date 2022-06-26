import re
import wheelly.envs.robot as robot

text = "sc -1.2"
exp = r"sc (-?\d*\.\d)"

x = re.search(exp, text)

print (x)
print (x.groups())
print (x.group(0))
print (x.group(1))


s = robot.parse_status("st 36517 0.000 0.000 1 0 0.22 0.000 0.000 0 13.12 1 1 0 1 0 0.00 0")
print (f"status={s}")
