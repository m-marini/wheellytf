
from typing import Callable

from wheelly.robots import RobotAPI

def halt(env:RobotAPI, start:float, stop:float) -> Callable[[float], None]:
    return lambda t: _halt(t=t, env=env, start=start, stop=stop)

def move(env:RobotAPI, start:float, stop:float, dir:int, speed: float) -> Callable[[float], None]:
    return lambda t: _fmove(t=t, env=env, start=start, stop=stop,
        dir=lambda _: dir, speed=lambda _: speed)

def fmove(env:RobotAPI, start:float, stop:float, dir:Callable[[float], int], speed: Callable[[float], float]) -> Callable[[float], None]:
    return lambda t: _fmove(t=t, env=env, start=start, stop=stop,
        dir=dir, speed=speed)

def scan(env:RobotAPI, start:float, stop:float, dir:int) -> Callable[[float], None]:
    return lambda t: _fscan(t=t, env=env, start=start, stop=stop, dir= lambda _: dir)

def fscan(env:RobotAPI, start:float, stop:float, dir:Callable[[float], int]) -> Callable[[float], None]:
    return lambda t: _fscan(t=t, env=env, start=start, stop=stop, dir=dir)

def concat(*behaviors: Callable[[float], None]) -> Callable[[int], None]:
    return lambda t : _concat(t, *behaviors)

def _concat(t: float, *behaviors: Callable[[float], None]):
    for b in behaviors:
        b(t)

def _fmove(t:int, env:RobotAPI, start:float, stop:float, dir:Callable[[float], int], speed: Callable[[float], float]):
    if t >= start and t <= stop:
        env.move(dir(t-start), speed(t-start))

def _halt(t:int, env:RobotAPI, start:float, stop:float):
    if t >= start and t <= stop:
        env.halt()

def _fscan(t:int, env:RobotAPI, start:float, stop:float, dir:Callable[[float], int]):
    if t >= start and t <= stop:
        env.scan(dir(t - start))
