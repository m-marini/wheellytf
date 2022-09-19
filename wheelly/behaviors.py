
from typing import Callable

from wheelly.robots import RobotAPI

def halt(env:RobotAPI, start:float, stop:float) -> Callable[[float], None]:
    def _halt(t:int):
        if t >= start and t < stop:
            env.halt()
    return _halt

def move(env:RobotAPI, start:float, stop:float, dir:int, speed: float) -> Callable[[float], None]:
    return fmove(env=env, start=start, stop=stop, dir=lambda _: dir, speed=lambda _: speed)

def fmove(env:RobotAPI, start:float, stop:float, dir:Callable[[float], int], speed: Callable[[float], float]) -> Callable[[float], None]:
    def _fmove(t:int):
        if t >= start and t < stop:
            env.move(dir(t - start), speed(t - start))
    return _fmove

def scan(env:RobotAPI, start:float, stop:float, dir:int) -> Callable[[float], None]:
    return fscan(env=env, start=start, stop=stop, dir= lambda _: dir)

def fscan(env:RobotAPI, start:float, stop:float, dir:Callable[[float], int]) -> Callable[[float], None]:
    def _fscan(t:int, env:RobotAPI, start:float, stop:float, dir:Callable[[float], int]):
        if t >= start and t < stop:
            env.scan(dir(t - start))
    return _fscan

def concat(*behaviors: Callable[[float], None]) -> Callable[[int], None]:
    def _concat(t: float):
        for b in behaviors:
            b(t)
    return _concat
