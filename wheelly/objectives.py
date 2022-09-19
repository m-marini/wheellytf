from typing import Any, Callable, Tuple

from wheelly.utils import defuzzy, fuzzy_and, fuzzy_not, fuzzy_pos, fuzzy_range


def no_move(velocity_threshold:float=0.01)-> Callable[[dict[str, Any]], float]:
    """Reward function of no move behaviours"""
    return lambda status: _no_move(status=status, velocity_threshold=velocity_threshold)

def stuck(distance:float=0.5, range:float=0.1)-> Callable[[dict[str, Any]], float]:
    """Reward function of stuck to obstacle"""
    return lambda status: _stuck(status=status, distance=distance, range=range)

def fuzzy_stuck(distances: Tuple[float, float, float, float], sensor_range:int)-> Callable[[dict[str, Any]], float]:
    """Reward function of stuck to obstacle"""
    return lambda status: _fuzzy_stuck(status=status, distances=distances, sensor_range=sensor_range)

def _no_move(status:dict[str, Any], velocity_threshold:float)->float:
    """Reward function of no move behaviours"""
    if status['canMoveForward'] == 0 or status["canMoveBackward"] == 0:
        return -1
    elif abs(status["left"]) < velocity_threshold and abs(status["right"]) < velocity_threshold and status['sensor'] == 0:
        return 1
    else:
        return 0

def _stuck(status:dict[str, Any], distance:float, range:float)->float:
    """Reward function of stuck to obstacle"""
    d = status['dist']
    if status['canMoveForward'] == 0 or status["canMoveBackward"] == 0 or d == 0:
        return -1
    elif abs(d - distance) <= range and status['sensor'] == 0:
        return 1
    else:
        return -0.1

def _fuzzy_stuck(status:dict[str, Any], distances:Tuple[float, float, float, float], sensor_range: int)->float:
    """Reward function of stuck to obstacle"""
    dist = status['dist']
    if status['canMoveForward'] == 0 or status["canMoveBackward"] == 0 or dist == 0:
        return -1
    else:   
        sensor =  status['sensor']
        isInRange = fuzzy_range(dist, limits=distances)
        isInDirection = fuzzy_not(fuzzy_pos(abs(sensor), sensor_range))
        isTarget = fuzzy_and(isInRange, isInDirection)
        return defuzzy((1, isTarget))
