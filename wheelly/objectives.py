from typing import Any, Callable

_VELOCITY_THRESHOLD = 0.01

def no_move()-> Callable[[dict[str, Any]], float]:
    """Reward function of no move behaviours"""
    return _no_move

def stuck(distance:float=0.5, range:float=0.1)-> Callable[[dict[str, Any]], float]:
    """Reward function of stuck to obstacle"""
    return lambda status: _stuck(status=status, distance=distance, range=range)

def _no_move(status:dict[str, Any])->float:
    """Reward function of no move behaviours"""
    if status['canMoveForward'] == 0 or status["canMoveBackward"] == 0:
        return -1
    elif abs(status["left"]) < _VELOCITY_THRESHOLD and abs(status["right"]) < _VELOCITY_THRESHOLD and status['sensor'] == 0:
        return 1
    else:
        return 0

def _stuck(status:dict[str, Any], distance:float=0.5, range:float=0.1)->float:
    """Reward function of stuck to obstacle"""
    d = status['dist']
    if status['canMoveForward'] == 0 or status["canMoveBackward"] == 0 or d == 0:
        return -1
    elif abs(d - distance) <= range:
        return 1
    else:
        return -0.1
