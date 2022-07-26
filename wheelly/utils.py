from math import atan2, floor, nan, pi

from numpy import ndarray, arctan2

def normalizeRad(angle: float):
    "Returns normalized radians angle"
    while angle < -pi:
        angle += pi * 2
    while angle > pi:
        angle -= pi * 2
    return angle

def normalizeDeg(angle: int | float):
    "Returns normalized degrees angle"
    while angle < -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle

def sign(x:float):
    """Return the sign of x"""
    return 1.0 if x > 0 else \
        -1.0 if x < 0 else \
        -0.0 if x == -0.0 else \
        0.0 if x == 0.0 else \
        nan

def lin_map(x: float, min_x: float, max_x: float, min_y: float, max_y: float):
    """Returns the linear map of x

    Arguments:
    x -- the value
    min_x -- min value of x
    max_x -- min value of x
    min_y -- min value of output y
    max_y -- max value of output y
    """
    return (x - min_x) * (max_y - min_y) / (max_x - min_x) + min_y

def clip(x:float, min_x: float, max_x: float):
    """Returns the clip value of x

    Arguments:
    x -- the value
    min_x -- min value of x
    max_x -- min value of x
    """
    return min(max(x, min_x), max_x)

def direction(vect: ndarray):
    """Returns the angle of a vector (RAD)"""
    return atan2(vect[1], vect[0])

def str_time_inter(interval: float, dec=1):
    """Returns the string representation of a time interval"""
    interval = round(interval, dec)
    mm = floor(interval / 60)
    ss = interval - mm * 60
    hh = floor(mm / 60)
    mm -= hh * 60
    gg = floor(hh / 24)
    hh -= gg * 24
    result = ""
    if gg > 0:
        result += f"{gg}g "
    if hh > 0 or hh > 0:
        result += f"{hh:02.0f}h "
    if mm > 0 or hh > 0 or gg > 0:
        result += f"{mm:02.0f}' "
    result += f'{ss:0{3 + dec}.{dec}f}"'
    return result
