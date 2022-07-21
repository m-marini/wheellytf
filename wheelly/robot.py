"""This module provides robot API and concrete class to connect to concrete robot or simulated robot.

Class:
Robot() -- Connect to concrete robot via socket
SimRobot() -- Simulated robot
"""
from __future__ import annotations

import logging
import re
import socket
import time
from math import degrees, nan, pi, radians
from typing import Any, Tuple

from Box2D import b2Body, b2Vec2, b2World

logger = logging.getLogger(__name__)

CONNECTION_TIMEOUT = 10.0
READ_TIMEOUT = 3.0

class RobotAPI:
    "API Interface for robot"
    def __init__(self):
        self._robot_pos = b2Vec2()
        self._robot_dir = 0
        self._sensor = 0
        self._time = 0.0
        self._status: dict[str, Any]= {}

    def start(self):
        "Start the robot interface"
        raise NotImplemented()

    def move(self, dir: int, speed: float) -> RobotAPI:
        """Moves robot to direction at speed
        
        dir -- the direction (DEG)
        speed -- the speed (-1 ... 1)
        """
        raise NotImplemented()

    def scan(self, dir: int) -> RobotAPI:
        """Moves the sensor to a direction
        
        Arguments:
        dir -- the sensor direction (DEG)
        """
        raise NotImplemented()

    def halt(self) -> RobotAPI:
        """Halt the robot"""
        raise NotImplemented()

    def tick(self, dt: float) -> RobotAPI:
        """Advances time by a time interval
        dt --- the interval (sec)
        """
        raise NotImplemented()

    def close(self) -> RobotAPI:
        """Closes the API interface"""
        raise NotImplemented()

    def robot_pos(self) -> b2Vec2:
        """Returns the robot position"""
        return self._robot_pos

    def robot_dir(self) -> int:
        """Returns the robot direction"""
        return self._robot_dir

    def sensor_dir(self) -> int:
        """Returns the sensor direction"""
        return self._sensor

    def time(self):
        """Returns the robot time"""
        return self._time

    def status(self):
        """Returns the robot status"""
        return self._status;


class Robot (RobotAPI):
    def __init__(self,
        robotHost:str ,
        robotPort: int,
        connectionTimeout=CONNECTION_TIMEOUT,
        readTimeout = READ_TIMEOUT):
        """Create a Robot object to comunicate to robot
        
        Arguments:
        robotHost -- the host name of robot
        robotPort -- the robot port
        connectionTimeout -- connection timeout (sec)
        readTimeout -- read timeout (sec)
        """
        super().__init__()
        self._host = robotHost
        self._port = robotPort
        self._connection_timeout = connectionTimeout
        self._read_timeout = readTimeout
        self._socket = None
        self._timestamp_offset = None
        self._time = time.time()
        self._status = None

    def start(self):
        return self._connect()

    def move(self, dir: int, speed: float) -> Any:
        return self._write_cmd(f"mv {dir} {speed}")

    def scan(self, dir):
        return self._write_cmd(f"sc {dir}")

    def halt(self):
        return self._write_cmd("al")

    def tick(self, dt: float):
        if self._socket:
            if self._timestamp_offset == None:
                self._sync()
            timeout = self._time + dt
            while self._time <= timeout:
                data = self._read_line()
                status = _parse_status(data) if data else None
                self._status = status
                if status:
                    self._robot_dir = status['dir']
                    self._robot_pos = b2Vec2(status['x'], status['y'])
                    self._sensor = status['sensor']
                    self._time = status['timestamp']
                else:
                    self._time = time.time()
        return self

    def close(self):
        if self._socket:
            self._socket.close()
        return self

    def _connect(self):
        """Connects the robot socket"""
        if self._socket == None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self._connection_timeout)
            self._socket.connect((self._host, self._port))
            self._file = self._socket.makefile("rb")
        self._time = time.time()
        return self

    def _sync(self):
        """Synchronizes the local clock with the remote clock"""
        now = time.time()
        ref = f"{now}"
        self._write_cmd(f"ck {ref}")
        while True:
            clock = _parse_clock(self._read_line())
            if clock and clock["ref"] == ref:
                break
        rp = float(clock["reply"] - clock["received"]) / 1000;
        elaps = clock["timestamp"] - now
        latency = (elaps - rp) / 2
        # offset + received - latency = now 
        # offset = now - received + latency
        offset = now - float(clock["received"]) / 1000 + latency;
        self._timestamp_offset = offset
        return self

    def _write_cmd(self, cmd):
        """Writes a command string to robot socket and return the command string.
        
        Arguments:
        cmd -- the command string
        """
        if self._socket:
            self._socket.settimeout(None)
            line = cmd + "\n"
            logger.debug("--> %s", cmd)
            self._socket.sendall(line.encode("utf-8"))
        return self    

    def _read_line(self):
        """Reads a line from robot socket and return a string"""
        if self._file:
            self._socket.settimeout(self._read_timeout)
            data  = self._file.readline()
            timestamp = time.time()
            line = data.decode("utf-8")
            logger.debug("<-- %s", line[0: -1])
            return line, timestamp
        else:
            return None

# ck 1656250845.8646197 12387 12389
# ck ref received_instant reply_instant
def _parse_clock(timed_line:Tuple[str, float]):
    """Parses the clock line and return the clock dictionary

    Argument:
    timed_line -- tuple of clock string and robot clock time
    """
    m = re.search(r"ck (.*) (\d*) (\d*)", timed_line[0])
    return {
        "timestamp": timed_line[1],
        "ref": m.group(1),
        "received": int(m.group(2)),
        "reply": int(m.group(3)),
    } if m else None

# st 36517 0.000 0.000 1 0 0.22 0.000 0.000 0 13.12 1 1 0 1 0 0.00 0
# st clk x y deg sens dist left right contacs voltage canMoveForw camMoveBack imuFailure halt moveDir moveSpeed nextSensor
def _parse_status(timed_line:tuple[str, float]):
    """Parses the status line and return the status dictionary

    Arguments:
    timed_line -- tuple of status string and robot clock time
    """
    m = re.search(r"st (\d*) (-?\d*\.\d*) (-?\d*\.\d*) (-?\d*) (-?\d*) (-?\d*\.\d*) (-?\d*\.\d*) (-?\d*\.\d*) (\d*) (-?\d*\.\d*) ([01]) ([01]) ([01]) ([01]) (-?\d*) (-?\d*\.\d*) (-?\d*)", timed_line[0])
    return {
        #"timestamp": float(m.group(1)) / 1000 + self._timestamp_offset,
        "timestamp": timed_line[1],
        "x": float(m.group(2)),
        "y": float(m.group(3)),
        "dir": int(m.group(4)),
        "sensor": int(m.group(5)),
        "dist": float(m.group(6)),
        "left": float(m.group(7)),
        "right": float(m.group(8)),
        "contacts": int(m.group(9)),
        "canMoveForward": int(m.group(11)),
        "canMoveBackward": int(m.group(12)),
    } if m else None

ROBOT_WIDTH = 0.18
ROBOT_LENGTH = 0.26
_ROBOT_MASS = 0.78
_ROBOT_DENSITY = _ROBOT_MASS / ROBOT_LENGTH / ROBOT_WIDTH
_ROBOT_FRICTION = 0.3

_ROBOT_TRACK = 0.136
_MAX_ACC = 1
_MAX_FORCE = _MAX_ACC * _ROBOT_MASS
_MAX_VELOCITY = 0.280

_VELOCITY_ITER = 10
_POSITION_ITER = 10

_TRACK = 0.136

_RAD_10 = radians(10)
_RAD_30 = radians(30)

class SimRobot(RobotAPI):
    """Simulated robot"""
    def __init__(self):
        """Create a Rsimulated robot envinment"""
        super().__init__()
        world: b2World = b2World(gravity=(0,0), doSleep=True)
        robot: b2Body = world.CreateDynamicBody(position=(0,0))
        box = robot.CreatePolygonFixture(box=(ROBOT_WIDTH / 2, ROBOT_LENGTH/ 2),
            density=_ROBOT_DENSITY,
            friction=_ROBOT_FRICTION)
        robot.angle = pi / 2

        self._distance = 0
        self._can_move_forward = 1
        self._can_move_backward = 1
        self._left = 0
        self._right = 0
        self._contacts = 0
        self.world = world
        self.robot = robot
        self.robotBox = box
        self._direction = 0
        self._speed = 0.0

    def start(self):
        return self

    def robot_pos(self) -> b2Vec2:
        return self.robot.position

    def robot_dir(self):
        return normalizeDeg(round(90 - degrees(self.robot.angle)))

    def move(self, dir: int, speed: float):
        self._direction = dir
        self._speed = speed
        return self

    def scan(self, dir: int):
        self._sensor = dir
        return self

    def halt(self):
        self._direction = self.robot_dir()
        self._speed = 0
        return self

    def status(self):
        return {
            "timestamp": self._time,
            "x": self.robot_pos().x,
            "y": self.robot_pos().x,
            "dir": self.robot_dir(),
            "sensor": self._sensor,
            "dist": self._distance,
            "left": self._left,
            "right": self._right,
            "contacts": self._contacts,
            "canMoveForward": self._can_move_forward,
            "canMoveBackward": self._can_move_backward,
        }

    def tick(self, dt:float):
        self._controller(dt)
        self.world.Step(dt, _VELOCITY_ITER, _POSITION_ITER)
        self._time += dt

    def close(self) -> RobotAPI:
        return self

    def _controller(self, dt:float):
        robot = self.robot
        delta_angle = normalizeRad(radians(90 - self._direction) - robot.angle)
        angular_velocity = clip(lin_map(delta_angle, -_RAD_10, _RAD_10, -1, 1), -1, 1)
        linear_velocity = self._speed * clip(lin_map(abs(delta_angle), 0, _RAD_30, 1, 0), 0, 1)

        left = clip((linear_velocity - angular_velocity) / 2, -1, 1)
        right = clip((linear_velocity + angular_velocity) / 2, -1, 1)

        self._left = left * _MAX_VELOCITY
        self._right = right * _MAX_VELOCITY

        forward_velocity = (left + right) / 2 * _MAX_VELOCITY
        angular_velocity = (right - left) * _MAX_VELOCITY / _TRACK
        target_velocity = robot.GetWorldVector((forward_velocity, 0))
        dv = target_velocity - robot.linearVelocity
        dq = dv * robot.mass
        force = dq / dt
        local_force = robot.GetLocalVector(force)
        if local_force.x > _MAX_FORCE:
            local_force.x = _MAX_FORCE
            force = robot.GetWorldVector(local_force)
        elif local_force.x < -_MAX_FORCE:
            local_force.x = -_MAX_FORCE
            force = robot.GetWorldVector(local_force)

        angular_impulse = (angular_velocity - robot.angularVelocity) * robot.inertia

        self.world.ClearForces()
        self.robot.ApplyForceToCenter(force=force, wake=True)
        self.robot.ApplyAngularImpulse(impulse = angular_impulse, wake=True)

def normalizeRad(angle: float):
    while angle < -pi:
        angle += pi * 2
    while angle > pi:
        angle -= pi * 2
    return angle

def normalizeDeg(angle: int | float):
    while angle < -180:
        angle += 360
    while angle > 180:
        angle -= 260
    return angle

def sign(x:float):
    return 1.0 if x > 0 else \
        -1.0 if x < 0 else \
        -0.0 if x == -0.0 else \
        0.0 if x == 0.0 else \
        nan

def lin_map(x: float, min_x: float, max_x: float, min_y: float, max_y: float):
    return (x - min_x) * (max_y - min_y) / (max_x - min_x) + min_y

def clip(x:float, min_x: float, max_x: float):
    return min(max(x, min_x), max_x)
