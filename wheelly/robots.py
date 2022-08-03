"""This module provides robot API and concrete class to connect to concrete robot or simulated robot.

Class:
Robot() -- Connect to concrete robot via socket
SimRobot() -- Simulated robot
"""
from __future__ import annotations

import random
import logging
import re
import socket
import time

from math import cos, degrees, pi, radians, sin
from os import stat
from typing import Any, Tuple

import numpy as np
from Box2D import (b2Body, b2Contact, b2ContactListener, b2PolygonShape,
                   b2Vec2, b2World)

from wheelly.sims import ObstacleMap
from wheelly.utils import clip, lin_map, normalizeDeg, normalizeRad

logger = logging.getLogger(__name__)

CONNECTION_TIMEOUT = 10.0
READ_TIMEOUT = 3.0
_OBSTACLE_SIZE = 0.2

class RobotAPI:
    "API Interface for robot"
    def __init__(self):
        pass

    def start(self) -> RobotAPI:
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

    def reset(self) -> RobotAPI:
        """Closes the API interface"""
        raise NotImplemented()

    def robot_pos(self) -> b2Vec2:
        """Returns the robot position"""
        raise NotImplemented()

    def robot_dir(self) -> int:
        """Returns the robot direction"""
        raise NotImplemented()

    def sensor_dir(self) -> int:
        """Returns the sensor direction"""
        raise NotImplemented()

    def sensor_distance(self) -> float:
        """Returns the sensor distance"""
        raise NotImplemented()
    
    def contacts(self) -> int:
        """Returns the contact sensors"""
        raise NotImplemented()

    def can_move_forward(self) -> bool:
        """Returns the move forward sensors"""
        raise NotImplemented()

    def can_move_backward(self) -> bool:
        """Returns the move backward sensors"""
        raise NotImplemented()

    def sensor_obstacle(self) -> b2Vec2 | None:
        """Returns the obstacle location"""
        dist = self.sensor_distance()
        if dist > 0:
            d = dist + _OBSTACLE_SIZE / 2
            angle = radians(90 - self.robot_dir() - self.sensor_dir())
            return b2Vec2(d * cos(angle), d * sin(angle)) + self.robot_pos()
        else:
            return None

    def time(self) -> float:
        """Returns the robot time"""
        raise NotImplemented()

    def elapsed(self) -> float:
        """Returns the time since last reset"""
        raise NotImplemented()

    def status(self) -> dict[str, Any]:
        """Returns the robot status"""
        raise NotImplemented()
    
    def obstaclesMap(self)-> ObstacleMap | None:
        """Returns the obstacle map if any"""
        return None

class Robot(RobotAPI):
    """Implements the interface to the real robot"""
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
        self._reset_time = self._time
        self._status = None

    def start(self):
        return self._connect()

    def move(self, dir: int, speed: float) -> Any:
        return self._write_cmd(f"mv {dir} {speed}")

    def scan(self, dir):
        return self._write_cmd(f"sc {dir}")

    def halt(self):
        return self._write_cmd("al")
    
    def reset(self):
        self._reset_time = self._time
        self._status = None
        return self

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
                    self._time = status['timestamp']
                else:
                    self._time = time.time()
        return self
    
    def robot_dir(self) -> int:
        return self._status['dir'] if self._status else 0
    
    def robot_pos(self) -> b2Vec2:
        return b2Vec2(self._status['x'], self._status['y']) if self._status else b2Vec2()

    def sensor_dir(self) -> int:
        return self._status['sensor'] if self._status else 0

    def sensor_distance(self) -> float:
        return self._status['dist'] if self._status else 0.0

    def contacts(self):
        return self._status['contacts'] if self._status else 0
    
    def can_move_forward(self):
        return self._status['canMoveForward'] == 1 if self._status else False

    def can_move_forward(self):
        return self._status['canMoveBackward'] == 1 if self._status else False
    
    def time(self):
        return self._time

    def elapsed(self):
        return self._time - self._reset_time

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
MAX_DISTANCE = 3.0
_ROBOT_MASS = 0.78
_ROBOT_DENSITY = _ROBOT_MASS / ROBOT_LENGTH / ROBOT_WIDTH
_ROBOT_FRICTION = 1
_ROBOT_RESTITUTION = 0
_SAFE_DISTANCE = 0.2

_ROBOT_TRACK = 0.136
_MAX_ACC = 1
_MAX_FORCE = _MAX_ACC * _ROBOT_MASS
_MAX_VELOCITY = 0.280

_VELOCITY_ITER = 10
_POSITION_ITER = 10

_RAD_10 = radians(10)
_RAD_30 = radians(30)

_SENSOR_GAP = 0.01

_FRONT_LEFT = [
    (_SENSOR_GAP, ROBOT_WIDTH / 2 + _SENSOR_GAP),
    (ROBOT_LENGTH / 2 + _SENSOR_GAP, ROBOT_WIDTH / 2 + _SENSOR_GAP),
    (ROBOT_LENGTH / 2 + _SENSOR_GAP, _SENSOR_GAP)
]
_FRONT_RIGHT = [
    (_SENSOR_GAP, -ROBOT_WIDTH / 2 - _SENSOR_GAP),
    (ROBOT_LENGTH / 2 + _SENSOR_GAP, -ROBOT_WIDTH / 2 - _SENSOR_GAP),
    (ROBOT_LENGTH / 2 + _SENSOR_GAP, -_SENSOR_GAP)
]
_REAR_LEFT = [
    (-_SENSOR_GAP, ROBOT_WIDTH / 2 + _SENSOR_GAP),
    (-ROBOT_LENGTH / 2 - _SENSOR_GAP, ROBOT_WIDTH / 2 + _SENSOR_GAP),
    (-ROBOT_LENGTH / 2 - _SENSOR_GAP, _SENSOR_GAP)
]
_REAR_RIGHT = [
    (-_SENSOR_GAP, -ROBOT_WIDTH / 2 - _SENSOR_GAP),
    (-ROBOT_LENGTH / 2 - _SENSOR_GAP, -ROBOT_WIDTH / 2 - _SENSOR_GAP),
    (-ROBOT_LENGTH / 2 - _SENSOR_GAP, -_SENSOR_GAP)
]

class SimRobot(RobotAPI, b2ContactListener):
    """Simulated robot"""
    
    def __init__(self, obstacles: ObstacleMap, err_sigma=0.05, err_sensor=0.05):
        """Create a simulated robot envinment"""
        RobotAPI.__init__(self)
        b2ContactListener.__init__(self)

        world: b2World = b2World(gravity=(0,0), doSleep=True,
            contactListener=self)
        robot: b2Body = world.CreateDynamicBody(position=(0,0))

        robot.CreatePolygonFixture(box=(ROBOT_WIDTH / 2, ROBOT_LENGTH/ 2),
            density=_ROBOT_DENSITY,
            friction=_ROBOT_FRICTION,
            restitution=_ROBOT_RESTITUTION)
        self._fl_sens = robot.CreateChainFixture(
            vertices=_FRONT_LEFT,
            isSensor=True)
        self._fr_sens = robot.CreateChainFixture(
            vertices=_FRONT_RIGHT,
            isSensor=True)
        self._rl_sens = robot.CreateChainFixture(
            vertices=_REAR_LEFT,
            isSensor=True)
        self._rr_sens = robot.CreateChainFixture(
            vertices=_REAR_RIGHT,
            isSensor=True)

        robot.angle = pi / 2

        for i in range(0, obstacles.num_obstacles()):
            pos = obstacles[i]
            s = obstacles.size() / 2
            world.CreateStaticBody(
                position=(pos[0], pos[1]),
                shapes=b2PolygonShape(box=(s,s)),
        )

        self._obstacles = obstacles
        self._speed = 0.0 # required speed
        self._left = 0 # real left motor speed
        self._right = 0 # real right motor speed
        self._direction = 0 #r equired directin
        self._sensor = 0 # sensor direction
        self._distance = 0.0 # sensor measured distance
        self._contacts = 0 # contact sensors value
        self._can_move_forward = True # move forward sensor
        self._can_move_backward = True # move backward sensor
        self._time = 0.0 # robot time
        self._reset_time = 0.0
        self._err_sigma = err_sigma
        self._err_sensor = err_sensor
        self.world = world
        self.robot = robot
    
    def robot_pos(self) -> b2Vec2:
        return self.robot.position
        
    def robot_dir(self) -> int:
        return normalizeDeg(round(90 - degrees(self.robot.angle)))
    
    def sensor_dir(self) -> int:
        return self._sensor
    
    def sensor_distance(self) -> float:
        return self._distance

    def contacts(self):
        return self._contacts

    def can_move_forward(self):
        return self._can_move_forward

    def can_move_backward(self):
        return self._can_move_backward

    def time(self):
        return self._time

    def elapsed(self):
        return self._time - self._reset_time

    def reset(self):
        self._speed = 0.0 # required speed
        self._left = 0 # real left motor speed
        self._right = 0 # real right motor speed
        self._direction = 0 #r equired directin
        self._sensor = 0 # sensor direction
        self._distance = 0.0 # sensor measured distance
        self._contacts = 0 # contact sensors value
        self._can_move_forward = True # move forward sensor
        self._can_move_backward = True # move backward sensor
        self._reset_time = self._time
        self.robot.position = (0,0)
        self.robot.linearVelocity = (0,0)
        self.robot.angle = pi / 2
        self.robot.angularVelocity = 0
        return self

    def start(self):
        return self

    def move(self, dir: int, speed: float):
        self._direction = dir
        self._speed = speed
        self._checkForSpeed()
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
            "y": self.robot_pos().y,
            "dir": self.robot_dir(),
            "sensor": self._sensor,
            "dist": self._distance,
            "left": self._left,
            "right": self._right,
            "contacts": self._contacts,
            "canMoveForward": 1 if self._can_move_forward else 0,
            "canMoveBackward": 1 if self._can_move_backward else 0
        }
    
    def obstaclesMap(self) -> ObstacleMap | None:
        return self._obstacles

    def tick(self, dt:float):
        self._controller(dt)
        self.world.Step(dt, _VELOCITY_ITER, _POSITION_ITER)
        self._time += dt
        robot_pos = self.robot_pos()
        sensor_deg = normalizeDeg(90 - self.robot_dir() - self._sensor)
        sensor_rad = radians(sensor_deg)
        _, dist = self._obstacles.nearest(location=np.array([robot_pos.x, robot_pos.y]),
            dir_rad=sensor_rad)
        dist = clip(dist - self._obstacles.size() / 2, 0, 3)
        self._distance = clip(dist + random.gauss(0, self._err_sensor), 0, MAX_DISTANCE) if dist < MAX_DISTANCE else 0.0
        self._can_move_forward = dist > _SAFE_DISTANCE and (self._contacts & 0xc) == 0
        self._can_move_backward = (self._contacts & 0x3) == 0
        self._checkForSpeed()
        return self

    def close(self):
        return self

    def _checkForSpeed(self):
        if (self._speed > 0 and self._left > 0 and self._right > 0 and not self._can_move_forward) \
            or (self._speed < 0 and self._left < 0 and self._right < 0 and not self._can_move_backward):
            self.halt()
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
        angular_velocity = (right - left) * _MAX_VELOCITY / _ROBOT_TRACK
        target_velocity = robot.GetWorldVector((forward_velocity, 0))
        dv = target_velocity - robot.linearVelocity
        dq = dv * robot.mass
        force = dq / dt
        local_force = robot.GetLocalVector(force)
        local_force *= 1 + random.gauss(0, self._err_sigma)
        if local_force.x > _MAX_FORCE:
            local_force.x = _MAX_FORCE
            force = robot.GetWorldVector(local_force)
        elif local_force.x < -_MAX_FORCE:
            local_force.x = -_MAX_FORCE
            force = robot.GetWorldVector(local_force)

        angular_impulse = (angular_velocity - robot.angularVelocity) * robot.inertia
        angular_impulse *= 1 + random.gauss(0, self._err_sigma)

        self.world.ClearForces()
        self.robot.ApplyForceToCenter(force=force, wake=True)
        self.robot.ApplyAngularImpulse(impulse = angular_impulse, wake=True)

    def _contactValue(self, contact: b2Contact):
        fa = contact.fixtureA
        fb = contact.fixtureB
        if fa == self._fl_sens or fb == self._fl_sens:
            return 8
        elif fa == self._fr_sens or fb == self._fr_sens:
            return 4
        elif fa == self._rl_sens or fb == self._rl_sens:
            return 2
        elif fa == self._rr_sens or fb == self._rr_sens:
            return 1
        return 0

    def BeginContact(self, contact: b2Contact):
        value = self._contactValue(contact=contact)
        self._contacts |= value

    def EndContact(self, contact):
        value = self._contactValue(contact=contact)
        self._contacts &= ~value
