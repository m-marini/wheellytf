import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from Box2D import b2World
from Box2D.b2 import (circleShape, contactListener, edgeShape, fixtureDef,
                      polygonShape, revoluteJointDef)
from gym.utils import EzPickle, colorize
from numpy.testing import assert_allclose, assert_equal

#from gym.utils.step_api_compatibility import step_api_compatibility

ROBOT_WIDTH = 0.18
ROBOT_DEPTH = 0.26
ROBOT_MASS = 0.78
ROBOT_DENSITY = ROBOT_MASS / ROBOT_DEPTH / ROBOT_WIDTH
ROBOT_FRICTION = 0.3
TIME_STEP = 0.3
FORCE = 4

def test_box2d():
    world = b2World(gravity=(0,0), doSleep=True)
    robot = world.CreateDynamicBody(position=(0,0))
    box = robot.CreatePolygonFixture(box=(ROBOT_WIDTH / 2, ROBOT_DEPTH/ 2), density=ROBOT_DENSITY, friction=ROBOT_FRICTION)
    assert_allclose(robot.mass, ROBOT_MASS)
    vel_iters, pos_iters = 10, 10

    robot.ApplyForce(force=(0, FORCE), point=robot.position, wake=True)

    world.Step(TIME_STEP, vel_iters, pos_iters)

    # Clear applied body forces. We didn't apply any forces, but you
    # should know about this function.
    world.ClearForces()
 
    # Now print the position and angle of the body.

    assert_allclose(robot.linearVelocity, (0, FORCE * TIME_STEP / ROBOT_MASS))
    assert_allclose(robot.position, (0, FORCE * TIME_STEP * TIME_STEP / ROBOT_MASS), rtol=1e-6)
    assert_allclose( robot.angle, 0)
