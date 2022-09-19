from math import cos, pi, sin

import numpy as np
import quaternion
from numpy.testing import assert_almost_equal, assert_equal


def test_hamilton():
    one = np.quaternion(1, 0, 0, 0)
    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    assert_equal(one * one, one)

    assert_equal(i*i, -one)
    assert_equal(j*j, -one)
    assert_equal(k*k, -one)

    assert_equal(i*j, k)
    assert_equal(j*i, -k)

    assert_equal(j*k, i)
    assert_equal(k*j, -i)

    assert_equal(k*i, j)
    assert_equal(i*k, -j)


def test_vect():
    v = np.array((1, 2, 3))
    q = quaternion.from_vector_part(v)
    assert_equal(q, np.quaternion(0, 1, 2, 3))


def test_conj():
    q = np.quaternion(1, 2, 3, 4)
    assert_equal(np.conjugate(q), np.quaternion(1, -2, -3, -4))


def test_conj():
    q = np.quaternion(1, 2, 3, 4)
    assert_equal(np.conjugate(q), np.quaternion(1, -2, -3, -4))


def test_asRot0():
    q = np.quaternion(1, 0, 0, 0)
    v = quaternion.as_rotation_vector(q)
    assert_equal(v, np.array((0, 0, 0)))


def test_asRotx():
    q = np.quaternion(cos(pi/4), sin(pi/4), 0, 0)
    v = quaternion.as_rotation_vector(q)
    assert_almost_equal(v, np.array((pi/2, 0, 0)))

def test_asRoty():
    q = np.quaternion(cos(pi/4), 0, sin(pi/4), 0)
    v = quaternion.as_rotation_vector(q)
    assert_almost_equal(v, np.array((0, pi/2, 0)))

def test_asRotz():
    q = np.quaternion(cos(pi/4), 0, 0, sin(pi/4))
    v = quaternion.as_rotation_vector(q)
    assert_almost_equal(v, np.array((0, 0, pi/2)))

def test_fromRot():
    v = np.array((1,0,0)) * (pi/2)
    q = quaternion.from_rotation_vector(v)
    assert_almost_equal(q, np.quaternion(cos(pi/4), sin(pi/4), 0, 0))

def test_trotx():
    vrot = np.array((1,0,0)) * (pi/2)
    q = quaternion.from_rotation_vector(vrot)
    qvy = np.quaternion(0, 0, 1, 0)
    qvt = q * qvy * q.conjugate()
    assert_almost_equal(qvt, np.quaternion(0, 0, 0, 1))

def test_tinvrotx():
    vrot = np.array((1,0,0)) * (pi/2)
    q = quaternion.from_rotation_vector(vrot)
    invq = q.inverse()
    qvy = np.quaternion(0, 0, 0, 1)
    qvt = invq * qvy * invq.conjugate()
    assert_almost_equal(qvt, np.quaternion(0, 0, 1, 0))
