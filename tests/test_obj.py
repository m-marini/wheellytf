from this import d
from numpy.testing import assert_approx_equal
from wheelly.objectives import fuzzy_stuck

def test_fuzzy_stuck():
    f = fuzzy_stuck(distances=(0.1, 0.3, 0.7, 2.0), sensor_range=90)
    # in target
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': 0,
        'dist': 0.5
    }), 1)
    # not in target, sensor too right
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': 90,
        'dist': 0.5
    }), 0)
    # not in target, sensor too left
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': -90,
        'dist': 0.5
    }), 0)
    # not in target, distance too short
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': 0,
        'dist': 0.1
    }), 0)
    # not in target, distance too far
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': 0,
        'dist': 2
    }), 0)

    # partial target, sensor left
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': -45,
        'dist': 0.5
    }), 0.5)

    # partial target, sensor right
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': 45,
        'dist': 0.5
    }), 0.5)

    # partial target, distance short
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': 0,
        'dist': 0.2
    }), 0.5)

    # partial target, distance far
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 1,
        'sensor': 0,
        'dist': (2 + 0.7) / 2
    }), 0.5)

    # contact
    assert_approx_equal(f({
        'canMoveForward': 0,
        'canMoveBackward': 1,
        'sensor': 45,
        'dist': 0.9
    }), -1)
    # contact
    assert_approx_equal(f({
        'canMoveForward': 1,
        'canMoveBackward': 0,
        'sensor': 45,
        'dist': 0.9
    }), -1)
