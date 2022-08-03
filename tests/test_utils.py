
from wheelly.utils import defuzzy, fuzzy_and, fuzzy_neg, fuzzy_not, fuzzy_or, fuzzy_pos, fuzzy_range

def test_fuzzy_pos():
    assert fuzzy_pos(-1, 2) == 0
    assert fuzzy_pos(0, 2) == 0
    assert fuzzy_pos(1, 2) == 0.5
    assert fuzzy_pos(2, 2) == 1
    assert fuzzy_pos(3, 2) == 1

def test_fuzzy_neg():
    assert fuzzy_neg(-3, 2) == 1
    assert fuzzy_neg(-2, 2) == 1
    assert fuzzy_neg(-1, 2) == 0.5
    assert fuzzy_neg(0, 2) == 0
    assert fuzzy_neg(1, 2) == 0

def test_fuzzy_not():
    assert fuzzy_not(0) == 1
    assert fuzzy_not(0.25) == 0.75
    assert fuzzy_not(0.75) == 0.25
    assert fuzzy_not(1) == 0

def test_fuzzy_and():
    assert fuzzy_and(0, 1) == 0
    assert fuzzy_and(0.25, 1) == 0.25
 
def test_fuzzy_or():
    assert fuzzy_or(0, 1) == 1
    assert fuzzy_or(0, 0.25) == 0.25

def test_defuzzy():
    assert defuzzy((10., 1.)) == 10
    assert defuzzy((10, 0)) == 0
    assert defuzzy((10, 0.5)) == 5

    assert defuzzy((10, 1), default_value=20) == 10
    assert defuzzy((10, 0), default_value=20) == 20
    assert defuzzy((10, 0.5), default_value=20) == 15

    assert defuzzy((10, 0.2), (20, 0.4)) == (10 * 0.2 + 20 * 0.4) / (0.2 + 0.4 + 0.6)

def test_fuzzy_range():
    assert fuzzy_range(0, (1, 3, 5, 9)) == 0
    assert fuzzy_range(1, (1, 3, 5, 9)) == 0
    assert fuzzy_range(2, (1, 3, 5, 9)) == 0.5
    assert fuzzy_range(3, (1, 3, 5, 9)) == 1
    assert fuzzy_range(4, (1, 3, 5, 9)) == 1
    assert fuzzy_range(5, (1, 3, 5, 9)) == 1
    assert fuzzy_range(7, (1, 3, 5, 9)) == 0.5
    assert fuzzy_range(9, (1, 3, 5, 9)) == 0
    assert fuzzy_range(10, (1, 3, 5, 9)) == 0
