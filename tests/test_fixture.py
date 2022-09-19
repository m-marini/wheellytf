import pytest
import tensorflow as tf

from tests.fixtures import generate_case, random_cases, random_tensor


@pytest.fixture
def cases():
    return random_cases(spec=dict(
        a=dict(
            type="uniform",
            shape=(10,)
        )
    ))


def test_tensor():
    random = tf.random.Generator.from_seed(1234)
    spec = dict(type="normal",
                shape=(10,))
    x = random_tensor(random=random, spec=spec)
    assert True


def test_case():
    random = tf.random.Generator.from_seed(1234)
    spec = dict(
        a=dict(type="normal",
               shape=(10,)),
        b=dict(type="uniform",
               shape=(10,)),
        c=dict(type="exp",
               shape=(10,),
               minval=0.1,
               maxval=10),
        d = dict(type="func",
               func=lambda random: random.uniform(shape=(1,))))
    x = generate_case(spec=spec, case_num=0, random=random)
    assert "a" in x
    assert "b" in x
    assert "c" in x
    assert "d" in x


def test_fix(cases):
    assert len(cases) == 30
