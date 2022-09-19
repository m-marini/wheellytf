from math import log
from typing import Any

import tensorflow as tf


def random_tensor(random: tf.random.Generator, spec: dict[str, Any]) -> Any:
    type = spec["type"]
    if type == "uniform":
        return random.uniform(shape=spec["shape"],
                              minval=spec.get("minval", 0),
                              maxval=spec.get("maxval", 1))
    elif type == "normal":
        return random.normal(shape=spec["shape"],
                             mean=spec.get("mean", 0),
                             stddev=spec.get("stddev", 1))
    elif type == "exp":
        return tf.exp(random.uniform(shape=spec["shape"],
                                     minval=log(spec["minval"]),
                                     maxval=log(spec["maxval"])))
    elif type == "func":
        return spec["func"](random)
    else:
        raise Exception(f'Wrong type "{type}"')


def generate_case(spec: dict[str, dict[str, Any]],
                  case_num: int,
                  random: tf.random.Generator) -> dict[str, Any]:
    case = {key: random_tensor(random, case_spec)
            for key, case_spec in spec.items()}
    case["case_num"] = case_num
    return case


def random_cases(spec: dict[str, dict[str, Any]],
                 seed: int = 1234,
                 num_test: int = 30) -> list[dict[str, Any]]:
    random = tf.random.Generator.from_seed(seed=seed)
    cases = [generate_case(spec=spec,
                           case_num=i,
                           random=random)
             for i in range(num_test)]
    return cases
