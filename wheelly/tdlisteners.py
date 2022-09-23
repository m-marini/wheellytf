from __future__ import annotations

from types import NoneType
from typing import Any, Callable

import numpy as np

from wheelly.tdagents import TDAgent
import os


class Kpi:
    @staticmethod
    def from_numpy(data: np.ndarray):
        n = np.size(a=data)
        x = np.arange(stop=n, dtype=np.float32)
        m_y = np.mean(data)
        std = float(np.std(data))
        lin_poly: np.polynomial.Polynomial = np.polynomial.Polynomial.fit(
            x, data, deg=1)
        lin_rms = float(np.std(data - lin_poly(x)))
        exp_poly = None
        exp_rms = None
        if (data > 0).all():
            exp_poly = np.polynomial.Polynomial = np.polynomial.Polynomial.fit(
                x, np.log(data), deg=1)
            exp_rms = float(np.std(data - np.exp(exp_poly(x))))

        return Kpi(num_samples=n,
                   mean=m_y,
                   std=std,
                   lin_poly=lin_poly,
                   lin_rms=lin_rms,
                   exp_poly=exp_poly,
                   exp_rms=exp_rms)

    def __init__(self,
                 num_samples: int,
                 mean: float,
                 std: float,
                 lin_poly: np.polynomial.Polynomial,
                 lin_rms: float,
                 exp_poly: np.polynomial.Polynomial | NoneType,
                 exp_rms:  float | NoneType = None):
        self.num_samples = num_samples
        self.avg = mean
        self.std = std
        self.lin_poly = lin_poly
        self.lin_rms = lin_rms
        self.exp_poly = exp_poly
        self.exp_rms = exp_rms


class DiscountConsumer:
    def __init__(self, discount: float) -> None:
        self._kpi = None
        self._discount = discount

    def __call__(self, x: np.ndarray | NoneType):
        if x is not None:
            if self._kpi is None:
                self._kpi = np.zeros_like(x)
            self._kpi = (self._kpi - x) * self._discount + x

    def kpi(self):
        return self._kpi


class CsvConsumer:
    def __init__(self, fname: str, buffer_size: int = 100) -> None:
        self.fname = fname
        self.buffer = None
        self.count = 0
        self._size = buffer_size
        if os.path.exists(fname):
            os.remove(fname)

    def __call__(self, x: np.ndarray | NoneType):
        if x is not None:
            if self.buffer is None:
                self.buffer = np.zeros((self._size, x.size))
            self.buffer[self.count] = x
            self.count = self.count + 1
            if self.count >= self._size:
                self.flush()

    def flush(self):
        if self.count > 0:
            with open(self.fname, "at") as f:
                np.savetxt(f, self.buffer[0: self.count, :])
        self.count = 0


class DataCollectorConsumer:
    def __init__(self):
        self._data = []

    def __call__(self, x: np.ndarray | NoneType):
        if x is not None:
            self._data.append(x)

    def data(self):
        return np.stack(arrays=self._data, axis=0)

    def to_csv(self, fname: str):
        np.savetxt(fname=fname, X=self.data(), delimiter=",")

    def kpi(self):
        return Kpi.from_numpy(self.data())


class KpiListenerBuilder:
    @staticmethod
    def any():
        return KpiListenerBuilder(func=lambda x: x)

    @staticmethod
    def getter(key: str):
        return KpiListenerBuilder(func=lambda x: x.get(key, None))

    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    def filter(self, filter: Callable[[Any], bool]):
        return self.map(mapper=lambda x: x if filter(x) else None)

    def map(self, mapper: Callable[[Any], Any]):
        def map(x: Any):
            y = self.func(x)
            return mapper(y) if y is not None else None

        return KpiListenerBuilder(func=map)

    def get(self, key: str):
        return self.map(lambda x: x.get(key, None))

    def build(self, consumer: Callable[[np.ndarray], None]):

        def listener(x: Any):
            value = self.func(x)
            if value is not None:
                if isinstance(value, float | int):
                    consumer(np.array(value))
                else:
                    consumer(value.numpy().flatten())

        return listener

    def register(self, agent: TDAgent, consumers: Callable[[np.ndarray], None]):
        agent.add_kpi_listener(self.build(consumers))
        return consumers
