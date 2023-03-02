from __future__ import annotations
from typing import Literal

from vulkpy.vkarray import Array
from .layers import Softmax

__all__ = [
    "SoftmaxCrossEntropyLoss",
    "MSELoss",
    "HuberLoss",
]

class Loss:
    def __init__(self, reduce: Literal["mean", "sum"] = "mean"):
        self.reduce = {
            "mean": lambda _L: _L.mean(axis=0),
            "sum": lambda _L: _L.sum(axis=0),
        }[reduce]

    def __call__(self, x: Array, y: Array) -> Array:
        self._x = x
        self._y = y
        self._L = self.forward(x, y)
        return self.reduce(self._L)

    def forward(self, x: Array, y: Array) -> Array:
        raise NotImplementedError

    def backward(self) -> Array:
        raise NotImplementedError


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sm = Softmax()

    def forward(self, x: Array, y: Array) -> Array:
        return self._sm(x) * y

    def backward(self) -> Array:
        return self._sm._y - self._y


class MSELoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, y: Array) -> Array:
        L = (y - x) # Allocate
        L.square(inplace=True)
        return L

    def backward(self) -> Array:
        dx = self._x - self._y # Allocate
        dx *= 2
        return dx


class HuberLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, y: Array) -> Array:
        delta = y - x # Allocate
        delta.abs(inplace=True)                 # |y-x|
        delta.min(delta.square(), inplace=True) # min(|y-x|^2, |y-x|)
        delta *= 0.5                            # 0.5 * min(|y-x|^2, |y-x|)
        return delta

    def backward(self) -> Array:
        delta = self._x - self._y
        delta.clamp(-1.0, 1.0, inplace=True)
        return delta
