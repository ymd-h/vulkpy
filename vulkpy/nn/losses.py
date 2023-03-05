from __future__ import annotations
from typing import Literal

from vulkpy.util import getShader
from vulkpy.vkarray import Array, DataShape, VectorParams
from .layers import Softmax

__all__ = [
    "CrossEntropyLoss",
    "SoftmaxCrossEntropyLoss",
    "MSELoss",
    "HuberLoss",
]

class Loss:
    def __init__(self, reduce: Literal["mean", "sum"] = "mean"):
        self.reduce, self.scale_backward = {
            "mean": (lambda _L: _L.mean(axis=0), lambda _dx: 1/_dx.shape[0]),
            "sum": (lambda _L: _L.sum(axis=0), None),
        }[reduce]

    def __call__(self, x: Array, y: Array) -> Array:
        self._x = x
        self._y = y
        L = self.forward(x, y)
        return self.reduce(L)

    def grad(self):
        dx = self.backward()
        if self.scale_backward is not None:
            dx *= self.scale_backward(dx)
        return dx

    def forward(self, x: Array, y: Array) -> Array:
        raise NotImplementedError

    def backward(self) -> Array:
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """
    Cross Entropy Loss
    """
    _forward = getShader("nn_cross_entropy.spv")
    _backward = getShader("nn_cross_entropy_backward.spv")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, y: Array):
        size = x.buffer.size()
        L = Array(x._gpu, shape=x.shape)
        L.job = x._gpu._submit(self._forward, 64, 1, 1,
                               [x, y, L],
                               DataShape(size, 1, 1),
                               VectorParams(size))
        L._keep.extend([x, y])
        return L.sum(axis=1)

    def backward(self):
        size = self._x.buffer.size()
        dx = Array(self._x._gpu, shape=self._x.shape)
        dx.job = self._x._gpu._submit(self._backward, 64, 1, 1,
                                      [self._x, self._y, dx],
                                      DataShape(size, 1, 1),
                                      VectorParams(size))
        dx._keep.extend([self._x, self._y])
        return dx


class SoftmaxCrossEntropyLoss(CrossEntropyLoss):
    """
    Softmax Cross Entropy Loss

    See Also
    --------
    Softmax : Softmax layer
    CrossEntropyLoss : Cross Entropy loss without Softmax
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sm = Softmax()

    def forward(self, x: Array, y: Array) -> Array:
        return super().forward(self._sm(x), y)

    def backward(self) -> Array:
        return self._sm._y - self._y


class MSELoss(Loss):
    """
    Mean Squared Loss
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, y: Array) -> Array:
        L = (y - x)          # Allocate
        L **= 2.0
        return L.sum(axis=1) # Allocate

    def backward(self) -> Array:
        dx = self._x - self._y # Allocate
        dx *= 2
        return dx


class HuberLoss(Loss):
    """
    Huber Loss
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, y: Array) -> Array:
        delta = y - x # Allocate
        delta.abs(inplace=True)               # |y-x|
        delta.min(delta ** 2.0, inplace=True) # min(|y-x|^2, |y-x|)
        delta *= 0.5                          # min(|y-x|^2, |y-x|) * 0.5
        return delta.sum(axis=1) # Allocate

    def backward(self) -> Array:
        delta = self._x - self._y
        delta.clamp(-1.0, 1.0, inplace=True)
        return delta
