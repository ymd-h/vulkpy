from __future__ import annotations
from typing import Literal

from vulkpy.vkarray import Array
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
        self._L = self.forward(x, y)
        return self.reduce(self._L)

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

    Notes
    -----
    .. math:: L = - \sum _i y_i \log x_i
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, y: Array):
        L = x + 1e-8          #          x+eps  # Allocate
        L.log(inplace=True)   #      log(x+eps)
        L *= y                #  y * log(x+eps)
        L *= -1.0             # -y * log(x+eps)
        return L.sum(axis=1)

    def backward(self):
        dx = self._x + 1e-8 #       x+eps  # Allocate
        dx *= -1.0          # -    (x+eps)
        dx = self._y / dx   # -y / (x+eps) # Allocate
        return dx


class SoftmaxCrossEntropyLoss(CrossEntropyLoss):
    """
    Softmax Cross Entropy Loss

    Notes
    -----
    This loss includes Softmax layer to compute gradient efficiently.

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, y: Array) -> Array:
        L = (y - x) # Allocate
        L **= 2.0
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
        delta.abs(inplace=True)               # |y-x|
        delta.min(delta ** 2.0, inplace=True) # min(|y-x|^2, |y-x|)
        delta *= 0.5                          # 0.5 * min(|y-x|^2, |y-x|)
        return delta

    def backward(self) -> Array:
        delta = self._x - self._y
        delta.clamp(-1.0, 1.0, inplace=True)
        return delta
