from __future__ import annotations

from vulkpy.vkarray import Array
from .layers import Softmax

class Loss:
    def __call__(self, x: Array, y: Array):
        self._L = self.forward(x, y)
        return self._L

    def forward(self, x: Array, y: Array):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, reduce="mean"):
        self._sm = Softmax()

    def forward(self, x: Array):
        return self._sm(x) * y

    def backward(self):
        return self.sm._y - y
