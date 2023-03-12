from __future__ import annotations
from typing import Iterable

from vulkpy.vkarray import GPU, Array


__all__ = [
    "OptimizerState",
    "Optimizer",
    "Regularizer",
    "Loss",
    "Module",
]


class OptimizerState:
    def grad2diff(self, grad: Array) -> Array:
        raise NotImplementedError

class Optimizer:
    def init_state(self, shape: Iterable[int]) -> OptimizerState:
        raise NotImplementedError

class Loss:
    def __call__(self, x: Array, y: Array) -> Array:
        raise NotImplementedError

    def grad(self) -> Array:
        raise NotImplementedError

class Regularizer:
    def loss(self, param: Array) -> Array:
        raise NotImplementedError

    def grad(self, param: Array) -> Array:
        raise NotImplementedError


class Module:
    def __init__(self):
        pass

    def __call__(self, x: Array) -> Array:
        """
        Call Module

        Parameters
        ----------
        x : vulkpy.Array
            Input

        Returns
        -------
        y : vulkpy.Array
            Output

        Raises
        ------
        ValueError
            If input (``x``) shape doesn't have at least 2-dimensions.

        Notes
        -----
        This function stores input (``x``) and output (``y``) for training.
        """
        if len(x.shape) < 2:
            raise ValueError("Input must have at least 2-dimensions.")

        self._x = x
        self._y = self.forward(x)
        return self._y

    def forward(self, x: Array) -> Array:
        raise NotImplementedError

    def backward(self, dy: Array) -> Array:
        raise NotImplementedError

    def zero_grad(self):
        """
        Reset accumulated gradients to 0.
        """
        pass

    def update(self):
        """
        Update parameters based on accumulated gradients
        """
        pass
