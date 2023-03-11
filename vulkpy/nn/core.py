from __future__ import annotations
from typing import Callable, Literal, Optional, Iterable, Tuple

from vulkpy.vkarray import GPU, Array


__all__ = [
    "OptimizerState",
    "Optimizer",
    "Module",
    "Loss",
]


class OptimizerState:
    def grad2diff(self, grad: Array) -> Array:
        raise NotImplementedError

class Optimizer:
    def init_state(self, shape: Iterable[int]) -> OptimizerState:
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

F = Callable[[Array], Array]
class Loss:
    def __init__(self, reduce: Literal["mean", "sum"] = "mean"):
        tmp: Tuple[F, Optional[F]] = {
            "mean": (lambda _L: _L.mean(axis=0), lambda _dx: 1/_dx.shape[0]),
            "sum": (lambda _L: _L.sum(axis=0), None),
        }[reduce]
        self.reduce, self.scale_backward = tmp


    def __call__(self, x: Array, y: Array) -> Array:
        r"""
        Compute Loss

        Parameters
        ----------
        x : vulkpy.Array
            Batch input features
        y : vulkpy.Array
            Batch labels/targets

        Returns
        -------
        loss : vulkpy.Array
            Loss
        """
        self._x = x
        self._y = y
        L = self.forward(x, y)
        return self.reduce(L)

    def grad(self) -> Array:
        r"""
        Compute Gradients

        Returns
        -------
        dx : vulkpy.Array
            Batch gradients of dL/dx

        Notes
        -----
        This method calculates gradients for the last ``__call__(x, y)``.
        """
        dx = self.backward()
        if self.scale_backward is not None:
            dx *= self.scale_backward(dx)
        return dx

    def forward(self, x: Array, y: Array) -> Array:
        raise NotImplementedError

    def backward(self) -> Array:
        raise NotImplementedError
