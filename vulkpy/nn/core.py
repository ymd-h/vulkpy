from __future__ import annotations
from typing import Callable, Iterable, Optional

from vulkpy.vkarray import GPU, Array
from .optimizers import Optimizer, Adam


__all__ = ["Parameter", "Module"]


class Parameter:
    """
    Neural Network Parameter
    """
    def __init__(self,
                 gpu: GPU,
                 shape: Iterable[int],
                 trainable: bool = True,
                 opt: Optional[Optimizer] = None,
                 initializer: Optional[Callable[[GPU, Iterable[int]], Array]]=None):
        """
        Initialize Parameter

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU
        shape : iterable of ints
            Shape of parameter
        trainable : bool, optional
            If ``True`` (default), track gradient
        initializer : callable, optional
            Initializer function. If ``None`` (default), initialized with ``0.0``.
        """
        if initializer is None:
            self.value = Array(gpu, shape=shape)
            self.value[:] = 0.0
        else:
            self.value = initializer(gpu, shape=shape)

        if trainable:
            self.grad = Array(gpu, shape=shape)
            self.grad[:] = 0.0

            if opt is None:
                opt = Adam(gpu)
            self.opt = opt
            self.opt_state = self.opt.init_state(shape)
        else:
            self.grad = None

    def is_trainable(self) -> bool:
        """
        Whether this parameter is trainable

        Returns
        -------
        bool
            Is trainable
        """
        return self.grad is not None

    def add_grad(self, grad: Array):
        """
        Add gradient

        Parameters
        ----------
        grad : vulkpy.Array
            Gradient to be accumulated
        """
        self.grad += grad

    def zero_grad(self):
        """
        Clear gradient to 0.0
        """
        self.grad[:] = 0.0

    def update(self):
        """
        Update value
        """
        if self.is_trainable():
            self.value += self.opt(self.grad, self.opt_state)

class Module:
    def __init__(self):
        pass

    def __call__(self, x: Array) -> Array:
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
        pass

    def update(self):
        pass
