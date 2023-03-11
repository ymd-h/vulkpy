from __future__ import annotations
from typing import Callable, Iterable, Optional

from vulkpy.vkarray import GPU, Array, zeros
from .core import Optimizer, OptimizerState
from .optimizers import Adam


__all__ = [
    "Parameter"
]


class Parameter:
    """
    Neural Network Parameter

    Attributes
    ----------
    value : vulkpy.Array
        Parameter value
    grad : vulkpy.Array, optional
        Accumulated gradients if parameter is trainable

    Methods
    -------
    is_trainable()
        Whether this parameter is trainable.
    add_grad(grad)
        Add gradients.
    zero_grad()
        Reset accumulated gradients to 0.
    update()
        Update parameter based on accumulated gradients.
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
        opt : vulkpy.nn.Optimizer, optional
            Optimizer. If ``None`` (default), ``vulkpy.nn.Adam`` is used.
        initializer : callable, optional
            Initializer function. If ``None`` (default), initialized with ``0.0``.
        """
        if initializer is None:
            initializer = zeros
        self.value: Array = initializer(gpu, shape)

        self.grad: Optional[Array] = None
        self.opt_state: Optional[OptimizerState] = None
        if trainable:
            self.grad = zeros(gpu, shape=shape)

            if opt is None:
                opt = Adam(gpu)
            self.opt_state = opt.init_state(shape)

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
        if self.is_trainable():
            self.grad += grad

    def zero_grad(self):
        """
        Clear gradient to 0.0
        """
        if self.is_trainable():
            self.grad[:] = 0.0

    def update(self):
        """
        Update value

        Update value with accumulated gradients only if this value is trainable.
        """
        if self.is_trainable():
            self.value += self.opt_state.grad2diff(self.grad)
