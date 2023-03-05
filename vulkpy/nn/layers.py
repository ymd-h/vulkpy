from __future__ import annotations
from typing import Callable, Iterable, Optional

from vulkpy.util import getShader
from vulkpy.vkarray import GPU, Array, DataShape, BatchAffineParams
from .optimizers import Optimizer
from .core import Parameter, Module
from .initializers import HeNormal


__all__ = ["Dense", "ReLU", "Sigmoid", "Softmax"]


class Dense(Module):
    """
    Dense
    """
    _batch_affine = getShader("batch_affine.spv")

    def __init__(self, gpu: GPU, input_dim: int, output_dim: int, *,
                 w_init: Optional[Callable[[GPU, Iterable[int]], Array]] = None,
                 b_init: Optional[Callable[[GPU, Iterable[int]], Array]] = None,
                 w_opt: Optional[Optimizer] = None,
                 b_opt: Optional[Optimizer] = None):
        """
        Initialize Dense

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU
        input_dim, output_dim : int
            Input / output dimension
        w_init, b_init : Callable, optional
            Weight / bias initializer.
        """
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        if w_init is None:
            w_init = HeNormal(gpu, self.input_dim)

        self.w = Parameter(gpu, shape=(self.output_dim, self.input_dim),
                           initializer=w_init, opt=w_opt)
        self.b = Parameter(gpu, shape=(self.output_dim,),
                           initializer=b_init, opt=b_opt)

    def forward(self, x: Array) -> Array:
        """
        Forward

        Parameters
        ----------
        x : vulkpy.Array
            Batch input

        Returns
        -------
        vulkpy.Array
            Batch output

        Notes
        -----
        .. math:: y = Wx + b
        """
        y = Array(x._gpu, shape=(x.shape[0], self.output_dim))
        y.job = x._gpu._submit(self._batch_affine, 1, 64, 1,
                               [self.w.value, self.b.value, x, y],
                               DataShape(x.shape[0], self.output_dim, 1),
                               BatchAffineParams(x.shape[0],
                                                 x.shape[1],
                                                 self.output_dim))
        y._keep.extend([self.w.value, self.b.value, x])
        return y

    def backward(self, dy: Array) -> Array:
        """
        Backward

        Parameters
        ----------
        dy : vulkpy.Array
            Batch grad

        Returns
        -------
        vulkpy.Array
            Batch grad

        Notes
        -----
        .. math::

            dx = dy @ W\\
            dW = dy ^T \cdot x\\
            db = dy
        """
        db = dy.sum(axis=0) # Allocate
        self.b.add_grad(db)

        x_shape = self._x.shape
        dy_shape = dy.shape
        dy.reshape((dy.shape[0], dy.shape[1], 1))
        self._x.reshape((self._x.shape[0], 1, self._x.shape[1]))

        dW = dy * self._x # Allocate
        dW = dW.sum(axis=0) # Allocate
        self.w.add_grad(dW)

        self._x.reshape(x_shape)
        dy.reshape(dy_shape)

        return dy @ self.w.value # Allocate

    def zero_grad(self):
        self.w.zero_grad()
        self.b.zero_grad()

    def update(self):
        self.w.update()
        self.b.update()


class ReLU(Module):
    """
    Rectified Linear Unit (ReLU)
    """
    def forward(self, x: Array) -> Array:
        """
        Forward

        Parameters
        ----------
        x : vulkpy.Array
            Batch input

        Returns
        -------
        vulkpy.Array
            Batch output

        Notes
        -----
        .. math:: y = \max(x, 0)
        """
        return x.max(0.0) # Allocate

    def backward(self, dy: Array) -> Array:
        """
        Backward

        Parameters
        ----------
        dy : vulkpy.Array
            Batch grad

        Returns
        -------
        vulkpy.Array
            Batch grad

        Notes
        -----
        .. math:: dx = dy * \max(sign(y), 0)

        if x == 0, dy/dx => 0
        """
        dx = self._y.sign() # Allocate
        dx.max(0.0, inplace=True)
        dx *= dy
        return dx


class Sigmoid(Module):
    """
    Sigmoid
    """
    def forward(self, x: Array) -> Array:
        """
        Forward

        Parameters
        ----------
        x : vulkpy.Array
            Batch input

        Returns
        -------
        vulkpy.Array
            Batch output

        Notes
        -----
        .. math:: y = 1/(1 + \exp (-x))
        """
        y = 0.0 - x # Allocate
        y.exp(inplace=True)
        y += 1.0
        y = 1.0 / y # Allocate
        return y

    def backward(self, dy: Array) -> Array:
        """
        Backward

        Parameters
        ----------
        dy : vulkpy.Array
            Batch grad

        Returns
        -------
        vulkpy.Array
            Batch grad

        Notes
        -----
        .. math:: dx = dy \times y(1 - y)
        """
        dx = 1.0 - self._y
        dx *= self._y
        dx *= dy
        return dx


class Softmax(Module):
    """
    SoftMax
    """
    def forward(self, x: Array) -> Array:
        """
        Forward

        Parameters
        ----------
        x : vulkpy.Array
            Batch input

        Returns
        -------
        vulkpy.Array
            Batch output

        Notes
        -----
        .. math:: y = \exp (x) / \sum _i \exp(x_i)
        """
        X = x - x.maximum(axis=1, rebroadcast=True)
        X.exp(inplace=True)
        X /= X.sum(axis=1, rebroadcast=True)
        return X

    def backward(self, dy: Array) -> Array:
        """
        Backward

        Parameters
        ----------
        dy : vulkpy.Array
            Batch grad

        Returns
        -------
        vulkpy.Array
            Batch grad

        Notes
        -----
        .. math:: dx = dy \times y(1 - y)
        """
        dx = 1.0 - self._y
        dx *= self._y
        dx *= dy
        return dx
