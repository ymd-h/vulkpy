"""
Neural Network Module (:mod:`vulkpy.nn`)
========================================
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

from .vkarray import GPU, Array, DataShape, BatchParams
from .util import getShader


class Parameter:
    """
    Neural Network Parameter
    """
    def __init__(self,
                 gpu: GPU,
                 shape: Iterable[int],
                 trainable: bool = True,
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


class Dense(Module):
    """
    Dense
    """
    _batch_affine = getShader("batch_affine.spv")

    def __init__(self, gpu: GPU, input_dim: int, output_dim: int):
        super().__init__(gpu)

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.w = Parameter(gpu, shape=(self.output_dim, self.input_dim))
        self.b = Parameter(gpu, shape=(self.output_dim,))

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
        y.job = self.w._gpu._submit(self._batch_affine, 1, 64, 1,
                                    [self.w.value, self.b.value, x, y],
                                    DataShape(x.shape[0], self.output_dim, 1),
                                    BatchParams(x.shape[0],
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
        dy.reshape(dy.shape[0], dy.shape[1], 1)
        self._x.reshape((self._x.shape[0], 1, self._x.shape[1]))

        dW = dy * self._x # Allocate
        dW = dW.sum(axis=0) # Allocate
        self.w.add_grad(dW)

        self._x.reshape(x_shape)
        dy.reshape(dy_shape)

        return dy @ self.w # Allocate


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
        X = x - x.max(axis=1, rebroadcast=True)
        X.exp(inplace=True)
        X /= X.sum(axis=1, rebroadcast=True)
        return X

    def backward(self, dy: Array) -> Array:
        pass


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
