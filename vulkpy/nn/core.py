"""
Neural Network Core Module (:mod:`vulkpy.nn.core`)
==================================================

This module provides abstract base classes for Neural Network.
"""
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
    """
    Abstract base class for Optimizer State

    See Also
    --------
    vulkpy.nn.Optimizer : Optimizer
    vulkpy.nn.SGDState : OptimizerState subclass for SGD
    vulkpy.nn.AdamState : OptimizerState subclass for Adam

    Notes
    -----
    Mutable per-parameter values are stored at this class instance,
    although static global parameters (e.g. learning rate) are
    stored at ``Optimizer`` class.

    Subclass of ``OptimizerState`` should implement ``Optimizer.grad2diff()``,
    which takes accumulated gradients and returns update difference.

    In standard design, ``OptimizerState`` holds a reference to
    its parent ``Optimizer`` in order to access global parameters.
    """
    def grad2diff(self, grad: Array) -> Array:
        """
        Compute update diff from gradient

        Parameters
        ----------
        grad : vulkpy.Array
            Accumulated gradient

        Returns
        -------
        diff : vulkpy.Array
            Update diff. (``v += opt_state.grad2diff(grad)``)

        Notes
        -----
        Subclass must implement this method.
        """
        raise NotImplementedError

class Optimizer:
    """
    Abstract base class for Optimizer

    See Also
    --------
    vulkpy.nn.OptimizerState : Optimizer State
    vulkpy.nn.SGD : Optimizer subclass for SGD
    vulkpy.nn.Adam : Optimizer subclass for Adam

    Notes
    -----
    ``Optimizer`` class is designed to pass to ``Parameter`` constructor
    through ``Module`` constructor.
    Inside ``Parameter`` constructor, ``Optimizer.init_state()`` is called and
    corresponding ``OptimizerState`` are stored at the ``Parameter`` instance.

    Mutable per-parameter values are stored at ``OptimizerState`` class instance,
    although static global parameters (e.g. learning rate) are
    stored at this class.

    To implement specific optimizer, Subclass of ``Optimizer`` should implement
    ``Optimizer.init_state()`` method, which returns corresponding subclass of
    ``OptimizerState``.

    Examples
    --------
    >>> import vulkpy as vk
    >>> gpu = vk.GPU()
    >>>
    >>> adam = vk.nn.Adam(gpu) # Optimizer
    >>> dense = vk.nn.Dense(gpu, 1, 1, w_opt=adam, b_opt=adam) # Module
    """
    def init_state(self, shape: Iterable[int]) -> OptimizerState:
        """
        Create OptimizerState

        Parameters
        ----------
        shape : iterable of ints
            Parameter Shape

        Returns
        -------
        opt_state : vulkpy.nn.OptimizerState
            Optimizer State

        Notes
        -----
        Subclass must implement this method.
        """
        raise NotImplementedError

class Loss:
    """
    Abstract base class for Loss

    See Also
    --------
    vulkpy.nn.CrossEntropyLoss : Cross Entropy Loss
    vulkpy.nn.SoftmaxCrossEntropyLoss : Softmax Cross Entropy Loss
    vulkpy.nn.HuberLoss : Huber Loss
    vulkpy.nn.MSELoss : MSE Loss
    vulkpy.nn.MixLoss : Mixing Loss

    Notes
    -----
    ``Loss`` is designed

    Subclass of ``Loss`` must implements ``__call__()`` and ``grad()``.
    """
    def __call__(self, x: Array, y: Array) -> Array:
        """
        Compute Loss

        Parameters
        ----------
        x : vulkpy.Array
            Input features
        y : vulkpy.Array
            Output target/label

        Returns
        -------
        loss : vulkpy.Array
            Loss

        Notes
        -----
        Subclass must implement this method.
        """
        raise NotImplementedError

    def grad(self) -> Array:
        """
        Compute Gradient

        Returns
        -------
        grad : vulkpy.Array
            Gradient

        Notes
        -----
        Subclass must implement this method.
        """
        raise NotImplementedError

class Regularizer:
    """
    Abstract base class for Regularizer

    See Also
    --------
    vulkpy.nn.Lasso : Lasso (L1) Regularizer
    vulkpy.nn.Ridge : Ridge (L2) Regularizer
    vulkpy.nn.Elastic : Elastic (L1 + L2) Regularizer

    Notes
    -----
    Subclass must implement ``loss()`` and ``grad()``.
    """
    def loss(self, param: Array) -> Array:
        """
        Compute Regularizer Loss

        Parameters
        ----------
        param : vulkpy.Array
            Parameters

        Returns
        -------
        loss : vulkpy.Array
            Loss

        Notes
        -----
        Subclass must implement this method.
        """
        raise NotImplementedError

    def grad(self, param: Array) -> Array:
        """
        Compute Gradient

        Parameters
        ----------
        param : vulkpy.Array
            Parameters

        Returns
        -------
        grad : vulkpy.Array
            Gradient

        Notes
        -----
        Subclass must implement this method.
        """
        raise NotImplementedError


class Module:
    """
    Abstract base class for Module

    See Also
    --------
    vulkpy.nn.Dense : Dense Layer (subclass)
    vulkpy.nn.ReLU : ReLU Layer (subclass)
    vulkpy.nn.Sigmoid : Sigmoid Layer (subclass)
    vulkpy.nn.Softmax : Softmax Layer (subclass)
    vulkpy.nn.Sequence : Sequential Model

    Notes
    -----
    ``Module`` is designed to for Neural Network Layer.

    Subclass must implement ``forward()`` and ``backward()``, and can implement
    ``zero_grad()`` and ``update()`` when it is necessary.
    """

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
        """
        Forward Calculation

        Parameters
        ----------
        x : vulkpy.Array
            Input features

        Returns
        -------
        y : vulkpy.Array
            Output

        Notes
        -----
        Subclass must implement this method.
        """
        raise NotImplementedError

    def backward(self, dy: Array) -> Array:
        """
        Backward Calculation

        Parameters
        ----------
        dy : vulkpy.Array
            dL/dy propagated from following layer

        Returns
        -------
        dx : vulkpy.Array
            dL/dx propagated to previous layer

        Notes
        -----
        Subclass must implement this method.
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Reset accumulated gradients to 0.

        Notes
        -----
        Base class implement no-operation.
        Subclass can customize this method.
        """
        pass

    def update(self):
        """
        Update parameters based on accumulated gradients

        Notes
        -----
        Base class implement no-operation.
        Subclass can customize this method.
        """
        pass
