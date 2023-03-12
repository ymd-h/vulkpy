"""
Neural Network Regularizer Module (:mod:`vulkpy.nn.regularizers`)
=================================================================
"""
from __future__ import annotations
from typing import Iterable, Tuple
from typing_extensions import Protocol

import wblog

from vulkpy import Array
from .core import Regularizer

__all__ = [
    "Lasso",
    "Ridge",
    "Elastic",
]

logger = wblog.getLogger()


class Lasso(Regularizer):
    r"""
    Lasso (L1) Regularization

    Notes
    -----
    .. math::

         L = coeff \times \sum_i |W_i|\\
         dL/dW_i = coeff \times sign(W_i)
    """
    def __init__(self, coeff: float = 1.0):
        """
        Initialize Lasso Regularizer

        Parameters
        ----------
        coeff : float, optional
            L1 Coefficient
        """
        logger.debug(f"Lasso(L1={coeff})")
        self.coeff: float = coeff

    def loss(self, param: Array) -> Array:
        """
        L1 Regularization Loss

        Parameters
        ----------
        param : vulkpy.Array
            Parameter

        Returns
        -------
        loss : vulkpy.Array
            L1 Regularization Loss
        """
        L = param.abs().sum()
        L *= self.coeff
        return L

    def grad(self, param: Array) -> Array:
        """
        Gradient of L1 Regularization Loss

        Parameters
        ----------
        param : vulkpy.Array
            Parameter

        Returns
        -------
        dW : vulkpy.Array
            Gradient for L1 Regularization Loss
        """
        return self.coeff * param.sign()

class Ridge(Regularizer):
    r"""
    Ridge (L2) Regularization

    Notes
    -----
    .. math::

         L = coeff \times \sum_i |W_i|^2\\
         dL/dW_i = 2 coeff \times W_i
    """
    def __init__(self, coeff: float = 1.0):
        """
        Initialize Ridge Regularizer

        Parameters
        ----------
        coef : float, optional
            L2 Coefficient
        """
        logger.debug(f"Ridge(L2={coeff})")
        self.coeff: float = coeff

    def loss(self, param: Array) -> Array:
        """
        L2 Regularization Loss

        Parameters
        ----------
        param : vulkpy.Array
            Parameter

        Returns
        -------
        loss : vulkpy.Array
            L2 Regularization Loss
        """
        L = (param ** 2).sum()
        L *= self.coeff
        return L

    def grad(self, param: Array) -> Array:
        """
        Gradient of L2 Regularization Loss

        Parameters
        ----------
        param : vulkpy.Array
            Parameter

        Returns
        -------
        dW : vulkpy.Array
            Gradient for L2 Regularization Loss
        """
        return (2 * self.coeff) * param

class Elastic(Regularizer):
    """
    Elastic (L1 + L2) Regularization
    """
    def __init__(self, L1: float = 1.0, L2: float = 1.0):
        """
        Initialize Elastic Regularizer

        Parameters
        ----------
        L1 : float, optional
            L1 Coefficient
        L2 : float, optional
            L2 Coefficient
        """
        self.L1 = Lasso(L1)
        self.L2 = Ridge(L2)

    def loss(self, param: Array) -> Array:
        """
        L1 + L2 Regularization Loss

        Parameters
        ----------
        param : vulkpy.Array
            Parameter

        Returns
        -------
        loss : vulkpy.Array
            L1 + L2 Regularization Loss
        """
        return self.L1.loss(param) + self.L2.loss(param)

    def grad(self, param: Array) -> Array:
        """
        Gradient of L1 + L2 Regularization Loss

        Parameters
        ----------
        param : vulkpy.Array
            Parameter

        Returns
        -------
        dW : vulkpy.Array
            Gradient for L1 + L2 Regularization Loss
        """
        return self.L1.grad(param) + self.L2.grad(param)
