"""
Neural Network Regularizer Module (:mod:`vulkpy.nn.regularizers`)
=================================================================
"""
from __future__ import annotations
from typing import Iterable, Tuple
from typing_extensions import Protocol

from vulkpy import Array
from .core import Regularizer
from .parameters import Parameter

__all__ = [
    "Ridge",
    "Lasso",
    "Elastic",
]


class _Ridge1(Regularizer):
    def __init__(self, param: Parameter, coeff: float = 1.0):
        self.coeff: float = coeff
        self.param: Parameter = param

    def loss(self) -> Array:
        L = (self.param.value ** 2).sum()
        L *= self.coeff
        return L

    def add_grad(self):
        self.param.add_grad(2 * self.coeff * self.param.value)

class _Lasso1(Regularizer):
    def __init__(self, param: Parameter, coeff: float = 1.0):
        self.coeff: float = coeff
        self.param: Parameter = param

    def loss(self) -> Array:
        L = self.param.value.abs().sum()
        L *= self.coeff
        return L

    def add_grad(self):
        self.param.add_grad(self.coeff * self.param.value.sign())


class SumRegularizerProtocol(Protocol):
    @property
    def R(self) -> Tuple[Regularizer, ...]: ...

class _SumRegularizer(Regularizer):
    def loss(self: SumRegularizerProtocol) -> Array:
        """
        Regularization Loss

        Returns
        -------
        loss : vulkpy.Array
            Regularization Loss
        """
        L = self.R[0].loss()
        for _R in self.R[1:]:
            L += _R.loss()

        L.reshape((1, 1))
        return L

    def add_grad(self: SumRegularizerProtocol):
        """
        Add regularization gradient to parameter
        """
        for _R in self.R:
            _R.add_grad()

class Ridge(_SumRegularizer):
    r"""
    Ridge (L2) Regularization

    Notes
    -----
    .. math::

         L = coeff \times \sum_i |W_i|^2\
         dL/dW_i = 2 coeff \times W_i
    """
    def __init__(self, params: Iterable[Tuple[float, Parameter]]):
        """
        Initialize Ridge Regularizer

        Parameters
        ----------
        params : iterable of tuple[float, vulkpy.nn.Parameter]
            Sets of L2 coeff and parameter.

        Raises
        ------
        ValueError
            When params is empty
        """
        self.R: Tuple[Regularizer, ...] = tuple(_Ridge1(p, c) for c, p in params)
        if len(self.R) < 1:
            raise ValueError(f"params must not be empty.")

class Lasso(_SumRegularizer):
    r"""
    Lasso (L1) Regularization

    Notes
    -----
    .. math::

         L = coeff \times \sum_i |W_i|\
         dL/dW_i = coeff \times sign(W_i)
    """
    def __init__(self, params: Iterable[Tuple[float, Parameter]]):
        """
        Initialize Lasso Regularizer

        Parameters
        ----------
        params : iterable of tuple[float, vulkpy.nn.Parameter]
            Sets of L1 coeff and parameter.

        Raises
        ------
        ValueError
            When params is empty
        """
        self.R: Tuple[Regularizer, ...] = tuple(_Lasso1(p, c) for c, p in params)
        if len(self.R) < 1:
            raise ValueError(f"params must not be empty.")

class Elastic(_SumRegularizer):
    """
    Elastic (L1 + L2) Regularization
    """
    def __init__(self, params: Iterable[Tuple[float, float, Parameter]]):
        """
        Initialize Elastic Regularizer

        Paramters
        ---------
        params : iterable of tuple[float, float, vulkpy.nn.Parameter]
            Sets of L1 coeff, L2 coeff, and parameter.

        Raises
        ------
        ValueError
            When params is empty
        """
        self.R: Tuple[Regularizer, ...] = (tuple(_Lasso1(p, c)
                                                 for c, _, p in params) +
                                           tuple(_Ridge1(p, c)
                                                 for _, c, p in params))
        if len(self.R) < 1:
            raise ValueError(f"params must not be empty.")
