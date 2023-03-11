"""
Neural Network Optimizer Module (:mod:`vulkpy.nn.optimizers`)
=============================================================
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Union

from wblog import getLogger

from vulkpy.vkarray import GPU, Array
from .core import Optimizer, OptimizerState

__all__ = [
    "SGD", "SGDState",
    "Adam", "AdamState",
    "Optimizer", "OptimizerState",
]

logger = getLogger()


@dataclass
class SGDState(OptimizerState):
    """
    Optimizer State for SGD

    Attributes
    ----------
    opt : vulkpy.SGD
        SGD Optimizer
    """
    opt: SGD

    def grad2diff(self, grad: Array) -> Array:
        """
        Compute diff from gradient

        Parameters
        ----------
        grad : vulkpy.Array
            Gradient

        Returns
        -------
        diff : vulkpy.Array
            Update diff
        """
        return (-self.opt.lr) * grad

@dataclass
class SGD(OptimizerState):
    """
    Stachostic Gradient Decent Optimizer

    Use constant learning rate

    Attributes
    ----------
    lr : float
        Learning rate
    """
    lr: float

    def __post_init__(self):
        logger.debug(f"SGD(lr={self.lr})")

    def init_state(self, shape: Iterable[int]) -> SGDState:
        """
        Initialize Optimizer state

        Parameters
        ----------
        shape : iterable of ints
            Shape of parameter

        Returns
        -------
        SGDState
            Optimizer state

        Notes
        -----
        Currently SGDState is empty, however,
        we might add some field like momentum in future.
        """
        return SGDState(self)


@dataclass
class AdamState(OptimizerState):
    """
    Optimizer State for Adam

    Attributes
    ----------
    opt : vulkpy.Adam
        Adam Optimizer
    m : vulkpy.Array
        Adam Parameter
    v : vulkpy.Array
        Adam Parameter
    beta1t : float
        ``beta1 ** t``
    beta2t : float
        ``beta2 ** t``
    """
    opt: Adam
    m: Array
    v: Array
    beta1t: float = 1.0
    beta2t: float = 1.0

    def __post_init__(self):
        self.m[:] = 0.0
        self.v[:] = 0.0

    def grad2diff(self, grad: Array) -> Array:
        """
        Compute diff from gradient

        Parameters
        ----------
        grad : vulkpy.Array
            Gradient

        Returns
        -------
        diff : vulkpy.Array
            Update diff
        """
        self.m *= self.opt.beta1
        self.m += (1 - self.opt.beta1) * grad        # Allocate

        self.v *= self.opt.beta2
        self.v += (1 - self.opt.beta2) * (grad ** 2) # Allocate

        self.beta1t *= self.opt.beta1
        self.beta2t *= self.opt.beta2

        mhat = self.m / (1 - self.beta1t) # Allocate
        vhat = self.v / (1 - self.beta2t) # Allocate

        vhat.sqrt(inplace=True) # sqrt(vhat)
        vhat += self.opt.eps    # sqrt(vhat) + eps

        mhat *= (-self.opt.lr)  # -lr * mhat
        mhat /= vhat            # -lr * mhat / (sqrt(vhat) + eps)

        return mhat


@dataclass
class Adam(Optimizer):
    """
    Adam Optimizer

    Attributes
    ----------
    gpu : vulkpy.GPU
        GPU
    lr : float, optional
        Learning rate. The default is 0.001
    beta1 : float, optional
        Adam parameter. The defaults are 0.9.
    beta2 : float, optional
        Adam parameter. The defaults is 0.999.
    eps : float, optional
        Adam parameter. The defaults is, 1e-8.
    """
    gpu: GPU
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        logger.debug(f"Adam(lr={self.lr}, beta1={self.beta1}, " +
                     f"beta2={self.beta2}, eps={self.eps})")

    def init_state(self, shape: Iterable[int]) -> AdamState:
        """
        Initialize Optimizer state

        Parameters
        ----------
        shape : iterable of ints
            Shape of parameter

        Returns
        -------
        AdamState
            Optimizer state
        """
        return AdamState(opt=self,
                         m=Array(self.gpu, shape=shape),
                         v=Array(self.gpu, shape=shape))
