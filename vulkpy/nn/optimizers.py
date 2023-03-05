from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Union

from wblog import getLogger

from vulkpy.vkarray import GPU, Array

__all__ = [
    "SGD", "SGDState",
    "Adam", "AdamState",
    "Optimizer", "OptimizerState",
]

logger = getLogger()

@dataclass
class SGDState:
    pass

class SGD:
    """
    Stachostic Gradient Decent Optimizer

    Use constant learning rate
    """
    def __init__(self, lr: float):
        """
        Initialize SGD

        Parameters
        ----------
        lr : float
            Learning rate
        """
        self.lr = lr
        logger.debug(f"SGD(lr={lr})")

    def __call__(self, grad: Array, state: SGDState) -> Array:
        """
        Compute update diff from gradient

        Parameters
        ----------
        grad : vulkpy.Array
            Gradient
        state : SGDState
            Optimizer state

        Returns
        -------
        vulkpy.Array
            Update diff
        """
        return (-self.lr) * grad

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
        return SGDState()


@dataclass
class AdamState:
    m: Array
    v: Array
    beta1t: float = 1.0
    beta2t: float = 1.0

class Adam:
    """
    Adam Optimizer
    """
    def __init__(self,
                 gpu: GPU,
                 lr: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):
        """
        Initialize Adam Optimizer

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU
        lr : float, optional
            Learning rate. The default is 0.001
        beta1, beta2, eps : float, optional
            Adam parameters. The defaults are 0.9, 0.999, 1e-8, respectively.
        """
        self.gpu = gpu
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        logger.debug(f"Adam(lr={lr}, beta1={beta1}, beta2={beta2}, eps={eps})")

    def __call__(self, grad: Array, state: AdamState) -> Array:
        """
        Compute update diff from gradient

        Parameters
        ----------
        grad : vulkpy.Array
            Gradient
        state : AdamState
            Optimizer state

        Returns
        -------
        vulkpy.Array
            Update diff
        """
        state.m *= self.beta1
        state.m += (1 - self.beta1) * grad # Allocate

        state.v *= self.beta2
        state.v += (1 - self.beta2) * (grad ** 2) # Allocate

        state.beta1t *= self.beta1
        state.beta2t *= self.beta2

        mhat = state.m / (1 - state.beta1t) # Allocate
        vhat = state.v / (1 - state.beta2t) # Allocate

        vhat.sqrt(inplace=True) # sqrt(vhat)
        vhat += self.eps        # sqrt(vhat) + eps

        mhat *= (-self.lr)      # -lr * mhat
        mhat /= vhat            # -lr * mhat / (sqrt(vhat) + eps)

        return mhat

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
        state = AdamState(m=Array(self.gpu, shape=shape),
                          v=Array(self.gpu, shape=shape))
        state.m[:] = 0
        state.v[:] = 0
        return state


Optimizer = Union[
    SGD,
    Adam,
]

OptimizerState = Union[
    SGDState,
    AdamState,
]
