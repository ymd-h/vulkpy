from __future__ import annotations
from dataclass import dataclass
from typing import Iterable

from vulkpy.vkarray import Array


@dataclass
class SGDState:
    pass

class SGD:
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, grad: Array, state: SGDState) -> Array:
        return (-self.lr) * grad

    def init_state(self, shape: Iterable[int]) -> SGDState:
        return SGDState()


@dataclass
class AdamState:
    m: Array
    v: Array
    beta1t: float = 1.0
    beta2t: float = 1.0

class Adam:
    def __init__(self,
                 gpu: GPU,
                 lr: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):
        self.gpu = gpu
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def __call__(self, grad: Array, state: AdamState) -> Array:
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
        state = AdamState(m=Array(self.gpu, shape=shape),
                          v=Array(self.gpu, shape=shape))
        state.m[:] = 0
        state.v[:] = 0
        return state
