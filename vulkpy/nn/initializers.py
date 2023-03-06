"""
Neural Network Initializer Module (:mod:`vulkpy.nn.initializers`)
=================================================================
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from vulkpy.vkarray import GPU, Array
from vulkpy.random import Xoshiro128pp


__all__ = ["Constant", "HeNormal"]

class Initializer:
    def __call__(self, gpu: GPU, shape: Iterable[int]) -> Array:
        raise NotImplementedError


class Constant(Initializer):
    """
    Constant Initializer
    """
    def __init__(self, value: float):
        """
        Initialize Constant Initializer

        Parameters
        ----------
        value : float
            Constant value
        """
        self.value = value

    def __call__(self, gpu: GPU, shape: Iterable[int]) -> Array:
        """
        Initialize new parameters

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU
        shape : iterable of ints
            Parameter shape
        """
        p = Array(gpu, shape=shape)
        p[:] = self.value
        return p


class HeNormal(Initializer):
    r"""
    He Normal Initializer

    Note
    ----
    Standard deviation :math:`\sigma` is following;

    .. math:: \sigma = \sqrt{2/d_{\text{in}}}
    """
    def __init__(self, gpu: GPU, input_dim: int, *, seed: Optional[int] = None):
        """
        Initialize He Normal Initializer

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU
        input_dim : int
            Input dimension
        seed : int, optional
            Initial seed for PRNG
        """
        self.rng = Xoshiro128pp(gpu, seed=seed)
        self.stddev = np.sqrt(2 / input_dim)

    def __call__(self, gpu: GPU, shape: Iterable[int]):
        """
        Initialize new parameters

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU
        shape : iterable of ints
            Parameter shape
        """
        return self.rng.normal(shape=shape, stddev=self.stddev)
