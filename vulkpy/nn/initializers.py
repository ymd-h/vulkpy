from __future__ import annotations
from typing import Optional

import numpy as np

from vulkpy.vkarray import GPU
from vulkpy.random import Xoshiro128pp


class HeNormal:
    """
    He Normal Initializer

    Note
    ----
    .. math:: \sigma = \sqrt(2/input_dim)
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
        return self.rng.normal(shape=shape, stddev=self.stddev)
