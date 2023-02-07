from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np

from . import _vkarray
from . import vkarray as vk
from .util import getShader

__all__ = ["Xoshiro128pp"]

class Xoshiro128pp:
    _spv = getShader("prng_xoshiro128pp.spv")

    def __init__(self, gpu: vk.GPU, size: int = 64, *, seed: Optional[int] = None):
        """
        xoshiro128++: Pseudo Random Number Generator

        Parameters
        ----------
        gpu : vulkpy.GPU
            GPU where PRNG allocates
        size : int
            Number of internal states. These states generate random number parallelly.
        seed : int, optional
            Random seed. If ``None`` (default), use hardware random instead.
        """
        self._gpu = gpu

        if seed is None:
            self.rng = _vkarray.Xoshiro128pp(self._gpu.gpu, self._spv, size)
        else:
            self.rng = _vkarray.Xoshiro128pp(self._gpu.gpu, self._spv, size, seed)

    def random(self, *,
               shape: Optional[Iterable[int]] = None,
               buffer: Optional[vk.Array] = None) -> vk.Array:
        """
        Generate [0, 1) floating numbers

        Parameters
        ----------
        shape : iterable of ints, optional
            If specified, new ``vulkpy.Array`` with ``shape`` will be returned.
        buffer : vulkpy.Array
            If specified, generated numbers will be stored.

        Returns
        -------
        vulkpy.Array
            Array which will get random numbers.
        """
        if buffer is None:
            if shape is None:
                raise ValueError("One of `shape` and  `buffer` must be specified.")

            buffer = vk.Array(self._gpu, shape=shape)
        else:
            # For safety, we wait output buffer job.
            buffer.wait()

        n = buffer.shape.prod()
        buffer.job = self.rng.random(n, buffer.buffer.info())
        return buffer
