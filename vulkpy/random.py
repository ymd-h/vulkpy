from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np

from . import _vkarray
from . import vkarray as vk
from .util import getShader

__all__ = ["Xoshiro128pp"]

class _ConvertMixin:
    _box_muller = getShader("prng_box_muller.spv")
    _ibox_muller = getShader("prng_ibox_muller.spv")

    def normal(self, *,
               shape: Optional[Iterable[int]] = None,
               buffer: Optional[vk.Array] = None,
               mean: float = 0.0,
               stddev: float = 1.0) -> vk.Array:
        """
        Generate Normal Distributing numbers

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

        Raises
        ------
        ValueError
            If neither `shape` or `buffer` are specified

        Notes
        -----
        This method first generates [0, 1) uniform random numbers,
        then transforms them to normal distribution with Box-Muller method.
        Box-Muller might have problem in terms of random number quality,
        however, it is quite GPU friendly.
        """
        _local_size = 64
        if buffer is None:
            if shape is None:
                raise ValueError("One of `shape` and  `buffer` must be specified.")

            buffer = vk.Array(self._gpu, shape=shape)
        else:
            # For safety, we wait output buffer job.
            buffer.wait()

        n = int(np.prod(buffer.shape))
        floor_n = n // 2
        dshape = _vkarray.DataShape(floor_n, 1, 1)
        p = _vkarray.VectorScalar2Params(n, mean, stddev)
        if n % 2 == 0:
            # Even: Reuse `buffer`
            buffer = self.random(buffer=buffer)
            buffer.job = self._gpu._submit(self._ibox_muller,
                                           _local_size, 1, 1,
                                           [buffer], dshape, p)
        else:
            # Odd: Require additional space for intermediate [0, 1)
            rng = self.random(shape=floor_n + 1)
            buffer.job = self._gpu._submit(self._box_muller,
                                           _local_size, 1, 1,
                                           [rng, buffer], dshape, p)

        return buffer

class Xoshiro128pp(_ConvertMixin):
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

        n = int(np.prod(buffer.shape))
        buffer.job = self.rng.random(n, buffer.buffer.info())
        return buffer
